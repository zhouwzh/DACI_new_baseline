from typing import List, Optional, Tuple, Dict
import os
import socket
import copy
import time
from abc import abstractmethod
import gc
import sys
import queue
from itertools import chain
from tqdm import tqdm

import numpy as np
import transformers
import torch
import torch.distributed
from torch import multiprocessing as mp
from torch.multiprocessing import Process

import cgen.dist_utils
from cgen.models import create_model, PinnedModelWeights
from cgen.models.weight_utils import hf_model_weights_iterator

from cgen.config import DistConfig, CacheConfig, Plan
from cgen.dist_utils import create_process_groups
from cgen.utils import logger, get_open_port, logger, Device, p2p_warm_up
from cgen.layers import ModelInput
from cgen.layers.cache import KVCacheManager, PageTable, SharedCacheManager, SharedCacheMetadata
from cgen.schedule import Scheduler
from cgen.prefetch import Prefetcher, PrefetchTask


class PerformanceRecord:
    def __init__(self):
        self.prefill_time = []
        self.decode_time = []

    @property
    def avg_decode_time(self):
        return np.mean(self.decode_time)
    
    @property
    def tot_decode_time(self):
        return sum(self.decode_time)

    def add_decode(self, t: float, bsz: int = 0):
        logger.debug(f"decode time: {t: .1f} ms")
        if bsz > 0:
            logger.debug(f"decode time: {bsz * 1e3 / t: .2f} tokens/s")
        self.decode_time.append(t)

    @property
    def avg_prefill_time(self):
        return np.mean(self.prefill_time)

    @property
    def tot_prefill_time(self):
        return sum(self.prefill_time)

    def add_prefill(self, t: float):
        self.prefill_time.append(t)

    def print(self):
        print(
            f"avg prefill time: {self.avg_prefill_time : .1f} ms | avg decode time: {self.avg_decode_time : .1f} ms"
        )
        print(
            f"tot prefill time: {self.tot_prefill_time : .1f} ms | tot decode time: {self.tot_decode_time : .1f} ms"
        )

class Worker:
    def __init__(
        self,
        model_name,
        init_method,
        queue,
        return_queue,
        prefetch_ret_q,
        dist_config_decode: DistConfig,
        dist_config_prefill: Optional[DistConfig] = None,
        log_level: str = "INFO",
        dummy_weights: bool = False
    ):
        self.log_level = log_level
        self.queue = queue
        self.return_queue = return_queue
        self.dist_config_decode: DistConfig = dist_config_decode

        dist_config_prefill = dist_config_prefill or dist_config_decode
        self.dist_config_prefill: DistConfig = dist_config_prefill
        self.dummy_weights = dummy_weights

        self.device = torch.device(f'cuda:{dist_config_prefill.rank()}')
        torch.cuda.set_device(self.device)
        # torch.set_num_threads(12)

        logger.info("loading model...")
        logger.info("partitioning model...")


        torch.distributed.init_process_group(
            init_method=init_method,
            rank=dist_config_prefill.rank(),
            world_size=dist_config_prefill.world_size(),
        )
        torch.distributed.barrier()
        create_process_groups(dist_config_prefill, dist_config_decode, self.device)
        p2p_warm_up(self.dist_config_prefill.rank(), self.dist_config_prefill.world_size(), "cuda")

        model_config = transformers.AutoConfig.from_pretrained(model_name)
        self.config = model_config
        self.model_name = model_name
        self.model = self._load_model(dist_config_prefill)

        if not dummy_weights:
            weight_loader = hf_model_weights_iterator(model_name)
            self._pinned_model_prefill = self.model.pinned_model_weights(weight_loader, model_config, dist_config_prefill)
            weight_loader = hf_model_weights_iterator(model_name)
            self._pinned_model_decode = self.model.pinned_model_weights(weight_loader, model_config, dist_config_decode)

            del weight_loader

        self._current_dist_config = self.dist_config_prefill
        self.cache_set = None

        self._pipeline_finished = False
        self.prefetch_ret_q = prefetch_ret_q

        self._prefill_time = 0
        self._decode_time = 0
     
    def rank(self):
        return self._current_dist_config.rank()
    
    def _load_model(self, dist_config:DistConfig, weight_loader=None, embed_tokens=None, lm_head=None):
        if not self.dummy_weights and weight_loader is None:
            weight_loader = hf_model_weights_iterator(self.model_name)
        if hasattr(self, 'model'):
            del self.model
        model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        model = create_model(
            model_config, dist_config, self.device, embed_tokens=embed_tokens, lm_head=lm_head
        )
        t0 = time.time()
        if not self.dummy_weights:
            model.load_weights(weight_loader) 
        logger.debug(
            f"model loaded, memory occupied: {torch.cuda.memory_allocated(self.device) / (1024**3):.2f} GB"
        )
        return model
    
    def to_prefill(self):
        t = time.time()
        embed_tokens = self.model.model.embed_tokens 
        lm_head = self.model.lm_head
        del self.model
        self._current_dist_config = self.dist_config_prefill
        self._pipeline_finished = False
        weight_loader = None if self.dummy_weights else self._pinned_model_prefill.weight_loader()
        self.model = self._load_model(self.dist_config_prefill, weight_loader, embed_tokens=embed_tokens, lm_head=lm_head)
        if self.rank() == 0:
            logger.info(f"convert from decode to prefill time: {time.time() - t: .2f} s")
        cgen.dist_utils.to_prefill()
        return 'finish'

    def to_decode(self):
        t = time.time()
        embed_tokens = self.model.model.embed_tokens 
        lm_head = self.model.lm_head
        del self.model
        self._current_dist_config = self.dist_config_decode
        weight_loader = None if self.dummy_weights else self._pinned_model_decode.weight_loader()
        self.model = self._load_model(self.dist_config_decode, weight_loader, embed_tokens=embed_tokens, lm_head=lm_head)
        if self.rank() == 0:
            logger.info(f"convert from prefill to decode time: {time.time() - t: .2f} s")
        cgen.dist_utils.to_decode()
        return 'finish'

    def main_loop(self):
        logger.remove()
        logger.add(sys.stderr, level=self.log_level)
        while True:
            torch.cuda.nvtx.range_push("wait for inst")
            inst, args, kwargs = self.queue.get()
            torch.cuda.nvtx.range_pop()
            if inst == 'quit':
                # torch.distributed.destroy_process_group()
                del self._shared_cache_manager
                return
            else:
                ret = getattr(self, inst)(*args, **kwargs)
                if ret is not None and self._current_dist_config.should_return():
                    self.return_queue.put(ret)

    def execute_model(
        self,
        inputs: ModelInput,
    ):
        inputs = copy.deepcopy(inputs)
        t = time.time()
        rank = self._current_dist_config.rank()
        logger.debug(f"rank: {rank} | batch_id: {inputs.batch_id} | is_prefill: {inputs.is_prefill}")
        # if self._current_dist_config.should_return():
            # return [0]*len(inputs.seq_ids)
        # else:
            # return None
        start_t = time.time()
        torch.cuda.nvtx.range_push("model execution")
        with torch.no_grad():
            cache_set = self.cache_set
            if inputs.is_prefill:
                cache_set = self._shared_cache_manager
            out = self.model(inputs, cache_set)
            if self._current_dist_config.should_return():
                if out is None:
                    ret = []
                else:
                    indptr = inputs.q_indptr[1:] - 1
                    ret = torch.argmax(out[indptr], dim=-1).cpu().share_memory_()
            else:
                ret = None
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        dur = (time.time() - start_t) * 1e3

        logger.debug(f"schedule {len(inputs.seq_ids)} {inputs.seq_ids} | is_prefill: {inputs.is_prefill} | ntokens: {inputs.x.shape[0]} | {dur: .1f} ms")
        if inputs.is_prefill:
            self._prefill_time += dur
        else:
            self._decode_time += dur
        return ret

    def init_kvcache(self, config: CacheConfig):
        if self.cache_set is not None:
            self.cache_set = None
        logger.info("initializing kvcache")
        dist_config = self.dist_config_decode
        nlayers = self.config.num_hidden_layers // dist_config.pp_size
        nheads = (
            getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
            // dist_config.tp_size
        )
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.cache_set = KVCacheManager(
            dist_config,
            nlayers,
            config.num_pages,
            config.page_size,
            nheads,
            head_dim,
            self.device,
            torch.half,
        )

        logger.info(
            f"kvcache initialized, memory occupied: {torch.cuda.memory_allocated(self.device) / (1024**3):.2f} GB"
        )
        return 'finish'
    
    def init_shared_cache(self, max_len, shared_mem: torch.Tensor, config: CacheConfig):
        logger.info("start allocating shared kvcache")
        nlayers = self.config.num_hidden_layers // self.dist_config_prefill.pp_size
        nheads =  getattr(self.config, "num_key_value_heads", self.config.num_attention_heads) \
            // self.dist_config_prefill.tp_size
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self._shared_cache_manager = SharedCacheManager(
            nlayers,
            self.dist_config_prefill,
            self.dist_config_decode,
            shared_mem,
            config.shared_cache_max_prompt_len,
            nheads,
            head_dim,
            self.device,
            torch.half
        )
        logger.info("launch prefetcher")
        logger.info(f"{torch.cuda.current_device()} {torch.cuda.current_stream()} ")
        self.prefetcher = Prefetcher(max_len, self.prefetch_ret_q, self.cache_set, shared_mem, self.dist_config_decode, self.device)
        return 'finish'

    def ping(self):
        return 'finish'

    def perf_record(self):
        return self._perf_record
    
    def reset_perf_record(self):
        self._perf_record = PerformanceRecord()

    def add_prefetch_task(self, task: PrefetchTask):
        self.prefetcher.add_task(task)
    
    def _clear_timing(self):
        self._prefill_time = 0
        self._decode_time = 0
        return 'finish'

def _work_proc(*args, **kwargs):
    worker = Worker(*args, **kwargs)
    worker.main_loop()

class LLMEngine():
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        plan: Plan,
        log_level: str = "INFO",
        dummy_weights: bool = False,
    ):
        logger.remove()
        logger.add(sys.stderr, level=log_level)
        self.model_name = model_name
        self.model_config = transformers.AutoConfig.from_pretrained(model_name)

        self.plan = plan
        self.dist_config_decode = plan.dist_config_decode
        self.dist_config_prefill = plan.dist_config_prefill or plan.dist_config_decode

        assert self.dist_config_prefill.pp_size >= self.dist_config_decode.pp_size

        self.n_gpus = plan.dist_config_decode.world_size()
        self.is_prefill = True

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        ctx = mp.get_context('spawn')
        mp.set_sharing_strategy('file_system')
        self._queues = [ctx.Queue() for _ in range(self.n_gpus)]
        self._return_queue = ctx.Queue()
        self.prefetch_ret_q = ctx.Queue()
        port = get_open_port()
        logger.info(f"init process group using port {port}")

        self._procs = [
            ctx.Process(
                target=_work_proc,
                args=(
                    model_name,
                    f"tcp://localhost:{port}",
                    self._queues[i],
                    self._return_queue,
                    self.prefetch_ret_q,
                    plan.dist_config_decode.with_rank(i),
                    self.dist_config_prefill.with_rank(i),
                    log_level,
                    dummy_weights
                ),
                # daemon=True
            )
            for i in range(self.n_gpus)
        ]
        for p in self._procs:
            p.start()

        cache_config = plan.cache_config
        self._init_tokenizer(model_name, tokenizer_name)
        self.call_and_wait('ping')
        self.call_and_wait('init_kvcache', cache_config)
        self.pagetable = PageTable(cache_config.num_pages, cache_config.page_size, self.plan.prefill_threshold)
        self.shared_table = SharedCacheMetadata(cache_config.shared_cache_tokens)

        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        shared_mem_kv_num = cache_config.shared_cache_tokens 
        self.shared_mem = torch.empty(
            [self.model_config.num_hidden_layers, 2, self.model_config.num_key_value_heads, shared_mem_kv_num, head_dim], dtype=torch.half
        )
        self.call_and_wait('init_shared_cache', self.plan.max_prefill_tokens, self.shared_mem, cache_config)



    def join(self):
        for q, p in zip(self._queues, self._procs):
            q.put(('quit', [], {}))
        for p in self._procs:
            p.join()
        del self.shared_mem

    def call_and_wait(self, inst, *args, **kwargs):
        for q in self._queues:
            q.put((inst, args, kwargs))
        return copy.deepcopy(self._return_queue.get())
    
    def call_no_wait(self, inst, *args, workers=None, **kwargs):
        logger.debug(f"{inst=} {workers=}")
        queues = self._queues if workers is None else [self._queues[i] for i in workers]
        for q in queues:
            q.put((inst, args, kwargs))

    def _clear_timing(self):
        self.call_and_wait('_clear_timing')
    
    def get_result(self):
        return copy.deepcopy(self._return_queue.get())

    def _clear_pipeline(self, scheduler, task_queue):
        logger.debug("clear pipeline")
        while task_queue:
            o = self.get_result()
            task = task_queue.pop(0)
            scheduler.append_output(task, o)
        logger.debug("pipeline cleared")

    def _init_tokenizer(self, model_name, tokenizer_name):
        if model_name == 'DEBUG':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                'meta-llama/Llama-2-7b-hf'
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def run(
        self,
        inputs: List[str],
        max_tokens: List[int],
    ):
        input_ids = self.tokenizer.batch_encode_plus(inputs).input_ids
        scheduler = Scheduler(self.plan, self.pagetable, self.shared_table, input_ids, max_tokens, self.prefetch_ret_q)
        task_queue = []
        empty_cnt = 0
        pbar = tqdm(
            total=len(inputs),
            desc="Processed prompts",
            dynamic_ncols=True,
            postfix=(f"est. speed input: {0:.2f} toks/s, "
                        f"output: {0:.2f} toks/s"),
            smoothing=0
        )
        finished_reqs = len(scheduler.finished_reqs)
        input_tokens = 0
        output_tokens = 0
        itr_cnt = 0
        detokenized_seq: Dict[int, str] = {}
        start_t = time.time()
        while not scheduler.finished():
            logger.debug("next iteration")
            itr_cnt += 1

            # if not scheduler.is_prefill:
            prefetch_tasks = scheduler.schedule_prefetch()
            for task in prefetch_tasks:
                self.call_no_wait("add_prefetch_task", task)

            batch = scheduler.schedule()
            if batch.is_prefill != self.is_prefill:
                # model transition
                # clear the pipeline
                self._clear_pipeline(scheduler, task_queue)
                if batch.is_prefill:
                    self.call_and_wait("to_prefill")
                else:
                    self.call_and_wait("to_decode")
            self.is_prefill = batch.is_prefill

            if len(batch.seq_ids) == 0:
                empty_cnt += 1
                if empty_cnt > self.dist_config_prefill.pp_size:
                    raise RuntimeError("Deadlock")
            else:
                empty_cnt = 0
            
            task_queue.append(batch.seq_ids)
            self.call_no_wait(
                "execute_model", batch
            )
                            
            if len(task_queue) >= scheduler.dist_config.pp_size:
                o = self.get_result()
                task = task_queue.pop(0)
                scheduler.append_output(task, o)
            
            if batch.is_prefill:
                input_tokens += batch.x.shape[0]
            else:
                output_tokens += batch.x.shape[0]
            in_spd = input_tokens / pbar.format_dict["elapsed"]
            out_spd = output_tokens / pbar.format_dict["elapsed"]
            pbar.postfix = (
                f"est. speed input: {in_spd:.2f} toks/s, "
                f"output: {out_spd:.2f} toks/s")
            pbar.update(len(scheduler.finished_reqs) - finished_reqs)
            finished_reqs = len(scheduler.finished_reqs)
        
        self._clear_pipeline(scheduler, task_queue)
        pbar.close()
        
        seqs = [seq for _, seq in scheduler.new_tokens.items()]
        t0 = time.time()
        ret = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
        t1 = time.time() - t0
        print(f"detokenizer takes {t1:.2f} s")
        dur = time.time() - start_t
        return ret, dur

    def perf_record(self):
        return self.call_and_wait('perf_record')

    def to_decode(self):
        return self.call_and_wait("reload_model")