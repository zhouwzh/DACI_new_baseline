from typing import List, Dict, Tuple, Set, Union, Optional
import itertools
import copy
import time

import torch
import torch.multiprocessing as mp

from cgen.layers import ModelInput, PageTable, NoSpaceError, SwapInfo, SharedCacheMetadata, LoadPrefill
from cgen.utils import logger
from cgen.config import Plan, DO_SANITY_CHECK
from cgen.prefetch import PrefetchTask


class PrefetcherCounter:
    # collect prefetecher returns and return if all works have finished prefetching
    def __init__(self, ret_q:mp.Queue, world_size):
        self.ret_q:mp.Queue = ret_q
        self.prefetch_cnt: Dict[int, int] = {}
        self.world_size = world_size
    
    def update(self) -> List[int]:
        if self.ret_q.empty():
            return []
        ret = []
        while not self.ret_q.empty():
            seq_id = self.ret_q.get_nowait()
            self.prefetch_cnt[seq_id] = self.prefetch_cnt.get(seq_id, 0) + 1
            if self.prefetch_cnt[seq_id] >= self.world_size:
                ret.append(seq_id)
                self.prefetch_cnt.pop(seq_id)
        return ret


class Scheduler:
    def __init__(self,
                 plan: Plan,
                 pagetable: PageTable,
                 shared_table: SharedCacheMetadata,
                 input_tokens: List[List[int]], 
                 max_tokens: List[int],
                 prefetch_ret_q: mp.Queue
                 ):
        self.plan = plan
        self.dist_config_prefill = plan.dist_config_prefill
        self.dist_config_decode = plan.dist_config_decode
        self.dist_config = self.dist_config_prefill
        self.is_prefill = True

        self.waiting_prompts: List[int] = []
        self.seq_tokens: Dict[int, List[int]] = {}
        self.new_tokens: Dict[int, List[int]] = {}
        self.finished_reqs: Set[int] = set()
        self.prefilling_reqs: Set[int] = set() # reqs whose kvcache is in share mem
        self.prefilled_reqs: Set[int] = set() # reqs whose kvcache is in share mem and ready
        self.prefetching_reqs: Set[int] = set() # reqs whose kvcache is being moved to gpu
        self.running_reqs: Set[int] = set() # reqs whose kvcache are in gpu
        self.swap_out_reqs: Set[int] = set()

        for seq_id, seq in enumerate(input_tokens):
            self.waiting_prompts.append(seq_id)
            self.seq_tokens[seq_id] = seq

        self.max_tokens: List[int] = max_tokens
        self.pagetable: PageTable = pagetable
        self.shared_table: SharedCacheMetadata = shared_table

        self.active_seqs: Set[int] = set() # reqs that are processing by the pipeline
        self.batch_cnt = 0
        self.prefetch_counter = PrefetcherCounter(prefetch_ret_q, self.dist_config_decode.world_size())
        self.max_prefetching_seqs = 1000

    def to_prefill(self):
        if not self.is_prefill:
            self.batch_cnt = 0
        self.is_prefill = True
        self.dist_config = self.dist_config_prefill
        logger.info(f"switch to prefill, seqs in CPU {len(self.prefilled_reqs)}, prefetching {len(self.prefetching_reqs)}, seqs in GPU {len(self.running_reqs)}")
    
    def to_decode(self):
        self.is_prefill = False
        self.dist_config = self.dist_config_decode
        logger.info(f"switch to decode, seqs in CPU {len(self.prefilled_reqs)}, prefetching {len(self.prefetching_reqs)}, seqs in GPU {len(self.running_reqs)}")
    
    def append_output(self, seq_ids: List[int], new_tokens: List[int]):
        if len(new_tokens) == 0:
            return
        assert len(seq_ids) == len(new_tokens)
        new_finished = []
        for seq_id, new_token in zip(seq_ids, new_tokens):
            seq = self.seq_tokens[seq_id]
            seq.append(new_token)
            new_tokens = self.new_tokens.get(seq_id, [])
            new_tokens.append(new_token)
            self.new_tokens[seq_id] = new_tokens
            self.active_seqs.remove(seq_id)
            if seq_id in self.prefilling_reqs:
                self.prefilling_reqs.remove(seq_id)
                self.prefilled_reqs.add(seq_id)
            
            if len(seq) >= self.max_tokens[seq_id]:
                logger.debug(f"seq {seq_id} finished")
                self.finished_reqs.add(seq_id)
                if seq_id in self.running_reqs:
                    self.running_reqs.remove(seq_id)
                    self.pagetable.free(seq_id)
                if seq_id in self.swap_out_reqs:
                    self.swap_out_reqs.remove(seq_id)
                    self.pagetable.free(seq_id)
                    # TODO: dangerous, check this
                if seq_id in self.prefilled_reqs:
                    self.prefilled_reqs.remove(seq_id)
                    self.shared_table.free(seq_id)
                new_finished.append(seq_id)

    def _try_schedule_prefill(self) -> List[int]:
        ret = []
        num_tokens = 0
        for seq_id in self.waiting_prompts:
            seq = self.seq_tokens[seq_id]
            num_tokens += len(seq) 
            if num_tokens > self.plan.max_prefill_tokens:
                if len(ret) == 0:
                    # logger.warning("PROMPT TOO LONG")
                    ret.append(seq_id)
                break
            shared_slots = self.shared_table.alloc(seq_id, len(seq))
            if shared_slots is not None:
                ret.append(seq_id)
                # self.running_reqs.add(seq_id)
            else:
                logger.debug("no slots for prefill")
                break
        self.waiting_prompts = self.waiting_prompts[len(ret):]
        self.prefilling_reqs.update(ret)
        return ret

    def _schedule_prefill(self, batch: List[int]) -> ModelInput:
        seqs = [self.seq_tokens[seq_id] for seq_id in batch]
        flatten_input_ids = list(itertools.chain(*seqs))
        pos_ids = [list(range(len(seq))) for seq in seqs]
        flatten_pos_ids = list(itertools.chain(*pos_ids))
        seqlen = [len(seq) for seq in seqs]
        q_indptr = [0] + seqlen
        q_indptr = torch.tensor(q_indptr, dtype=torch.int32).cumsum(dim=0)

        batch = ModelInput(
            is_prefill=True,
            batch_id = self.batch_cnt,
            seq_ids = batch,
            x=torch.tensor(flatten_input_ids, dtype=torch.int32),
            pos=torch.tensor(flatten_pos_ids, dtype=torch.int32),
            q_indptr=q_indptr,
            kv_indptr=None,
            kv_indices=None,
            kv_last_page_len=None,
            kv_slot_ids=[self.shared_table.slot(seq_id) for seq_id in batch],
            seqlen=[len(seq) for seq in seqs],
            swap_info=SwapInfo([], []),
        )
        self.batch_cnt += 1
        logger.debug(f"#scheduled prefill: {self.shared_table.num_seqs()}")
        return batch
    
    def _schedule_decode(self, swap_in) -> ModelInput:
        exclude = [seq_id for seq_id, _ in swap_in]
        chosen_seqs = []
        running_reqs = copy.copy(self.running_reqs)
        swap_out = []
        pp_size = self.dist_config_decode.pp_size
        for seq_id in running_reqs:
            max_seqs = len(self.running_reqs) // pp_size + 1

            # too many sequences
            if len(chosen_seqs) >= max_seqs:
                break

            seq = self.seq_tokens[seq_id]
            
            # remove finished requests (why we need this?)
            if seq_id not in self.running_reqs:
                continue
            assert seq_id in self.pagetable.seqs() 

            # do not schedule requests that are being processed by pp stages
            if seq_id in self.active_seqs:
                continue
            chosen_seqs.append(seq_id)

            # allocate page for newly generated tokens
            if len(seq) % self.pagetable.page_size == 1: # need new page!!!
                if self.pagetable.num_free_pages(fraction=pp_size) < 1:
                    try:
                        evict_ret = self.pagetable.evict(exclude=chosen_seqs + exclude + list(self.prefetching_reqs))
                        self.running_reqs.remove(evict_ret[0])
                        self.swap_out_reqs.add(evict_ret[0])
                        swap_out.append(evict_ret)
                    except NoSpaceError:
                        # logger.warning("no valid sequence to evict")
                        chosen_seqs.pop()
                        continue
                self.pagetable.alloc(seq_id, 1)
        
        # assert len(chosen_seqs) > 0, "No sequence is scheduled!"
        seqs = [self.seq_tokens[seq_id] for seq_id in chosen_seqs]
        x = torch.tensor([seq[-1] for seq in seqs], dtype=torch.int32)
        seqlen = [len(seq) for seq in seqs]
        pos = torch.tensor([seqlen - 1 for seqlen in seqlen], dtype=torch.int32)
        pages = [self.pagetable.seq_pages(seq_id) for seq_id in chosen_seqs]
        
        kv_indptr = [0] + [len(page) for page in pages]
        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32).cumsum(dim=0)
        kv_indices = list(itertools.chain(*pages))
        
        page_size = self.pagetable.page_size
        kv_last_page_len = [(l - 1) % page_size + 1 for l in seqlen]
        assert len(kv_last_page_len) + 1 == kv_indptr.shape[0]

        batch = ModelInput(
            is_prefill=False,
            batch_id=self.batch_cnt,
            seq_ids=chosen_seqs,
            x=x,
            pos=pos,
            q_indptr=torch.arange(len(chosen_seqs) + 1, dtype=torch.int32),
            kv_indptr=kv_indptr,
            kv_indices=torch.tensor(kv_indices, dtype=torch.int32),
            kv_last_page_len=torch.tensor(kv_last_page_len, dtype=torch.int32),
            kv_slot_ids=None,
            seqlen=seqlen,
            swap_info=SwapInfo(swap_in, swap_out),
        )
        self.batch_cnt += 1
        return batch

    def schedule_prefill(self):
        chosen_seqs = self._try_schedule_prefill()
        if len(chosen_seqs) == 0:
            self.to_decode()
            return self.schedule_decode()
        return  self._schedule_prefill(chosen_seqs)
        
    def update_prefetch(self):
        if len(self.prefetching_reqs) == 0:
            return

        while len(prefetched_seqs := self.prefetch_counter.update()) == 0:
            if len(self.running_reqs) > 0:
                return
            if self.is_prefill:
                return
            logger.info("waiting for prefetching...")
            time.sleep(0.1)
         
        for seq_id in prefetched_seqs:
            self.running_reqs.add(seq_id)
            self.prefetching_reqs.remove(seq_id)
            self.shared_table.free(seq_id)

    def schedule_prefetch(self) -> List[PrefetchTask]:
        if len(self.prefetching_reqs) > self.max_prefetching_seqs:
            return []
        self.update_prefetch()
        chosen = []
        ret = []
        pp_size = self.dist_config_decode.pp_size
        for seq_id in self.prefilled_reqs:
            seq = self.seq_tokens[seq_id]
            free_pages = self.pagetable.num_free_pages(fraction=self.dist_config_decode.pp_size)
            required_pages = self.pagetable.required_num_pages(len(seq) - 1)
            if free_pages - required_pages >= self.pagetable.watermark_blocks // pp_size:
                self.pagetable.alloc(seq_id, required_pages)
                chosen.append(seq_id)
        for seq_id in chosen:
            seqlen = len(self.seq_tokens[seq_id]) - 1
            self.prefilled_reqs.remove(seq_id)
            page_size = self.pagetable.page_size
            kv_last_page_len = (seqlen - 1) % page_size + 1 
            ret.append(PrefetchTask(
                seq_id,
                self.shared_table.slot(seq_id),
                seqlen,
                torch.tensor(self.pagetable.seq_pages(seq_id), dtype=torch.int32),
                torch.tensor([kv_last_page_len], dtype=torch.int32)
            ))
            self.prefetching_reqs.add(seq_id)
        return ret
    

    def schedule_decode(self):
        if (len(self.prefilled_reqs) == 0 or self.shared_table.used_percentage() < 0.2) and\
           len(self.waiting_prompts) > 0: #    len(self.prefetching_reqs) == 0 and
            self.to_prefill()
            return self.schedule_prefill()

        self.update_prefetch()
        
        swap_in = self.pagetable.try_swap_in(self.dist_config_decode.pp_size)
        for seq_id, _ in swap_in:
            self.running_reqs.add(seq_id)
            self.swap_out_reqs.remove(seq_id)        
        return self._schedule_decode(swap_in=swap_in)


    def schedule(self) -> Union[ModelInput, LoadPrefill]:
        if self.is_prefill:
            batch = self.schedule_prefill()
        else:
            batch = self.schedule_decode()
        if isinstance(batch, ModelInput): 
            self.active_seqs.update(batch.seq_ids)
        self._sanity_check(batch)
        return batch
        
    def _sanity_check(self, batch: ModelInput):
        if not DO_SANITY_CHECK:
            return
        batch_size = len(batch.seq_ids)
        if batch.is_prefill:
            for i in range(batch_size):
                seq_id = batch.seq_ids[i]
                seqlen1 = len(self.seq_tokens[seq_id])
                seqlen2 = batch.q_indptr[i + 1] - batch.q_indptr[i]
                assert seqlen1 == seqlen2
        else:
            assert batch_size == batch.x.shape[0]
            assert batch_size == batch.pos.shape[0]
            for i in range(batch_size):
                seq_id = batch.seq_ids[i]
                pos_id = batch.pos[i]
                seq_len = len(self.seq_tokens[seq_id])
                assert pos_id + 1 == seq_len, f"{pos_id + 1} vs {seq_len}"

                block_size = self.pagetable.page_size
                num_blocks = (seq_len - 1) // block_size + 1
                assert num_blocks == batch.kv_indptr[i + 1] - batch.kv_indptr[i]
                

    def finished(self):
        return len(self.waiting_prompts) == 0 and\
               len(self.prefilled_reqs) == 0 and \
               len(self.prefilling_reqs) == 0 and \
               len(self.prefetching_reqs) == 0 and \
               len(self.running_reqs) == 0 and\
               len(self.swap_out_reqs) == 0