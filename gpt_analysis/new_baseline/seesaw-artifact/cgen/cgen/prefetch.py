from dataclasses import dataclass
import torch
import queue
import threading
import torch.multiprocessing as mp
import flashinfer
import copy
import time

from cgen.config import DistConfig, CacheConfig
from cgen.layers.cache import KVCacheManager
from cgen.utils import logger
from cgen.page import append_page_kv

@dataclass
class PrefetchTask:
    seq_id: int
    slot_id: int # slot id in shared memory
    seq_len: int
    kv_indices: torch.Tensor
    kv_last_page_len: int

class Prefetcher:
    def __init__(self, max_prefill_len:int, out_queue: mp.Queue, kvcache: KVCacheManager, shared_mem:torch.Tensor,
                 dist_config: DistConfig, device):
        self.in_queue = queue.Queue()
        self.out_queue = out_queue 
        self.shared_mem = shared_mem
        self.kvcache = kvcache

        self.dist_config = dist_config
        self.nlayers = kvcache.nlayers
        self.layer_offset = self.nlayers * dist_config.pp_rank

        self.nheads = kvcache.nheads
        self.dim = kvcache.dim
        self.tp_rank = dist_config.tp_rank

        self.device = device

        self.max_prefill_len = max_prefill_len
        
        _threads = [threading.Thread(target=self.thread_fn, args=(i, self.in_queue, ), daemon=True) for i in range(1)]
        for t in _threads:
            t.start()

        
    def thread_fn(self, tid, in_queue):
        # logger.info(f"(in sub thread) prefetch thread cuda device {torch.cuda.current_device()}")
        torch.cuda.set_device(self.device) 
        copy_stream = torch.cuda.Stream()
        local_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(local_stream)
        torch.set_num_threads(8)
        _pinned_buf = torch.empty((2, self.nlayers * self.nheads * self.max_prefill_len * self.dim), 
                                dtype=torch.half, device='cpu', pin_memory=True)
        while True:
            task: PrefetchTask = copy.deepcopy(in_queue.get())
            t0 = time.time()

            k_shared = self.shared_mem[self.layer_offset
            : self.layer_offset + self.nlayers, 0, self.nheads * self.tp_rank : self.nheads * (self.tp_rank + 1), task.slot_id : task.slot_id + task.seq_len]
            v_shared = self.shared_mem[self.layer_offset
            : self.layer_offset + self.nlayers, 1, self.nheads * self.tp_rank : self.nheads * (self.tp_rank + 1), task.slot_id : task.slot_id + task.seq_len]
            k_size = k_shared.nelement()

            _pin_buf_view_k = _pinned_buf[0, :k_size].view(self.nlayers, self.nheads, task.seq_len, self.dim)
            _pin_buf_view_v = _pinned_buf[1, :k_size].view(self.nlayers, self.nheads, task.seq_len, self.dim)
            _pin_buf_view_k.copy_(k_shared)
            _pin_buf_view_v.copy_(v_shared)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(local_stream)
                k = _pin_buf_view_k.to(self.device, non_blocking=True)
                v = _pin_buf_view_v.to(self.device, non_blocking=True)
            copy_stream.synchronize()
            local_stream.wait_stream(copy_stream) # guarantee the copying is finished
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            kv_indices = task.kv_indices.to(device=self.device, dtype=torch.int32)
            for layer_id in range(self.nlayers):
                append_page_kv(
                    k[layer_id], v[layer_id], kv_indices, self.kvcache.get(layer_id)
                )
            
            local_stream.synchronize()
            dur = time.time() - t0
            size = k.numel() * 2 * 2
            tput = size / dur
            logger.debug(f"{self.device=} prefetecher {tid} seq_id={task.seq_id}, {dur * 1e3 :.2f} ms | {size / 1e6} MB | {tput / 1e6} MB/s")
            self.out_queue.put(task.seq_id)
    
    def add_task(self, task: PrefetchTask):
        self.in_queue.put(task)