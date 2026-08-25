from typing import Dict, List, Tuple, Set, Optional
import torch
import copy
import threading
import bisect
import time

from cgen.utils import Device, logger
from cgen.config import DistConfig, DO_SANITY_CHECK


class NoSpaceError(Exception):
    pass

class PageTable:
    def __init__(self, 
                 num_pages: int,
                 page_size: int,
                 watermark: float = 0.1,
                ):
        self.num_pages: int = num_pages
        self.page_size: int = page_size
        self._map: Dict[int, List[int]] = {}
        self._owner: Dict[int, int] = {}
        self._swap: Dict[int, int] = {}
        self.watermark_blocks = max(1, watermark * num_pages)
        print(f"{self.watermark_blocks=}")
        
    def _sanity_check(self):
        if not DO_SANITY_CHECK:
            return
        visited_seq_id = set()
        for seq_id, blocks in self._map.items():
            visited_seq_id.add(seq_id)
            for block_id in blocks:
                assert self._owner[block_id] == seq_id
        for block_id, seq_id in self._owner.items():
            assert seq_id in visited_seq_id 
            assert block_id in self._map[seq_id]

    def seqs(self) -> List[int]:
        return list(self._map.keys())

    def seq_pages(self, seq_id: int):
        return self._map[seq_id]
    
    def alloc(self, seq_id:int, require_pages: int) -> List[int]:
        assert require_pages <= self.num_free_pages()
        seq_pages = self._map.get(seq_id, [])
        new_pages = []
        for i in range(self.num_pages):
            if i not in self._owner:
                new_pages.append(i)
                self._owner[i] = seq_id
                if len(new_pages) == require_pages:
                    break
        else:
            raise NoSpaceError()
        self._map[seq_id] = seq_pages + new_pages
        self._sanity_check()
        return seq_pages + new_pages

    def evict(self, exclude: Set[int]=set()) -> Tuple[int, List[int]]:
        for chosen_seq, chosen_pages in self._map.items():
            if chosen_seq not in exclude:
                break
        else:
            raise NoSpaceError()
        freed_blocks = self._map[chosen_seq]
        self.free(chosen_seq)
        self._swap[chosen_seq] = len(chosen_pages)
        self._sanity_check()
        return chosen_seq, chosen_pages
    
    def free(self, seq_id: int):
        if seq_id in self._swap:
            self._swap.pop(seq_id)
            return
        assert seq_id in self._map
        for page in self._map[seq_id]:
            self._owner.pop(page)
        self._map.pop(seq_id)
        self._sanity_check()

    def num_free_pages(self, fraction=1):
        return min(self.num_pages // fraction, self.num_pages - len(self._owner))

    def swap_in(self, seq_id: int):
        assert seq_id in self._swap
        npages = self._swap[seq_id]
        assert npages <= self.num_free_pages()
        self.alloc(seq_id, npages)
        self._swap.pop(seq_id)
        logger.debug(f"swap in {seq_id=}, {self._map[seq_id]}")

    def try_swap_in(self, fraction=1) -> List[Tuple[int, int]]:
        chosen_seqs = []
        swap = copy.copy(self._swap)
        for seq_id, npages in swap.items():
            num_free_pages = self.num_free_pages(fraction=fraction)
            if num_free_pages - npages >= self.watermark_blocks:
                self.swap_in(seq_id)
                chosen_seqs.append((seq_id, self._map[seq_id]))
        return chosen_seqs

    def required_num_pages(self, seq_len: int) -> int:
        return (seq_len + self.page_size - 1) // self.page_size

class KVCacheManager:
    def __init__(self,
                 dist_config: DistConfig,
                 nlayers: int,
                 npages: int, 
                 page_size: int,
                 nheads: int,
                 dim: int,
                 device: Device,
                 dtype: torch.dtype = torch.float):
        self.page_size = page_size
        self.dist_config = dist_config
        self.nheads = nheads
        self.dim = dim
        self.kvshape = [page_size, nheads, dim]
        self._cpu_buf: Dict[int, List[torch.Tensor]] = {} 
        self.kvcache = [torch.empty((npages, 2, page_size, nheads, dim), device=device, dtype=dtype) for _ in range(nlayers)]
        self.nlayers = nlayers
    
    def swap_out(self, seq_id: int, pages: List[int]):
        num_pages = len(pages)
        kvs = []
        nbytes = 0
        logger.debug(f"swap out {seq_id=} {pages=}")
        for layer in range(self.nlayers):
            _buf = torch.empty((num_pages, 2, *self.kvshape) , device='cpu', pin_memory=True)
            nbytes += _buf
            for i, page in enumerate(pages):
                _buf[i].copy_(self.kvcache[layer][page])
            kvs.append(_buf)
        self._cpu_buf[seq_id] = kvs
    
    def swap_in(self, seq_id: int, pages: List[int]):
        assert seq_id in self._cpu_buf
        for layer in range(self.nlayers):
            assert self._cpu_buf[seq_id][layer].shape[0] == len(pages)
            for i, page in enumerate(pages):
                self.kvcache[layer][page].copy_(self._cpu_buf[seq_id][layer][i])
        self._cpu_buf.pop(seq_id)

    def get(self, layer) -> torch.Tensor:
        return self.kvcache[layer]
    
    def num_free_pages(self):
        return self.pagetable.num_free_pages()
    
    def seq_pages(self, seq_id: int) -> List[int]:
        return self.pagetable.seq_pages(seq_id)

class LayerSharedCache:
    def __init__(self,
            layer_idx: int,
            nlayers: int,
            dist_config_prefill: DistConfig,
            dist_config_decode: DistConfig,
            shared_mem: torch.Tensor,
            seq_len: int,
            nheads: int,
            dim: int,
            device: Device,
            dtype: torch.dtype = torch.half):
        self.dist_config_prefill = dist_config_prefill
        self.dist_config_decode = dist_config_decode
        self.shared_mem = shared_mem
        self.layer_idx = layer_idx
        self.copy_stream = torch.cuda.Stream(device=device)
        self.seq_len = seq_len
        self.nlayers = nlayers
        self.nheads = nheads
        self.dim = dim
        self.dtype = dtype

        # double buffering
        self._buf = torch.empty((2, nheads, seq_len, dim), device='cpu', dtype=dtype, pin_memory=True)
        self._gpu_buf = torch.empty((2, self.nheads, seq_len, dim), device='cuda', dtype=self.dtype)
        self.copy_thread = None
        self.device = device

    def copy_fn(self, k, v, slot_ids, kv_indptr, ntokens):
        # return
        t0 = time.time()
        torch.cuda.set_device(self.device)
        buf = self._buf
        default_stream = torch.cuda.current_stream()
        copy_stream = torch.cuda.Stream()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(default_stream)
            self._gpu_buf[0, :, :ntokens].copy_(k.transpose(0,1))
            self._gpu_buf[1, :, :ntokens].copy_(v.transpose(0,1))
            buf.copy_(self._gpu_buf, non_blocking=True)
            copy_stream.synchronize()
            
            global_layer_idx = self.layer_idx + self.dist_config_prefill.pp_rank * self.nlayers 
            head_offset = self.nheads * self.dist_config_prefill.tp_rank
            start_ptr = 0
            end_ptr = kv_indptr[1]
            last_slot_start = slot_ids[0]
            last_slot_end = slot_ids[0] 
            shard_mem_tp_shard = self.shared_mem[global_layer_idx, :, head_offset : head_offset + self.nheads]
            for i, slot_id in enumerate(slot_ids):
                kv_start = kv_indptr[i]
                kv_end = kv_indptr[i + 1]
                seqlen = kv_end - kv_start
                if last_slot_end == slot_id:
                    end_ptr = kv_end
                    last_slot_end += seqlen
                else:
                    shard_mem_tp_shard[:, :, last_slot_start: last_slot_end].copy_(
                        buf[:, :, start_ptr:end_ptr], non_blocking=True
                    )
                    last_slot_start = slot_id
                    last_slot_end = slot_id + seqlen
                    start_ptr = kv_start
                    end_ptr = kv_end
            else:
                shard_mem_tp_shard[:, :, last_slot_start: last_slot_end].copy_(
                    buf[:, :, start_ptr:end_ptr], non_blocking=True
                )
            copy_stream.synchronize()
            logger.debug(f"total time used: {(time.time() - t0) * 1e3: .3f} ms")

    
    def stash(self, k, v, slot_ids, kv_indptr):
        logger.debug(f"{self.layer_idx} stash")
        ntokens = min(k.shape[0], self.seq_len)
        self.copy_thread = threading.Thread(target=self.copy_fn, args=(k, v, slot_ids, kv_indptr, ntokens))
        self.copy_thread.start()

    def wait_stash(self):
        logger.debug(f"{self.layer_idx} wait stash")
        self.copy_thread.join()
        self.copy_thread = None
    


class SharedCacheManager:
    def __init__(self,
                nlayers: int,
                *args, **kwargs):
        args = (nlayers,) + args
        self.layers = [LayerSharedCache(i, *args, **kwargs)
                       for i in range(nlayers)]
    
    def get(self, layer_idx):
        return self.layers[layer_idx]

    def wait(self):
        self._copy_thread.wait()
    
    

class SharedCacheMetadata:
    def __init__(self, num_tokens:int):
        self.num_tokens: int = num_tokens
        self.free_slots: List[List[int]] = [[0, num_tokens]]
        self.seq_map: Dict[int, Tuple[int, int]] = {} # seq_id -> (start, end)
        self.used_tokens = 0

    def alloc(self, seq_id: int, num_tokens: int):
        for i, slot in enumerate(self.free_slots):
            start, end = slot
            if end - start >= num_tokens:
                break
        else:
            return None
        if end - start > num_tokens:
            slot[0] = slot[0] + num_tokens
        else:
            del self.free_slots[i]
        self.seq_map[seq_id] = (start, start + num_tokens)
        self.used_tokens += num_tokens
        return start
        
    def free(self, seq_id):
        start, end = self.seq_map[seq_id]
        self.used_tokens -= end - start
        slot_id = bisect.bisect(self.free_slots, start, key=lambda x:x[1])
        
        # if only one
        if len(self.free_slots) == 0:
            self.free_slots.append([start, end])
        
        self.free_slots.insert(slot_id, [start, end])
        
        # merge with succ
        if slot_id < len(self.free_slots) - 1:
            if end == self.free_slots[slot_id + 1][0]:
                self.free_slots[slot_id][1] = self.free_slots[slot_id + 1][1]
                del self.free_slots[slot_id + 1]
        
        # merge with prev
        if slot_id > 0:
            if start == self.free_slots[slot_id - 1][1]:
                self.free_slots[slot_id - 1][1] = self.free_slots[slot_id][1]
                del self.free_slots[slot_id]
        
        del self.seq_map[seq_id]
        logger.debug(f"remove {seq_id}: {start}, {end} | {self.free_slots}")
    
    def num_seqs(self):
        return len(self.seq_map)

    def slot(self, sed_id):
        return self.seq_map[sed_id][0]

    def used_percentage(self):
        return self.used_tokens / self.num_tokens