from typing import List, Tuple, Optional
import torch

from dataclasses import dataclass

@dataclass
class SwapInfo:
    swap_in: List[Tuple[int, List[int]]] # seq_id, page ids
    swap_out: List[Tuple[int, List[int]]] # seq_id, page ids

class ModelInput:
    def __init__(self, 
                 is_prefill: bool,
                 batch_id: int,
                 seq_ids: List[int],
                 x: torch.Tensor, 
                 pos: torch.Tensor,
                 q_indptr: torch.Tensor,
                 kv_indptr: torch.Tensor,
                 kv_indices: torch.Tensor,
                 kv_last_page_len: torch.Tensor,
                 kv_slot_ids: Optional[List[int]],
                 seqlen: List[int],
                 swap_info: SwapInfo,
                 ):
        self.is_prefill: bool = is_prefill
        self.batch_id = batch_id
        self.seq_ids = seq_ids
        self.x: torch.Tensor = x
        self.pos: torch.Tensor = pos
        self.q_indptr: torch.Tensor = q_indptr.int()
        self.kv_indptr: torch.Tensor = kv_indptr
        self.kv_indices: torch.Tensor = kv_indices
        self.kv_last_page_len: torch.Tensor = kv_last_page_len
        self.kv_slot_ids: List[int] = kv_slot_ids
        self.seqlen: List[int] = seqlen
        self.swap_info = swap_info

        self.attn_wrapper = None

        assert not ((kv_slot_ids is not None) ^ is_prefill)
    
    def __str__(self):
        return f"ModelInput(is_prefill={self.is_prefill}, batch_id={self.batch_id}, "\
               f"seq_ids={self.seq_ids}, "\
               f"x={self.x}, pos={self.pos}, q_indptr={self.q_indptr}, kv_indptr={self.kv_indptr}, "\
               f"kv_last_page_len={self.kv_last_page_len}, "\
               f"kv_indices={self.kv_indices}, seqlen={self.seqlen}, "\
               f"kv_slot_ids={self.kv_slot_ids}, "\
               f"swap_info={self.swap_info})"


class LoadPrefill:
    def __init__(self,
                 slot_ids: List[int],
                 seq_lens: List[int],
                 kv_indptr: torch.Tensor,
                 kv_indices: torch.Tensor,
                 kv_last_page_len: torch.Tensor,
                 ):
        assert kv_indptr.shape[0] == len(slot_ids)
        assert kv_indptr.shape[1] == len(seq_lens)
        self.slot_ids = slot_ids
        self.seq_len = seq_lens
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_len = kv_last_page_len

class ShardedWeight:
    def __init__(self, x: torch.Tensor, dtype=torch.float16):
        self.x = x.clone().to(dtype=dtype).pin_memory()
    
    def get(self):
        return self.x
    
    def numel(self):
        return self.x.numel()    