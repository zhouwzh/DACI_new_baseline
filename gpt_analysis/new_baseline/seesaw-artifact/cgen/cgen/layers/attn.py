from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.distributed as dist
import flashinfer
import copy

from cgen.config import DistConfig, PipelineStatus
from cgen.utils import Device, logger
from cgen.layers.cache import KVCacheManager, SharedCacheManager
from cgen.layers.base import ModelInput

class AttentionLayer:
    """
    A class that contains attention computation and kvcache.
    We do CPU-GPU data parallel in this op.
    """

    def __init__(
        self,
        dist_config: DistConfig,
        device: Device,
        dtype: torch.dtype,
        layer_idx: int = 0,
    ):
        self.dist_config = dist_config
        self.device = device
        self.dtype = dtype
        self.layer_idx = layer_idx

    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        inputs: ModelInput,
        kvcache: Union[KVCacheManager, SharedCacheManager],
    ):
        kvdata = kvcache.get(self.layer_idx)
        if inputs.is_prefill:
            assert isinstance(kvcache, SharedCacheManager)
            kvdata.stash(k, v, inputs.kv_slot_ids, inputs.q_indptr) # TODO: kuso, nani kore
            return inputs.attn_wrapper.forward(q, k, v, pos_encoding_mode="ROPE_LLAMA" )
        else:
            flashinfer.page.append_paged_kv_cache(k, v, inputs.q_indptr, kvdata, inputs.kv_indices, inputs.kv_indptr, inputs.kv_last_page_len)
            return inputs.attn_wrapper.forward(
                q, kvdata, pos_encoding_mode="ROPE_LLAMA" 
            )