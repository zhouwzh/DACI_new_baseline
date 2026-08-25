from typing import Optional, Tuple
from enum import Enum, auto
from pydantic import BaseModel
import copy

DO_SANITY_CHECK = False

class DistConfig(BaseModel):
    tp_size: int
    pp_size: int

    tp_rank: int = -1
    pp_rank: int = -1

    def rank(self):
        assert (
            self.tp_rank >= 0 and self.pp_rank >= 0
        ), f"{self.tp_rank}, {self.pp_rank}"
        return self.tp_rank + self.pp_rank * self.tp_size

    def world_size(self):
        return self.tp_size * self.pp_size

    def is_last_pp(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp(self):
        return self.pp_rank == 0

    def pp_pred(self):
        assert self.pp_rank > 0
        return (self.pp_rank - 1) * self.tp_size + self.tp_rank

    def pp_succ(self):
        assert self.pp_rank < self.pp_size - 1
        return (self.pp_rank + 1) * self.tp_size + self.tp_rank

    def should_return(self):
        return self.tp_rank == 0 and self.is_last_pp()

    def with_rank(self, rank):
        return DistConfig(
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            tp_rank=rank % self.tp_size,
            pp_rank=rank // self.tp_size,
        )

class HardwareSpec(BaseModel):
    ngpus: int
    gpu_mem: float
    cpu_mem: float
    host_device_copy: float
    gpu_bandwith: float


class PipelineStatus(BaseModel):
    stage: str = 'running' # can be 'idle', 'running' or 'finished'
    block_idx: int = 0
    layer_idx: int = 0

    def set_stage(self, stage: str):
        ret = copy.copy(self)
        ret.stage = stage
        return ret

    def set_block(self, block_idx: int):
        ret = copy.copy(self)
        ret.block_idx = block_idx
        return ret

    def set_batch_and_layer(self, batch_idx, layer_idx):
        ret = copy.copy(self)
        ret.batch_idx = batch_idx
        ret.layer_idx = layer_idx
        return ret

class CacheConfig(BaseModel):
    num_pages: int
    page_size: int

    # shared_cache_gb: int = 180
    shared_cache_tokens: int = 64_000
    shared_cache_max_prompt_len: int = 4096

class Plan(BaseModel):
    model: str
    tokenizer: str
    max_prefill_tokens: int
    transition_threshold: int = 16
    evict_recompte: bool = True
    prefill_threshold: float = 0.1
    # max_pre

    cache_config: CacheConfig
    dist_config_decode: DistConfig
    dist_config_prefill: Optional[DistConfig] = None

    @property
    def num_reqs(self):
        return self.schedule_decode.num_reqs

    def str_dist(self, dp):
        tp_d = self.dist_config_decode.tp_size
        pp_d = self.dist_config_decode.pp_size 
        ret = f"D{dp}T{tp_d}P{pp_d}"

        if self.dist_config_prefill and self.dist_config_decode != self.dist_config_prefill:
            tp_p = self.dist_config_prefill.tp_size
            pp_p = self.dist_config_prefill.pp_size 
            ret = f"D{dp}T{tp_p}P{pp_p}->{ret}"
        return ret

