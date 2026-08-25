from typing import Optional, List
import copy
import time

import torch
from transformers import PretrainedConfig
import flashinfer
from torch.distributed import ReduceOp

from cgen.config import DistConfig, PipelineStatus
from cgen.utils import Device, logger
from cgen.dist_utils import pp_group, tp_group

from cgen.layers.cache import KVCacheManager
from cgen.layers.base import ModelInput, SwapInfo, ShardedWeight
from cgen.layers.emb import DistEmbed
from cgen.layers.linear import LinearLayer

def _split_weight(weight: torch.Tensor, split_row: bool, dist_config: DistConfig):
    n = dist_config.tp_size
    i = dist_config.tp_rank
    nrow, ncol = weight.shape
    if split_row:
        assert nrow % n == 0
        part_row = nrow // n
        return weight[part_row * i : part_row * (i + 1)]
    else:
        assert ncol % n == 0
        part_col = ncol // n
        return weight[:, part_col * i : part_col * (i + 1)]


def _split_bias(bias: torch.Tensor, dist_config: DistConfig):
    if not isinstance(bias, torch.Tensor) or len(bias.shape) == 0:
        return bias
    n = dist_config.tp_size
    i = dist_config.tp_rank
    part_row = bias.shape[0] // n
    return bias[part_row * i : part_row * (i + 1)]


class DistModuleBase(torch.nn.Module):
    def __init__(
        self,
        dist_config: DistConfig,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.dist_config = dist_config
        self.device = device


class DistModelBase(torch.nn.Module):
    def __init__(
        self,
        model_config: PretrainedConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.half,
        embed_tokens = None
    ):
        super().__init__()
        self.dist_config = dist_config
        self.hsz = model_config.hidden_size
        self.model_config = model_config
        self.nheads = model_config.num_attention_heads // dist_config.tp_size
        self.nkvheads = (
            getattr(model_config, "num_key_value_heads", self.nheads)
            // dist_config.tp_size
        )
        self.dim = model_config.hidden_size // model_config.num_attention_heads

        self.device = device

        # create model
        if dist_config.is_first_pp():
            self.embed_tokens = embed_tokens or torch.nn.Embedding(
                model_config.vocab_size,
                model_config.hidden_size,
                model_config.pad_token_id,
                device=device,
                dtype=dtype,
            )
        else:
            self.embed_tokens = None

        self.copy_stream = torch.cuda.Stream(device=device)

        # Fot OPT
        self.embed_positions = None
        self.word_emb_dim = getattr(model_config, 'word_embed_proj_dim', self.hsz)
        if self.word_emb_dim != self.hsz:
            assert False, "You need to fix this bug"
        self._workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.attn_wrapper_prefill = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self._workspace_buffer, "NHD")
        self.attn_wrapper_decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace_buffer, "NHD")

    def _emb_or_recv(
        self,
        inputs: ModelInput,
    ) -> torch.Tensor:
        if self.dist_config.is_first_pp():
            h = self.embed_tokens(inputs.x)
            if self.embed_positions is not None:
                pos_emb = self.embed_positions(inputs.pos)
                h += pos_emb
        else:
            h = torch.empty(
                (inputs.x.shape[0], self.hsz),
                dtype=torch.half,
                device=self.device,
            )  # TODO: dtype
            torch.distributed.recv(h, self.dist_config.pp_pred())
        return h

    def _build_flashinfer_wrapper(self, inputs: ModelInput, page_size: int=None):
        if inputs.is_prefill:
            inputs.attn_wrapper = self.attn_wrapper_prefill
            inputs.attn_wrapper.begin_forward(inputs.q_indptr, inputs.q_indptr, self.nheads, self.nkvheads, self.dim)
        else:
            inputs.attn_wrapper = self.attn_wrapper_decode
            inputs.attn_wrapper.begin_forward(inputs.kv_indptr,
                                              inputs.kv_indices,
                                              inputs.kv_last_page_len,
                                              self.nheads,
                                              self.nkvheads,
                                              self.dim,
                                              page_size,
                                              pos_encoding_mode="ROPE_LLAMA"
                                              )

    # @torch.compile(mode=COMPILE_MODE)
    def forward(
        self,
        inputs: ModelInput,
        kvcache: KVCacheManager,  # pp, block, layers
    ):
        # print("forward pass", inputs)
        h = self._emb_or_recv(inputs)
        torch.cuda.nvtx.range_push("build flash infer")
        self._build_flashinfer_wrapper(inputs, page_size = kvcache.page_size if isinstance(kvcache, KVCacheManager) else None)
        torch.cuda.nvtx.range_pop()

        for layer_idx, decoder_layer in enumerate(self.layers):
            h = decoder_layer(
                h,
                inputs=inputs,
                kvcache=kvcache,
            )
        inputs.attn_wrapper.end_forward()
        del inputs.attn_wrapper

        # if inputs.is_prefill:
            # for layer_idx, _ in enumerate(self.layers):
                # kvcache.get(layer_idx).wait_stash()

        if self.dist_config.is_last_pp():
            if self.norm is not None:
                h = self.norm(h)
            return h
        else:
            torch.distributed.send(
                h, self.dist_config.pp_succ()
            )
            return None


class DistModelBaseForCausalLM(torch.nn.Module):
    def __init__(
        self,
        model_config: PretrainedConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        dtype=torch.half,
        lm_heads=None
    ):
        super().__init__()
        self.model_config = model_config
        self.dist_config = dist_config
        self.model = None  # need to be filled by subclass

        if dist_config.is_last_pp():
            self.lm_head = lm_heads or torch.nn.Linear(
                model_config.hidden_size,
                model_config.vocab_size,
                device=device,
                bias=False,
                dtype=dtype,
            )
        else:
            self.lm_head = None
        self.device = device


    def load_weights(self, weight_loader):
        nlayers = self.model_config.num_hidden_layers
        pp_stage_layers = (nlayers - 1) // self.dist_config.pp_size + 1
        layer_start_idx = pp_stage_layers * self.dist_config.pp_rank
        params_dict = dict(self.named_parameters())
        tot_params = 0
        for name, weight in weight_loader:
            # pp dirty work
            tot_params += weight.numel()
            name = name.split('.')
            if len(name) > 2 and name[2].isnumeric():
                name[2] = str(int(name[2]) - layer_start_idx)
            name = '.'.join(name)

            if "rotary_emb.inv_freq" in name:
                continue
            # if name == 'model.embed_tokens.weight':
            #     self.model.embed_tokens.load_weights(weight)
            # elif name == 'lm_head.weight':
            #     if isinstance(weight, ShardedWeight):
            #         weight = weight.get()
            #     else:
            #         weight = _split_weight(weight, True, self.dist_config)
            #     self.lm_head.load_weight(weight)
            if name in params_dict:
                param = params_dict[name]
                param.data.copy_(weight)
            elif name.startswith("model.layer"):
                layer_idx = int(name.split('.')[2])
                if layer_idx >= 0 and layer_idx < len(self.model.layers):
                    self.model.layers[layer_idx].load_weight(name, weight)
            # else:
                # logger.warning(f"weight {name} not found in model")
        # logger.debug(f"Total number of parameters: {tot_params}")

    def _swap(self, kvcache: KVCacheManager, swap_info: SwapInfo):
        for seq_id, pages in swap_info.swap_out:
            kvcache.swap_out(seq_id, pages)
        
        for seq_id, pages in swap_info.swap_in:
            kvcache.swap_in(seq_id, pages)
    
    def forward(
        self,
        inputs: ModelInput,
        kvcache: KVCacheManager,  # pp, block, layers
    ):
        if len(inputs.seq_ids) == 0:
            return None
        # torch.cuda.nvtx.range_push("move input to cuda")
        for child_name in inputs.__dict__:
            child = getattr(inputs, child_name)
            if isinstance(child, torch.Tensor):
                dtype = torch.int32 if child.dtype in (torch.int32, torch.int64) else child.dtype
                setattr(inputs, child_name, child.to(device=self.device, dtype=dtype))
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push("swap")
        self._swap(kvcache, inputs.swap_info)
        # torch.cuda.nvtx.range_pop()
        o = self.model(
            inputs, kvcache
        )
        if self.dist_config.is_last_pp():
            return self.lm_head(o)
        else:
            return None

class PinnedModelWeights:
    # TODO: move all parameters to pinned memory for faster mode switching
    def __init__(self, weight_loader, dist_config: DistConfig):
        self.params = {}
        for name, weight in weight_loader:
            self.params[name] = weight.pin_memory()
    
    def weight_loader(self):
        return self.params.items()

