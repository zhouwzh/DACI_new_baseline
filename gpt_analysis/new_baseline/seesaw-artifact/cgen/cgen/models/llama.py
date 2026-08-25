from typing import Optional, Tuple, List

from tqdm import tqdm
import torch
import torch.distributed
from torch.distributed import ReduceOp
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.activation import SiluAndMul

from cgen.config import DistConfig, PipelineStatus
from cgen.dist_utils import tp_group, all_reduce
from cgen.models.base import (
    _split_weight,
    DistModuleBase,
    DistModelBase,
    DistModelBaseForCausalLM,
    ShardedWeight
)
from cgen.layers import LinearLayer, AttentionLayer, KVCacheManager, ModelInput
from cgen.utils import Device, logger

def _load_weight(weight_buf, weight, part_out, dist_config):
    if isinstance(weight, ShardedWeight):
        weight = weight.get()
    else:
        weight = _split_weight(weight, part_out, dist_config)
    weight_buf.load_weight(weight)

def _get_weight(weight, part_out, dist_config):
    if isinstance(weight, ShardedWeight):
        weight = weight.get()
    else:
        weight = _split_weight(weight, part_out, dist_config)
    return weight

class DistLlamaAttn(torch.nn.Module):
    """
    LlamaSdpaAttention
    """

    def __init__(
        self,
        model_config: LlamaConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        layer_idx: int = 0,
    ):
        super().__init__()
        self.dist_config = dist_config
        self.device = device

        self.model_config = model_config
        self.nheads = model_config.num_attention_heads // dist_config.tp_size
        self.nkvheads = model_config.num_key_value_heads // dist_config.tp_size
        self.hsz = model_config.hidden_size
        self.dim = model_config.hidden_size // model_config.num_attention_heads
        self.rope_theta = model_config.rope_theta


        self.qkv_proj = LinearLayer(
            self.hsz,
            self.nheads * self.dim + 2 * self.nkvheads * self.dim,
            False,
            device,
            torch.float16,
        )
        self.o_proj = LinearLayer(
            self.nheads * self.dim,
            self.hsz,
            False,
            device,
            torch.float16,
        )
        self.attn = AttentionLayer(
            dist_config, device, torch.float16, layer_idx=layer_idx
        )

    def forward(
        self,
        x: torch.Tensor,
        inputs: ModelInput,
        kvcache: KVCacheManager,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        q_shape = x.shape[:-1]
        qkv = self.qkv_proj(x)
        q = qkv[:, : self.nheads * self.dim]
        k = qkv[:, self.nheads * self.dim : (self.nheads + self.nkvheads) * self.dim]
        v = qkv[:, (self.nheads + self.nkvheads) * self.dim : ]
        q_size = self.nheads * self.dim
        kv_size = self.nkvheads * self.dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(*q_shape, self.nheads, self.dim).contiguous()
        k = k.view(*q_shape, self.nkvheads, self.dim).contiguous()
        v = v.view(*q_shape, self.nkvheads, self.dim).contiguous()

        out = self.attn.forward(
            q, k, v, inputs, kvcache
        )

        out = out.view(*q_shape, self.nheads * self.dim)
        out = self.o_proj(out)
        assert(out.is_contiguous())

        torch.distributed.all_reduce(out, group=tp_group().group, op=ReduceOp.SUM)
        return out

    def load_weight(self, name, weight):
        q_size = self.nheads * self.dim
        kv_size = self.nkvheads * self.dim
        if name == 'rotary_emb':
            return
        if name == 'q_proj':
            weight = _get_weight(weight, True, self.dist_config)
            self.qkv_proj.weight[:q_size].copy_(weight)
        elif name == 'k_proj':
            weight = _get_weight(weight, True, self.dist_config)
            self.qkv_proj.weight[q_size:q_size + kv_size].copy_(weight)
        elif name == 'v_proj':
            weight = _get_weight(weight, True, self.dist_config)
            self.qkv_proj.weight[q_size + kv_size :].copy_(weight)
        elif name == 'o_proj':
            _load_weight(getattr(self, name), weight, False, self.dist_config)
        else:
            raise ValueError(name)


class DistLlamaMLP(DistModuleBase):
    def __init__(
        self,
        model_config: LlamaConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
    ):
        super().__init__(dist_config, device)

        hsz = model_config.hidden_size
        isz = model_config.intermediate_size // dist_config.tp_size
        self.isz = isz
        self.model_config = model_config

        self.gate_proj = LinearLayer(hsz, isz, False, device, torch.half)
        self.up_proj = LinearLayer(hsz, isz, False, device, torch.half)
        self.down_proj = LinearLayer(isz, hsz, False, device, torch.half)

        self.act_fn = model_config.hidden_act
        if isinstance(self.act_fn, str):
            self.act_fn = torch.nn.functional.__dict__[self.act_fn]


    def forward(self, x):
        raise RuntimeError("You should not directly call this function")
        return o

    def load_weight(self, name, weight):
        if name == "gate_proj":
            weight = _get_weight(weight, True, self.dist_config)
            self.gate_proj.weight.copy_(weight)
        elif name == "up_proj":
            weight = _get_weight(weight, True, self.dist_config)
            self.up_proj.weight.copy_(weight)
        elif name == 'down_proj':
            _load_weight(getattr(self, name), weight, False, self.dist_config)


class DistLlamaDecoder(DistModuleBase):
    def __init__(
        self,
        model_config: LlamaConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        dtype: torch.device = torch.half,
        layer_idx: int = 0,
    ):
        super().__init__(dist_config, device)

        hsz = model_config.hidden_size
        eps = model_config.rms_norm_eps

        self.input_layernorm = RMSNorm(hsz, eps=eps).to(device, dtype)
        self.post_attention_layernorm = RMSNorm(hsz, eps=eps).to(device, dtype)
        self.self_attn = DistLlamaAttn(
            model_config, dist_config, device, layer_idx=layer_idx
        )
        self.mlp = DistLlamaMLP(
            model_config, dist_config, device
        )
        self.layer_idx = layer_idx
        

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs: ModelInput,
        kvcache: KVCacheManager,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states, inputs, kvcache
        )

        hidden_states.add_(residual)
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        up_proj = self.mlp.up_proj(hidden_states)
        hidden_states = self.mlp.gate_proj(hidden_states)
        self.mlp.act_fn(hidden_states, inplace=True)
        hidden_states.mul_(up_proj)
        del up_proj
        hidden_states = self.mlp.down_proj(hidden_states)
        torch.distributed.all_reduce(hidden_states, group=tp_group().group, op=ReduceOp.SUM)

        hidden_states.add_(residual)
        del residual
        if inputs.is_prefill:
            kvcache.get(self.layer_idx).wait_stash()
        return hidden_states

    def load_weight(self, name: str, weight: torch.Tensor):
        submodule, *name = name.split('.')[3:]
        getattr(self, submodule).load_weight(name[0], weight)


class DistLlamaModel(DistModelBase):
    pos_mode = "ROPE_LLAMA"

    def __init__(
        self,
        model_config: LlamaConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        dtype: torch.dtype = torch.half,
        embed_tokens=None
    ):
        super().__init__(model_config, dist_config, device, dtype, embed_tokens=embed_tokens)
        # partition model
        rank = self.dist_config.rank()
        stage_id = dist_config.pp_rank
        stage_num = dist_config.pp_size
        nlayers = model_config.num_hidden_layers
        stage_size = (nlayers - 1) // stage_num + 1
        self.stage_size = stage_size

        self.global_layer_idx = stage_size * stage_id
        self.global_num_layer = model_config.num_hidden_layers
        layers = []
        for layer_idx in range(stage_size):
            global_layer_idx = stage_size * stage_id + layer_idx
            if global_layer_idx >= nlayers:
                break
            layers.append(
                DistLlamaDecoder(
                    model_config,
                    dist_config,
                    device,
                    layer_idx=layer_idx,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.norm = RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps).to(
            device, dtype
        )


class DistLlamaForCausalLM(DistModelBaseForCausalLM):
    def __init__(
        self,
        model_config: LlamaConfig,
        dist_config: DistConfig,
        device: Device = 'cpu',
        dtype=torch.half,
        embed_tokens=None,
        lm_head=None
    ):
        super().__init__(model_config, dist_config, device, dtype, lm_heads=None)
        self.model = DistLlamaModel(
            model_config, dist_config, device, dtype, embed_tokens=None
        )
    
    @property
    def pinned_model_weights(self):
        return LlamaPinnedModelWeights

class LlamaPinnedModelWeights:
    # move all parameters to pinned memory for faster mode switching
    IN_PART = ["mlp.down_proj", "self_attn.o_proj"]
    OUT_PART = ["mlp.gate_proj", "mlp.up_proj", "self_atten.k_proj", "self.attn.q_proj", "self_attn.v_proj"]
    def __init__(self, weight_loader, model_config, dist_config: DistConfig, dtype=torch.half):
        self.params = {}
        num_layers = model_config.num_hidden_layers
        pp_num_layers = num_layers // dist_config.pp_size
        pp_rank = dist_config.pp_rank
        tp_rank = dist_config.tp_rank
        tot_params = 0
        for name, weight in tqdm(weight_loader):
            if 'layers' in name:
                layer_idx = int(name.split('.')[2])
                if layer_idx >= pp_num_layers * (pp_rank + 1) or layer_idx < pp_num_layers * pp_rank:
                    continue
            shard = None
            for in_part_layer in self.IN_PART:
                if in_part_layer in name:
                    # logger.info(f"load shard of {name}")
                    tp_dim_size = weight.shape[1] // dist_config.tp_size
                    shard = weight[:, tp_rank * tp_dim_size : (tp_rank + 1) * tp_dim_size]
                    shard = ShardedWeight(shard, dtype=dtype)
                    break
            for out_part_layer in self.OUT_PART:
                if shard is None and out_part_layer in name:
                    # logger.info(f"load shard of {name}")
                    tp_dim_size = weight.shape[0] // dist_config.tp_size
                    shard = weight[tp_rank * tp_dim_size : (tp_rank + 1) * tp_dim_size]
                    shard = ShardedWeight(shard, dtype=dtype)
                    break
            if shard is None:
                shard = weight.to(dtype=dtype).pin_memory()
            self.params[name] = shard
            tot_params += shard.numel()

    def weight_loader(self):
        return self.params.items()


