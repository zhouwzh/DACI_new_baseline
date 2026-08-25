import torch
import torch.distributed
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from cgen.config import DistConfig
from cgen.utils import logger

TP_GROUP = None
PP_GROUP = None

TP_GROUP_PREFILL = None
PP_GROUP_PREFILL = None

TP_GROUP_DECODE = None
PP_GROUP_DECODE = None

def _create_process_groups(
    dist_config: DistConfig,
    device
):
    assert torch.distributed.is_initialized()
    tp_groups = []
    pp_groups = []
    tp = dist_config.tp_size
    pp = dist_config.pp_size
    logger.info(
        f"create proc groups tp={dist_config.tp_rank}/{tp} "
        f"pp={dist_config.pp_rank}/{pp}"
    )
    world_size = dist_config.world_size()
    for tp_group_id in range(dist_config.pp_size):
        ranks = list(range(tp_group_id * tp, (tp_group_id + 1) * tp))
        tp_groups.append(torch.distributed.new_group(ranks))
    for pp_group_id in range(dist_config.tp_size):
        ranks = list(range(pp_group_id, world_size, tp))
        pp_groups.append(torch.distributed.new_group(ranks))
    tp_group = tp_groups[dist_config.pp_rank]
    tp_group = PyNcclCommunicator(tp_group, device)
    tp_group.disabled = False
    pp_group = pp_groups[dist_config.tp_rank]
    return tp_group, pp_group   

def all_reduce(x, op):
    group = tp_group()
    if group.world_size > 1:
        group.all_reduce(x, op=op, stream=torch.cuda.current_stream())
        # dist.all_reduce(x, op=op, group=group.group)

def create_process_groups(
        dist_config_prefill: DistConfig,
        dist_config_decode: DistConfig,
        device
        ):
    global TP_GROUP, PP_GROUP, TP_GROUP_PREFILL, PP_GROUP_PREFILL, TP_GROUP_DECODE, PP_GROUP_DECODE
    TP_GROUP_PREFILL, PP_GROUP_PREFILL = _create_process_groups(dist_config_prefill, device)
    TP_GROUP_DECODE, PP_GROUP_DECODE = _create_process_groups(dist_config_decode, device)
    TP_GROUP, PP_GROUP = TP_GROUP_PREFILL, PP_GROUP_PREFILL

def to_prefill(): 
    global TP_GROUP, PP_GROUP
    TP_GROUP = TP_GROUP_PREFILL
    PP_GROUP = TP_GROUP_PREFILL

def to_decode(): 
    global TP_GROUP, PP_GROUP
    TP_GROUP = TP_GROUP_DECODE
    PP_GROUP = PP_GROUP_DECODE

def tp_group():
    return TP_GROUP


def pp_group():
    return PP_GROUP
