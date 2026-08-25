from typing import Union
import sys

import torch
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

Device = Union[str, torch.device]


def quick_random(shape, device='cpu', dtype=torch.float):
    return torch.randint(10, shape, dtype=dtype, device=device) / 10


def get_open_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def p2p_warm_up(rank, world_size, device):
    import torch.distributed as dist
    send_buf = torch.empty([1], device=device, dtype=torch.half)
    recv_buf = torch.empty([1], device=device, dtype=torch.half)
    dst = (rank + 1) % world_size
    src = (rank - 1 + world_size) % world_size
    send_op = dist.P2POp(dist.isend, send_buf, dst)
    recv_op = dist.P2POp(dist.irecv, recv_buf, src)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()