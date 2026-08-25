import torch
import torch.nn.functional as F

from cgen.utils import Device


class LinearLayer:
    # _IN_GPU_LAYERS: int = 0

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias: bool,
        device: Device,
        dtype: torch.dtype,
    ):
        self.device = device

        self.n_gpu = out_feats
        self.weight = torch.empty(
            (self.n_gpu, in_feats), dtype=dtype, device=device
        )
        self.bias = bias
        if bias:
            self._gpu_bias = torch.empty((self.n_gpu,), dtype=dtype, device=device)
        else:
            self._gpu_bias = None

    def load_weight(self, weight: torch.Tensor):
        self.weight.copy_(weight[: self.n_gpu])

    def load_bias(self, bias: torch.Tensor):
        self.weight.copy_(bias[: self.n_gpu])

    def forward(self, x):
        o = F.linear(x, self.weight, self._gpu_bias)
        return o

    __call__ = forward