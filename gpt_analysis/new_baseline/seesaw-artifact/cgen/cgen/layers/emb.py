import torch

import cgen.dist_utils
from cgen.layers.base import ShardedWeight

class DistEmbed(torch.nn.Module):
    def __init__(self, vocab_size,
                hidden_size,
                pad_token_id,
                dist_config,
                device,
                dtype):
        super().__init__()
        tp_size = dist_config.tp_size
        self.dist_config = dist_config
        assert hidden_size % tp_size == 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._local_emb = torch.nn.Embedding(
            vocab_size,
            hidden_size // tp_size,
            pad_token_id,
            device=device,
            dtype=dtype,
        )
    
    def forward(self, x):
        local_o = self._local_emb(x)
        o_list = [torch.empty_like(local_o) for _ in range(self.dist_config.tp_size)]
        torch.distributed.all_gather(o_list, local_o, group=cgen.dist_utils.TP_GROUP)

        # [token, embed]
        return torch.cat(o_list, dim=1)

    def load_weights(self, w):
        tp_size = self.dist_config.tp_size
        tp_rank = self.dist_config.tp_rank
        if isinstance(w, ShardedWeight):
            w = w.get()
        else:
            dim = w.shape[1] // tp_size
            w = w[:, dim * tp_rank : dim * (tp_rank + 1)]
        self._local_emb.weight.data.copy_(w.data)