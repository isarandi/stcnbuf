import math

import einops
import torch
import numpy as np


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    values -= values.max(dim=1, keepdim=True)[0]
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype))  # B * THW * HW
    return x


class MemoryBank:
    def __init__(self, k, top_k=20, memory_size=np.inf):
        self.top_k = top_k
        self.CK = None
        self.CV = None
        self.mem_k = None
        self.mem_v = None
        self.num_objects = k
        self.memory_size = memory_size

    def _global_matching(self, mk, qk):
        CK = mk.shape[1]
        mk = mk.squeeze(0)
        a_sq = mk.pow(2).sum(0)
        ab = torch.einsum('cm,bcq->bmq', mk, qk)
        affinity = (2 * ab - a_sq.unsqueeze(-1)) / math.sqrt(CK)  # BATCH, MEMPIX, QUERYPIX
        return softmax_w_top(affinity, top=self.top_k)  # BATCH, MEMPIX, QUERYPIX

    def match_memory(self, qk):
        h = qk.shape[2]
        qk = einops.rearrange(qk, 'batch chan h w -> batch chan (h w)')

        if self.temp_k is not None:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)
        readout_mem = torch.einsum('ocm,bmq->obcq', mv, affinity)
        return einops.rearrange(readout_mem, 'obj batch chan (h w) -> obj batch chan h w', h=h)

    def add_memory(self, key, value, is_temp=False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed
        self.temp_k = None
        self.temp_v = None

        key = einops.rearrange(key, 'chan h w -> 1 chan (h w)')
        value = einops.rearrange(value, 'obj chan h w -> obj chan (h w)')

        if self.mem_k is None:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            n_pixels = key.shape[2]
            n_items_stored = self.mem_k.shape[2] // n_pixels

            if is_temp:
                self.temp_k = key
                self.temp_v = value
            elif n_items_stored >= self.memory_size:
                index = np.random.randint(0, n_items_stored)
                ids = slice(index * n_pixels, (index + 1) * n_pixels)
                self.mem_k[:, :, ids] = key
                self.mem_v[:, :, ids] = value
            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
