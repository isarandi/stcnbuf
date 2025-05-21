import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class MemoryBank(nn.Module):
    def __init__(self, k, top_k=20, memory_size=np.inf, temp_memory_size=3):
        super().__init__()
        self.num_objects = k
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = torch.zeros(0)
        self.mem_v = torch.zeros(0)
        self.has_mem = False
        self.memory_size = memory_size

        self.temp_k = torch.zeros(0)
        self.temp_v = torch.zeros(0)
        self.has_temp_mem = False
        self.temp_memory_size = temp_memory_size
        self.i_temp_mem = 0

    def _global_matching(self, mk, qk):
        CK = mk.shape[1]
        mk = mk.squeeze(0)
        a_sq = mk.pow(2).sum(0)
        ab = torch.einsum('cm,bcq->bmq', mk, qk)
        affinity = (2 * ab - a_sq.unsqueeze(-1)) / math.sqrt(CK)  # BATCH, MEMPIX, QUERYPIX
        return softmax_w_top(affinity, top=self.top_k)  # BATCH, MEMPIX, QUERYPIX

    # @torch.jit.export
    def match_memory(self, qk):
        h = qk.shape[2]
        # qk = einops.rearrange(qk, 'batch chan h w -> batch chan (h w)')
        qk = qk.view(qk.shape[0], qk.shape[1], qk.shape[2] * qk.shape[3])

        if self.has_temp_mem:
            mk = torch.cat([self.mem_k, self.temp_k], 2)
            mv = torch.cat([self.mem_v, self.temp_v], 2)
        else:
            mk = self.mem_k
            mv = self.mem_v

        affinity = self._global_matching(mk, qk)
        readout_mem = torch.einsum('ocm,bmq->obcq', mv, affinity)
        # return einops.rearrange(readout_mem, 'obj batch chan (h w) -> obj batch chan h w', h=h)
        return readout_mem.view(
            readout_mem.shape[0],
            readout_mem.shape[1],
            readout_mem.shape[2],
            h,
            readout_mem.shape[3] // h,
        )

    # @torch.jit.export
    def add_memory(self, key, value, is_temp: bool = False):
        # Temp is for "last frame"
        # Not always used
        # But can always be flushed

        # key = einops.rearrange(key, 'chan h w -> 1 chan (h w)')
        # value = einops.rearrange(value, 'obj chan h w -> obj chan (h w)')
        key = key.view(1, key.shape[0], key.shape[1] * key.shape[2])
        value = value.view(value.shape[0], value.shape[1], value.shape[2] * value.shape[3])

        if not self.has_mem:
            # First frame, just shove it in
            self.mem_k = key
            self.mem_v = value
            self.CK = key.shape[1]
            self.CV = value.shape[1]
            self.has_mem = True
        else:
            n_pixels = key.shape[2]

            if is_temp:
                if not self.has_temp_mem:
                    self.temp_k = key
                    self.temp_v = value
                    self.has_temp_mem = True
                else:
                    n_items_stored = self.temp_k.shape[2] // n_pixels
                    if n_items_stored >= self.temp_memory_size:
                        i_slice = slice(
                            self.i_temp_mem * n_pixels, (self.i_temp_mem + 1) * n_pixels
                        )
                        self.temp_k[:, :, i_slice] = key
                        self.temp_v[:, :, i_slice] = value
                        self.i_temp_mem = (self.i_temp_mem + 1) % self.temp_memory_size
                    else:
                        self.temp_k = torch.cat([self.temp_k, key], 2)
                        self.temp_v = torch.cat([self.temp_v, value], 2)

            else:
                n_items_stored = self.mem_k.shape[2] // n_pixels
                if n_items_stored >= self.memory_size:
                    # ids = np.random.choice(np.arange(self.mem_k.shape[2]), n_pixels, replace=False)
                    # now with pytorch:
                    ids = torch.randperm(self.mem_k.shape[2])[:n_pixels]
                    self.mem_k[:, :, ids] = key
                    self.mem_v[:, :, ids] = value
                else:
                    self.mem_k = torch.cat([self.mem_k, key], 2)
                    self.mem_v = torch.cat([self.mem_v, value], 2)


def softmax_w_top(x, top: int):
    values, indices = torch.topk(x, k=top, dim=1)
    values -= values.max(dim=1, keepdim=True)[0]
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # The types should be the same already
    # some people report an error here so an additional guard is added
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype))  # B * THW * HW
    return x
