"""
eval_network.py - Evaluation version of the network
The logic is basically the same
but with top-k and some implementation optimization

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import einops
import torch
import torch.nn as nn

from model.modules import KeyEncoder, ValueEncoder, KeyProjection
from model.network import Decoder


class STCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder = Decoder()

    def encode_value(self, frame, kf16, masks):
        # Extract memory key/value for a frame with multiple masks
        num_masks = masks.shape[0]
        frame = einops.repeat(frame, 'batch chan h w -> (obj batch) chan h w', obj=num_masks)
        masks = einops.rearrange(masks, 'obj batch h w -> (obj batch) 1 h w')
        if num_masks != 1:
            others = torch.cat([
                torch.sum(masks[[j for j in range(num_masks) if i != j]], dim=0, keepdim=True)
                for i in range(num_masks)], 0)
        else:
            others = torch.zeros_like(masks)

        kf16 = einops.repeat(kf16, 'batch chan h w -> (obj batch) chan h w', obj=num_masks)
        f16 = self.value_encoder(frame, kf16, masks, others)
        return einops.repeat(f16, '(obj batch) chan h w -> obj batch chan h w', obj=num_masks)

    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)
        return k16, f16_thin, f16, f8, f4

    def segment_with_query(self, mem_bank, qf8, qf4, qk16, qv16):
        readout_mem = mem_bank.match_memory(qk16)
        readout_mem = einops.rearrange(
            readout_mem, 'obj batch chan h w -> (obj batch) chan h w', obj=mem_bank.num_objects)
        qv16 = einops.repeat(
            qv16, 'batch chan h w -> (obj batch) chan h w', obj=mem_bank.num_objects)
        qv16 = torch.cat([readout_mem, qv16], 1)
        prob = torch.sigmoid(self.decoder(qv16, qf8, qf4))
        return einops.rearrange(
            prob, '(obj batch) 1 h w -> obj batch h w', obj=mem_bank.num_objects)
