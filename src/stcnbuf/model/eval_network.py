import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from stcnbuf.inference_memory_bank import MemoryBank
from stcnbuf.model.modules import KeyEncoder, KeyProjection, ResBlock, UpsampleBlock, ValueEncoder


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

        frame = frame.repeat(num_masks, 1, 1, 1)
        masks = masks.view(-1, 1, masks.shape[2], masks.shape[3])
        # others is the sum of the nonself masks
        # thats same as sum of all masks - self mask
        others = torch.sum(masks, dim=0, keepdim=True) - masks
        kf16 = kf16.repeat(num_masks, 1, 1, 1)
        f16 = self.value_encoder(frame, kf16, masks, others)
        return f16.view(num_masks, -1, f16.shape[1], f16.shape[2], f16.shape[3])

    @torch.jit.export
    def encode_key(self, frame):
        f16, f8, f4 = self.key_encoder(frame)
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)
        return k16, f16_thin, f16, f8, f4

    @torch.jit.export
    def segment_with_query(self, readout_mem, qf8, qf4, qk16, qv16):
        num_obj = readout_mem.shape[0]
        # readout_mem = einops.rearrange(
        #   readout_mem, 'obj batch chan h w -> (obj batch) chan h w', obj=num_obj
        # )

        readout_mem = readout_mem.view(
            -1, readout_mem.shape[2], readout_mem.shape[3], readout_mem.shape[4]
        )

        # qv16 = einops.repeat(
        #     qv16, 'batch chan h w -> (obj batch) chan h w', obj=mem_bank.num_objects
        # )
        qv16 = qv16.repeat(num_obj, 1, 1, 1)

        qv16 = torch.cat([readout_mem, qv16], 1)
        prob = torch.sigmoid(self.decoder(qv16, qf8, qf4))
        # return einops.rearrange(
        #    prob, '(obj batch) 1 h w -> obj batch h w', obj=mem_bank.num_objects
        # )
        return prob.view(num_obj, -1, prob.shape[2], prob.shape[3])


class STCNInference(nn.Module):
    def __init__(self, prop_model, mem_size, temp_mem_size, topk, target_resolution):
        super().__init__()
        self.memory_bank: Optional[MemoryBank] = None
        self.prop_model = prop_model
        self.topk = topk
        self.mem_size = mem_size
        self.temp_mem_size = temp_mem_size
        self.n_objects = None

        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def initialize(self, frame, mask):
        n_objects = len(mask)
        bg_mask = 1 - torch.sum(mask, dim=0, keepdim=True)
        mask = torch.cat([bg_mask, mask], 0).unsqueeze(1)
        mask, _ = pad_divide_by(mask, 16)
        prob = aggregate(mask[1:])[1:]

        frame = self.im_transform(frame)
        frame = frame.unsqueeze(0).cuda()
        frame, pad = pad_divide_by(frame, 16)
        key, _, qf16, _, _ = self.prop_model.encode_key(frame)

        value = self.prop_model.encode_value(frame, qf16, prob)
        self.memory_bank = MemoryBank(
            k=n_objects,
            top_k=self.topk,
            memory_size=self.mem_size,
            temp_memory_size=self.temp_mem_size,
        )
        self.memory_bank.add_memory(key.squeeze(0), value.squeeze(1))

    @torch.jit.export
    def predict_batch(
        self, frame: torch.Tensor, add_last_to_memory: bool = True, is_temp: bool = False
    ):
        assert self.memory_bank is not None
        frame, pad = pad_divide_by(frame, 16)
        k16, qv16, qf16, qf8, qf4 = self.prop_model.encode_key(frame)

        readout_mem = self.memory_bank.match_memory(k16)
        mask = self.prop_model.segment_with_query(readout_mem, qf8, qf4, k16, qv16)
        mask = aggregate(mask)

        if add_last_to_memory:
            last_value = self.prop_model.encode_value(frame[-1:], qf16[-1:], mask[1:, -1:])
            last_key = k16[-1]
            self.memory_bank.add_memory(last_key, last_value.squeeze(1), is_temp=is_temp)

        return unpad(mask, pad)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)
        x = self.pred(F.relu(x))
        x = F.interpolate(x, scale_factor=4.0, mode='bilinear', align_corners=False)
        return x


# Soft aggregation from STM
def aggregate(object_probs):
    bg_prob = torch.prod(1 - object_probs, dim=0, keepdim=True)
    new_prob = torch.cat([bg_prob, object_probs], 0).clamp(1e-7, 1 - 1e-7)
    odds = new_prob / (1 - new_prob)
    return odds / odds.sum(dim=0, keepdim=True)


def pad_divide_by(in_img: torch.Tensor, d: int):
    h, w = in_img.shape[-2:]
    hp_total = (-h) % d
    hp_before = hp_total // 2
    hp_after = hp_total - hp_before

    wp_total = (-w) % d
    wp_before = wp_total // 2
    wp_after = wp_total - wp_before

    pad_array = (wp_before, wp_after, hp_before, hp_after)
    return F.pad(in_img, pad_array), pad_array


def unpad(img, pad: tuple[int, int, int, int]):
    return img[:, :, pad[2] : img.shape[2] - pad[3], pad[0] : img.shape[3] - pad[1]]
