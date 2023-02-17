import collections

import torch

from inference_memory_bank import MemoryBank
from model.aggregate import aggregate
from model.eval_network import STCN
from util.tensor_util import pad_divide_by, unpad
from util.plotting import plot_with_masks
import numpy as np
import torch.nn.functional as F
from dataset.range_transform import inv_im_trans


class InferenceCore:
    def __init__(self, prop_net: STCN, images, num_objects, top_k=20,
                 mem_every=5, include_last=False, req_frames=None):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        # We HAVE to get the output for these frames
        # None if all frames are required
        self.req_frames = req_frames
        self.top_k = top_k

        # True dimensions
        self.t = images.shape[1]
        self.h, self.w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
        self.nh, self.nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'
        self.k = num_objects

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros(
            (self.k + 1, self.t, 1, self.nh, self.nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7
        self.kh = self.nh // 16
        self.kw = self.nw // 16

        # list of objects with usable memory
        self.enabled_obj = []
        self.mem_banks = collections.defaultdict(lambda: MemoryBank(k=1, top_k=self.top_k))

    def do_pass(self, key_k, key_v, idx, end_idx):
        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            self.mem_banks[oi].add_memory(key_k, key_v[i:i + 1])

        last_ti = idx

        for ti in range(idx + 1, end_idx):
            is_mem_frame = (abs(ti - last_ti) >= self.mem_every)
            # Why even work on it if it is not required for memory/output
            if (not is_mem_frame) and (not self.include_last) and (
                    self.req_frames is not None) and (ti not in self.req_frames):
                continue

            k16, qv16, qf16, qf8, qf4 = self.prop_net.encode_key(self.images[:, ti].cuda())

            # After this step all keys will have the same size
            out_mask = torch.cat([
                self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
                for oi in self.enabled_obj], 0)

            out_mask = aggregate(out_mask)
            self.prob[0, ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi, ti] = out_mask[i + 1]

            prob = self.prob[:, ti]
            prob = unpad(prob, self.pad)
            prob = F.interpolate(prob, (self.h, self.w), mode='bilinear', align_corners=False)
            mask = (torch.argmax(prob, dim=0).detach().cpu().numpy()[0]).astype(np.uint8)
            images = unpad(self.images[:, ti], self.pad)
            image = (inv_im_trans(images[0]).detach().cpu().numpy().transpose(
                [1, 2, 0]) * 255).astype(np.uint8)
            plot_with_masks(image, mask)

            if ti != end_idx - 1 and (self.include_last or is_mem_frame):
                prev_value = self.prop_net.encode_value(
                    self.images[:, ti].cuda(), qf16, out_mask[1:])
                prev_key = k16.unsqueeze(2)
                for i, oi in enumerate(self.enabled_obj):
                    self.mem_banks[oi].add_memory(
                        prev_key, prev_value[i:i + 1], is_temp=not is_mem_frame)

                if is_mem_frame:
                    last_ti = ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        # In youtube mode, we interact with a subset of object id at a time
        mask, _ = pad_divide_by(mask.cuda(), 16)

        # update objects that have been labeled
        self.enabled_obj.extend(obj_idx)

        # Set other prob of mask regions to zero
        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]
        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx])[1:]

        # KV pair for the interacting frame
        key_k, _, qf16, _, _ = self.prop_net.encode_key(self.images[:, frame_idx].cuda())
        key_v = self.prop_net.encode_value(
            self.images[:, frame_idx].cuda(), qf16, self.prob[self.enabled_obj, frame_idx].cuda())
        key_k = key_k.unsqueeze(2)

        # Propagate
        self.do_pass(key_k, key_v, frame_idx, end_idx)
