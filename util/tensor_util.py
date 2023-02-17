import torch
import torch.nn.functional as F
import numpy as np


def compute_tensor_iu(seg, gt):
    intersection = torch.count_nonzero(seg & gt)
    union = torch.count_nonzero(seg | gt)
    return intersection, union


def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    return (intersection + 1e-6) / (union + 1e-6)


# STM
def pad_divide_by(in_img, d, in_size=None):
    shape = np.array(in_img.shape[-2:] if in_size is None else in_size)
    pad_total = (-shape) % d
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad_array = (pad_before[1], pad_after[1], pad_before[0], pad_after[0])
    return F.pad(in_img, pad_array), pad_array


def unpad(img, pad):
    return img[:, :, pad[2]:img.shape[2] - pad[3], pad[0]:img.shape[3] - pad[1]]
