import functools
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import simplepyutils as spu

import rlemasklib


@functools.lru_cache()
def get_structuring_element(shape, ksize, anchor=None):
    if not isinstance(ksize, tuple):
        ksize = (ksize, ksize)
    return cv2.getStructuringElement(shape, ksize, anchor)


def dilate(mask, kernel_size, iterations=1):
    if kernel_size == 1:
        return mask
    elem = get_structuring_element(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, elem, iterations=iterations)


def erode(mask, kernel_size, iterations=1):
    if kernel_size == 1:
        return mask
    elem = get_structuring_element(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, elem, iterations=iterations)


def get_outline(mask, d1=1, d2=3):
    if mask.dtype == bool:
        return get_outline(mask.astype(np.uint8), d1, d2).astype(bool)
    return dilate(mask, d2) - dilate(mask, d1)


def get_inline(mask, d1=1, d2=3):
    if mask.dtype == bool:
        return get_inline(mask.astype(np.uint8), d1, d2).astype(bool)
    return erode(mask, d1) - erode(mask, d2)


def plot_with_masks(img, label_map):
    colors = itertools.cycle(plt.get_cmap('tab10')(range(10)))
    n_labels = np.max(label_map) + 1

    img = img.copy()
    for label, color in zip(range(1, n_labels), colors):
        mask_color = np.array(color[:3]).astype(np.float64) * 255
        mask = label_map == label
        # img = highlight_mask(img, mask)
        outline = get_inline(mask, 1, 5)
        imcolor = img[mask].astype(np.float64)
        img[mask] = np.clip(mask_color * 0.3 + imcolor * 0.7, 0, 255).astype(np.uint8)
        img[outline] = mask_color
    return img


def plot_with_mask_stack(img, mask_stack):
    colors = itertools.cycle(plt.get_cmap('tab10')(range(10)))

    img = img.copy()
    for mask, color in zip(mask_stack, colors):
        mask_color = np.array(color[:3]).astype(np.float64) * 255
        mask = mask > 0.5
        outline = get_inline(mask, 1, 5)
        imcolor = img[mask].astype(np.float64)
        img[mask] = np.clip(mask_color * 0.3 + imcolor * 0.7, 0, 255).astype(np.uint8)
        img[outline] = mask_color
    return img


def highlight_mask(image, mask, inplace=True):
    r_outline = get_outline(mask, 3, 5)
    g_outline = get_outline(mask, 5, 7)
    b_outline = get_outline(mask, 7, 9)
    if not inplace:
        image = image.copy()

    image[r_outline] = [255, 0, 0]
    image[g_outline] = [0, 255, 0]
    image[b_outline] = [0, 0, 255]

    return image


def fill_polygon(img, pts, color):
    pts = pts.reshape((-1, 1, 2))
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, [pts], color)


def mask_iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(np.logical_or(mask1, mask2))
    return intersection / union


def pose_to_mask(pose2d, imshape, joint_info, thickness, thresh=0.2):
    result = np.zeros(imshape[:2], dtype=np.uint8)
    if pose2d.shape[1] == 3:
        is_valid = pose2d[:, 2] > thresh
    else:
        is_valid = np.ones(shape=[pose2d.shape[0]], dtype=bool)

    for i_joint1, i_joint2 in joint_info.stick_figure_edges:
        if pose2d.shape[1] != 3 or (is_valid[i_joint1] and is_valid[i_joint2]):
            line(
                result, pose2d[i_joint1, :2], pose2d[i_joint2, :2], color=(1, 1, 1),
                thickness=thickness)

    j = joint_info.ids
    torso_joints = [j.lhip, j.rhip, j.rsho, j.lsho]
    if np.all(is_valid[torso_joints]):
        fill_polygon(result, pose2d[torso_joints, :2], (1, 1, 1))
    return result


def line(im, p1, p2, *args, **kwargs):
    if np.asarray(p1).shape[-1] != 2 or np.asarray(p2).shape[-1] != 2:
        raise Exception('Wrong dimensionality of point in line drawing')

    cv2.line(im, spu.rounded_int_tuple(p1), spu.rounded_int_tuple(p2), *args, **kwargs)


def encode_label_map(label_map, n_objects):
    return [rlemasklib.encode(label_map == i_obj + 1) for i_obj in range(n_objects)]


def to_numpy(x, dtype=None):
    x = x.detach().cpu().numpy()
    if dtype:
        x = x.astype(dtype)
    return x


def masks_to_label_map(masks):
    h, w = masks[0].shape
    final_mask = np.zeros([h, w], np.uint8)
    i_instance = 1
    for mask in masks:
        final_mask[mask > 0.5] = i_instance
        i_instance += 1
    return final_mask
