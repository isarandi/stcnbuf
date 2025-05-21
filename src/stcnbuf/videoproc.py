import functools

import numba
import numpy as np


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_decoder_lut():
    lut = np.zeros(256, np.float64)
    for i in numba.prange(256):
        x = i / 255
        if x <= 0.04045:
            lut[i] = x / 12.92
        else:
            lut[i] = ((x + 0.055) / 1.055) ** 2.4
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1
    return (lut * (1 << 16 - 1)).astype(np.uint16)


@functools.lru_cache
@numba.njit(error_model='numpy', cache=True)
def get_srgb_encoder_lut():
    lut = np.zeros(1 << 16, np.float64)
    for i in numba.prange(1 << 16):
        x = i / (1 << 16 - 1)
        if x <= 0.0031308:
            lut[i] = x * 12.92
        else:
            lut[i] = 1.055 * x ** (1 / 2.4) - 0.055
        if lut[i] < 0:
            lut[i] = 0
        elif lut[i] > 1:
            lut[i] = 1
    return (lut * 255).astype(np.uint8)


@numba.njit(error_model='numpy', cache=True, nogil=True)
def LUT(im, lut, dst):
    out = np.empty(im.shape, lut.dtype) if dst is None else dst
    im_flat = im.reshape(-1)
    out_flat = out.reshape(-1)
    for i in numba.prange(im_flat.shape[0]):
        out_flat[i] = lut[im_flat[i]]
    return out


def encode_srgb(im, dst=None):
    if dst is not None and dst.dtype != np.uint8:
        raise ValueError("The destination dtype must be np.uint8")
    if not im.dtype == np.uint16:
        raise ValueError("The input dtype must be np.uint16")
    if dst is not None and im.size != dst.size:
        raise ValueError("The input and destination arrays must have the same size")

    return LUT(im, get_srgb_encoder_lut(), dst)


def decode_srgb(im, dst=None):
    if dst is not None and dst.dtype != np.uint16:
        raise ValueError("The destination dtype must be np.uint16")
    if not im.dtype == np.uint8:
        raise ValueError("The input dtype must be np.uint8")
    if dst is not None and im.size != dst.size:
        raise ValueError("The input and destination arrays must have the same size")

    return LUT(im, get_srgb_decoder_lut(), dst)
