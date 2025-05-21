from argparse import ArgumentParser

import rlemasklib
import simplepyutils as spu
from simplepyutils import FLAGS


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    spu.argparse.initialize(parser)

    rle_masks_in = spu.load_pickle(
        FLAGS.input_path
    )  # this is a list of lists, rle_masks_in[i_frame][i_person]
    rle_masks_out = [
        [process_mask_dict(x) for x in rle_masks_of_frame]
        for rle_masks_of_frame in spu.progressbar(rle_masks_in)
    ]
    rle_masks_out = [
        [x for x in rle_masks_of_frame if x is not None] for rle_masks_of_frame in rle_masks_out
    ]
    spu.dump_pickle(rle_masks_out, FLAGS.output_path)


def process_mask_inplace(rle):
    rle.dilate5x5(inplace=True)
    rle.dilate5x5(inplace=True)
    rle.erode5x5(inplace=True)
    rle.erode5x5(inplace=True)
    # rle.remove_small_components(100, inplace=True)
    # rle.fill_small_holes(100, inplace=True)
    bbox = rle.largest_connected_component().bbox()
    bbox = [bbox[0] - bbox[2] * 0.15, bbox[1] - bbox[3] * 0.15, bbox[2] * 1.3, bbox[3] * 1.3]
    rle &= rlemasklib.RLEMask.from_bbox(bbox, rle.shape)
    return rle


def process_mask_dict(d):
    rle = rlemasklib.RLEMask.from_dict(d)
    rle = process_mask_inplace(rle)
    return rle.to_dict()


def process_mask_arr(arr):
    rle = rlemasklib.RLEMask.from_array(arr)
    rle = process_mask_inplace(rle)
    return rle.to_array()


if __name__ == '__main__':
    main()
