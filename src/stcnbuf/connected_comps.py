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
        [rlemasklib.largest_connected_component(x) for x in rle_masks_of_frame]
        for rle_masks_of_frame in spu.progressbar(rle_masks_in)
    ]
    rle_masks_out = [
        [x for x in rle_masks_of_frame if x is not None] for rle_masks_of_frame in rle_masks_out
    ]
    spu.dump_pickle(rle_masks_out, FLAGS.output_path)


if __name__ == '__main__':
    main()
