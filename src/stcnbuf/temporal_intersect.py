from argparse import ArgumentParser

import rlemasklib
import simplepyutils as spu
from simplepyutils import FLAGS


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    spu.argparse.initialize(parser)

    rle_masks_in = spu.load_pickle(FLAGS.input_path) # this is a list of lists, rle_masks_in[i_frame][i_person]
    n_frames = len(rle_masks_in)
    n_persons = len(rle_masks_in[0])
    rle_masks_in_transposed = [[] for _ in range(n_persons)]
    for i_frame, masks_frame in enumerate(rle_masks_in):
        for i_person, mask in enumerate(masks_frame):
            rle_masks_in_transposed[i_person].append(mask)
    # let's intersect 3 consecutive frames. the first output will just intersect 2 frames etc

    rle_masks_in_transposed_filtered = [
        [rlemasklib.intersection(ms[max(0,i-1):min(n_frames,i+2)]) for i in range(n_frames)]
        for ms in rle_masks_in_transposed]

    rle_masks_out = [[] for _ in range(n_frames)]
    for i_person, masks_person in enumerate(rle_masks_in_transposed_filtered):
        for i_frame, mask in enumerate(masks_person):
            rle_masks_out[i_frame].append(mask)


    spu.dump_pickle(rle_masks_out, FLAGS.output_path)


if __name__ == '__main__':
    main()
