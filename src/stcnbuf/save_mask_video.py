import argparse
import itertools

import cv2
import numpy as np
import rlemasklib
import simplepyutils as spu
import framepump
from simplepyutils import FLAGS
from stcnbuf import myutils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask-path', type=str)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)

    video = framepump.VideoFrames(FLAGS.video_path)[FLAGS.start_frame :]

    writer = (
        framepump.VideoWriter(FLAGS.out_video_path, fps=video.fps) if FLAGS.out_video_path else None
    )
    masks_all = spu.load_pickle(FLAGS.mask_path)
    for frame, encoded_masks in zip(video, masks_all):
        masks = np.stack(
            [rlemasklib.decode(encoded_mask) for encoded_mask in encoded_masks], axis=0
        )
        label_map = myutils.masks_to_label_map(masks)
        frame = cv2.resize(
            frame, (label_map.shape[1], label_map.shape[0]), interpolation=cv2.INTER_AREA
        )
        visu = myutils.plot_with_masks(frame, label_map)
        if FLAGS.out_video_path:
            writer.append_data(visu)

        cv2.imshow('image', visu[..., ::-1])
        cv2.waitKey(1)

    if FLAGS.out_video_path:
        writer.close()


if __name__ == '__main__':
    main()
