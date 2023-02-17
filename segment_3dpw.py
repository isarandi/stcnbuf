#!/usr/bin/env python3
import argparse
import glob
import itertools
import os

# https://github.com/pytorch/pytorch/issues/21956
os.environ['OMP_NUM_THREADS'] = '1'

import cv2
import imageio
import more_itertools
import numpy as np
import scipy.optimize
import torch
import torch.utils.data
from attrdict import AttrDict

import eval_video
import mask_init
import myutils
from progressbar import progressbar

DATA_ROOT = os.environ['DATA_ROOT']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saves/stcn.pth')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--mem-every', default=5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--mem-size', default=25, type=int)
    parser.add_argument('--start-frame', default=0, type=int)
    parser.add_argument('--max-persons', default=np.inf, type=float)
    parser.add_argument('--resolution', default=320, type=int)
    FLAGS = parser.parse_args()

    seq_filepaths = sorted(glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl'))
    seq_names = [os.path.basename(p).split('.')[0] for p in seq_filepaths]
    ji2d = JointInfo(
        'nose,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,reye,leye,lear,rear',
        'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,lear-leye-nose-reye-rear')
    torch.autograd.set_grad_enabled(False)

    # We use a Mask R-CNN for obtaining initial person segments
    init_segmenter = mask_init.MaskRCNN().cuda().eval()
    prop_model = eval_video.load_prop_model(FLAGS.model)
    model = eval_video.STCNInference(
        prop_model, FLAGS.mem_size, FLAGS.top, FLAGS.resolution).cuda().eval()

    for seq_name, seq_filepath in progressbar(zip(seq_names, seq_filepaths)):
        out_path = f'{FLAGS.output_dir}/{seq_name}.pkl'
        if os.path.exists(out_path):
            continue

        frame_paths = sorted(glob.glob(f'{DATA_ROOT}/3dpw/imageFiles/{seq_name}/image_*.jpg'))
        poses2d_true = get_poses2d_3dpw(seq_name)
        preds = predict_sequence(
            init_segmenter, model, frame_paths, poses2d_true, ji2d, FLAGS.mem_every)
        myutils.dump_pickle(preds, out_path)


def predict_sequence(init_segmenter, model, frame_paths, poses2d_true, ji2d, batch_size):
    n_tracks = poses2d_true.shape[1]

    # Get the first frame of the video
    frames = (imageio.imread(p) for p in frame_paths)
    frames = more_itertools.peekable(frames)

    # Obtain first-frame segmentation via Mask R-CNN
    initial_frame = frames.peek()
    initial_masks = myutils.to_numpy(init_segmenter.predict(initial_frame))

    # Generate masks based on the ground-truth 2D poses provided in 3DPW
    initial_posemasks = np.array([
        myutils.pose_to_mask(p, initial_frame.shape, ji2d, 30) for p in poses2d_true[0]])

    # Now associate the Mask R-CNN output masks to the 3DPW ground-truth people, so that we only
    # track people for whom we have ground-truth pose
    iou_matrix = np.array([
        [myutils.mask_iou(m1, m2)
         for m2 in initial_posemasks]
        for m1 in initial_masks])
    pred_indices, gt_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
    indices = [pred_indices[list(gt_indices).index(i)] for i in range(n_tracks)]
    initial_masks = torch.from_numpy(initial_masks[indices])

    # Initialize STCN
    model.initialize(initial_frame, initial_masks)

    # Set up the input stream of frames
    ds = eval_video.VideoDataset(progressbar(frames), model.im_transform)
    frame_loader = torch.utils.data.DataLoader(
        ds, num_workers=1, batch_size=batch_size, prefetch_factor=5)

    # Collect the predictions for each frame
    results = []
    for frame_batch in frame_loader:
        mask_batch = model.predict_batch(frame_batch.cuda())
        label_map_batch = torch.argmax(mask_batch, dim=0)
        label_map_batch = myutils.to_numpy(label_map_batch).astype(np.uint8)
        results += [myutils.encode_label_map(lm, n_tracks) for lm in label_map_batch]

        ## For visualization:
        # frame_batch = (inv_im_trans(frame_batch).detach().cpu().numpy().transpose(
        #     [0, 2, 3, 1]) * 255).astype(np.uint8)
        # for frame, label_map in zip(frame_batch, label_map_batch):
        #     visu = myutils.plot_with_masks(frame, label_map)
        #     cv2.imshow('image', visu[..., ::-1])
        #     cv2.waitKey(1)

    return results


def get_poses2d_3dpw(seq_name):
    seq_filepaths = glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    filepath = next(p for p in seq_filepaths if os.path.basename(p) == f'{seq_name}.pkl')
    seq = myutils.load_pickle(filepath)
    return np.transpose(np.array(seq['poses2d']), [1, 0, 3, 2])  # [Frame, Track, Joint, Coord]


class JointInfo:
    def __init__(self, joints, edges=()):
        if isinstance(joints, dict):
            self.ids = joints
        elif isinstance(joints, (list, tuple, np.ndarray)):
            self.ids = JointInfo.make_id_map(joints)
        elif isinstance(joints, str):
            self.ids = JointInfo.make_id_map(joints.split(','))
        else:
            raise Exception

        self.names = list(sorted(self.ids.keys(), key=self.ids.get))
        self.stick_figure_edges = []
        self.add_edges(edges)

    @property
    def n_joints(self):
        return len(self.ids)

    def add_edges(self, edges):
        if isinstance(edges, str):
            for path_str in edges.split(','):
                joint_names = path_str.split('-')
                for joint_name1, joint_name2 in more_itertools.pairwise(joint_names):
                    if joint_name1 in self.ids and joint_name2 in self.ids:
                        edge = (self.ids[joint_name1], self.ids[joint_name2])
                        self.stick_figure_edges.append(edge)
        else:
            self.stick_figure_edges.extend(edges)

    @staticmethod
    def make_id_map(names):
        return AttrDict(dict(zip(names, itertools.count())))


if __name__ == '__main__':
    main()
