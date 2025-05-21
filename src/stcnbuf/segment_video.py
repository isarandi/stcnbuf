import os

# https://github.com/pytorch/pytorch/issues/21956
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import rlemasklib
import simplepyutils as spu
import torch
import torch.utils.data
from simplepyutils import FLAGS
import stcnbuf.morph
from stcnbuf import mask_init, myutils
from stcnbuf.model.eval_network import STCN, STCNInference
import framepump
import torchvision.transforms.v2.functional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--output', type=str)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--mem-every', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--mem-size', default=128, type=int)
    parser.add_argument('--temp-mem-size', default=2, type=int)
    parser.add_argument('--init-frame', default='0', type=str)
    parser.add_argument('--permute-frames', action=spu.argparse.BoolAction)
    parser.add_argument('--max-persons', default=None, type=int)
    parser.add_argument('--resolution', default=320, type=int)
    parser.add_argument('--initial-threshold', default=0.5, type=float)
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--gui-selection', action=spu.argparse.BoolAction)
    parser.add_argument('--morph-cleanup', action=spu.argparse.BoolAction)
    parser.add_argument('--skip-existing', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--constant-framerate', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--passes', type=int, default=1)
    spu.argparse.initialize(parser)

    if osp.exists(FLAGS.output) and FLAGS.skip_existing:
        print('Already done')
        return

    if FLAGS.model_path is None:
        FLAGS.model_path = osp.join(osp.dirname(__file__), '../../saves/stcn.pth')

    with torch.inference_mode(), torch.amp.autocast('cuda'):
        init_segmenter = mask_init.MaskRCNN(FLAGS.initial_threshold).cuda().eval()
        prop_model = load_prop_model(FLAGS.model_path)
        stcn = (
            STCNInference(
                prop_model, FLAGS.mem_size, FLAGS.temp_mem_size, FLAGS.top, FLAGS.resolution
            )
            .cuda()
            .eval()
        )
        results = process_video(init_segmenter, stcn)

    if FLAGS.output:
        spu.dump_pickle(results, FLAGS.output)


def process_video(init_segmenter, vos_model):
    spu.logger.info(f'Processing {FLAGS.video_path}...')
    frames, index_unshuffler, resized_frames = get_frames(max(1, FLAGS.passes - 1))
    n_frames = framepump.num_frames(FLAGS.video_path)

    # Segment the first frame (which is outside the scope of STCN)
    frames = more_itertools.peekable(frames)
    initial_frame = frames.peek()
    initial_masks = init_segmenter.predict(initial_frame)

    ms = initial_masks.cpu().numpy()
    maxval = np.max(ms, axis=0, keepdims=True)
    ms[ms != maxval] = 0
    initial_masks = np.stack([stcnbuf.morph.process_mask_arr(m > 0.5) for m in ms], axis=0)

    if FLAGS.gui_selection:
        indices = visualize_and_select_objects(initial_frame, initial_masks)
        initial_masks = initial_masks[indices]
    elif FLAGS.max_persons != -1 and FLAGS.max_persons is not None:
        initial_masks = initial_masks[: FLAGS.max_persons]

    initial_masks = torch.as_tensor(initial_masks.astype(np.float32)).cuda()

    n_objects = len(initial_masks)
    vos_model.initialize(initial_frame, initial_masks)
    vos_model_script = torch.jit.script(vos_model)
    desc1 = 'Segmenting' if FLAGS.passes == 1 else 'Burn-in'
    ds = VideoDataset(
        spu.progressbar(frames, total=n_frames * max(1, FLAGS.passes - 1), desc=desc1),
        vos_model.im_transform,
    )
    frame_loader = torch.utils.data.DataLoader(
        ds, num_workers=1, batch_size=FLAGS.mem_every, prefetch_factor=5
    )

    if FLAGS.passes > 1:
        spu.logger.info(f'Running {FLAGS.passes - 1} burn-in passes...')
        for frame_batch in frame_loader:
            vos_model_script.predict_batch(frame_batch.cuda())

        # frames, index_unshuffler = get_frames_resized_already(
        #    n_passes=1, resized_frames=resized_frames, permute_frames=False
        # )
        frames, index_unshuffler = get_frames_resized_already(
            n_passes=1, resized_frames=resized_frames, permute_frames=False
        )
        # frames_interleaved = more_itertools.interleave(frames, frames_perm)

        ds = VideoDataset(
            spu.progressbar(frames, total=n_frames, desc='Segmenting'), vos_model.im_transform
        )
        frame_loader = torch.utils.data.DataLoader(
            ds, num_workers=1, batch_size=FLAGS.mem_every, prefetch_factor=5
        )

    results = []

    video_writer = (
        framepump.VideoWriter(FLAGS.out_video_path, fps=framepump.get_fps(FLAGS.video_path))
        if FLAGS.out_video_path
        else None
    )

    spu.logger.info('Running main pass...')
    for i, frame_batch in enumerate(frame_loader):
        frame_batch: torch.Tensor
        mask_batch = vos_model_script.predict_batch(
            frame_batch.cuda(), add_last_to_memory=True, is_temp=FLAGS.passes > 1
        )
        # if i % 2 == 1:
        #     continue
        # maxval_batch, label_map_batch = torch.max(mask_batch, dim=0)
        label_map_batch = torch.argmax(mask_batch, dim=0)
        # if the max is not large enough, set it to bg
        # label_map_batch[maxval_batch < 0.75] = 0
        label_map_batch = myutils.to_numpy(label_map_batch, np.uint8)
        results += [myutils.encode_label_map(lm, n_objects) for lm in label_map_batch]
        if (FLAGS.viz or FLAGS.out_video_path) and not FLAGS.permute_frames:
            frame_batch = undo_imagenet_normalization(frame_batch)
            frame_batch = (myutils.to_numpy(frame_batch).transpose([0, 2, 3, 1]) * 255).astype(
                np.uint8
            )
            for frame, label_map in zip(frame_batch, label_map_batch):
                visu = myutils.plot_with_masks(frame, label_map)
                if FLAGS.out_video_path:
                    writer.append_data(visu)
                if FLAGS.viz:
                    cv2.imshow('image', visu[..., ::-1])
                    cv2.waitKey(1)

    if FLAGS.morph_cleanup:
        results = [[stcnbuf.morph.process_mask_dict(m) for m in ms] for ms in results]

    if FLAGS.permute_frames:
        spu.logger.info('Unpermuting...')
        results = [results[i] for i in index_unshuffler]
        frames = [frames[i] for i in index_unshuffler]

        if FLAGS.viz or FLAGS.out_video_path:
            for frame, encoded_masks in zip(frames, results):
                masks = np.stack(
                    [rlemasklib.decode(encoded_mask) for encoded_mask in encoded_masks], axis=0
                )
                label_map = myutils.masks_to_label_map(masks)
                visu = myutils.plot_with_masks(frame, label_map)

                if FLAGS.out_video_path:
                    visu_padded = cv2.copyMakeBorder(
                        visu, 0, visu.shape[0] % 2, 0, visu.shape[1] % 2, cv2.BORDER_CONSTANT
                    )
                    video_writer.append_data(visu_padded)

                if FLAGS.viz:
                    cv2.imshow('image', visu[..., ::-1])
                    cv2.waitKey(1)

    if FLAGS.out_video_path:
        video_writer.close()

    return results


def get_frames(n_passes=1):
    video = framepump.VideoFrames(FLAGS.video_path, constant_framerate=FLAGS.constant_framerate)
    h, w = video.imshape
    if w > h:
        new_shape = (FLAGS.resolution, int(round(w * FLAGS.resolution / h)))
    else:
        new_shape = (int(h * FLAGS.resolution / w), FLAGS.resolution)

    # clahe = improc.Clahe()
    resized_frames = list(
        spu.progressbar(video.resized(new_shape), total=len(video), desc='Resizing')
    )

    # resized_frames = [
    #     resize_frame(frame, new_size)
    #     for frame in spu.progressbar(frames, total=n_frames, desc='Resizing')
    # ]
    # the above resizing is slow. lets do it faster via thread pool
    # resized_frames = spu.parallel_map_with_progbar(
    #     resize_frame2,
    #     zip(frames, itertools.repeat(new_size)),
    #     use_threads=True,
    #     total=n_frames,
    #     desc='Resizing',
    # )

    index_shuffler = []

    if FLAGS.permute_frames:
        for i in range(n_passes):
            index_shuffler += list(np.random.permutation(len(resized_frames)))
    else:
        for i in range(n_passes):
            index_shuffler += list(range(len(resized_frames)))

    start_frames = [int(x) for x in FLAGS.init_frame.split(',')]
    for s in start_frames:
        index_shuffler.remove(s)
    index_shuffler = start_frames + index_shuffler
    index_unshuffler = np.argsort(index_shuffler)
    frames = [resized_frames[i] for i in index_shuffler]

    return frames, index_unshuffler, resized_frames


def get_frames_resized_already(n_passes=1, resized_frames=None, permute_frames=None):
    index_shuffler = []

    if permute_frames is None:
        permute_frames = FLAGS.permute_frames

    if permute_frames:
        for i in range(n_passes):
            index_shuffler += list(np.random.permutation(len(resized_frames)))
    else:
        for i in range(n_passes):
            index_shuffler += list(range(len(resized_frames)))

    start_frames = [int(x) for x in FLAGS.init_frame.split(',')]
    for s in start_frames:
        index_shuffler.remove(s)
    index_shuffler = start_frames + index_shuffler
    index_unshuffler = np.argsort(index_shuffler)
    resized_frames = [resized_frames[i] for i in index_shuffler]
    return resized_frames, index_unshuffler


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable, transform):
        self.iterable = iterable
        self.transform = transform

    def __iter__(self):
        for frame in self.iterable:
            yield self.transform(frame)


def load_prop_model(model_path):
    prop_model = STCN().cuda().eval()
    prop_saved = torch.load(model_path, weights_only=True)
    # name = 'value_encoder.conv1.weight'
    # if name in prop_saved and prop_saved[name].shape[1] == 4:
    #     print('Patching the first layer of the value encoderaaaaaa...')
    #     pads = torch.zeros((64, 1, 7, 7), device=prop_saved[name].device)
    #     prop_saved[name] = torch.cat([prop_saved[name], pads], 1)
    prop_model.load_state_dict(prop_saved)
    return prop_model


def undo_imagenet_normalization(im):
    return torchvision.transforms.v2.functional.normalize(
        im, [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225]
    )


def visualize_and_select_objects(initial_frame, mask_stack):
    """
    Visualizes the initial frame with a stack of binary masks, allows the user to click on objects,
    and returns the indices of the masks corresponding to the clicked points.

    Args:
        initial_frame: The image/frame to be visualized.
        mask_stack: A mask stack of shape [n_objs, h, w], where each mask corresponds to an object.

    Returns:
        selected_masks_indices: A list of unique mask indices corresponding to the clicked points.
    """
    visu = myutils.plot_with_mask_stack(initial_frame, mask_stack)
    fig, ax = plt.subplots()
    ax.imshow(visu)
    selected_points = []

    def on_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            selected_points.append((x, y))
            ax.plot(x, y, 'ro')
            plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click once on each person to track, then close the window")
    plt.show()

    selected_masks_indices = []
    for point in selected_points:
        x, y = point
        for i, mask in enumerate(mask_stack):
            if mask[y, x] > 0:
                selected_masks_indices.append(i)

    selected_masks_indices = list(set(selected_masks_indices))
    return selected_masks_indices


if __name__ == '__main__':
    main()
