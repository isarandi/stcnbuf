import os

# https://github.com/pytorch/pytorch/issues/21956
os.environ['OMP_NUM_THREADS'] = '1'
import itertools
from argparse import ArgumentParser

import cv2
import imageio
import more_itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import mask_init
import myutils
from dataset.range_transform import im_normalization, inv_im_trans
from inference_memory_bank import MemoryBank
from model.aggregate import aggregate
from model.eval_network import STCN
from util.tensor_util import pad_divide_by, unpad


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', default='saves/stcn.pth')
    parser.add_argument('--output', type=str)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--mem-every', default=5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--mem-size', default=25, type=int)
    parser.add_argument('--start-frame', default=0, type=int)
    parser.add_argument('--max-persons', default=None, type=int)
    parser.add_argument('--resolution', default=320, type=int)
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()
    torch.autograd.set_grad_enabled(False)

    init_segmenter = mask_init.MaskRCNN().cuda().eval()
    prop_model = load_prop_model(args.model)
    stcn = STCNInference(prop_model, args.mem_size, args.top, args.resolution).cuda().eval()

    with torch.cuda.amp.autocast():
        process_video(
            args.video_path, init_segmenter, stcn, args.start_frame, args.out_video_path,
            args.max_persons, args.mem_every, visualize=args.viz)


def process_video(
        video_path, init_segmenter, vos_model, start_frame, out_video_path, max_persons, mem_every,
        visualize):
    print(f'Processing {video_path}...')
    frames = imageio.get_reader(video_path)
    frames = itertools.islice(frames, start_frame, None)

    writer = imageio.get_writer(
        out_video_path, fps=25, codec='h264', ffmpeg_params=['-crf', '17'],
        macro_block_size=None) if out_video_path else None

    # Segment the first frame (which is outside the scope of STCN)
    frames = more_itertools.peekable(frames)
    initial_frame = frames.peek()
    initial_mask = init_segmenter.predict(initial_frame)[:max_persons]
    n_objects = len(initial_mask)  # np.max(initial_label_map)
    vos_model.initialize(initial_frame, initial_mask)
    ds = VideoDataset(tqdm.tqdm(frames), vos_model.im_transform)
    frame_loader = torch.utils.data.DataLoader(
        ds, num_workers=1, batch_size=mem_every, prefetch_factor=5)

    results = []
    for frame_batch in frame_loader:
        mask_batch = vos_model.predict_batch(frame_batch.cuda())
        label_map_batch = torch.argmax(mask_batch, dim=0)
        label_map_batch = myutils.to_numpy(label_map_batch).astype(np.uint8)
        results += [myutils.encode_label_map(lm, n_objects) for lm in label_map_batch]

        if visualize:
            frame_batch = (inv_im_trans(frame_batch).detach().cpu().numpy().transpose(
                [0, 2, 3, 1]) * 255).astype(np.uint8)
            for frame, label_map in zip(frame_batch, label_map_batch):
                visu = myutils.plot_with_masks(frame, label_map)
                if out_video_path:
                    writer.append_data(visu)
                cv2.imshow('image', visu[..., ::-1])
                cv2.waitKey(1)
    if out_video_path:
        writer.close()

    return results


class STCNInference(nn.Module):
    def __init__(self, prop_model, mem_size, topk, target_resolution):
        super().__init__()
        self.memory_bank = None
        self.prop_model = prop_model
        self.topk = topk
        self.mem_size = mem_size
        self.n_objects = None
        self.im_transform = transforms.Compose([
            transforms.ToTensor(), im_normalization,
            transforms.Resize(target_resolution, interpolation=InterpolationMode.BICUBIC)])
        self.mask_transform = transforms.Compose([
            transforms.Resize(target_resolution, interpolation=InterpolationMode.NEAREST)])

    def initialize(self, frame, mask):
        self.n_objects = len(mask)
        mask = self.mask_transform(mask).cuda()
        bg_mask = 1 - torch.sum(mask, dim=0, keepdim=True)
        mask = torch.cat([bg_mask, mask], 0).unsqueeze(1)
        mask, _ = pad_divide_by(mask, 16)
        prob = aggregate(mask[1:])[1:]

        frame = self.im_transform(frame)
        frame = frame.unsqueeze(0).cuda()
        frame, pad = pad_divide_by(frame, 16)
        key, _, qf16, _, _ = self.prop_model.encode_key(frame)

        value = self.prop_model.encode_value(frame, qf16, prob)
        self.memory_bank = MemoryBank(k=self.n_objects, top_k=self.topk, memory_size=self.mem_size)
        self.memory_bank.add_memory(key.squeeze(0), value.squeeze(1))

    def predict_batch(self, frame, add_last_to_memory=True):
        frame, pad = pad_divide_by(frame, 16)
        k16, qv16, qf16, qf8, qf4 = self.prop_model.encode_key(frame)
        mask = self.prop_model.segment_with_query(self.memory_bank, qf8, qf4, k16, qv16)
        mask = aggregate(mask)

        if add_last_to_memory:
            last_value = self.prop_model.encode_value(frame[-1:], qf16[-1:], mask[1:, -1:])
            last_key = k16[-1]
            self.memory_bank.add_memory(last_key, last_value.squeeze(1))

        return unpad(mask, pad)


class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable, transform):
        self.iterable = iterable
        self.transform = transform

    def __iter__(self):
        for frame in self.iterable:
            yield self.transform(frame)


def load_prop_model(model_path):
    prop_model = STCN()
    prop_model = prop_model.cuda()
    prop_model = prop_model.eval()
    prop_saved = torch.load(model_path)
    name = 'value_encoder.conv1.weight'
    if name in prop_saved and prop_saved[name].shape[1] == 4:
        pads = torch.zeros((64, 1, 7, 7), device=prop_saved[name].device)
        prop_saved[name] = torch.cat([prop_saved[name], pads], 1)
    prop_model.load_state_dict(prop_saved)
    return prop_model


if __name__ == '__main__':
    main()
