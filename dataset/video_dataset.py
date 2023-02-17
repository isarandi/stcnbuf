import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class GenericTestDataset(Dataset):
    def __init__(self, data_root, out_dir, res=480):
        self.image_dir = os.path.join(data_root, 'JPEGImages')
        self.mask_dir = os.path.join(data_root, 'Annotations')
        self.shape = {}
        self.frame_filenames = {}
        self.video_names = sorted(
            [x for x in os.listdir(self.image_dir) if not all_frames_done(x, data_root, out_dir)])

        for video_name in self.video_names:
            self.frame_filenames[video_name] = sorted(glob.glob(f'{self.image_dir}/{video_name}/*'))
            first_mask = os.listdir(os.path.join(self.mask_dir, video_name))[0]
            _mask = np.array(
                Image.open(os.path.join(self.mask_dir, video_name, first_mask)).convert("P"))
            self.shape[video_name] = np.shape(_mask)

        if res is not None:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(), im_normalization,
                transforms.Resize(res, interpolation=InterpolationMode.BICUBIC)])
            self.mask_transform = transforms.Compose([
                transforms.Resize(res, interpolation=InterpolationMode.NEAREST)])
        else:
            self.im_transform = transforms.Compose([transforms.ToTensor(), im_normalization])
            self.mask_transform = transforms.Compose([])

    def __getitem__(self, idx):
        video = self.video_names[idx]
        info = dict(
            name=video, frames=self.frame_filenames[video], size=self.shape[video], gt_obj={})
        vid_im_path = os.path.join(self.image_dir, video)
        vid_gt_path = os.path.join(self.mask_dir, video)
        frame_filenames = self.frame_filenames[video]

        frames = []
        masks = []
        for i, frame_filename in enumerate(frame_filenames):
            frame = Image.open(os.path.join(vid_im_path, frame_filename)).convert('RGB')
            frames.append(self.im_transform(frame))
            mask_path = os.path.join(vid_gt_path, frame_filename.replace('.jpg', '.png'))

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('P')
                palette = mask.getpalette()
                masks.append(np.array(mask, dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels != 0]
                info['gt_obj'][i] = this_labels
            else:
                masks.append(np.zeros(self.shape[video]))

        frames = torch.stack(frames, 0)
        masks = np.stack(masks, 0)

        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels != 0]
        info['labels'] = labels
        info['label_convert'] = {l: idx + 1 for idx, l in enumerate(labels)}
        info['label_backward'] = {idx + 1: l for idx, l in enumerate(labels)}
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        masks = self.mask_transform(masks).unsqueeze(2)
        return dict(rgb=frames, gt=masks, info=info, palette=np.array(palette))

    def __len__(self):
        return len(self.video_names)


def all_frames_done(video_name, data_root, out_dir):
    image_paths = glob.glob(f'{data_root}/JPEGImages/{video_name}/*.jpg')
    pred_mask_paths = glob.glob(f'{out_dir}/{video_name}/*.png')
    return len(image_paths) == len(pred_mask_paths)
