import argparse
import glob
import os

# https://github.com/pytorch/pytorch/issues/21956
os.environ['OMP_NUM_THREADS'] = '1'

from eval_video import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--file-pattern', type=str, default='**/*.mp4')
    parser.add_argument('--file-list-file', type=str)
    parser.add_argument('--videos-per-task', type=int, default=1)
    parser.add_argument('--model', default='saves/stcn.pth')
    parser.add_argument('--output', type=str)
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--mem-every', default=5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--mem-size', default=25, type=int)
    parser.add_argument('--max-persons', default=None, type=int)
    parser.add_argument('--resolution', default=320, type=int)
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()
    torch.autograd.set_grad_enabled(False)

    if args.file_list_file:
        with open(args.file_list_file) as f:
            lines = f.read().splitlines()

        video_paths = sorted([p if os.path.isabs(p) else f'{args.video_dir}/{p}' for p in lines])
    else:
        globs = [glob.glob(f'{args.video_dir}/{p}', recursive=True)
                 for p in args.file_pattern.split(',')]
        video_paths = sorted([x for l in globs for x in l])

    relpaths = [os.path.relpath(video_path, args.video_dir) for video_path in video_paths]
    output_paths = [f'{args.output_dir}/{os.path.splitext(relpath)[0]}.pkl' for relpath in relpaths]

    get_group = get_group_pku if 'pku/' in args.video_dir else get_group_ntu
    video_path_groups = myutils.groupby(video_paths, get_group)
    output_path_groups = myutils.groupby(output_paths, get_group)

    group_ids = list(video_path_groups.keys())
    print(len(group_ids))
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    group_id = group_ids[i_task]
    video_paths_group = video_path_groups[group_id]
    output_paths_group = output_path_groups[group_id]

    if all(os.path.exists(p) for p in output_paths_group):
        return

    init_segmenter = mask_init.MaskRCNN().cuda().eval()
    prop_model = load_prop_model(args.model)
    stcn = STCNInference(prop_model, args.mem_size, args.top, args.resolution).cuda().eval()

    initialized = False
    n_objects = None
    with torch.cuda.amp.autocast():
        for video_path, output_path in zip(video_paths_group, output_paths_group):
            print(f'Processing {video_path}...', flush=True)
            frames = imageio.get_reader(video_path)
            if not initialized:
                # Segment the first frame (which is outside the scope of STCN)
                frames = more_itertools.peekable(frames)
                initial_frame = frames.peek()
                initial_mask = init_segmenter.predict(initial_frame)[:args.max_persons]
                n_objects = len(initial_mask)  # np.max(initial_label_map)
                stcn.initialize(initial_frame, initial_mask)
                initialized = True

            ds = VideoDataset(frames, stcn.im_transform)
            frame_loader = torch.utils.data.DataLoader(
                ds, num_workers=1, batch_size=args.mem_every, prefetch_factor=5)

            results = []
            for frame_batch in frame_loader:
                mask_batch = stcn.predict_batch(frame_batch.cuda())
                label_map_batch = torch.argmax(mask_batch, dim=0)
                label_map_batch = myutils.to_numpy(label_map_batch).astype(np.uint8)
                results += [myutils.encode_label_map(lm, n_objects) for lm in label_map_batch]

                if args.viz:
                    frame_batch = (inv_im_trans(frame_batch).detach().cpu().numpy().transpose(
                        [0, 2, 3, 1]) * 255).astype(np.uint8)
                    for frame, label_map in zip(frame_batch, label_map_batch):
                        visu = myutils.plot_with_masks(frame, label_map)
                        cv2.imshow('image', visu[..., ::-1])
                        cv2.waitKey(1)
            myutils.dump_pickle(results, output_path)


def get_group_ntu(p):
    name = os.path.basename(p)
    return name[:4] + name[8:12]

def get_group_pku(p):
    name = os.path.basename(p)
    return name[:4]


if __name__ == '__main__':
    main()
