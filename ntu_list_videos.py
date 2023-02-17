import glob
import os


def main():
    data_root = os.environ['DATA_ROOT']
    video_root = f'{data_root}/ntu/nturgb+d_rgb'
    relpaths = [
        os.path.relpath(p, video_root)
        for p in sorted(glob.glob(f'{video_root}/**/*.avi', recursive=True))]

    single_person_actions = [*range(1, 50), *range(61, 106)]
    single_person_relpaths = [p for p in relpaths if get_action(p) in single_person_actions]
    with open(f'{data_root}/ntu/single_person_videos.txt', 'w') as f:
        f.write('\n'.join(single_person_relpaths) + '\n')

    two_person_actions = [*range(50, 61), *range(106, 121)]
    two_person_relpaths = [p for p in relpaths if get_action(p) in two_person_actions]
    with open(f'{data_root}/ntu/two_person_videos.txt', 'w') as f:
        f.write('\n'.join(two_person_relpaths) + '\n')


def get_action(path):
    return int(path[-11:-8])


if __name__ == '__main__':
    main()
