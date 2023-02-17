import glob
import os


def main():
    data_root = os.environ['DATA_ROOT']
    video_root = f'{data_root}/pku/RGB_VIDEO'
    relpaths = [
        os.path.relpath(p, video_root)
        for p in sorted(glob.glob(f'{video_root}/*.avi'))]

    two_person_seq_numbers = [
        5, 6, 15, 16, 25, 26, 35, 36, 45, 46, 55, 56, 65, 66, 75, 76, 85, 86, 95, 96, 105, 106, 115,
        116, 125, 126, 135, 136, 145, 146, 155, 156, 165, 166, 175, 176, 185, 186, 195, 196, 205,
        206, 215, 216, 225, 226, 235, 236, 245, 246, 255, 256, 261, 262, 271, 272, 285, 286, 299,
        300, 309, 310, 319, 320, 329, 330, 339, 340, 349, 350, 359, 360]

    two_person_relpaths = [p for p in relpaths if get_seq_number(p) in two_person_seq_numbers]
    with open(f'{data_root}/pku/two_person_videos.txt', 'w') as f:
        f.write('\n'.join(two_person_relpaths) + '\n')

    single_person_relpaths = [p for p in relpaths if p not in two_person_relpaths]
    with open(f'{data_root}/pku/single_person_videos.txt', 'w') as f:
        f.write('\n'.join(single_person_relpaths) + '\n')


def get_seq_number(path):
    return int(path[-10:-6])


if __name__ == '__main__':
    main()
