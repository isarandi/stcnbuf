# Buffered STCN for Online Segmentation of Long Videos

This is a fork of the STCN video object segmentation method. It is optimized for online prediction for long
videos. That is, feeding a video stream to the network frame-by-frame and obtaining the predictions for each
frame immediately. By contrast, the original repository only allows batch processing where the whole video needs
to be loaded first and the result is given for the whole sequence at once.

To avoid running out of memory in case of long videos, I extended STCN with a buffer mechanism.
Once the memory bank buffer reaches a specified number of frames, each new insertion replaces a random
existing element in the buffer. This randomization results in an exponentially distributed sample, favoring
recent time steps.

## Usage Example

Check out the file `segment_3dpw.py` to see how this code can be used.
To generate segmentation results for the annotated people in the 3DPW dataset, use:

```bash
DATA_ROOT=some_path # we assume that 3dpw is stored under $DATA_ROOT/3dpw
python segment_3dpw.py --output-dir=$some_path --mem-every=3 --mem-size=128
```

## Acknowledgments

[Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

Original repo: https://github.com/hkchengrex/STCN

BibTeX:

```bibtex
@inproceedings{cheng2021stcn,
  title={Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{cheng2021mivos,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```
