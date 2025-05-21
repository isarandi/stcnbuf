# Buffered STCN for Online and Offline Segmentation of Long Videos

This is a fork of the STCN video object segmentation method. It is optimized for online prediction for long
videos. That is, feeding a video stream to the network frame-by-frame and obtaining the predictions for each
frame immediately. By contrast, the original repository only allows batch processing where the whole video needs
to be loaded first and the result is given for the whole sequence at once.

To avoid running out of memory in case of long videos, I extended STCN with a buffer mechanism.
Once the memory bank buffer reaches a specified number of frames, each new insertion replaces a random
existing element in the buffer. This randomization results in an exponentially distributed sample, favoring
recent time steps.

For offline use, permuting the frames randomly and performing a burn-in pass (or several) achieves better segmentation quality. This way the memory will contain samples from the whole video sequence, which helps if the subject undergoes some appearance changes, such as lighting change, clothing change, distance or zoom, turning around, etc. The final pass can be done in order, keeping the previous few frames also in the memory in addition to the burn-in result.

For initializing the segmentation, the intial timestamp can be provided; otherwise the middle frame is used. Mask-RCNN is run on this frame and then either the user can click the subjects to track using a GUI, or the highest-scoring detections are used.

----

The code has been refactored in many other aspects as well, including removing einops usage in order to support TorchScript compilation.

## References

Original STCN:

[Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.

https://github.com/hkchengrex/STCN

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
