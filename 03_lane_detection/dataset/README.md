# Dataset: Udacity CarND Lane-Lines test set

- **Source:** https://github.com/udacity/CarND-LaneLines-P1 (MIT license)
- **Contents:** `test_images/` (6 × 960×540 JPG) and `test_videos/` (`solidWhiteRight.mp4`, `solidYellowLeft.mp4`, `challenge.mp4`)

```bash
git clone --depth 1 https://github.com/udacity/CarND-LaneLines-P1 data
```
The scripts expect `data/test_images` and `data/test_videos`.

**Optional larger benchmark (Kaggle):** TuSimple lane dataset: https://www.kaggle.com/datasets/manideep1108/tusimple
(`kaggle datasets download -d manideep1108/tusimple`). Use any clip folder's frames with `src/lane_detector.py --input <folder>`.

`sample_images/` holds the 6 test images so that the project runs without downloading anything.
