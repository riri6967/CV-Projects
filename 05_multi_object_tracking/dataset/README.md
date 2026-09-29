# Dataset: MOTChallenge sequences (MOT16 + 2D MOT 2015)

- **Official site:** https://motchallenge.net (MOT16: https://motchallenge.net/data/MOT16/, 2D MOT 2015: https://motchallenge.net/data/MOT15/)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0
- **Kaggle mirror (MOT17 = MOT16 videos with extra detections):** https://www.kaggle.com/datasets/wenhoujinjust/mot-17

Sequences used: `MOT16-09`, `MOT16-11`, `PETS09-S2L1`, `TUD-Stadtmitte`, `TUD-Campus`, `KITTI-17`.

## Expected layout
```
data/sequences/<SEQ>/img1/000001.jpg ...
data/sequences/<SEQ>/gt/gt.txt          (frame,id,x,y,w,h,flag,class,visibility)
data/sequences/<SEQ>/seqinfo.ini
```

## Download options
**A. Official MOTChallenge zips** (MOT16.zip ≈ 1.9 GB, MOT15.zip ≈ 1.3 GB): copy the six sequence folders from `train/` into `data/sequences/`.

**B. Small GitHub copy of exactly these six sequences (≈ 350 MB)**, the copy used for these runs:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/st235/HSE.DeepLearning tmp_mot
cd tmp_mot && git sparse-checkout set data/sequences && cd ..
mkdir -p data && mv tmp_mot/data/sequences data/sequences
```

**C. Kaggle MOT17:** `kaggle datasets download -d wenhoujinjust/mot-17 --unzip`, then copy e.g. `MOT17-09-FRCNN` into `data/sequences/` (the code works with any MOT16/17 sequence).

`sample_images/` holds the first frame of each sequence.
