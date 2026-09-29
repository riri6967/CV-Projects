# Dataset: Drowsiness_dataset (Kaggle)

- **Link:** https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset (169 MB zip)
- **Classes:** `Closed` (726), `Open` (726), `yawn` (723), `no_yawn` (725)
- The yawn / no_yawn frames come from the YawDD in-car video dataset (Abtahi et al., 2014).

## Download
```bash
pip install kaggle          # API token: kaggle.com → Settings → API → Create New Token → kaggle.json
kaggle datasets download -d dheerajperumandla/drowsiness-dataset -p raw --unzip
python src/prepare_data.py --src raw
```
Or download the zip in the browser, unzip it into `raw/`, and run the last line.

## Leak-free split
`prepare_data.py` groups near-duplicate images (video frames of the same person) and assigns whole groups to one split.
`split_groups.csv` in this folder lists every file with its split and group id, so the exact split can be checked.

| class | train | val | test |
|---|---|---|---|
| Closed | 494 | 103 | 129 |
| Open | 522 | 115 | 89 |
| yawn | 495 | 127 | 101 |
| no_yawn | 508 | 98 | 119 |

`sample_images/` holds one test image per class.
