# Dataset: PlantVillage (tomato subset)

- **Kaggle:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Original / GitHub:** https://github.com/spMohanty/PlantVillage-Dataset
- **License:** CC BY-SA 3.0. Paper: Hughes & Salathé, *An open access repository of images on plant health*, arXiv:1511.08060.

The images are **not** stored in this repository (too large). Download them with one of the options below, then run `prepare_data.py`.

### Option A: Kaggle API
```bash
pip install kaggle            # put kaggle.json in ~/.kaggle (Windows: C:\Users\<you>\.kaggle)
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p raw --unzip
python src/prepare_data.py --src "raw/plantvillage dataset/color"
```

### Option B: GitHub (only the tomato folders, ≈650 MB)
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset raw_pv
cd raw_pv && git sparse-checkout set raw/color && cd ..
python src/prepare_data.py --src raw_pv/raw/color
```

### Split used (seed 42)
| class | train | val | test |
|---|---|---|---|
| each of 9 classes | 250 | 50 | 100 |
| Tomato_mosaic_virus | 233 | 46 | 94 |
| **total** | 2483 | 496 | 994 |

`sample_images/` holds one example test image per class.
