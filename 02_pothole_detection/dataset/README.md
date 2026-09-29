# Dataset: Pothole detection (YOLO format)

**Used in this project:** merged dataset of 4,054 images / 5,601 boxes from
https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model (folder `dataset/`, MIT license).

```bash
# download only the dataset folder (~1.2 GB) into ./data
git clone --depth 1 --filter=blob:none --sparse https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model tmp_pothole
cd tmp_pothole && git sparse-checkout set dataset && cd ..
mv tmp_pothole/dataset data           # Windows: move tmp_pothole\dataset data
```
Expected layout: `data/images/{train,val,test}/*.jpg` and `data/labels/{train,val,test}/*.txt` (YOLO: `class xc yc w h`, normalised).

**Kaggle alternatives** (same task; convert VOC → YOLO if needed):
- https://www.kaggle.com/datasets/andrewmvd/pothole-detection (665 images, Pascal VOC XML): `kaggle datasets download -d andrewmvd/pothole-detection`
- https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset

| split | images |
|---|---|
| train | 2,749 |
| val | 654 |
| test | 651 (833 boxes) |

`sample_images/` holds 6 test images with their label files.
