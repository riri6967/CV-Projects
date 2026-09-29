# Project Report: AI-Based Crop Disease Detection

## 1. Introduction
Tomato plants suffer from bacterial, fungal, viral and pest diseases whose first signs appear on the leaves. Manual scouting is slow and needs expertise. This project builds two automatic pipelines that take a leaf photo and decide whether the leaf is diseased and which disease it has:

- a **low-level** pipeline that works only on pixels (colour spaces, filtering, thresholding, morphology, contours, texture descriptors), and
- a **high-level** pipeline that learns semantic features with a convolutional neural network.

## 2. Dataset
- **Source:** PlantVillage (Hughes & Salathé, 2015). Kaggle mirror `abdallahalidev/plantvillage-dataset`; original on GitHub `spMohanty/PlantVillage-Dataset`.
- **Subset:** the 10 tomato classes: Bacterial spot, Early blight, Late blight, Leaf mould, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy.
- **Split:** 250 / 50 / 100 images per class for train / val / test (mosaic virus 233/46/94). That is 2,483 / 496 / 994 images, drawn at random with seed 42.
- **Properties:** 256×256 RGB photos of one leaf on a grey or purple background, taken in controlled lighting.

## 3. Pre-processing and analysis
- Resize to 256×256 (low-level) or 224×224 (CNN).
- Median blur (5×5) removes sensor noise before colour thresholding.
- HSV conversion separates colour (hue) from illumination (value), so the thresholds are less sensitive to brightness.
- Leaf-mask erosion (9×9) removes the shadowed rim of the leaf, which otherwise looked like necrosis.
- The CNN uses flip augmentation and small saturation/value jitter. Hue jitter is almost zero, because yellowing is the symptom itself.
- **Analysis of pixel statistics:** in healthy leaves 90 % of the pixels have hue between 35 and 58. Diseased classes such as early/late blight extend down to hue ≈ 11 (brown), and leaf mould up to ≈ 113 (grey-purple). This analysis set the healthy-green band (35–65).

## 4. Methodology
### 4.1 Low-level
1. **Leaf segmentation:** HSV threshold → closing (7×7 ellipse, 2 iterations) → opening → largest external contour, filled.
2. **Lesion segmentation:** inside the eroded leaf, pixels with S > 40 and hue outside 35–65 → 3×3 opening.
3. **Severity** = lesion area / leaf area. The rule predicts *diseased* if severity > T, with T = 5.3 % chosen on the validation set by balanced accuracy (the test set is 9:1 diseased:healthy).
4. **Hand-crafted descriptor (86-D):** 30-bin hue, 16-bin saturation and 16-bin value histograms inside the leaf, a 16-bin Sobel gradient-magnitude histogram, a 32-bin Local Binary Pattern histogram, Laplacian variance and severity. These are standardised and classified by an **SVM with RBF kernel** (C = 10).

### 4.2 High-level
1. **Model:** YOLOv8n-cls, a 1.4 M-parameter CNN pre-trained on ImageNet, with the last layer replaced for 10 classes (transfer learning).
2. **Training:** 8 epochs, batch 32, 224×224, Ultralytics default optimiser and LR schedule (`optimizer=auto`). CPU training time was about 11 minutes (2 cores).
3. **Explainability:** Grad-CAM on layer `model[-2]` (last C2f block). Heat-maps are overlaid on the images and compared with the low-level lesion masks.

## 5. Results
| Method | Test metric | Value |
|---|---|---|
| Severity rule (low-level) | balanced accuracy (healthy vs diseased) | 78.8 % |
| | ROC-AUC | 86.1 % |
| Hand-crafted features + SVM (low-level) | 10-class accuracy | 87.8 % |
| | macro F1 | 87.8 % |
| CNN (high-level) | 10-class accuracy | **94.6 %** |
| | macro precision / recall / F1 | 94.8 / 94.6 / 94.6 % |
| | healthy vs diseased accuracy | 99.4 % |
| | CPU inference | 9.3 ms / image |

- CNN validation accuracy rose from 65 % (epoch 1) to 96.2 % (epoch 8) without over-fitting (`screenshots/cnn_training_curves.png`).
- The CNN's weakest classes are Early blight and Target spot (F1 89.4 %), which it confuses with each other because both show brown concentric lesions. The SVM had the same difficulty (Early/Late blight F1 76.8 %).
- Grad-CAM: 12.8 % of the heat falls on lesion pixels that cover 10.3 % of the leaf. The network focuses on lesions more than chance, and also on leaf texture and shape.

## 6. Limitations and challenges
- **Lab images:** PlantVillage has one leaf per image on a clean background. Accuracy is known to drop sharply on field photos with clutter, several leaves and uneven sunlight.
- **Colour-only lesions:** the low-level rule can't see viral mosaic or spider-mite damage, which change texture but not hue. It also flags healthy leaves with yellowish veins or glare (334 false negatives / 5 false positives at the chosen threshold).
- **Thresholds are hand-tuned** for this background and camera, so a new setting needs re-tuning.
- **Compute:** we trained on CPU with a subset (≈250 images per class). The full 18k tomato images on a GPU would likely push accuracy above 98 %.
- **Dataset access:** Kaggle was not reachable from the build machine, so the identical GitHub original of PlantVillage was used.

## 7. Conclusion
Low-level vision gives an interpretable severity measure and a reasonable classical baseline (87.8 %). High-level CNN features are clearly more discriminative (94.6 %, 99.4 % for healthy vs diseased). Using the CNN for diagnosis and the low-level mask for severity combines the strengths of both.
