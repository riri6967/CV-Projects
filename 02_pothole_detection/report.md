# Project Report: Road Pothole Detection

## 1. Introduction
Automatic pothole detection lets road agencies find and prioritise repairs without manual surveys. It is an **object detection** problem: find every pothole in an image and draw a box around it. This project compares a hand-designed low-level image-processing detector with a learned high-level detector (YOLOv8n).

## 2. Dataset
- 4,054 RGB road images with 5,601 pothole bounding boxes in YOLO format (class, x-centre, y-centre, width, height, normalised), merged from three public pothole datasets (image name prefixes `pothole_dataset_1/2/3`).
- Split: train 2,749 / val 654 / test 651 images (test: 833 boxes, 2 images without potholes).
- **Content analysis:** the images range from close-ups of one pothole to street scenes with many small potholes. Potholes may be dry, water-filled, snowy or patched, on dark asphalt or light concrete. Dataset 3 has 4 augmented crops per original photo, and all crops of one photo are in the same split.

## 3. Pre-processing
- **Low-level:** resize to 640 px width (keeping aspect), grayscale, CLAHE (clip 2.0, 8×8 tiles) to normalise the very different exposures, and 5×5 Gaussian blur against asphalt grain.
- **YOLO:** letterbox resize to 416×416; online augmentation with mosaic, HSV jitter, horizontal flip, scale/translate (Ultralytics defaults).

## 4. Methodology
### 4.1 Low-level detector
A pothole is an **anomaly** in the road surface. Three anomaly maps are combined:
1. **Local contrast:** \|I − G_σ=31 * I\|, then smoothed (σ = 5). This responds to rims and interior shading.
2. **Texture:** Canny (60/160) edge density in 31×31 windows, taken as the absolute difference from the image median. Water-filled potholes are *smoother* than asphalt; broken ones are *rougher*.
3. **Intensity:** \|G_σ=9 * I − median(I)\|. Potholes are darker (holes, water) or lighter (dust, concrete base).

Score = 0.3·(1) + 0.4·(2) + 0.3·(3), each min-max normalised. The weights were chosen on the **validation** set: AP@0.5 (on every 3rd val image) was 0.5 % with local contrast + raw edge density, 7.5 % with local contrast + texture anomaly, and 11.1 % with all three maps. Otsu thresholding, closing and opening give candidate regions. Contours are filtered by area, solidity and aspect ratio. The top 3 boxes are kept, with confidence equal to the mean score inside the region.

### 4.2 YOLOv8n
One-stage, anchor-free CNN detector (CSP backbone, PAN neck, decoupled head), pretrained on COCO and fine-tuned for 10 epochs (batch 16, 416 px, SGD/AdamW auto, CPU, ≈ 2 h). Best weights were selected by validation fitness (mAP@0.5 = 86.5 %, mAP@0.5:0.95 = 59.3 % on val).

### 4.3 Evaluation protocol
The same code (`src/evaluate.py`) evaluates both methods. Predictions are sorted by confidence and greedily matched to ground truth at **IoU ≥ 0.5**, and AP is computed as the area under the interpolated precision-recall curve (VOC all-point). Precision, recall and F1 are reported at the confidence that maximises F1. YOLO is also evaluated with the official Ultralytics validator.

## 5. Results
| | AP@0.5 | P | R | F1 | ms/img |
|---|---|---|---|---|---|
| Low-level | 11.2 % | 21.4 % | 36.3 % | 26.9 % | 106 |
| YOLOv8n | **85.4 %** | 83.4 % | 79.7 % | 81.5 % | 48 |
| YOLOv8n (Ultralytics val) | mAP50 85.5 %, mAP50-95 59.7 % | 83.8 % | 79.5 % | – | 46 |

**Training behaviour:** validation mAP@0.5 rose from 57 % (epoch 1) to 86.5 % (epoch 10) and was still increasing. Training longer on a GPU would likely improve it further.

**Qualitative (`screenshots/comparison_examples.jpg`):** the low-level method usually covers the pothole area but with oversized boxes that merge neighbouring defects, and it fires on shadows, cracks and road edges. YOLO boxes are tight. YOLO misses some small or shallow potholes in crowded scenes, and occasionally boxes rubble or patched asphalt.

## 6. Limitations and challenges
- **Label ambiguity:** whether a shallow depression, crack cluster or patch counts as a pothole differs between the source datasets, which caps both methods.
- **Low-level method:** the global thresholds and weights can't adapt to every road texture. Box localisation is poor because anomaly regions spread along cracks and shadows. It doesn't use shape or 3-D cues.
- **Small objects:** at 416 px, distant potholes in street scenes are only a few pixels wide. Higher resolution or tiling would help.
- **Compute:** CPU-only training limited us to 10 epochs of the nano model. YOLOv8s/m on a GPU for 50–100 epochs typically gains 5–10 mAP points.
- **Monocular 2-D:** neither method estimates depth or severity. A stereo or depth camera would be needed for that.

## 7. Conclusion
Pothole appearance is too variable for fixed low-level rules (AP 11 %). A learned high-level detector captures the concept and reaches 85 % AP@0.5 while also running faster. Low-level anomaly maps remain useful for visualising why a region looks suspicious.
