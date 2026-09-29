# Project Report: Driver Drowsiness Detection

## 1. Introduction
Drowsy driving slows reaction time much as alcohol does. Two robust visual indicators are **prolonged eye closure** (PERCLOS) and **yawning**. This project detects both from images, compares low-level and high-level computer vision, and packages the result as a real-time monitor.

## 2. Dataset
- Kaggle `dheerajperumandla/drowsiness-dataset`: 2,900 images in 4 classes (Closed 726, Open 726, yawn 723, no_yawn 725).
- Eye classes are close-up eye crops of varying size (≈ 100–1,200 px) from many people. Yawn classes are 640×480 in-car frames (from the YawDD video dataset) of a limited set of drivers.
- **Split:** near-duplicate grouping (24×18 thumbnails, mean-subtracted, agglomerative clustering with average linkage; distance threshold 3 for eyes and 8 for faces) → 1,122 eye groups and 73 face groups. Whole groups were assigned to train/val/test. Final counts: eyes 1,016 / 218 / 218; faces 1,003 / 225 / 220.
- **Why:** with a naive random split the low-level HOG-SVM reached 99 % on yawns only because it had seen almost the same frame during training. With the grouped split it drops to 72 %. This was the main data-analysis finding of the project.

## 3. Pre-processing
- Eye crops: resize 64×64 → grayscale → CLAHE (clip 2, 4×4 tiles) to equalise skin tone and lighting → 3×3 Gaussian blur.
- Face frames: resize 320×240 for skin segmentation; 160×120 grayscale + CLAHE for HOG.
- CNN: 224×224, horizontal flips and HSV jitter. No vertical flips, since an upside-down face is unrealistic.

## 4. Methodology
### 4.1 Low-level
**Eye cues:** (1) threshold at 0.55 × median → opening → largest dark blob, whose height/width is high for a round iris and low for a lash line; (2) Sobel-y / Sobel-x energy ratio (a closed lid gives strong horizontal edges); (3) sclera share: HSV pixels with low saturation and high value; (4) blob fill ratio; (5) darkest-pixel share.
**Mouth cues:** YCrCb skin mask (Cr 135–175, Cb 85–130) → opening/closing → face-shaped blob → lower-face ROI (60–90 % of the face height) → CLAHE → dark "cavity" pixels (< 0.5 × median) → share, blob height/width; plus holes inside the skin mask.
**Classifiers:** (A) the single most discriminative cue with a threshold fitted on train; (B) HOG (9 bins, 8×8 cells, 16×16 blocks) plus all cues → standardised → SVM (RBF, C = 10).

### 4.2 High-level
- **YOLOv8n-cls CNN** fine-tuned on 4 classes: 10 epochs, batch 32, CPU (≈ 11 min). Best validation top-1 was 99.1 %.
- **MediaPipe Face Mesh** (468 landmarks, attention-refined eyes and lips) → EAR and MAR. The MAR threshold (0.35) was chosen on the training frames.

### 4.3 Real-time decision logic
Eye closed if EAR < 0.21 (or CNN/SVM on the landmark-cropped eyes). DROWSY if closed for ≥ 15 consecutive frames or PERCLOS(90 frames) > 40 %. YAWN if MAR > 0.35 for ≥ 10 frames.

## 5. Results
| Task | Low-level best | High-level best |
|---|---|---|
| Eye closed/open | 99.1 % (HOG + SVM); single cue 74.3 % | **100 %** (CNN) |
| Yawn | 72.3 % (HOG + SVM); single cue 53.6 % | **98.6 %** (CNN); 96.4 % (MAR, AUC 99.7 %) |
| 4-class | – | **99.3 %** (CNN) |

- Best individual low-level eye cues (test AUC): dark-blob h/w 78.5 %, Sobel ratio 77.4 %, sclera share 71.8 %.
- MediaPipe failed to find a face in 1 of 220 test frames.
- The monitor was tested offline on a sequence built from test frames (`results/highlevel/demo_sequence_cnn.mp4`). The live webcam demo is run with `drowsiness_monitor.py`.

## 6. Limitations and challenges
- **Few drivers in the yawn data:** only about 73 person/session groups, so results may not generalise to new faces, cameras or night-time IR images.
- **Eye crops vs live crops:** the eye CNN was trained on tight, well-lit crops. On landmark crops from cabin video it is less reliable: sunglasses are read as "closed" (see the GIF), and squinting while yawning looks like closed eyes.
- **Skin-colour segmentation** (low-level yawn) fails with skin-coloured seats, strong sunlight, beards and hands on the face.
- **Static images:** real drowsiness is temporal. Proper evaluation needs labelled driving videos (e.g. NTHU-DDD, UTA-RLDD).
- **MediaPipe version:** the legacy `solutions` API needs `mediapipe==0.10.14` (Python ≤ 3.12).

## 7. Conclusion
Low-level vision works when the object is already localised and has a clear shape (an eye crop), but fails when it must first find a small part (the mouth) in a cluttered scene. High-level models solve localisation and appearance together and reach 96–100 % on both tasks. A practical system uses high-level landmarks for localisation and simple, explainable measures (EAR, MAR, PERCLOS) for the final decision.
