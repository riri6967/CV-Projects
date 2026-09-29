# Project Report: Multi-Object Tracking System

## 1. Introduction
Multi-object tracking (MOT) estimates the trajectories of all objects of a class, here pedestrians, through a video, keeping one identity per person. It combines **detection** (where are the people in this frame?) with **association** (which detection is which person from the previous frame?). We compare a classical low-level pipeline built on motion segmentation with a modern high-level tracking-by-detection system.

## 2. Dataset
Six public MOTChallenge sequences with ground-truth trajectories (MOT16 and 2D MOT 2015 benchmarks), 2,615 frames and 21,378 GT boxes of 140 different people:

| Sequence | Frames | People (IDs) | Setting |
|---|---|---|---|
| MOT16-09 | 525 | 25 | static, street level, 1080p, crowded pavement |
| PETS09-S2L1 | 795 | 19 | static, elevated view, well separated walkers |
| TUD-Stadtmitte | 179 | 10 | static, low view, heavy mutual occlusion |
| TUD-Campus | 71 | 8 | static, low view, people entering from the side |
| MOT16-11 | 900 | 69 | moving (hand-held), shopping mall |
| KITTI-17 | 145 | 9 | camera on a car, pedestrians at a crossing |

GT formats: MOT16 rows carry a class and a "consider" flag, and we evaluate only class 1 (pedestrian) with flag 1. 2D MOT 2015 rows contain only pedestrians.

## 3. Pre-processing and analysis
- **Low-level:** frames are used at native resolution. MOG2 is warmed up on the first 30 frames (learning rate 0.05) to build a background model. Foreground masks are cleaned with shadow removal (MOG2 labels shadows 127), a 5×5 median filter, opening, closing with a tall 9×15 ellipse (joins head, torso and legs) and dilation.
- **High-level:** YOLOv8n letterboxes each frame to 960 px (a larger input helps with the small people in the 1080p MOT16 sequences). Only the *person* class is kept.
- **Analysis:** pedestrians in these sequences range from ≈ 20 px (PETS, far away) to > 600 px tall (MOT16-11, close-up). TUD sequences have people constantly overlapping, which is the hardest case for blob-based methods.

## 4. Methodology
### 4.1 Low-level tracker
1. MOG2 background subtraction (mixture of Gaussians per pixel, adaptive).
2. Morphological clean-up → connected components → boxes.
3. Geometric filtering: area between 0.06 % and 8 % of the frame, height/width between 1.1 and 4.5.
4. Tracking: constant-velocity prediction of each track box, cost = 1 − IoU, optimal assignment with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), gate IoU ≥ 0.2, EMA update of box and velocity. Tracks are confirmed after 3 hits and removed after 8 misses.

Three parameter settings were compared (aspect range, minimum area, IoU gate, confirmation hits). The default above gave the best average MOTA.

### 4.2 High-level tracker
- **Detector:** YOLOv8n (COCO-pretrained, not fine-tuned on MOT), person class, confidence ≥ 0.1.
- **Tracker:** ByteTrack (Ultralytics implementation). A Kalman filter predicts each track. Detections are split into high- and low-confidence sets: high ones are matched first by IoU, then leftover tracks are matched with low-confidence boxes, which are often occluded people. Unmatched high boxes start new tracks, and lost tracks are kept for a buffer of frames.

### 4.3 Metrics
CLEAR-MOT and identity metrics computed with `motmetrics` (IoU ≥ 0.5 match): MOTA, MOTP, IDF1, precision, recall, ID switches, FP, FN, mostly tracked (≥ 80 % of the trajectory covered) and mostly lost (≤ 20 %).

## 5. Results
| Setting | Method | MOTA | IDF1 | Prec. | Recall | IDSW | FPS |
|---|---|---|---|---|---|---|---|
| Static (4 seq.) | Low-level | 8.1 % | 25.6 % | 54.7 % | 37.8 % | 101 | 39.9 |
| Static (4 seq.) | High-level | **64.8 %** | **67.4 %** | **81.9 %** | **84.8 %** | 99 | 5.9 |
| Moving (2 seq.) | High-level | 48.2 % | 55.7 % | 79.2 % | 67.9 % | 42 | 9.6 |

- The low-level tracker's best case is PETS09-S2L1 (MOTA 32.2 %, precision 71 %): an elevated camera with isolated walkers is exactly what background subtraction assumes. Its worst case is TUD-Stadtmitte (MOTA −51 %): people walk in groups close to the camera, so blobs merge and split and produce more false positives than true detections.
- The high-level tracker is best on PETS09 (MOTA 81.4 %, recall 93 %) and TUD-Stadtmitte (74.7 %). MOT16-09 is harder (42.0 %) because of many small, partly occluded people at the image edges. TUD-Campus has lower precision (59.5 %) because YOLO also detects people who are cut off at the image border and are not annotated in the GT.
- Mostly tracked / mostly lost trajectories on static sequences: low-level 6 / 29, high-level 41 / 3.

## 6. Limitations and challenges
- **Background subtraction needs a static camera** and fails with camera motion, lighting changes, stationary people and crowds. It also can't tell a person from a car or a moving shadow.
- **Blob merging:** occluding people form one blob, causing misses and ID switches. Separating them needs shape or appearance models, which is exactly what the CNN learns.
- **Detector not fine-tuned:** YOLOv8n was trained on COCO, not on MOT data. Fine-tuning on MOT17, a larger model (YOLOv8s/m) or an appearance re-ID model (DeepSORT / BoT-SORT) would reduce ID switches.
- **Speed:** the high-level tracker runs at about 6 FPS on a 2-core CPU at 960 px; real time needs a GPU or smaller input.
- **Evaluation simplification:** MOTChallenge's official evaluation also ignores distractor classes (static persons, reflections). Our evaluation keeps only pedestrian GT boxes, so matches to distractors count as false positives, which slightly lowers precision for both methods.

## 7. Conclusion
Low-level motion cues give a fast, training-free tracker that works only in simple, static, uncrowded scenes. High-level semantic detection plus a motion-aware association algorithm (ByteTrack) is far more accurate (MOTA +57 points) and robust to occlusion and camera motion, at a higher computational cost.
