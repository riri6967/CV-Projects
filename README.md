# Computer Vision Projects: Low-Level vs High-Level Vision

Five independent computer vision projects submitted together. Each project solves a different real-world problem with a different CV concept, and **each one implements both**:

- a **low-level vision** algorithm: pixel-level image processing with no learning (colour spaces, filtering, edges, thresholding, morphology, contours, background subtraction, hand-crafted features), and
- a **high-level vision** algorithm: learned semantic understanding (CNN classification, object detection, facial landmarks, tracking-by-detection).

Both levels are evaluated on the **same held-out test data** with standard metrics, so the comparison is fair.

| | |
|---|---|
| **Student** |Rishi Gerolaga Kumar|
| **Register no.** |URK24RA3001|
| **Course** | Computer Vision  |
| **Institution** | Karunya Institute of Technology and Sciences |

---

## The five projects at a glance

| # | Project | CV concept | Low-level algorithm | High-level algorithm | Dataset | Key result (test set) |
|---|---|---|---|---|---|---|
| 1 | [**Crop Disease Detection**](01_crop_disease_detection/) | Image **classification** + segmentation | HSV colour segmentation, morphology, lesion severity %; colour/LBP/gradient features + SVM | CNN transfer learning (YOLOv8n-cls) + **Grad-CAM** explainability | PlantVillage (10 tomato classes) | Low: 87.8 % → **High: 94.6 %** accuracy |
| 2 | [**Pothole Detection**](02_pothole_detection/) | **Object detection** | CLAHE, anomaly maps (local contrast, Canny texture, intensity), Otsu, morphology, contour filtering | YOLOv8n detector (fine-tuned) | 4,054 road images, YOLO format | AP@0.5: Low 11.2 % → **High 85.4 %** |
| 3 | [**Lane Detection & Departure Warning**](03_lane_detection/) | **Geometric** feature extraction, video | HLS colour mask, Canny, ROI, **Hough transform**, line fitting, offset-based warning | YOLOv8n vehicle detection + ego-lane assignment + forward-collision warning | Udacity lane test videos | Lane found in 100 % of frames; warning **F1 98.8 %** |
| 4 | [**Driver Drowsiness Detection**](04_driver_drowsiness_detection/) | **Face analysis**, real-time monitoring | CLAHE, thresholding, Sobel, sclera cue, skin segmentation; **HOG** + SVM | CNN (4 classes) + **MediaPipe Face Mesh** landmarks (EAR / MAR) + PERCLOS alert logic | Kaggle Drowsiness dataset (leak-free split) | Eyes: 99.1 % → **100 %**; Yawn: 72.3 % → **98.6 %** |
| 5 | [**Multi-Object Tracking**](05_multi_object_tracking/) | **Motion analysis / tracking** | **MOG2 background subtraction**, morphology, blobs, Hungarian IoU tracker | YOLOv8n + **ByteTrack** | 6 MOTChallenge sequences | MOTA: Low 8.1 % → **High 64.8 %** |

**Overall finding:** low-level methods are fast, explainable and need no training data, and they work well when the target has a simple, fixed appearance (lane paint, cropped eyes, a static camera). High-level learned methods are needed when appearance varies (potholes, diseases, people, faces in a cabin), where they gain 7 to 74 percentage points.

---

## Repository structure
Every project folder has **the same layout**, so each one can be read the same way:

```
CV-Projects/
├── README.md                          ← this overview
├── SETUP_AND_RUN.md                   ← installation, Kaggle API token, how to run everything
├── 01_crop_disease_detection/
├── 02_pothole_detection/
├── 03_lane_detection/
├── 04_driver_drowsiness_detection/
└── 05_multi_object_tracking/
    ├── README.md          ← title, problem, objective, dataset, methodology, tools, results, conclusion
    ├── report.md          ← brief project report (intro, data, pre-processing, method, results, limitations)
    ├── requirements.txt   ← Python packages for this project
    ├── src/               ← source code (lowlevel_*.py = low-level, train/highlevel_*.py = high-level, evaluate*.py)
    ├── dataset/           ← README.md with dataset link + download commands, sample_images/
    ├── models/            ← trained weights (where training was done)
    ├── results/           ← metrics (JSON/CSV), output images and videos
    └── screenshots/       ← figures used in the README (pipelines, confusion matrices, demos)
```

## Assignment checklist
| Requirement | Where to find it (in every project folder) |
|---|---|
| Problem defined clearly | `README.md` → Problem statement / Objective |
| Dataset identified and described | `README.md` → Dataset, `dataset/README.md` (link, download, split) |
| CV methodology implemented | `src/` (low-level and high-level), `README.md` → Methodology |
| Pre-processing and analysis | `report.md` → section 3 |
| Meaningful results | `results/` (metrics files, output images, videos) |
| Performance evaluation with suitable metrics | `README.md` → Results, `results/*.json` |
| Sample input/output images and videos | `dataset/sample_images/`, `screenshots/`, `results/` |
| Limitations and challenges | `report.md` → section 6 |
| README, source code, dataset info, requirements.txt, results, evaluation, screenshots, report | all present in each folder |

## Quick start
```bash
# Python 3.10–3.12 recommended
cd 03_lane_detection
pip install -r requirements.txt
git clone --depth 1 https://github.com/udacity/CarND-LaneLines-P1 data
python src/lane_detector.py --input data/test_images --output results/images
```
See [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md) for all five projects.

## Notes
- Datasets are **not** stored in the repository (several GB). Each `dataset/README.md` gives the Kaggle and/or GitHub download commands, and `sample_images/` has examples so the code can be tried at once.
- All experiments were run on a CPU (2 cores). Training commands accept `--device 0` to use a GPU (e.g. Google Colab).
- Fixed random seeds (42) are used for data splits and training.
