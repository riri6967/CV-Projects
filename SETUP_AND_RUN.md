# Setup and Run Guide

## 1. Install Python and a virtual environment
Use Python **3.10–3.12** (MediaPipe 0.10.14 in project 4 does not support 3.13).

```bash
python -m venv cvenv
# Windows:  cvenv\Scripts\activate        Linux/macOS:  source cvenv/bin/activate
pip install --upgrade pip
```
Each project has its own `requirements.txt`. Install the one for the project you want to run:
```bash
pip install -r 01_crop_disease_detection/requirements.txt
```

## 2. Kaggle API token (for Kaggle downloads)
1. Log in at https://www.kaggle.com → click your profile picture → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Put it in `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS).
3. `pip install kaggle`, then test it with `kaggle datasets list -s pothole`.

## 3. Get the data and run each project
Run all commands from inside the project folder.

### 01: Crop disease detection
```bash
cd 01_crop_disease_detection
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p raw --unzip
python src/prepare_data.py --src "raw/plantvillage dataset/color"
python src/lowlevel_segmentation.py --demo         # low-level segmentation examples
python src/lowlevel_evaluate.py                    # low-level metrics
python src/train.py --epochs 8                     # high-level CNN (≈ 11 min CPU)
python src/evaluate.py                             # CNN metrics + comparison chart
python src/gradcam.py                              # Grad-CAM explanations
```

### 02: Pothole detection
```bash
cd 02_pothole_detection
# dataset → ./data (see dataset/README.md)
python src/lowlevel_detector.py --image dataset/sample_images/pothole_dataset_2_00608.jpg
python src/train.py --epochs 10                    # ≈ 2 h CPU, ≈ 5 min on a GPU (--device 0)
python src/evaluate.py                             # low-level vs YOLO on the test split
```

### 03: Lane detection and departure warning
```bash
cd 03_lane_detection
git clone --depth 1 https://github.com/udacity/CarND-LaneLines-P1 data
python src/evaluate.py                             # low-level: videos, metrics, drift test
python src/scene_understanding.py                  # high-level: vehicles + ego lane + FCW
```

### 04: Driver drowsiness detection
```bash
cd 04_driver_drowsiness_detection
kaggle datasets download -d dheerajperumandla/drowsiness-dataset -p raw --unzip
python src/prepare_data.py --src raw               # leak-free grouped split
python src/lowlevel_evaluate.py
python src/train_cnn.py --epochs 10                # or use models/drowsy_cls_yolov8n.pt
python src/highlevel_evaluate.py
python src/drowsiness_monitor.py --method cnn      # LIVE webcam demo, press q to quit
```

### 05: Multi-object tracking
```bash
cd 05_multi_object_tracking
# sequences → ./data/sequences (see dataset/README.md)
python src/evaluate.py                             # both trackers on all sequences
python src/highlevel_tracker.py --seq PETS09-S2L1  # single sequence
```

## 4. Demonstration tips
- **Live demo:** project 4's `drowsiness_monitor.py` uses your laptop webcam. Close your eyes for about 1 second to trigger the alert, or open your mouth wide to trigger the yawn alert.
- **Videos:** `03_lane_detection/results/videos/*.mp4`, `03_lane_detection/results/scene/*.mp4` and `05_multi_object_tracking/results/*/*.mp4` show the output of each algorithm.
- **Modifying:** thresholds and parameters are named constants at the top of each `src/*.py` file (e.g. `DEPARTURE_THRESH` in `lane_detector.py`, `HEALTHY_HUE` in `lowlevel_segmentation.py`), so you can change one and re-run the evaluation.
