"""Real-time driver drowsiness monitor (webcam or video file). Use this for the live demo.

Per frame:
  1. MediaPipe Face Mesh -> eye / mouth landmarks (high-level)
  2. Eye state:
       --method ear     : EAR < EAR_THRESH                      (landmark geometry)
       --method cnn     : YOLOv8n-cls on the two eye crops      (high-level CNN)
       --method lowlevel: HOG + geometric cues + SVM on crops   (low-level features)
  3. Yawn: MAR > MAR_THRESH
  4. Temporal logic:
       * DROWSY ALERT when the eyes are closed for >= CLOSED_FRAMES consecutive frames
       * PERCLOS = % of the last 90 frames with closed eyes; alert when > 40 %
       * YAWN ALERT when MAR is high for >= YAWN_FRAMES consecutive frames

Usage:
    python src/drowsiness_monitor.py                      # webcam 0, EAR method
    python src/drowsiness_monitor.py --method cnn --source my_drive.mp4 --save results/demo.mp4
Press q to quit.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import FaceLandmarks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THRESH_FILE = ROOT / "results" / "highlevel" / "thresholds.json"
EAR_THRESH, MAR_THRESH = 0.21, 0.5
CLOSED_FRAMES, YAWN_FRAMES, PERCLOS_WIN, PERCLOS_ALERT = 15, 10, 90, 0.40


class EyeClassifier:
    def __init__(self, method):
        self.method = method
        if method == "cnn":
            from ultralytics import YOLO
            self.model = YOLO(str(ROOT / "models" / "drowsy_cls_yolov8n.pt"))
        elif method == "lowlevel":
            import joblib
            import lowlevel_features as F
            self.F = F
            self.model = joblib.load(ROOT / "models" / "eye_hog_svm.joblib")

    def closed(self, frame, lm):
        if self.method == "ear":
            return lm["ear"] < EAR_THRESH
        votes = []
        for pts in (lm["left_eye"], lm["right_eye"]):
            crop = FaceLandmarks.eye_crop(frame, pts)
            if crop.size == 0:
                continue
            if self.method == "cnn":
                r = self.model.predict(crop, imgsz=224, verbose=False)[0]
                votes.append(self.model.names[int(r.probs.top1)] == "Closed")
            else:
                x = np.hstack([self.F.eye_hog(crop), self.F.eye_geometric(crop)])[None]
                votes.append(bool(self.model.predict(x)[0] == 1))
        return bool(votes) and np.mean(votes) >= 0.5


def main():
    global EAR_THRESH, MAR_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--method", choices=["ear", "cnn", "lowlevel"], default="ear")
    ap.add_argument("--save", default=None, help="optional output video path")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()
    if THRESH_FILE.exists():
        t = json.loads(THRESH_FILE.read_text())
        MAR_THRESH = t.get("mar_threshold", MAR_THRESH)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    face = FaceLandmarks(static=False)
    eye = EyeClassifier(args.method)
    closed_run = yawn_run = 0
    hist = collections.deque(maxlen=PERCLOS_WIN)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = face(frame)
        lines = []
        if lm is None:
            lines.append("no face")
        else:
            is_closed = eye.closed(frame, lm)
            closed_run = closed_run + 1 if is_closed else 0
            yawn_run = yawn_run + 1 if lm["mar"] > MAR_THRESH else 0
            hist.append(is_closed)
            perclos = float(np.mean(hist))
            lines += [f"EAR {lm['ear']:.2f}  MAR {lm['mar']:.2f}  eyes: {'CLOSED' if is_closed else 'open'} ({args.method})",
                      f"PERCLOS {perclos*100:.0f}%"]
            if closed_run >= CLOSED_FRAMES or (len(hist) == PERCLOS_WIN and perclos > PERCLOS_ALERT):
                lines.append("DROWSINESS ALERT!")
            if yawn_run >= YAWN_FRAMES:
                lines.append("YAWN ALERT")
            frame = FaceLandmarks.draw(frame, lm, lines)
        if lm is None:
            cv2.putText(frame, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, frame.shape[1::-1])
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Drowsiness monitor (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
