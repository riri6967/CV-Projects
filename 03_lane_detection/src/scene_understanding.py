"""HIGH-LEVEL vision: vehicle detection + ego-lane reasoning on top of the lane detector.

The low-level pipeline (lane_detector.py) only knows *where the paint is*.
This module adds semantic understanding of the road scene:
  1. YOLOv8n (pre-trained on COCO) detects cars / trucks / buses / motorcycles.
  2. Each vehicle is assigned to EGO lane / LEFT lane / RIGHT lane using the lane
     lines from the low-level detector (ground-contact point = bottom-centre of the box).
  3. Forward-collision warning (FCW) when an ego-lane vehicle is close
     (box height > FCW_HEIGHT of the frame height, a monocular distance proxy).

Usage:
    python src/scene_understanding.py --video data/test_videos/solidWhiteRight.mp4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from lane_detector import LaneDetector, line_points  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FCW_HEIGHT = 0.18
COLOURS = {"ego": (0, 0, 255), "left": (255, 200, 0), "right": (255, 0, 255), "other": (200, 200, 200)}


def lane_x_at(params, y):
    m, c = params
    return m * y + c


def assign_lane(box, left, right):
    """Return 'ego' / 'left' / 'right' / 'other' for a vehicle box (x1,y1,x2,y2)."""
    if left is None or right is None:
        return "other"
    cx, by = (box[0] + box[2]) / 2, box[3]
    lx, rx = lane_x_at(left, by), lane_x_at(right, by)
    if lx <= cx <= rx:
        return "ego"
    return "left" if cx < lx else "right"


def run(video, out_dir, model_path="yolov8n.pt", conf=0.35, imgsz=640):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{video.stem}_scene.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
    lanes = LaneDetector(smooth=True)
    log, t0, n = [], time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        info = lanes.detect(frame)
        vis = LaneDetector.draw(frame, info)
        res = model.predict(frame, conf=conf, imgsz=imgsz, classes=list(VEHICLES), verbose=False)[0]
        counts = {"ego": 0, "left": 0, "right": 0, "other": 0}
        fcw = False
        for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                           res.boxes.conf.cpu().numpy()):
            lane = assign_lane(b, info["left"], info["right"])
            counts[lane] += 1
            close = lane == "ego" and (b[3] - b[1]) / h > FCW_HEIGHT
            fcw |= close
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOURS[lane], 2)
            cv2.putText(vis, f"{VEHICLES[int(c)]} {s:.2f} [{lane}]", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[lane], 2)
        cv2.putText(vis, f"vehicles  ego:{counts['ego']} left:{counts['left']} right:{counts['right']}",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if fcw:
            cv2.putText(vis, "FORWARD COLLISION WARNING", (w // 2 - 250, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        writer.write(vis)
        if n % 30 == 1:
            cv2.imwrite(str(out_dir / f"{video.stem}_scene_frame{n:04d}.jpg"), vis)
        log.append(dict(frame=n, fcw=bool(fcw), **counts, n_det=int(len(res.boxes))))
    cap.release()
    writer.release()
    fps = n / (time.time() - t0)
    return log, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="*", default=None)
    ap.add_argument("--model", default="yolov8n.pt")
    args = ap.parse_args()
    videos = [Path(v) for v in args.video] if args.video else sorted((ROOT / "data/test_videos").glob("*.mp4"))
    out_dir = ROOT / "results" / "scene"
    summary = {}
    for v in videos:
        log, fps = run(v, out_dir, args.model)
        det = np.array([r["n_det"] for r in log])
        summary[v.name] = dict(frames=len(log), fps=round(fps, 1),
                               mean_vehicles_per_frame=round(float(det.mean()), 2),
                               frames_with_ego_vehicle_pct=round(100 * np.mean([r["ego"] > 0 for r in log]), 1),
                               fcw_frames_pct=round(100 * np.mean([r["fcw"] for r in log]), 1))
        print(v.name, summary[v.name])
        (out_dir / f"{v.stem}_log.json").write_text(json.dumps(log))
    (out_dir / "scene_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
