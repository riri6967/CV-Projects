"""HIGH-LEVEL multi-object tracker: YOLOv8n person detector (COCO-pretrained) + ByteTrack.

ByteTrack associates detections to tracks with a Kalman filter + Hungarian matching, and
uses LOW-confidence boxes in a second association round to keep occluded people alive.
Works with a moving camera (no background model).

Usage: python src/highlevel_tracker.py --seq MOT16-09 [--model yolov8n.pt]
"""
import argparse

import cv2
from ultralytics import YOLO

from mot_utils import ROOT, draw, evaluate, frame_paths, save_results


def run(seq, model_path="yolov8n.pt", imgsz=960, conf=0.1, save_video=True):
    model = YOLO(model_path)
    paths = frame_paths(seq)
    out_dir = ROOT / "results" / "highlevel"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, writer = [], None
    for f, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        r = model.track(img, persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=imgsz,
                        conf=conf, verbose=False)[0]
        tracks = []
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
                tracks.append((int(i), x1, y1, x2 - x1, y2 - y1))
        rows += [(f, *t) for t in tracks]
        if save_video:
            vis = draw(img, tracks, f"high-level (YOLOv8n+ByteTrack)  frame {f}")
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{seq}_highlevel.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         15, (img.shape[1] // 2, img.shape[0] // 2))
            writer.write(cv2.resize(vis, (img.shape[1] // 2, img.shape[0] // 2)))
            if f in (60, len(paths) // 2):
                cv2.imwrite(str(out_dir / f"{seq}_highlevel_f{f}.jpg"), vis)
    if writer:
        writer.release()
    # reset tracker state between sequences
    model.predictor = None
    save_results(rows, out_dir / f"{seq}.txt")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MOT16-09")
    ap.add_argument("--model", default="yolov8n.pt")
    a = ap.parse_args()
    rows = run(a.seq, a.model)
    print(evaluate(rows, a.seq, "highlevel"))
