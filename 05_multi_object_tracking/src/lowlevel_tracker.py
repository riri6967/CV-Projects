"""LOW-LEVEL multi-object tracker: background subtraction + blob analysis + IoU/centroid tracking.

No learning / no object model:
  1. MOG2 background subtraction (Gaussian-mixture model of every pixel over time)
  2. Remove shadows (MOG2 marks them 127), median filter
  3. Morphology: opening (noise) -> closing + dilation (join body parts into one blob)
  4. Connected components -> bounding boxes, filtered by area and pedestrian-like aspect ratio
  5. Tracking-by-detection: Hungarian assignment on (1 - IoU) with a centroid-distance gate,
     constant-velocity prediction, tracks deleted after MAX_AGE missed frames.

Works only for a STATIC camera (the background model assumes a fixed view).
Usage: python src/lowlevel_tracker.py --seq MOT16-09
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mot_utils import ROOT, draw, evaluate, frame_paths, save_results

MIN_AREA_FRAC = 0.0006     # blob area / image area
MAX_AREA_FRAC = 0.08
ASPECT = (1.1, 4.5)        # h / w of a standing person
MAX_AGE, MIN_HITS = 8, 3
IOU_GATE = 0.2


def blobs(fg, img_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = []
    for x, y, w, h, a in stats[1:]:
        if not (MIN_AREA_FRAC * img_area < a < MAX_AREA_FRAC * img_area):
            continue
        if not (ASPECT[0] <= h / max(w, 1) <= ASPECT[1]):
            continue
        out.append([x, y, w, h])
    return np.array(out, float).reshape(-1, 4)


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[:, None, 0] + a[:, None, 2], a[:, None, 1] + a[:, None, 3], b[None, :, 0] + b[None, :, 2], b[None, :, 1] + b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter + 1e-9)


class Track:
    def __init__(self, tid, box):
        self.id, self.box, self.vel = tid, box.copy(), np.zeros(2)
        self.age, self.hits, self.miss = 0, 1, 0

    def predict(self):
        self.box[:2] += self.vel
        self.age += 1
        return self.box

    def update(self, box):
        self.vel = 0.6 * self.vel + 0.4 * (box[:2] - self.box[:2])
        self.box = 0.3 * self.box + 0.7 * box  # smooth size & position
        self.hits += 1
        self.miss = 0


class IoUTracker:
    def __init__(self):
        self.tracks, self.next_id = [], 1

    def step(self, dets):
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        matched_t, matched_d = set(), set()
        if len(preds) and len(dets):
            cost = 1 - iou_xywh(preds, dets)
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if cost[i, j] <= 1 - IOU_GATE:
                    self.tracks[i].update(dets[j]); matched_t.add(i); matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.miss += 1
        for j in range(len(dets)):
            if j not in matched_d:
                self.tracks.append(Track(self.next_id, dets[j])); self.next_id += 1
        self.tracks = [t for t in self.tracks if t.miss <= MAX_AGE]
        return [(t.id, *t.box) for t in self.tracks if t.miss == 0 and t.hits >= MIN_HITS]


def run(seq, save_video=True):
    paths = frame_paths(seq)
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    trk = IoUTracker()
    rows, writer = [], None
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15))
    out_dir = ROOT / "results" / "lowlevel"
    out_dir.mkdir(parents=True, exist_ok=True)
    # warm-up: learn the background from the first frames
    for p in paths[:30]:
        mog.apply(cv2.imread(str(p)), learningRate=0.05)
    for f, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        fg = mog.apply(img)
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)          # drop shadows (127)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
        fg = cv2.dilate(fg, k_open, iterations=2)
        dets = blobs(fg, img.shape[0] * img.shape[1])
        tracks = trk.step(dets)
        rows += [(f, *t) for t in tracks]
        if save_video:
            vis = draw(img, tracks, f"low-level (MOG2+IoU)  frame {f}")
            mask_small = cv2.resize(cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR), (img.shape[1] // 4, img.shape[0] // 4))
            vis[-mask_small.shape[0]:, -mask_small.shape[1]:] = mask_small
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{seq}_lowlevel.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         15, (img.shape[1] // 2, img.shape[0] // 2))
            writer.write(cv2.resize(vis, (img.shape[1] // 2, img.shape[0] // 2)))
            if f in (60, len(paths) // 2):
                cv2.imwrite(str(out_dir / f"{seq}_lowlevel_f{f}.jpg"), vis)
    if writer:
        writer.release()
    save_results(rows, out_dir / f"{seq}.txt")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MOT16-09")
    a = ap.parse_args()
    rows = run(a.seq)
    print(evaluate(rows, a.seq, "lowlevel"))
