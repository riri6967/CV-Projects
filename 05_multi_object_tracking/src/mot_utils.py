"""Shared helpers: read MOTChallenge sequences, write/read MOT-format results, CLEAR-MOT / IDF1 evaluation.

Sequence layout (MOTChallenge): data/sequences/<SEQ>/{img1/*.jpg, gt/gt.txt, seqinfo.ini}
GT formats handled:
  MOT16/17 : frame,id,x,y,w,h,consider_flag,class,visibility   -> keep class 1 (pedestrian) with flag 1
  2DMOT2015: frame,id,x,y,w,h,1,wx,wy,wz (no class column)       -> keep all boxes
"""
from pathlib import Path

import cv2
import motmetrics as mm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sequences"


def frame_paths(seq):
    d = DATA / seq / "img1"
    return sorted(d.glob("*.jpg"))


def load_gt(seq):
    """GT pedestrian boxes -> {frame: [(id, x, y, w, h), ...]}."""
    gt = {}
    arr = np.loadtxt(DATA / seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
    mot16 = seq.startswith(("MOT16", "MOT17", "MOT20"))
    for row in arr:
        f, i, x, y, w, h, conf = row[:7]
        if mot16 and not (int(conf) == 1 and int(row[7]) == 1):
            continue
        gt.setdefault(int(f), []).append((int(i), x, y, w, h))
    return gt


def save_results(rows, path):
    """rows: list of (frame, id, x, y, w, h) -> MOTChallenge txt format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for f, i, x, y, w, h in rows:
            fh.write(f"{f},{i},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")


def iou_distance(a, b, max_iou=0.5):
    """1 - IoU between xywh boxes; NaN (= not allowed to match) when IoU < max_iou.
    (Same as motmetrics.distances.iou_matrix, re-implemented for NumPy 2 compatibility.)"""
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    d = 1 - inter / np.maximum(union, 1e-9)
    d[d > max_iou] = np.nan
    return d


def evaluate(rows, seq, name):
    """Return a motmetrics summary row (MOTA, IDF1, ...) for one sequence."""
    gt = load_gt(seq)
    hyp = {}
    for f, i, x, y, w, h in rows:
        hyp.setdefault(int(f), []).append((int(i), x, y, w, h))
    acc = mm.MOTAccumulator(auto_id=True)
    n_frames = len(frame_paths(seq)) or max(gt)
    for f in range(1, n_frames + 1):
        g, p = gt.get(f, []), hyp.get(f, [])
        gb = np.array([b[1:] for b in g]).reshape(-1, 4)
        pb = np.array([b[1:] for b in p]).reshape(-1, 4)
        dist = iou_distance(gb, pb, max_iou=0.5)
        acc.update([b[0] for b in g], [b[0] for b in p], dist)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=["num_frames", "mota", "motp", "idf1", "precision", "recall",
                                    "num_switches", "num_false_positives", "num_misses",
                                    "mostly_tracked", "mostly_lost"], name=name)


def colour(i):
    rng = np.random.default_rng(int(i) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(frame, tracks, label=""):
    out = frame.copy()
    for i, x, y, w, h in tracks:
        c = colour(i)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
        cv2.putText(out, str(int(i)), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    if label:
        cv2.putText(out, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return out
