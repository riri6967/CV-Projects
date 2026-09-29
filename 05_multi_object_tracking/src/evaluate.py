"""Run LOW-LEVEL and HIGH-LEVEL trackers on MOTChallenge sequences and compute CLEAR-MOT + IDF1.

Static-camera sequences are used for the head-to-head comparison, because background
subtraction assumes a fixed camera. The high-level tracker is also run on the
moving / vehicle-mounted camera sequences.

Usage: python src/evaluate.py [--static MOT16-09 ...] [--moving MOT16-11 ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import highlevel_tracker  # noqa: E402
import lowlevel_tracker  # noqa: E402
from mot_utils import ROOT, evaluate, frame_paths  # noqa: E402

STATIC = ["MOT16-09", "PETS09-S2L1", "TUD-Stadtmitte", "TUD-Campus"]
MOVING = ["MOT16-11", "KITTI-17"]
COLS = ["num_frames", "mota", "idf1", "precision", "recall", "num_switches", "num_false_positives",
        "num_misses", "mostly_tracked", "mostly_lost"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", nargs="*", default=STATIC)
    ap.add_argument("--moving", nargs="*", default=MOVING)
    args = ap.parse_args()
    out = ROOT / "results"
    rows, fps = [], {}
    for seq in args.static + args.moving:
        if not frame_paths(seq):
            print("skip (no frames):", seq); continue
        n = len(frame_paths(seq))
        methods = [("high-level YOLOv8n+ByteTrack", highlevel_tracker.run)]
        if seq in args.static:
            methods.insert(0, ("low-level MOG2+IoU", lowlevel_tracker.run))
        for name, fn in methods:
            t0 = time.time()
            res = fn(seq)
            fps[(seq, name)] = n / (time.time() - t0)
            m = evaluate(res, seq, name)
            r = m.iloc[0][COLS].to_dict()
            r.update(sequence=seq, method=name, camera="static" if seq in args.static else "moving",
                     fps=round(fps[(seq, name)], 1))
            rows.append(r)
            print(seq, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    df = pd.DataFrame(rows)
    for c in ("mota", "idf1", "precision", "recall"):
        df[c] = (df[c] * 100).round(1)
    df.to_csv(out / "per_sequence_metrics.csv", index=False)

    summary = {}
    for (cam, meth), g in df.groupby(["camera", "method"]):
        # frame-weighted means of the ratio metrics, sums of the counts
        w = g["num_frames"]
        summary[f"{cam} | {meth}"] = dict(
            sequences=list(g["sequence"]),
            MOTA=round(float((g["mota"] * w).sum() / w.sum()), 1),
            IDF1=round(float((g["idf1"] * w).sum() / w.sum()), 1),
            precision=round(float((g["precision"] * w).sum() / w.sum()), 1),
            recall=round(float((g["recall"] * w).sum() / w.sum()), 1),
            ID_switches=int(g["num_switches"].sum()), FP=int(g["num_false_positives"].sum()),
            FN=int(g["num_misses"].sum()), mostly_tracked=int(g["mostly_tracked"].sum()),
            mostly_lost=int(g["mostly_lost"].sum()), mean_fps=round(float(g["fps"].mean()), 1))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    st = df[df.camera == "static"]
    if len(st):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
        for ax, m in zip(axes, ["mota", "idf1", "precision", "recall"]):
            p = st.pivot(index="sequence", columns="method", values=m)
            p.plot.bar(ax=ax, rot=20, color=["#2e8b57", "#d9822b"][:p.shape[1]] if p.shape[1] == 2 else None, legend=(m == "mota"))
            ax.set_title(m.upper() + " (%)"); ax.set_xlabel("")
        plt.suptitle("Static-camera sequences: low-level vs high-level tracking")
        plt.tight_layout(); plt.savefig(out / "comparison.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
