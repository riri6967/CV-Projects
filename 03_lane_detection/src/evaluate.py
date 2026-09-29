"""Run the lane detector on all test videos and compute evaluation metrics.

Metrics (no pixel-level ground truth ships with these clips, so we use
label-free metrics + a manual visual check of sampled frames):
  * detection rate      – % frames where left / right / both lanes were found in the raw frame
  * temporal jitter     – mean |Δx| (pixels) of each lane's bottom point between consecutive frames
  * warning rate        – % frames that raised a departure warning (should be ~0 on these
                          clips, since the car stays in its lane: every warning is a false alarm)
  * throughput          – frames per second on CPU
  * manual accuracy     – from results/manual_check.csv (frames checked by eye)

Usage:  python src/evaluate.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lane_detector import ROI_TOP, line_points, process_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = sorted((ROOT / "data" / "test_videos").glob("*.mp4"))


def bottom_x(params, h):
    return None if params is None else line_points(np.array(params), h)[0][0]


def drift_test(out_dir):
    """Simulated lane-departure test with known ground truth.

    We take real frames and apply a horizontal *shear anchored at the horizon*
    (vanishing point fixed, road at the bottom of the image moved sideways).
    For a flat road this is exactly what a sideways drift of the camera looks
    like, so the true offset is known: offset_true = offset_base - shift / lane_width.
    A warning is 'truly needed' when |offset_true| > DEPARTURE_THRESH.
    """
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    from lane_detector import DEPARTURE_THRESH, LaneDetector

    frames = [cv2.imread(str(p)) for p in sorted((ROOT / "data" / "test_images").glob("*.jpg"))]
    for vid in VIDEOS:
        cap = cv2.VideoCapture(str(vid))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % 30 == 0:
                frames.append(f)
            i += 1
    shifts = np.linspace(-0.35, 0.35, 15)  # fraction of lane width
    y_true, y_pred, errs = [], [], []
    for f in frames:
        det = LaneDetector(smooth=False)
        base = det.detect(f)
        if base["offset"] is None:
            continue
        h, w = f.shape[:2]
        lx = line_points(base["left"], h)[0][0]
        rx = line_points(base["right"], h)[0][0]
        lane_w = rx - lx
        for s in shifts:
            px = s * lane_w
            # vanishing point = intersection of the two fitted lines x = m*y + c
            (ml, cl), (mr, cr) = base["left"], base["right"]
            y_h = (cr - cl) / (ml - mr) if ml != mr else ROI_TOP * h
            k = px / (h - y_h)
            M = np.float32([[1, k, -k * y_h], [0, 1, 0]])
            shifted = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            probe = LaneDetector(smooth=False)
            probe.lane_w = det.lane_w  # lane width learned while driving normally
            info = probe.detect(shifted)
            true_off = base["offset"] - s
            y_true.append(abs(true_off) > DEPARTURE_THRESH)
            y_pred.append(bool(info["warning"]))
            if info["offset"] is not None:
                errs.append(abs(info["offset"] - true_off))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred).tolist()
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    res = dict(frames=len(frames), samples=len(y_true), warning_accuracy=round(acc * 100, 1),
               warning_precision=round(p * 100, 1), warning_recall=round(r * 100, 1),
               warning_f1=round(f1 * 100, 1),
               offset_MAE_pct_lane_width=round(float(np.mean(errs)) * 100, 2),
               offset_estimated_pct=round(100 * len(errs) / len(y_true), 1))
    res["confusion_matrix [[TN, FP], [FN, TP]]"] = cm
    print("drift test:", res)
    return res


def main():
    out_dir = ROOT / "results"
    summary = {"simulated_drift_test": drift_test(out_dir)}
    fig, axes = plt.subplots(len(VIDEOS), 1, figsize=(10, 3 * len(VIDEOS)))
    for ax, vid in zip(np.atleast_1d(axes), VIDEOS):
        h = int(cv2.VideoCapture(str(vid)).get(4))
        stats, fps = process_video(vid, out_dir / "videos")
        n = len(stats)
        lf = np.array([s["left_found"] for s in stats])
        rf = np.array([s["right_found"] for s in stats])
        warn = np.array([s["warning"] for s in stats])
        lx = np.array([bottom_x(s["left"], h) or np.nan for s in stats], float)
        rx = np.array([bottom_x(s["right"], h) or np.nan for s in stats], float)
        off = np.array([np.nan if s["offset"] is None else s["offset"] for s in stats])
        summary[vid.name] = dict(
            frames=n,
            fps=round(fps, 1),
            left_detection_rate=round(lf.mean() * 100, 1),
            right_detection_rate=round(rf.mean() * 100, 1),
            both_detection_rate=round((lf & rf).mean() * 100, 1),
            left_jitter_px=round(float(np.nanmean(np.abs(np.diff(lx)))), 2),
            right_jitter_px=round(float(np.nanmean(np.abs(np.diff(rx)))), 2),
            false_warning_rate=round(warn.mean() * 100, 1),
            mean_abs_offset_pct=round(float(np.nanmean(np.abs(off))) * 100, 1),
        )
        ax.plot(off * 100, label="offset (% lane width)")
        ax.axhline(15, color="r", ls="--", lw=0.8)
        ax.axhline(-15, color="r", ls="--", lw=0.8, label="warning threshold")
        ax.set_title(vid.name)
        ax.set_xlabel("frame")
        ax.legend(loc="upper right")
        print(vid.name, summary[vid.name])
    plt.tight_layout()
    plt.savefig(out_dir / "offset_over_time.png", dpi=110)

    manual = out_dir / "manual_check.csv"
    if manual.exists():
        rows = list(csv.DictReader(manual.open()))
        per = {}
        for r in rows:
            per.setdefault(r["video"], []).append(r["correct"] == "1")
        for v, ok in per.items():
            summary.setdefault(v, {})["manual_accuracy_pct"] = round(100 * sum(ok) / len(ok), 1)
            summary[v]["manual_frames_checked"] = len(ok)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("saved", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
