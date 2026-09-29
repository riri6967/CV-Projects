"""Evaluate LOW-LEVEL (image processing) vs HIGH-LEVEL (YOLOv8) pothole detection on the test split.

Same metric code for both methods:
  * AP@0.5 (area under the precision-recall curve, VOC all-point interpolation)
  * precision / recall / F1 at IoU >= 0.5 for the confidence threshold that maximises F1
  * inference time per image
The YOLO model is also evaluated with Ultralytics' built-in validator (mAP@0.5, mAP@0.5:0.95).

Usage: python src/evaluate.py [--weights runs/pothole_yolov8n/weights/best.pt]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
import lowlevel_detector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def load_gt(img_path, w, h):
    lbl = Path(str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().split("\n"):
            if line.strip():
                _, x, y, bw, bh = map(float, line.split()[:5])
                boxes.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(boxes).reshape(-1, 4)


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def match(preds, gts, thr=0.5):
    """Return list of (confidence, is_true_positive) and number of GT boxes."""
    out, n_gt = [], 0
    for p, g in zip(preds, gts):
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        for box in p[np.argsort(-p[:, 4])] if len(p) else []:
            ok = False
            if len(g):
                ious = iou(box[:4], g)
                ious[used] = 0
                j = int(ious.argmax())
                if ious[j] >= thr:
                    used[j] = ok = True
            out.append((box[4], ok))
    return out, n_gt


def pr_curve(matches, n_gt):
    if not matches:
        return np.array([0]), np.array([0]), 0.0, {}
    m = sorted(matches, key=lambda t: -t[0])
    tp = np.cumsum([t[1] for t in m]); fp = np.cumsum([not t[1] for t in m])
    rec, prec = tp / max(n_gt, 1), tp / (tp + fp)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    k = int(f1.argmax())
    best = dict(conf_threshold=round(float(m[k][0]), 3), precision=round(float(prec[k]) * 100, 2),
                recall=round(float(rec[k]) * 100, 2), f1=round(float(f1[k]) * 100, 2))
    return rec, prec, ap, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/pothole_yolov8n/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=416)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    files = sorted((ROOT / "data/images/test").glob("*.jpg"))
    model = YOLO(args.weights)

    gts, low, high, t_low, t_high = [], [], [], 0.0, 0.0
    for f in files:
        img = cv2.imread(str(f)); h, w = img.shape[:2]
        gts.append(load_gt(f, w, h))
        t = time.time(); low.append(lowlevel_detector.detect(img)); t_low += time.time() - t
        t = time.time()
        r = model.predict(img, imgsz=args.imgsz, conf=0.001, verbose=False)[0]
        t_high += time.time() - t
        high.append(np.hstack([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]]))

    results = {}
    plt.figure(figsize=(6, 5))
    for name, preds, tt in (("low_level_image_processing", low, t_low), ("high_level_yolov8n", high, t_high)):
        m, n_gt = match(preds, gts)
        rec, prec, ap50, best = pr_curve(m, n_gt)
        results[name] = dict(AP50=round(ap50 * 100, 2), **best, ms_per_image=round(1000 * tt / len(files), 1))
        plt.plot(rec, prec, label=f"{name} (AP50={ap50*100:.1f})")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Pothole detection PR curve (test, IoU 0.5)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(OUT / "pr_curve_comparison.png", dpi=120)

    # Ultralytics' own validator on the test split (COCO-style mAP)
    v = model.val(data=str(ROOT / "data.yaml"), split="test", imgsz=args.imgsz, verbose=False,
                  project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    results["high_level_yolov8n"].update(ultralytics_mAP50=round(float(v.box.map50) * 100, 2),
                                         ultralytics_mAP50_95=round(float(v.box.map) * 100, 2),
                                         ultralytics_precision=round(float(v.box.mp) * 100, 2),
                                         ultralytics_recall=round(float(v.box.mr) * 100, 2))
    results["test_images"] = len(files)
    results["test_gt_boxes"] = int(sum(len(g) for g in gts))
    print(json.dumps(results, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    # side-by-side qualitative examples: GT (green) | low-level (red) | YOLO (blue)
    rng = np.random.default_rng(3)
    rows = []
    thr = results["high_level_yolov8n"]["conf_threshold"]
    lthr = results["low_level_image_processing"].get("conf_threshold", 0)
    for i in rng.choice(len(files), 6, replace=False):
        img = cv2.imread(str(files[i]))
        tiles = []
        for boxes, col, title, t in ((gts[i], (0, 255, 0), "ground truth", None),
                                     (low[i], (0, 0, 255), "low-level", lthr),
                                     (high[i], (255, 0, 0), "YOLOv8n", thr)):
            im = img.copy()
            for b in boxes:
                if t is not None and b[4] < t:
                    continue
                cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 4)
            im = cv2.resize(im, (400, 300))
            cv2.putText(im, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(im)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(OUT / "comparison_examples.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
