"""Evaluate the HIGH-LEVEL CNN on the held-out test split + compare with the low-level baselines.

Outputs (results/highlevel/):
  metrics.json, confusion_matrix.png, sample_predictions.jpg, comparison.png
Usage: python src/evaluate.py [--weights runs/leaf_cls/weights/best.pt]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "highlevel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/leaf_cls/weights/best.pt"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    classes = sorted(d.name for d in (ROOT / "data/test").iterdir())
    name_to_idx = {v: k for k, v in model.names.items()}

    files, y_true = [], []
    for ci, c in enumerate(classes):
        for p in sorted((ROOT / "data/test" / c).glob("*")):
            files.append(p); y_true.append(ci)
    y_pred, conf = [], []
    t0 = time.time()
    for i in range(0, len(files), 64):
        for r in model.predict([str(f) for f in files[i:i + 64]], imgsz=224, verbose=False):
            top = int(r.probs.top1)
            y_pred.append(classes.index(model.names[top]))
            conf.append(float(r.probs.top1conf))
    ms = 1000 * (time.time() - t0) / len(files)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    healthy = classes.index("healthy")
    bin_t, bin_p = (y_true != healthy), (y_pred != healthy)
    metrics = dict(
        test_images=len(files),
        accuracy=round(accuracy_score(y_true, y_pred) * 100, 2),
        macro_precision=round(rep["macro avg"]["precision"] * 100, 2),
        macro_recall=round(rep["macro avg"]["recall"] * 100, 2),
        macro_f1=round(rep["macro avg"]["f1-score"] * 100, 2),
        healthy_vs_diseased_accuracy=round(float((bin_t == bin_p).mean()) * 100, 2),
        inference_ms_per_image_cpu=round(ms, 1),
        per_class={c: {k: round(rep[c][k] * 100, 1) for k in ("precision", "recall", "f1-score")} for c in classes},
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(
        ax=ax, xticks_rotation=60, colorbar=False, cmap="Greens")
    ax.set_title(f"CNN (YOLOv8n-cls) test confusion matrix — acc {metrics['accuracy']}%")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=120); plt.close()

    # sample predictions grid (2 per class)
    tiles = []
    for ci, c in enumerate(classes):
        idx = np.where(y_true == ci)[0][:2]
        for i in idx:
            im = cv2.resize(cv2.imread(str(files[i])), (224, 224))
            ok = y_pred[i] == y_true[i]
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.rectangle(im, (0, 0), (223, 223), col, 4)
            cv2.putText(im, f"GT: {c[:18]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(im, f"PR: {classes[y_pred[i]][:18]} {conf[i]:.2f}", (6, 212),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
            tiles.append(im)
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(OUT / "sample_predictions.jpg"), np.vstack(rows))

    # low-level vs high-level comparison chart
    low = json.loads((ROOT / "results/lowlevel/lowlevel_metrics.json").read_text())
    names = ["Low-level:\nseverity rule\n(healthy vs diseased,\nbalanced acc.)", "Low-level:\nhand-crafted\nfeatures + SVM\n(10-class acc.)",
             "High-level:\nCNN\n(10-class acc.)"]
    vals = [low["severity_rule"]["balanced_accuracy"], low["handcrafted_svm"]["accuracy"], metrics["accuracy"]]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, vals, color=["#c9a227", "#d9822b", "#2e8b57"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
    plt.ylim(0, 105); plt.ylabel("test score (%)"); plt.title("Crop disease: low-level vs high-level vision")
    plt.tight_layout(); plt.savefig(OUT / "comparison.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
