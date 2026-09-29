"""Evaluate the LOW-LEVEL pipeline.

A) Rule-based healthy-vs-diseased decision from lesion severity alone
   (threshold picked on the val split, reported on test).
B) Hand-crafted features (colour histograms + gradient + LBP texture + severity)
   + SVM (RBF) for the full 10-class problem — the classical 'pre-deep-learning' baseline.

Usage: python src/lowlevel_evaluate.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from lowlevel_segmentation import handcrafted_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lowlevel"


def load(split):
    X, y, sev, classes = [], [], [], sorted(d.name for d in (ROOT / "data" / split).iterdir())
    for ci, c in enumerate(classes):
        for p in sorted((ROOT / "data" / split / c).glob("*")):
            f, s = handcrafted_features(cv2.imread(str(p)))
            X.append(f); y.append(ci); sev.append(s)
    return np.array(X), np.array(y), np.array(sev), classes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _, classes = load("train")
    Xva, yva, sva, _ = load("val")
    Xte, yte, ste, _ = load("test")
    healthy = classes.index("healthy")

    # ---- A) severity threshold: diseased if severity > t ----
    bva, bte = (yva != healthy).astype(int), (yte != healthy).astype(int)
    ts = np.linspace(0, 30, 301)
    # classes are 9:1 imbalanced, so pick the threshold by balanced accuracy (Youden's J)
    best_t = max(ts, key=lambda t: balanced_accuracy_score(bva, (sva > t).astype(int)))
    pred = (ste > best_t).astype(int)
    rule = dict(threshold_pct=round(float(best_t), 2),
                accuracy=round(accuracy_score(bte, pred) * 100, 2),
                balanced_accuracy=round(balanced_accuracy_score(bte, pred) * 100, 2),
                f1_diseased=round(f1_score(bte, pred) * 100, 2),
                roc_auc=round(roc_auc_score(bte, ste) * 100, 2),
                confusion_matrix=confusion_matrix(bte, pred).tolist())
    print("severity rule:", rule)
    plt.figure(figsize=(7, 4))
    plt.hist(ste[yte == healthy], bins=40, alpha=0.7, label="healthy")
    plt.hist(ste[yte != healthy], bins=40, alpha=0.7, label="diseased")
    plt.axvline(best_t, color="k", ls="--", label=f"threshold {best_t:.1f}%")
    plt.xlabel("lesion severity (% of leaf area)"); plt.ylabel("images"); plt.legend()
    plt.title("Low-level severity: healthy vs diseased (test set)")
    plt.tight_layout(); plt.savefig(OUT / "severity_histogram.png", dpi=120); plt.close()

    # ---- B) hand-crafted features + SVM ----
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    clf.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    yp = clf.predict(Xte)
    rep = classification_report(yte, yp, target_names=classes, output_dict=True)
    svm = dict(accuracy=round(accuracy_score(yte, yp) * 100, 2),
               macro_f1=round(f1_score(yte, yp, average="macro") * 100, 2),
               per_class_f1={c: round(rep[c]["f1-score"] * 100, 1) for c in classes})
    print("SVM:", svm)
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(confusion_matrix(yte, yp), display_labels=classes).plot(
        ax=ax, xticks_rotation=60, colorbar=False, cmap="Oranges")
    ax.set_title(f"Low-level features + SVM (test acc {svm['accuracy']}%)")
    plt.tight_layout(); plt.savefig(OUT / "svm_confusion_matrix.png", dpi=120); plt.close()

    (OUT / "lowlevel_metrics.json").write_text(json.dumps(dict(severity_rule=rule, handcrafted_svm=svm), indent=2))


if __name__ == "__main__":
    main()
