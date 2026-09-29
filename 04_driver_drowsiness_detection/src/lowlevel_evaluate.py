"""Evaluate the LOW-LEVEL drowsiness pipeline on the test split.

EYE  : (A) single-cue rule - threshold on the dark-blob height/width (picked on train)
       (B) geometric cues + HOG descriptor -> SVM (RBF)
MOUTH: (A) single-cue rule - threshold on the dark mouth-cavity share
       (B) geometric cues + HOG of the frame -> SVM (RBF)
Outputs: results/lowlevel/{lowlevel_metrics.json, *confusion*.png, eye_pipeline.jpg}
Usage: python src/lowlevel_evaluate.py
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
import lowlevel_features as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lowlevel"


def load(split, pos, neg, geom_fn, hog_fn):
    G, H, y = [], [], []
    for c, lbl in ((pos, 1), (neg, 0)):
        for p in sorted((ROOT / "data" / split / c).glob("*")):
            im = cv2.imread(str(p))
            G.append(geom_fn(im)); H.append(hog_fn(im)); y.append(lbl)
    return np.array(G), np.array(H), np.array(y)


def rule(train_x, train_y, test_x):
    """Best single threshold (either direction) on the training set."""
    ts = np.unique(np.percentile(train_x, np.linspace(1, 99, 197)))
    best = max(((t, s) for t in ts for s in (1, -1)),
               key=lambda ts_: accuracy_score(train_y, (ts_[1] * (train_x - ts_[0]) > 0).astype(int)))
    return (best[1] * (test_x - best[0]) > 0).astype(int), best


def run_task(name, pos, neg, geom_fn, hog_fn, rule_idx, geom_names, labels):
    Gtr, Htr, ytr = load("train", pos, neg, geom_fn, hog_fn)
    Gva, Hva, yva = load("val", pos, neg, geom_fn, hog_fn)
    Gte, Hte, yte = load("test", pos, neg, geom_fn, hog_fn)
    # (A) single-cue rule
    pa, (thr, sign) = rule(Gtr[:, rule_idx], ytr, Gte[:, rule_idx])
    auc = roc_auc_score(yte, sign * Gte[:, rule_idx])
    # (B) hand-crafted descriptor + SVM
    Xtr = np.hstack([np.vstack([Htr, Hva]), np.vstack([Gtr, Gva])])
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=False))
    clf.fit(Xtr, np.concatenate([ytr, yva]))
    pb = clf.predict(np.hstack([Hte, Gte]))
    if name == "eye":  # used by drowsiness_monitor.py --method lowlevel (mouth model is 70 MB, not saved)
        (ROOT / "models").mkdir(exist_ok=True)
        joblib.dump(clf, ROOT / "models" / "eye_hog_svm.joblib")
    single_auc = {n: round(max(roc_auc_score(yte, Gte[:, i]), 1 - roc_auc_score(yte, Gte[:, i])) * 100, 1)
                  for i, n in enumerate(geom_names)}
    res = dict(
        single_cue_rule=dict(cue=geom_names[rule_idx], threshold=round(float(thr), 4),
                             direction="> means positive" if sign > 0 else "< means positive",
                             accuracy=round(accuracy_score(yte, pa) * 100, 2), f1=round(f1_score(yte, pa) * 100, 2),
                             roc_auc=round(auc * 100, 2)),
        hog_geometric_svm=dict(accuracy=round(accuracy_score(yte, pb) * 100, 2), f1=round(f1_score(yte, pb) * 100, 2),
                               confusion_matrix=confusion_matrix(yte, pb).tolist()),
        per_cue_test_auc=single_auc,
        test_images=int(len(yte)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, p, t in ((axes[0], pa, f"single-cue rule ({geom_names[rule_idx]})"), (axes[1], pb, "HOG + cues + SVM")):
        ConfusionMatrixDisplay(confusion_matrix(yte, p), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Oranges")
        ax.set_title(f"{t}\nacc {accuracy_score(yte, p)*100:.1f}%", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / f"{name}_lowlevel_confusion.png", dpi=120); plt.close()
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eye = run_task("eye", "Closed", "Open", F.eye_geometric, F.eye_hog, 1, F.EYE_GEOM_NAMES, ["Open", "Closed"])
    print("eye:", eye)
    mouth = run_task("mouth", "yawn", "no_yawn", F.mouth_geometric, F.frame_hog, 0, F.MOUTH_GEOM_NAMES, ["no_yawn", "yawn"])
    print("mouth:", mouth)
    (OUT / "lowlevel_metrics.json").write_text(json.dumps(dict(eye_closed_vs_open=eye, yawn_vs_no_yawn=mouth), indent=2))
    rows = []
    for c in ("Open", "Closed"):
        for p in sorted((ROOT / "data/test" / c).glob("*"))[:3]:
            rows.append(F.eye_debug(cv2.imread(str(p))))
    grid = np.vstack(rows)
    head = np.zeros((30, grid.shape[1], 3), np.uint8)
    for i, t in enumerate(["input", "CLAHE", "dark mask", "Sobel-y", "HOG"]):
        cv2.putText(head, t, (i * 128 + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / "eye_pipeline.jpg"), np.vstack([head, grid]))


if __name__ == "__main__":
    main()
