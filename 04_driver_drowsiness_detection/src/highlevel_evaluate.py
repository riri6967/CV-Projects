"""Evaluate the HIGH-LEVEL drowsiness pipeline on the (leak-free) test split.

  (1) CNN (YOLOv8n-cls, 4 classes) -> 4-class accuracy + eye (Closed/Open) and mouth (yawn/no_yawn) accuracy
  (2) MediaPipe Face Mesh MAR rule for yawning (threshold picked on train frames)
  (3) Qualitative: landmarks + EAR/MAR + CNN eye state on test face frames
  (4) Comparison chart low-level vs high-level
Usage: python src/highlevel_evaluate.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score)
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from landmarks import FaceLandmarks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "highlevel"
CLASSES = ["Closed", "Open", "no_yawn", "yawn"]


def files(split, classes):
    return [(p, c) for c in classes for p in sorted((ROOT / "data" / split / c).glob("*"))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "models" / "drowsy_cls_yolov8n.pt"))
    res = {}

    # ---------- (1) CNN ----------
    test = files("test", CLASSES)
    y_true = np.array([CLASSES.index(c) for _, c in test])
    y_pred = []
    t0 = time.time()
    for i in range(0, len(test), 64):
        for r in model.predict([str(p) for p, _ in test[i:i + 64]], imgsz=224, verbose=False):
            y_pred.append(CLASSES.index(model.names[int(r.probs.top1)]))
    ms = 1000 * (time.time() - t0) / len(test)
    y_pred = np.array(y_pred)
    eye_m, mouth_m = y_true < 2, y_true >= 2
    res["cnn"] = dict(
        four_class_accuracy=round(accuracy_score(y_true, y_pred) * 100, 2),
        macro_f1=round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        eye_closed_vs_open_accuracy=round(accuracy_score(y_true[eye_m], y_pred[eye_m]) * 100, 2),
        eye_closed_f1=round(f1_score(y_true[eye_m] == 0, y_pred[eye_m] == 0) * 100, 2),
        yawn_vs_no_yawn_accuracy=round(accuracy_score(y_true[mouth_m], y_pred[mouth_m]) * 100, 2),
        yawn_f1=round(f1_score(y_true[mouth_m] == 3, y_pred[mouth_m] == 3) * 100, 2),
        ms_per_image_cpu=round(ms, 1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"CNN 4-class test confusion (acc {res['cnn']['four_class_accuracy']}%)")
    plt.tight_layout(); plt.savefig(OUT / "cnn_confusion_matrix.png", dpi=120); plt.close()

    # ---------- (2) MAR rule ----------
    fm = FaceLandmarks(static=True)

    def mars(split):
        v, y, miss = [], [], 0
        for p, c in files(split, ["yawn", "no_yawn"]):
            r = fm(cv2.imread(str(p)))
            if r is None:
                miss += 1; v.append(0.0)   # no face found -> treated as 'no yawn'
            else:
                v.append(r["mar"])
            y.append(int(c == "yawn"))
        return np.array(v), np.array(y), miss
    mtr, ytr, _ = mars("train")
    ts = np.linspace(0.05, 1.5, 146)
    thr = float(max(ts, key=lambda t: accuracy_score(ytr, (mtr > t).astype(int))))
    mte, yte, miss = mars("test")
    pm = (mte > thr).astype(int)
    res["mediapipe_mar_rule"] = dict(threshold=round(thr, 3), accuracy=round(accuracy_score(yte, pm) * 100, 2),
                                     f1=round(f1_score(yte, pm) * 100, 2), roc_auc=round(roc_auc_score(yte, mte) * 100, 2),
                                     faces_not_found=int(miss), test_images=int(len(yte)))
    (OUT / "thresholds.json").write_text(json.dumps(dict(mar_threshold=round(thr, 3), ear_threshold=0.21), indent=2))
    plt.figure(figsize=(7, 4))
    plt.hist(mte[yte == 0], bins=30, alpha=0.7, label="no_yawn"); plt.hist(mte[yte == 1], bins=30, alpha=0.7, label="yawn")
    plt.axvline(thr, color="k", ls="--", label=f"threshold {thr:.2f}")
    plt.xlabel("mouth aspect ratio (MAR)"); plt.ylabel("test frames"); plt.legend(); plt.title("MediaPipe MAR on test frames")
    plt.tight_layout(); plt.savefig(OUT / "mar_histogram.png", dpi=120); plt.close()

    # ---------- (3) qualitative ----------
    tiles = []
    for c in ("yawn", "no_yawn"):
        for p in sorted((ROOT / "data/test" / c).glob("*"))[::25][:4]:
            im = cv2.imread(str(p))
            lm = fm(im)
            if lm is None:
                continue
            states = []
            for pts in (lm["left_eye"], lm["right_eye"]):
                crop = FaceLandmarks.eye_crop(im, pts)
                if crop.size:
                    states.append(model.names[int(model.predict(crop, imgsz=224, verbose=False)[0].probs.top1)])
            lines = [f"GT: {c}", f"EAR {lm['ear']:.2f}  MAR {lm['mar']:.2f}",
                     f"CNN eyes: {'/'.join(states)}", "YAWN ALERT" if lm["mar"] > thr else "no yawn"]
            tiles.append(cv2.resize(FaceLandmarks.draw(im, lm, lines), (480, 360)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT / "landmark_examples.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

    # CNN sample predictions
    grid = []
    for c in CLASSES:
        idx = [i for i, (_, cc) in enumerate(test) if cc == c][:4]
        row = []
        for i in idx:
            im = cv2.resize(cv2.imread(str(test[i][0])), (200, 160))
            ok = y_pred[i] == y_true[i]
            cv2.rectangle(im, (0, 0), (199, 159), (0, 200, 0) if ok else (0, 0, 255), 4)
            cv2.putText(im, f"GT {c} / PR {CLASSES[y_pred[i]]}", (6, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            row.append(im)
        grid.append(np.hstack(row))
    cv2.imwrite(str(OUT / "cnn_sample_predictions.jpg"), np.vstack(grid))

    # ---------- (4) comparison ----------
    low = json.loads((ROOT / "results/lowlevel/lowlevel_metrics.json").read_text())
    names = ["eye: single cue\n(low)", "eye: HOG+SVM\n(low)", "eye: CNN\n(high)",
             "yawn: cavity cue\n(low)", "yawn: HOG+SVM\n(low)", "yawn: MAR\n(high)", "yawn: CNN\n(high)"]
    vals = [low["eye_closed_vs_open"]["single_cue_rule"]["accuracy"], low["eye_closed_vs_open"]["hog_geometric_svm"]["accuracy"],
            res["cnn"]["eye_closed_vs_open_accuracy"], low["yawn_vs_no_yawn"]["single_cue_rule"]["accuracy"],
            low["yawn_vs_no_yawn"]["hog_geometric_svm"]["accuracy"], res["mediapipe_mar_rule"]["accuracy"],
            res["cnn"]["yawn_vs_no_yawn_accuracy"]]
    cols = ["#d9822b", "#d9822b", "#2e8b57"] + ["#d9822b", "#d9822b", "#2e8b57", "#2e8b57"]
    plt.figure(figsize=(11, 4.5))
    b = plt.bar(names, vals, color=cols)
    for bb, v in zip(b, vals):
        plt.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    plt.ylim(0, 105); plt.ylabel("test accuracy (%)"); plt.title("Drowsiness: low-level (orange) vs high-level (green)")
    plt.tight_layout(); plt.savefig(OUT / "comparison.png", dpi=120); plt.close()

    print(json.dumps(res, indent=2))
    (OUT / "highlevel_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
