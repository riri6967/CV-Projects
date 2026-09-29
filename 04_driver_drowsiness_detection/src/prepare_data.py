"""Split the Kaggle 'Drowsiness_dataset' (dheerajperumandla/drowsiness-dataset) into train/val/test
WITHOUT near-duplicate leakage.

The yawn / no_yawn images are frames cut from a few in-car videos (YawDD), so a random split
puts almost identical frames of the same person in train and test and inflates accuracy.
We therefore:
  1. make a 24x18 thumbnail of every image (mean-subtracted),
  2. group near-identical images with agglomerative clustering (average linkage,
     distance threshold 8 for the face frames, 3 for the eye crops) - one group ~ one
     person/session,
  3. assign whole GROUPS to train / val / test (~70 / 15 / 15 % of the images).

Usage: python src/prepare_data.py --src "path/to/unzipped/dataset"
Creates data/{train,val,test}/{Closed,Open,yawn,no_yawn}/ and data/split_groups.csv
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(("Closed", "Open"), 3.0), (("yawn", "no_yawn"), 8.0)]


def class_files(src, c):
    return sorted(p for d in src.rglob(c) if d.is_dir() and d.name == c
                  for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    src, out = Path(args.src), ROOT / "data"
    rng = random.Random(args.seed)
    log = []
    for classes, thr in PAIRS:
        items = [(p, c) for c in classes for p in class_files(src, c)]
        thumbs = np.array([cv2.resize(cv2.imread(str(p)), (24, 18), interpolation=cv2.INTER_AREA)
                           .astype(np.float32).ravel() / 255 for p, _ in items])
        thumbs -= thumbs.mean(1, keepdims=True)
        groups = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                         linkage="average").fit_predict(thumbs)
        gids = list(range(groups.max() + 1))
        rng.shuffle(gids)
        split_of, count, n = {}, 0, len(items)
        for g in gids:  # fill test, then val, then train with whole groups
            split_of[g] = "test" if count < 0.15 * n else "val" if count < 0.30 * n else "train"
            count += int((groups == g).sum())
        counters = {}
        for (p, c), g in zip(items, groups):
            s = split_of[g]
            d = out / s / c
            d.mkdir(parents=True, exist_ok=True)
            i = counters.get((s, c), 0)
            counters[(s, c)] = i + 1
            name = f"{c}_{s}_{i:04d}{p.suffix.lower()}"
            shutil.copy(p, d / name)
            log.append((name, c, s, int(g)))
        print(f"{classes}: {n} images in {groups.max() + 1} groups")
        for c in classes:
            print(f"  {c:8s} " + " ".join(f"{s}={counters.get((s, c), 0)}" for s in ("train", "val", "test")))
    with open(out / "split_groups.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "class", "split", "group"]); w.writerows(log)


if __name__ == "__main__":
    main()
