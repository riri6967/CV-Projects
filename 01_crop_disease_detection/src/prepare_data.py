"""Build a balanced train/val/test split of the PlantVillage tomato classes.

Source folder = the 'color' folder of PlantVillage, which contains one folder per class
(e.g. Tomato___Early_blight). Works with both the Kaggle copy and the GitHub original:
  Kaggle : kaggle datasets download -d abdallahalidev/plantvillage-dataset  -> plantvillage dataset/color
  GitHub : https://github.com/spMohanty/PlantVillage-Dataset                -> raw/color

Usage:
    python src/prepare_data.py --src "path/to/color" --train 250 --val 50 --test 100
Creates data/{train,val,test}/<class>/*.jpg (copies).
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prefix", default="Tomato___", help="only use classes starting with this")
    ap.add_argument("--train", type=int, default=250)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = ROOT / "data"
    classes = sorted(d for d in Path(args.src).iterdir() if d.is_dir() and d.name.startswith(args.prefix))
    for cdir in classes:
        name = cdir.name.replace(args.prefix, "").replace(" ", "_")
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        need = args.train + args.val + args.test
        if len(files) < need:  # small class: keep the same proportions
            k = len(files) / need
            n_tr, n_va = int(args.train * k), int(args.val * k)
        else:
            n_tr, n_va = args.train, args.val
        n_te = min(args.test, len(files) - n_tr - n_va)
        splits = dict(train=files[:n_tr], val=files[n_tr:n_tr + n_va], test=files[n_tr + n_va:n_tr + n_va + n_te])
        for split, fl in splits.items():
            d = out / split / name
            d.mkdir(parents=True, exist_ok=True)
            for f in fl:
                shutil.copy(f, d / f.name)
        print(f"{name:40s} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


if __name__ == "__main__":
    main()
