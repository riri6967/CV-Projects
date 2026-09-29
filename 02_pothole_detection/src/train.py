"""Train a YOLOv8n pothole detector.

Usage (from the project folder):
    python src/train.py --epochs 10 --imgsz 416
Use --device 0 on a GPU (e.g. Google Colab) for much faster training.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def write_data_yaml() -> Path:
    """Create a data.yaml that points at ./data (images/{train,val,test}, labels/...)."""
    data_dir = (ROOT / "data").resolve()
    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {data_dir.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: ['pothole']\n"
    )
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained COCO weights")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_yaml = write_data_yaml()
    model = YOLO(args.model)  # transfer learning from COCO
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name="pothole_yolov8n",
        exist_ok=True,
        patience=20,
        workers=2,
        plots=True,
        seed=42,
    )
    print("Best weights:", ROOT / "runs/pothole_yolov8n/weights/best.pt")


if __name__ == "__main__":
    main()
