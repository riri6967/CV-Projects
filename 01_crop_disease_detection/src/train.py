"""HIGH-LEVEL vision: fine-tune an ImageNet-pretrained CNN (YOLOv8n-cls) on tomato leaf images.

Usage (from the project folder):
    python src/train.py --epochs 8            # CPU works; use --device 0 on a GPU / Colab
Data layout (created by prepare_data.py): data/{train,val,test}/<class>/*.jpg
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-cls.pt", help="ImageNet-pretrained classifier")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name="leaf_cls",
        exist_ok=True,
        workers=2,
        seed=42,
        # light augmentation: flips + colour jitter (hue kept small: colour IS the symptom)
        fliplr=0.5, flipud=0.5, hsv_h=0.005, hsv_s=0.3, hsv_v=0.3,
    )
    print("Best weights:", ROOT / "runs/leaf_cls/weights/best.pt")


if __name__ == "__main__":
    main()
