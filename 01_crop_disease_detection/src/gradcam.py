"""Grad-CAM explanation for the CNN: which leaf regions drove the prediction?

Hooks the last convolutional stage of YOLOv8n-cls, back-propagates the predicted class
score and builds a class-activation heat-map. Also reports how much of the Grad-CAM
energy falls on the low-level lesion mask (links high-level decisions to low-level evidence).

Usage: python src/gradcam.py [--n 2]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from lowlevel_segmentation import analyse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "highlevel"


def gradcam(net, x, target_layer):
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    out = net(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    cls = int(out.argmax(1))
    net.zero_grad()
    torch.log(out[0, cls] + 1e-9).backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam, cls, float(out[0, cls])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/leaf_cls/weights/best.pt"))
    ap.add_argument("--n", type=int, default=1, help="images per class")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(args.weights)
    net = yolo.model.float().eval()
    for p in net.parameters():
        p.requires_grad_(True)
    target = net.model[-2]  # last conv block before the classification head
    rows, overlap, area = [], [], []
    for cdir in sorted((ROOT / "data/test").iterdir()):
        for p in sorted(cdir.glob("*"))[:args.n]:
            bgr = cv2.resize(cv2.imread(str(p)), (224, 224))
            x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255.0
            cam, cls, prob = gradcam(net, x, target)
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            blend = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)
            a = analyse(bgr)
            if cdir.name != "healthy" and a["lesion"].sum() > 0:
                overlap.append(float((cam * (a["lesion"] > 0)).sum() / (cam * (a["leaf"] > 0)).sum()))
                area.append(float((a["lesion"] > 0).sum() / max((a["leaf"] > 0).sum(), 1)))
            les = bgr.copy(); les[a["lesion"] > 0] = (0, 0, 255)
            cv2.putText(blend, f"{yolo.names[cls][:20]} {prob:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(bgr, cdir.name[:22], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            cv2.putText(les, "low-level lesions", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            rows.append(np.hstack([bgr, blend, les]))
    grid = np.vstack([np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else np.hstack([rows[i], np.zeros_like(rows[i])])
                      for i in range(0, len(rows), 2)])
    cv2.imwrite(str(OUT / "gradcam_examples.jpg"), grid)
    # if Grad-CAM were spread uniformly over the leaf, energy-on-lesions would equal lesion area share
    res = dict(mean_gradcam_energy_on_lesions_pct=round(100 * float(np.mean(overlap)), 1) if overlap else None,
               mean_lesion_area_share_pct=round(100 * float(np.mean(area)), 1) if area else None,
               images=len(overlap))
    (OUT / "gradcam_stats.json").write_text(json.dumps(res, indent=2))
    print(res, "saved", OUT / "gradcam_examples.jpg")


if __name__ == "__main__":
    main()
