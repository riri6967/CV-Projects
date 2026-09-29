"""LOW-LEVEL vision pothole detector (no learning, pure image processing).

Idea: a pothole is a region whose intensity and texture differ from the typical road
surface (it can be smoother - water/dust - or rougher - broken gravel). Pipeline:
  1. Resize, grayscale, CLAHE (contrast-limited histogram equalisation)
  2. Local anomaly   = |I - large-Gaussian background|
  3. Texture anomaly = |edge density - median edge density| (Canny + box filter)
  4. Intensity anomaly = |smoothed I - median I|
  5. Score map = 0.3*local + 0.4*texture + 0.3*intensity (each min-max normalised)
  6. Otsu threshold -> morphological closing (fill) + opening (remove speckle)
  7. Contours -> keep blobs with plausible area / solidity / aspect ratio
  8. Box confidence = mean score inside the blob; keep top-K boxes

Usage:
    python src/lowlevel_detector.py --image data/images/test/xxx.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

WORK_W = 640
MIN_AREA, MAX_AREA = 0.004, 0.6      # fraction of image area
MIN_SOLIDITY = 0.45
TOP_K = 3


def norm(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def score_map(img):
    scale = WORK_W / img.shape[1]
    small = cv2.resize(img, (WORK_W, int(img.shape[0] * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=31)
    anomaly = cv2.absdiff(gray, background)
    anomaly = cv2.GaussianBlur(anomaly, (0, 0), 5)
    edges = cv2.Canny(gray, 60, 160)
    density = cv2.boxFilter(edges.astype(np.float32) / 255.0, -1, (31, 31))
    texture = np.abs(density - np.median(density))
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 9)
    intensity = np.abs(smooth - np.median(smooth))
    s = norm(0.3 * norm(anomaly) + 0.4 * norm(texture) + 0.3 * norm(intensity))
    return small, gray, anomaly, edges, s, scale


def detect(img, return_debug=False):
    small, gray, anomaly, edges, s, scale = score_map(img)
    s8 = (s * 255).astype(np.uint8)
    _, mask = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    H, W = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_AREA * H * W <= area <= MAX_AREA * H * W):
            continue
        hull = cv2.convexHull(c)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h
        if solidity < MIN_SOLIDITY or not (0.2 < ar < 6):
            continue
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        conf = float(s[blob > 0].mean())
        boxes.append([x / scale, y / scale, (x + w) / scale, (y + h) / scale, conf])
    boxes = sorted(boxes, key=lambda b: -b[4])[:TOP_K]
    if return_debug:
        return np.array(boxes).reshape(-1, 5), dict(gray=gray, anomaly=norm(anomaly), edges=edges, score=s, mask=mask)
    return np.array(boxes).reshape(-1, 5)


def debug_figure(img, boxes, dbg):
    g = lambda m: cv2.cvtColor((norm(m) * 255).astype(np.uint8) if m.dtype != np.uint8 else m, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(img, (dbg["gray"].shape[1], dbg["gray"].shape[0]))
    sc = small.shape[1] / img.shape[1]
    out = small.copy()
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(out, (int(x1 * sc), int(y1 * sc)), (int(x2 * sc), int(y2 * sc)), (0, 0, 255), 3)
        cv2.putText(out, f"{c:.2f}", (int(x1 * sc), int(y1 * sc) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    tiles = [small, g(dbg["gray"]), g(dbg["anomaly"]), g(dbg["edges"]),
             cv2.applyColorMap((dbg["score"] * 255).astype(np.uint8), cv2.COLORMAP_JET), g(dbg["mask"]), out]
    names = ["input", "CLAHE gray", "|I - background|", "Canny edges", "score map", "Otsu + morphology", "detections"]
    tiles = [cv2.putText(cv2.resize(t, (320, 240)), n, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
             for t, n in zip(tiles, names)]
    tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="lowlevel_debug.jpg")
    a = ap.parse_args()
    im = cv2.imread(a.image)
    b, d = detect(im, return_debug=True)
    cv2.imwrite(a.out, debug_figure(im, b, d))
    print(b)
