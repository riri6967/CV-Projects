"""LOW-LEVEL vision for crop disease: leaf + lesion segmentation and severity (no learning).

Steps
  1. Denoise with a median filter, convert BGR -> HSV.
  2. Leaf mask   : colour threshold in HSV (hue in the green-yellow-brown band, enough
                   saturation/brightness; the grey/purple background falls outside)
                   -> morphological closing/opening -> keep the largest contour.
  3. Lesion mask : inside the (eroded) leaf, saturated pixels whose hue is NOT healthy
                   green (yellow chlorosis, brown necrosis, grey-purple mould)
                   -> morphological opening to remove speckle.
  4. Severity    : lesion area / leaf area (%).
  5. Decision    : 'diseased' if severity > threshold (threshold chosen on the val split).

Also exports hand-crafted features (colour histograms, texture, severity) used by
classical_classifier.py.

Usage:
    python src/lowlevel_segmentation.py --demo          # save segmentation examples
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LEAF_HUE = (8, 95)        # hue band of leaf tissue incl. lesions (OpenCV hue 0-179)
HEALTHY_HUE = (35, 65)    # hue range of healthy green tissue
MIN_SAT = 40              # ignore unsaturated (grey / specular) pixels for lesions
ERODE = 9                 # erode leaf mask so shadowed margins are not counted


def leaf_mask(img):
    blur = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = (((h >= LEAF_HUE[0]) & (h <= LEAF_HUE[1]) & (s > 30) & (v > 30)) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, 255, cv2.FILLED)
    return out, hsv


def lesion_mask(hsv, leaf):
    h, s, v = cv2.split(hsv)
    inner = cv2.erode(leaf, np.ones((ERODE, ERODE), np.uint8))
    not_green = ((h < HEALTHY_HUE[0]) | (h > HEALTHY_HUE[1])) & (s > MIN_SAT)
    m = (not_green & (inner > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return m


def analyse(img):
    leaf, hsv = leaf_mask(img)
    lesion = lesion_mask(hsv, leaf)
    inner = cv2.erode(leaf, np.ones((ERODE, ERODE), np.uint8))
    leaf_area = max(int((inner > 0).sum()), 1)
    severity = 100.0 * (lesion > 0).sum() / leaf_area
    return dict(leaf=leaf, lesion=lesion, hsv=hsv, severity=severity)


def handcrafted_features(img):
    """Colour + texture + shape descriptor (all low-level)."""
    img = cv2.resize(img, (256, 256))
    a = analyse(img)
    leaf, hsv = a["leaf"], a["hsv"]
    feats = []
    # colour histograms of hue / saturation / value inside the leaf
    for ch, bins, rng in ((0, 30, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))):
        hist = cv2.calcHist([hsv], [ch], leaf, [bins], rng).ravel()
        feats.append(hist / (hist.sum() + 1e-6))
    # texture: Laplacian variance, Sobel gradient-magnitude histogram inside the leaf
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mh, _ = np.histogram(mag[leaf > 0], bins=16, range=(0, 400))
    feats.append(mh / (mh.sum() + 1e-6))
    # local binary pattern (8-neighbour) histogram — classic texture descriptor
    lbp = lbp_image(gray)
    lh, _ = np.histogram(lbp[leaf > 0], bins=32, range=(0, 256))
    feats.append(lh / (lh.sum() + 1e-6))
    feats.append(np.array([lap[leaf > 0].var() / 1000.0, a["severity"] / 100.0]))
    return np.concatenate(feats).astype(np.float32), a["severity"]


def lbp_image(gray):
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offs):
        nb = g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << i)
    return np.pad(code, 1)


def visualise(img, a):
    overlay = img.copy()
    overlay[a["lesion"] > 0] = (0, 0, 255)
    blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cnts, _ = cv2.findContours(a["leaf"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blend, cnts, -1, (0, 255, 0), 2)
    cv2.putText(blend, f"severity {a['severity']:.1f}%", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    g = lambda m: cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return np.hstack([img, g(a["leaf"]), g(a["lesion"]), blend])


def demo(n_per_class=1):
    out = ROOT / "results" / "lowlevel"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cdir in sorted((ROOT / "data" / "test").iterdir()):
        for p in sorted(cdir.glob("*"))[:n_per_class]:
            img = cv2.resize(cv2.imread(str(p)), (256, 256))
            v = visualise(img, analyse(img))
            cv2.putText(v, cdir.name, (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            rows.append(v)
    cv2.imwrite(str(out / "segmentation_examples.jpg"), np.vstack(rows))
    print("saved", out / "segmentation_examples.jpg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--image")
    args = ap.parse_args()
    if args.image:
        img = cv2.imread(args.image)
        a = analyse(img)
        cv2.imwrite("segmentation.jpg", visualise(img, a))
        print(f"severity = {a['severity']:.1f}%")
    else:
        demo()
