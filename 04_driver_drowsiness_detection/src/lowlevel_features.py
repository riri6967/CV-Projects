"""LOW-LEVEL vision for drowsiness: pixel operations and hand-crafted descriptors (no CNN).

EYE (Open vs Closed, eye-crop images)
  pre-processing : resize 64x64, grayscale, CLAHE, Gaussian blur
  geometric cues : * dark-blob shape  - threshold at 0.55 x median -> opening -> largest blob;
                     an open eye has a round iris (height/width high), a closed eye a thin lash line
                   * gradient ratio   - Sobel vertical / horizontal energy (closed = horizontal lid)
                   * sclera ratio     - bright, unsaturated pixels (white of the eye) in HSV
                   * darkest-pixel share, blob fill ratio
  descriptor     : HOG (9 orientations, 8x8 cells, 16x16 blocks) on the CLAHE image

MOUTH (yawn vs no_yawn, full in-car frames)
  face localisation : skin segmentation in YCrCb (Cr 135-175, Cb 85-130) -> morphology ->
                      largest face-shaped blob
  mouth cues        : lower-face ROI -> CLAHE -> dark 'cavity' pixels (< 0.5 x median) ->
                      largest dark blob area / height-width; holes in the skin mask
  descriptor        : HOG of the grayscale frame (160x120)
"""
import cv2
import numpy as np

CLAHE_EYE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
CLAHE_FACE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
HOG_EYE = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
HOG_FRAME = cv2.HOGDescriptor((160, 120), (16, 16), (8, 8), (8, 8), 9)
EYE_GEOM_NAMES = ["dark_share", "blob_h_over_w", "blob_fill", "sclera_share", "grad_v_over_h", "darkest_share"]
MOUTH_GEOM_NAMES = ["cavity_share", "cavity_h_over_w", "cavity_blob_area", "roi_contrast", "skin_hole_area"]


def eye_preprocess(bgr):
    b = cv2.resize(bgr, (64, 64))
    g = CLAHE_EYE.apply(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    return b, cv2.GaussianBlur(g, (3, 3), 0)


def eye_geometric(bgr):
    b, g = eye_preprocess(bgr)
    c = g[10:54, 6:58]
    dark = (c < 0.55 * np.median(c)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    if n > 1:
        k = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        h, w, a = st[k, 3], st[k, 2], st[k, 4]
        hw, fill = h / max(w, 1), a / max(h * w, 1)
    else:
        hw = fill = 0.0
    hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[10:54, 6:58]
    sclera = ((hsv[..., 1] < 60) & (hsv[..., 2] > 0.9 * np.percentile(hsv[..., 2], 90))).mean()
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad_ratio = float((gy ** 2).sum() / ((gx ** 2).sum() + 1e-6))
    darkest = (c < np.percentile(g, 5) + 10).mean()
    return np.array([dark.mean(), hw, fill, sclera, grad_ratio, darkest], np.float32)


def eye_hog(bgr):
    return HOG_EYE.compute(eye_preprocess(bgr)[1]).ravel()


def face_box(bgr):
    """Skin-colour face localisation. Returns resized image, skin mask, (x, y, w, h, label) or None."""
    im = cv2.resize(bgr, (320, 240))
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (0, 135, 85), (255, 175, 130))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin)
    best, best_s = None, 0
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < 800 or h < 30:
            continue
        s = a * (1.0 if 0.6 < h / w < 2.2 else 0.3) * (1.2 if y < 120 else 0.6)
        if s > best_s:
            best_s, best = s, (x, y, w, min(h, int(1.2 * w)), k)
    return im, skin, lab, best


def mouth_geometric(bgr):
    im, skin, lab, b = face_box(bgr)
    if b is None:
        return np.zeros(5, np.float32)
    x, y, w, h, k = b
    y0, y1, x0, x1 = y + int(0.6 * h), y + int(0.9 * h), x + int(0.25 * w), x + int(0.75 * w)
    roi = im[y0:y1, x0:x1]
    if roi.size == 0:
        return np.zeros(5, np.float32)
    g = cv2.GaussianBlur(CLAHE_FACE.apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)), (3, 3), 0)
    dark = (g < 0.5 * np.median(g)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    if n > 1:
        j = 1 + int(st[1:, cv2.CC_STAT_AREA].argmax())
        hw, area = st[j, 3] / max(st[j, 2], 1), st[j, 4] / g.size
    else:
        hw = area = 0.0
    face = (lab == k).astype(np.uint8) * 255
    filled = face.copy()
    cnts, _ = cv2.findContours(face, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
    holes = cv2.subtract(filled, face)[y + int(0.55 * h):y + h, x:x + w]
    hole_area = (holes > 0).mean() if holes.size else 0.0
    return np.array([dark.mean(), hw, area, g.std() / 255, hole_area], np.float32)


def frame_hog(bgr):
    g = CLAHE_FACE.apply(cv2.cvtColor(cv2.resize(bgr, (160, 120)), cv2.COLOR_BGR2GRAY))
    return HOG_FRAME.compute(g).ravel()


def eye_debug(bgr):
    """Visualise the low-level eye pipeline: input | CLAHE | dark mask | vertical gradient | HOG."""
    b, g = eye_preprocess(bgr)
    dark = ((g < 0.55 * np.median(g[10:54, 6:58])) * 255).astype(np.uint8)
    gy = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    hog_vis = hog_image(g)
    tiles = [b] + [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (g, dark, gy, hog_vis)]
    return np.hstack([cv2.resize(t, (128, 128), interpolation=cv2.INTER_NEAREST) for t in tiles])


def hog_image(g, cell=8, bins=9):
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180
    out = np.zeros((g.shape[0] * 2, g.shape[1] * 2), np.uint8)
    for cy in range(0, g.shape[0], cell):
        for cx in range(0, g.shape[1], cell):
            hist, _ = np.histogram(ang[cy:cy + cell, cx:cx + cell], bins=bins, range=(0, 180),
                                   weights=mag[cy:cy + cell, cx:cx + cell])
            hist = hist / (hist.max() + 1e-6)
            c = (cx * 2 + cell, cy * 2 + cell)
            for bi, v in enumerate(hist):
                t = np.deg2rad(bi * 180 / bins + 90)
                d = np.array([np.cos(t), np.sin(t)]) * cell * v
                cv2.line(out, (int(c[0] - d[0]), int(c[1] - d[1])), (int(c[0] + d[0]), int(c[1] + d[1])), int(255 * v), 1)
    return cv2.resize(out, g.shape[::-1])
