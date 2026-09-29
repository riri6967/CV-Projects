"""Classical lane detection + lane departure warning (no deep learning).

Pipeline per frame
  1. Colour mask   : keep white + yellow pixels (HLS colour space)
  2. Grayscale + Gaussian blur
  3. Canny edge detection
  4. Region-of-interest (trapezoid in front of the car)
  5. Probabilistic Hough transform -> line segments
  6. Split segments into left / right by slope, fit one line per side
  7. Temporal smoothing (exponential moving average) for video
  8. Departure warning: offset of lane centre from image centre

Usage:
    python src/lane_detector.py --input data/test_images --output results/images
    python src/lane_detector.py --input data/test_videos/solidWhiteRight.mp4 --output results/videos
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------- tunable parameters ----------------
CANNY_LOW, CANNY_HIGH = 50, 150
BLUR_K = 5
HOUGH = dict(rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=100)
MIN_ABS_SLOPE = 0.4          # ignore near-horizontal segments
ROI_TOP = 0.60               # top of ROI as a fraction of image height
SMOOTH_ALPHA = 0.2           # EMA weight of the new measurement (video)
DEPARTURE_THRESH = 0.15      # |offset| / lane width that triggers a warning


def colour_mask(img):
    """Keep white and yellow lane paint."""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (10, 0, 100), (40, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.bitwise_and(img, img, mask=mask)


def roi_vertices(h, w):
    return np.array([[(int(0.05 * w), h), (int(0.45 * w), int(ROI_TOP * h)),
                      (int(0.55 * w), int(ROI_TOP * h)), (int(0.95 * w), h)]], dtype=np.int32)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def fit_side(segments, h):
    """Least-squares fit x = m*y + c through all segment endpoints of one side."""
    if not segments:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    m, c = np.polyfit(ys, xs, 1)
    return np.array([m, c], dtype=np.float64)


def line_points(params, h):
    m, c = params
    y1, y2 = h, int(ROI_TOP * h)
    return (int(m * y1 + c), y1), (int(m * y2 + c), y2)


class LaneDetector:
    def __init__(self, smooth=True):
        self.smooth = smooth
        self.left = None
        self.right = None
        self.lane_w = None   # learned lane width at the image bottom (pixels)

    def _update(self, old, new):
        if new is None:
            return old
        if old is None or not self.smooth:
            return new
        return (1 - SMOOTH_ALPHA) * old + SMOOTH_ALPHA * new

    def detect(self, frame):
        """Return dict with debug images, lane params and departure info."""
        h, w = frame.shape[:2]
        masked = colour_mask(frame)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        roi = region_of_interest(edges, roi_vertices(h, w))
        lines = cv2.HoughLinesP(roi, **HOUGH)

        left_seg, right_seg = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < MIN_ABS_SLOPE:
                    continue
                # image y grows downward: left lane has negative slope
                if slope < 0:
                    left_seg.append((x1, y1, x2, y2))
                else:
                    right_seg.append((x1, y1, x2, y2))

        l_new, r_new = fit_side(left_seg, h), fit_side(right_seg, h)
        self.left = self._update(self.left, l_new)
        self.right = self._update(self.right, r_new)

        info = dict(left_found=l_new is not None, right_found=r_new is not None,
                    left=self.left, right=self.right, offset=None, warning=False,
                    n_segments=0 if lines is None else len(lines))
        lx = None if self.left is None else line_points(self.left, h)[0][0]
        rx = None if self.right is None else line_points(self.right, h)[0][0]
        if lx is not None and rx is not None and rx - lx > 0.2 * w:
            # both lines found: measure lane width and remember it
            lane_w = rx - lx
            self.lane_w = lane_w if self.lane_w is None else 0.9 * self.lane_w + 0.1 * lane_w
        elif self.lane_w is not None and (lx is not None or rx is not None):
            # single-line fallback: infer the missing side from the learned lane width
            lane_w = self.lane_w
            if lx is None:
                lx = rx - lane_w
            else:
                rx = lx + lane_w
            info["single_line_fallback"] = True
        else:
            lx = rx = None
        if lx is not None:
            lane_centre = (lx + rx) / 2
            offset = (w / 2 - lane_centre) / lane_w  # + => car right of centre
            info["offset"] = float(offset)
            info["warning"] = abs(offset) > DEPARTURE_THRESH
        info["debug"] = dict(mask=masked, edges=edges, roi=roi)
        return info

    @staticmethod
    def draw(frame, info):
        out = frame.copy()
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        pts = []
        for side, colour in (("left", (0, 0, 255)), ("right", (255, 0, 0))):
            if info[side] is not None:
                p1, p2 = line_points(info[side], h)
                cv2.line(overlay, p1, p2, colour, 10)
                pts.append((p1, p2))
        if len(pts) == 2:
            poly = np.array([pts[0][0], pts[0][1], pts[1][1], pts[1][0]], np.int32)
            cv2.fillPoly(overlay, [poly], (0, 180, 0))
        out = cv2.addWeighted(out, 1.0, overlay, 0.4, 0)
        if info["offset"] is not None:
            txt = f"Offset: {info['offset']*100:+.1f}% of lane width"
            cv2.putText(out, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if info["warning"]:
            cv2.putText(out, "LANE DEPARTURE WARNING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 3)
        cv2.polylines(out, roi_vertices(h, w), True, (0, 255, 255), 1)
        return out


def process_image(path, out_dir):
    img = cv2.imread(str(path))
    det = LaneDetector(smooth=False)
    info = det.detect(img)
    out = det.draw(img, info)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_lanes.jpg"), out)
    # pipeline stages figure: original | colour mask | edges | ROI edges | result
    small = lambda im: cv2.resize(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), (480, 270))
    d = info["debug"]
    strip = np.hstack([small(img), small(d["mask"]), small(d["edges"]), small(d["roi"]), small(out)])
    cv2.imwrite(str(out_dir / f"{path.stem}_pipeline.jpg"), strip)
    return info


def process_video(path, out_dir, max_frames=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{path.stem}_lanes.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    det = LaneDetector(smooth=True)
    stats = []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and len(stats) >= max_frames):
            break
        info = det.detect(frame)
        out = det.draw(frame, info)
        writer.write(out)
        stats.append(dict(left_found=info["left_found"], right_found=info["right_found"],
                          offset=info["offset"], warning=info["warning"],
                          left=None if info["left"] is None else info["left"].tolist(),
                          right=None if info["right"] is None else info["right"].tolist()))
        if len(stats) % 30 == 1:
            cv2.imwrite(str(out_dir / f"{path.stem}_frame{len(stats):04d}.jpg"), out)
    elapsed = time.time() - t0
    cap.release()
    writer.release()
    return stats, len(stats) / max(elapsed, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image, folder of images, or video")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    src, out = Path(args.input), Path(args.output)
    if src.is_dir():
        for p in sorted(src.glob("*.jpg")):
            info = process_image(p, out)
            print(p.name, "left:", info["left_found"], "right:", info["right_found"],
                  "offset:", info["offset"])
    elif src.suffix.lower() in (".mp4", ".avi", ".mov"):
        stats, fps = process_video(src, out)
        print(f"{src.name}: {len(stats)} frames, {fps:.1f} FPS")
    else:
        process_image(src, out)


if __name__ == "__main__":
    main()
