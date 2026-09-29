"""HIGH-LEVEL vision (part 2): MediaPipe Face Mesh landmarks -> EAR / MAR.

MediaPipe Face Mesh is a CNN that regresses 468 3-D facial landmarks. From them:
  EAR (eye aspect ratio)   = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)   -> small when the eye is closed
  MAR (mouth aspect ratio) = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 |p1-p5|) -> large when yawning
(Soukupova & Cech 2016 formulation, mapped to Face Mesh indices.)

Requires mediapipe==0.10.14 (legacy `solutions` API, model files bundled in the wheel).
"""
import cv2
import numpy as np

try:
    import mediapipe as mp
    _FM = mp.solutions.face_mesh
except (ImportError, AttributeError) as e:  # pragma: no cover
    raise ImportError("pip install mediapipe==0.10.14  (newer versions removed mp.solutions)") from e

# Face Mesh indices: p1..p6 around each eye, p1..p8 around the inner lips
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]


def _d(a, b):
    return float(np.linalg.norm(a - b))


def ear(p):
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2 * _d(p[0], p[3]) + 1e-6)


def mar(p):
    return (_d(p[1], p[7]) + _d(p[2], p[6]) + _d(p[3], p[5])) / (2 * _d(p[0], p[4]) + 1e-6)


class FaceLandmarks:
    def __init__(self, static=True):
        self.fm = _FM.FaceMesh(static_image_mode=static, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.4)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        r = self.fm.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks:
            return None
        pts = np.array([(l.x * w, l.y * h) for l in r.multi_face_landmarks[0].landmark])
        le, re, mo = pts[LEFT_EYE], pts[RIGHT_EYE], pts[MOUTH]
        return dict(points=pts, ear=(ear(le) + ear(re)) / 2, mar=mar(mo), left_eye=le, right_eye=re, mouth=mo)

    @staticmethod
    def eye_crop(bgr, eye_pts, pad=0.6):
        x0, y0 = eye_pts.min(0); x1, y1 = eye_pts.max(0)
        cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * (1 + pad)
        x0, x1 = int(max(cx - s / 2, 0)), int(min(cx + s / 2, bgr.shape[1]))
        y0, y1 = int(max(cy - s / 2, 0)), int(min(cy + s / 2, bgr.shape[0]))
        return bgr[y0:y1, x0:x1]

    @staticmethod
    def draw(bgr, lm, text_lines=()):
        out = bgr.copy()
        for grp, col in ((lm["left_eye"], (0, 255, 0)), (lm["right_eye"], (0, 255, 0)), (lm["mouth"], (0, 200, 255))):
            cv2.polylines(out, [grp.astype(np.int32)], True, col, 2)
        if text_lines:
            w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] for t in text_lines)
            cv2.rectangle(out, (0, 0), (w + 20, 12 + 30 * len(text_lines)), (0, 0, 0), cv2.FILLED)
        for i, t in enumerate(text_lines):
            cv2.putText(out, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if "ALERT" in t else (255, 255, 255), 2)
        return out
