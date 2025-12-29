from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp

from .smoothing import EMA


# MediaPipe Face Mesh landmark indices (common / stable)
# (Using FaceMesh because it’s easiest + reliable for MVP.)
LM = {
    # Eye corners / lids for blink (approx)
    "L_EYE_OUTER": 33,
    "L_EYE_INNER": 133,
    "L_EYE_TOP": 159,
    "L_EYE_BOT": 145,
    "R_EYE_OUTER": 263,
    "R_EYE_INNER": 362,
    "R_EYE_TOP": 386,
    "R_EYE_BOT": 374,
    # Mouth
    "MOUTH_L": 61,
    "MOUTH_R": 291,
    "MOUTH_TOP": 13,
    "MOUTH_BOT": 14,
    # Nose / chin for pose
    "NOSE_TIP": 1,
    "CHIN": 152,
    "L_MOUTH": 61,
    "R_MOUTH": 291,
    "L_EYE": 33,
    "R_EYE": 263,
}


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _pt3(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h, lm.z], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(outer, inner, top, bot) -> float:
    # Simple EAR-like ratio: vertical / horizontal
    horiz = _dist(outer, inner) + 1e-6
    vert = _dist(top, bot)
    return vert / horiz


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class FaceSignals:
    # Smoothed signals
    yaw: float = 0.0   # degrees
    pitch: float = 0.0 # degrees
    roll: float = 0.0  # degrees
    blink_l: float = 0.0  # 0=open, 1=closed
    blink_r: float = 0.0
    mouth_open: float = 0.0 # 0..1
    smile: float = 0.0      # 0..1


class FaceTracker:
    """
    Face tracking + signal extraction for MVP.
    Uses MediaPipe FaceMesh (468 landmarks).
    """

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Smoothers (tuned to reduce jitter without feeling laggy)
        self.s_pose = {
            "yaw": EMA(alpha=0.35),
            "pitch": EMA(alpha=0.35),
            "roll": EMA(alpha=0.35),
        }
        self.s_fast = {
            "blink_l": EMA(alpha=0.55),
            "blink_r": EMA(alpha=0.55),
            "mouth_open": EMA(alpha=0.45),
            "smile": EMA(alpha=0.45),
        }

        # Blink calibration defaults (EAR thresholds). We refine over time in M3 if needed.
        self.ear_open = 0.28
        self.ear_closed = 0.16

        # Mouth open ratio thresholds (vertical / width)
        self.mouth_open_min = 0.02
        self.mouth_open_max = 0.10

        # Smile ratio thresholds (mouth width / nose-chin approx)
        self.smile_min = 0.32
        self.smile_max = 0.42

    def process(self, frame_bgr: np.ndarray) -> Tuple[FaceSignals | None, Dict[str, float]]:
        """
        Returns (signals, debug_values).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return None, {}

        landmarks = results.multi_face_landmarks[0].landmark

        # Anchor point for puppet placement (use nose tip as stable anchor)
        nose_xy = _pt(landmarks, LM["NOSE_TIP"], w, h)
        anchor_xy = (int(nose_xy[0]), int(nose_xy[1]))

        # --- Compute raw measurements ---
        # Eye EARs
        l_outer = _pt(landmarks, LM["L_EYE_OUTER"], w, h)
        l_inner = _pt(landmarks, LM["L_EYE_INNER"], w, h)
        l_top   = _pt(landmarks, LM["L_EYE_TOP"], w, h)
        l_bot   = _pt(landmarks, LM["L_EYE_BOT"], w, h)

        r_outer = _pt(landmarks, LM["R_EYE_OUTER"], w, h)
        r_inner = _pt(landmarks, LM["R_EYE_INNER"], w, h)
        r_top   = _pt(landmarks, LM["R_EYE_TOP"], w, h)
        r_bot   = _pt(landmarks, LM["R_EYE_BOT"], w, h)

        ear_l = _eye_aspect_ratio(l_outer, l_inner, l_top, l_bot)
        ear_r = _eye_aspect_ratio(r_outer, r_inner, r_top, r_bot)

        # Convert EAR to blink 0..1 (1=closed)
        def ear_to_blink(ear: float) -> float:
            t = (self.ear_open - ear) / (self.ear_open - self.ear_closed + 1e-6)
            return _clamp01(t)

        blink_l = ear_to_blink(ear_l)
        blink_r = ear_to_blink(ear_r)

        # Mouth open ratio: vertical / width
        m_l = _pt(landmarks, LM["MOUTH_L"], w, h)
        m_r = _pt(landmarks, LM["MOUTH_R"], w, h)
        m_t = _pt(landmarks, LM["MOUTH_TOP"], w, h)
        m_b = _pt(landmarks, LM["MOUTH_BOT"], w, h)

        mouth_w = _dist(m_l, m_r) + 1e-6
        mouth_h = _dist(m_t, m_b)
        mouth_ratio = mouth_h / mouth_w

        mouth_open = _clamp01((mouth_ratio - self.mouth_open_min) / (self.mouth_open_max - self.mouth_open_min + 1e-6))

        # Smile approx: mouth width normalized by face height (nose tip to chin)
        nose = _pt(landmarks, LM["NOSE_TIP"], w, h)
        chin = _pt(landmarks, LM["CHIN"], w, h)
        face_h = _dist(nose, chin) + 1e-6
        smile_ratio = mouth_w / face_h
        smile = _clamp01((smile_ratio - self.smile_min) / (self.smile_max - self.smile_min + 1e-6))

        # Head pose (solvePnP) using a tiny set of points
        # 3D model points are generic face model approximations (units arbitrary).
        image_points = np.array([
            _pt(landmarks, LM["NOSE_TIP"], w, h),  # Nose tip
            _pt(landmarks, LM["CHIN"], w, h),      # Chin
            _pt(landmarks, LM["L_EYE"], w, h),     # Left eye corner
            _pt(landmarks, LM["R_EYE"], w, h),     # Right eye corner
            _pt(landmarks, LM["L_MOUTH"], w, h),   # Left mouth corner
            _pt(landmarks, LM["R_MOUTH"], w, h),   # Right mouth corner
        ], dtype=np.float32)

        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.0, -12.0),    # Chin
            (-43.0, 32.0, -26.0),   # Left eye
            (43.0, 32.0, -26.0),    # Right eye
            (-28.0, -28.0, -24.0),  # Left mouth
            (28.0, -28.0, -24.0),   # Right mouth
        ], dtype=np.float32)

        focal = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1],
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        # Use solvePnP to get rotation
        ok, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        yaw = pitch = roll = 0.0
        if ok:
            rmat, _ = cv2.Rodrigues(rvec)
            # Convert rotation matrix to Euler angles (approx)
            sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0.0

            pitch = float(np.degrees(x))
            yaw = float(np.degrees(y))
            roll = float(np.degrees(z))

        # --- Smooth signals ---
        yaw_s = self.s_pose["yaw"].update(yaw)
        pitch_s = self.s_pose["pitch"].update(pitch)
        roll_s = self.s_pose["roll"].update(roll)

        blink_l_s = self.s_fast["blink_l"].update(blink_l)
        blink_r_s = self.s_fast["blink_r"].update(blink_r)
        mouth_open_s = self.s_fast["mouth_open"].update(mouth_open)
        smile_s = self.s_fast["smile"].update(smile)

        signals = FaceSignals(
            yaw=yaw_s,
            pitch=pitch_s,
            roll=roll_s,
            blink_l=blink_l_s,
            blink_r=blink_r_s,
            mouth_open=mouth_open_s,
            smile=smile_s,
        )

        debug = {
            "ear_l": ear_l,
            "ear_r": ear_r,
            "mouth_ratio": mouth_ratio,
            "smile_ratio": smile_ratio,
        }
        debug["anchor_x"] = float(anchor_xy[0])
        debug["anchor_y"] = float(anchor_xy[1])
        return signals, debug

    def draw_debug(self, frame_bgr: np.ndarray, signals: FaceSignals, debug: Dict[str, float]) -> None:
        """Draw signal values on frame."""
        y0 = 80
        dy = 28
        lines = [
            f"yaw: {signals.yaw:6.1f}  pitch: {signals.pitch:6.1f}  roll: {signals.roll:6.1f}",
            f"blink L: {signals.blink_l:0.2f}   blink R: {signals.blink_r:0.2f}",
            f"mouth_open: {signals.mouth_open:0.2f}   smile: {signals.smile:0.2f}",
        ]
        lines2 = [
            f"ear_l: {debug.get('ear_l', 0.0):0.3f}  ear_r: {debug.get('ear_r', 0.0):0.3f}",
            f"mouth_ratio: {debug.get('mouth_ratio', 0.0):0.3f}  smile_ratio: {debug.get('smile_ratio', 0.0):0.3f}",
        ]

        for i, t in enumerate(lines):
            cv2.putText(frame_bgr, t, (20, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        for i, t in enumerate(lines2):
            cv2.putText(frame_bgr, t, (20, y0 + (len(lines) + i) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
