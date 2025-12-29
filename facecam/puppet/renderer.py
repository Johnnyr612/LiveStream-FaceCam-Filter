from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def _load_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Missing asset: {path}")
    # Ensure RGBA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _alpha_blit(dst_bgr: np.ndarray, src_rgba: np.ndarray, x: int, y: int) -> None:
    """
    Alpha-blend src_rgba onto dst_bgr at top-left (x,y).
    Clips automatically if it goes out of bounds.
    """
    h, w = dst_bgr.shape[:2]
    sh, sw = src_rgba.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + sw)
    y1 = min(h, y + sh)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    src = src_rgba[sy0:sy1, sx0:sx1]
    src_bgr = src[:, :, :3].astype(np.float32)
    alpha = (src[:, :, 3:4].astype(np.float32)) / 255.0

    dst_roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)

    out = alpha * src_bgr + (1.0 - alpha) * dst_roi
    dst_bgr[y0:y1, x0:x1] = out.astype(np.uint8)


def _rotate_rgba(img: np.ndarray, degrees: float, pivot: Tuple[float, float]) -> np.ndarray:
    """
    Rotate RGBA image about pivot (in image coords).
    Keeps same canvas size.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(pivot, degrees, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


@dataclass
class PuppetRig:
    name: str
    base_scale: float
    pivot: Tuple[int, int]          # pivot inside puppet canvas
    canvas_size: Tuple[int, int]    # (w,h) for sanity
    layers: Dict[str, str]          # layer_name -> filename (png)


class PuppetRenderer:
    """
    Loads a puppet pack from assets/puppets/<name>/rig.json + PNG layers.
    Renders and animates it using face signals.
    """

    def __init__(self, puppets_root: str = "assets/puppets") -> None:
        self.puppets_root = Path(puppets_root)
        self.current_name: str | None = None
        self.rig: PuppetRig | None = None
        self.images: Dict[str, np.ndarray] = {}
        self.enabled: bool = True

    def load(self, puppet_name: str) -> None:
        pack = self.puppets_root / puppet_name
        rig_path = pack / "rig.json"
        if not rig_path.exists():
            raise FileNotFoundError(f"Missing rig.json for puppet '{puppet_name}' at {rig_path}")

        data = json.loads(rig_path.read_text(encoding="utf-8"))
        self.rig = PuppetRig(
            name=data["name"],
            base_scale=float(data.get("base_scale", 1.0)),
            pivot=tuple(data["pivot"]),
            canvas_size=tuple(data["canvas_size"]),
            layers=dict(data["layers"]),
        )

        self.images.clear()
        for layer_name, filename in self.rig.layers.items():
            self.images[layer_name] = _load_rgba(pack / filename)

        self.current_name = puppet_name

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def render_on(
        self,
        frame_bgr: np.ndarray,
        anchor_xy: Tuple[int, int],
        signals,
    ) -> None:
        """
        anchor_xy: where the puppet pivot should land on the frame.
        signals: FaceSignals from tracker (yaw/pitch/roll/blink/mouth_open/smile)
        """
        if not self.enabled or self.rig is None:
            return

        # We use roll as a natural "head tilt" for a 2D puppet.
        # Keep it subtle.
        roll = float(getattr(signals, "roll", 0.0))
        yaw = float(getattr(signals, "yaw", 0.0))
        pitch = float(getattr(signals, "pitch", 0.0))

        # Clamp rotation to avoid wild spins
        roll = max(-20.0, min(20.0, roll))

        # Tiny positional parallax from yaw/pitch (subtle, cute)
        dx = int(max(-30, min(30, yaw)) * 0.6)
        dy = int(max(-25, min(25, pitch)) * 0.6)

        # Choose eye layer based on blink
        blink_l = float(getattr(signals, "blink_l", 0.0))
        blink_r = float(getattr(signals, "blink_r", 0.0))
        eyes_closed = (blink_l > 0.55) or (blink_r > 0.55)

        # Choose mouth
        mouth_open = float(getattr(signals, "mouth_open", 0.0))
        mouth_is_open = mouth_open > 0.25

        # Smile affects a small scale of mouth layer (optional)
        smile = float(getattr(signals, "smile", 0.0))
        smile_scale = 1.0 + 0.15 * max(0.0, min(1.0, smile))

        rig = self.rig
        pivot = (float(rig.pivot[0]), float(rig.pivot[1]))

        # Compose layers into a single puppet canvas first (RGBA)
        # Order: body -> eyes -> mouth
        body = self.images["body"].copy()

        eyes = self.images["eyes_closed"] if eyes_closed else self.images["eyes_open"]
        _alpha_blit(body[:, :, :3], eyes, 0, 0)  # blend eyes onto body

        mouth = self.images["mouth_open"] if mouth_is_open else self.images["mouth_closed"]

        # Apply smile scaling to mouth layer (scale around pivot for simplicity)
        if abs(smile_scale - 1.0) > 1e-3:
            mh, mw = mouth.shape[:2]
            mouth_scaled = cv2.resize(mouth, (int(mw * smile_scale), int(mh * smile_scale)), interpolation=cv2.INTER_LINEAR)
            # center scaled mouth onto canvas by blitting with offset
            offx = (mw - mouth_scaled.shape[1]) // 2
            offy = (mh - mouth_scaled.shape[0]) // 2
            # Make a temp canvas same size as body
            mouth_canvas = np.zeros_like(body, dtype=np.uint8)
            # Put scaled mouth onto canvas, clipped by alpha blit
            _alpha_blit(mouth_canvas[:, :, :3], mouth_scaled, offx, offy)
            mouth = mouth_canvas

        _alpha_blit(body[:, :, :3], mouth, 0, 0)

        # Rotate puppet canvas around pivot
        puppet_rgba = _rotate_rgba(body, degrees=roll, pivot=pivot)

        # Scale puppet for output
        if abs(rig.base_scale - 1.0) > 1e-6:
            ph, pw = puppet_rgba.shape[:2]
            puppet_rgba = cv2.resize(puppet_rgba, (int(pw * rig.base_scale), int(ph * rig.base_scale)), interpolation=cv2.INTER_LINEAR)

        # Compute top-left position so pivot lands at anchor
        ax, ay = anchor_xy
        ax += dx
        ay += dy

        # Pivot shifts if scaled
        px = int(rig.pivot[0] * rig.base_scale)
        py = int(rig.pivot[1] * rig.base_scale)

        x = ax - px
        y = ay - py

        _alpha_blit(frame_bgr, puppet_rgba, x, y)
