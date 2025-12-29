from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class Background:
    def __init__(self, path: str = "assets/backgrounds/bg.png") -> None:
        self.path = Path(path)
        self._img_bgr: np.ndarray | None = None

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Background image not found at: {self.path}\n"
                f"Create it (suggested 1280x720) and re-run."
            )
        img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read background image: {self.path}")
        self._img_bgr = img

    def render(self, size_wh: tuple[int, int]) -> np.ndarray:
        """Return a BGR background canvas resized to (w,h)."""
        if self._img_bgr is None:
            self.load()
        w, h = size_wh
        bg = cv2.resize(self._img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        return bg
