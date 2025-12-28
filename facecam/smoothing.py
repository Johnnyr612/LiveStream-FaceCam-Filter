from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EMA:
    """Exponential moving average smoother."""
    alpha: float
    value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value
