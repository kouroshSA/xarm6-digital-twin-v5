# teleop_hand/filters.py
"""One Euro filter for the palm position, slerp low-pass for its rotation.

One Euro (Casiez et al., CHI 2012): a low-pass whose cutoff rises with speed,
so a still hand is smoothed hard (tremor gone) and a moving one barely lags.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x: Optional[np.ndarray] = None
        self.dx: Optional[np.ndarray] = None

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None or dt <= 0:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + _alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


class RotLowPass:
    """First-order low-pass on SO(3): slerp a fraction of the way each step."""

    def __init__(self, cutoff_hz: float):
        self.cutoff_hz = cutoff_hz
        self.reset()

    def reset(self) -> None:
        self.r: Optional[Rotation] = None

    def __call__(self, rot, dt: float) -> np.ndarray:
        r = Rotation.from_matrix(np.asarray(rot, float))
        if self.r is None or dt <= 0:
            self.r = r
        else:
            a = _alpha(self.cutoff_hz, dt)
            both = Rotation.concatenate([self.r, r])
            self.r = Slerp([0.0, 1.0], both)([a])[0]
        return self.r.as_matrix()
