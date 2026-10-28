# teleop_hand/gestures.py
"""Hand-gesture classification and the pointing ray. Pure NumPy.

Used by the tracker (which has MediaPipe) and by the tests (which do not), so
this module imports neither MediaPipe nor MuJoCo.

Two gestures beyond the pinch that drives the gripper:

  fist   all four fingertips closer to the wrist than their MCPs
         (`palm.is_fist`, the same test the left-hand clutch uses).
  point  index extended and straight; middle, ring and pinky curled.
         The thumb is ignored: people point with it up, tucked, or out.

A pointing ray is built from the index MCP's DEPTH-MEASURED position and the
DIRECTION of MediaPipe's metric world landmarks (MCP -> tip). Not from the
fingertip's depth: a fingertip is a few pixels wide, so its depth sample is
usually the background behind it, and a ray through a point 300 mm too far
away is wrong in a way no filter fixes. A position on the hand is measured; a
direction is what the landmark model is good at.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from teleop_hand import config, palm

INDEX = (5, 6, 7, 8)                       # MCP, PIP, DIP, tip
OTHERS = ((9, 12), (13, 16), (17, 20))     # (MCP, tip) of middle, ring, pinky
NONE, FIST, POINT = "none", "fist", "point"


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def index_extended(kp: np.ndarray, straight_deg: float = config.POINT_STRAIGHT_DEG) -> bool:
    mcp, pip, _dip, tip = (kp[i] for i in INDEX)
    w = kp[0]
    if np.linalg.norm(tip - w) <= np.linalg.norm(pip - w):
        return False
    return _angle_deg(pip - mcp, tip - pip) < straight_deg


def others_curled(kp: np.ndarray) -> bool:
    w = kp[0]
    return all(np.linalg.norm(kp[t] - w) < np.linalg.norm(kp[m] - w) for m, t in OTHERS)


def classify_gesture(kp: np.ndarray) -> str:
    """`kp` is (21, 3) metric landmarks (MediaPipe world landmarks)."""
    kp = np.asarray(kp, float)
    if palm.is_fist(kp):
        return FIST
    if index_extended(kp) and others_curled(kp):
        return POINT
    return NONE


def pointing_ray(kp: np.ndarray, kp_px: np.ndarray, depth_m: Optional[np.ndarray],
                 k: np.ndarray):
    """(origin_cam_m, dir_cam) or (None, None). Camera frame, OpenCV axes.

    Origin: index MCP deprojected with its measured depth. Direction: unit
    MCP -> tip from the world landmarks. `config.MEDIAPIPE_TO_CAMERA` maps
    MediaPipe's world axes to camera axes (measured identity on a D435)."""
    if depth_m is None:
        return None, None
    z = palm.depth_at(depth_m, kp_px[5, 0], kp_px[5, 1])
    if z <= 0:
        return None, None
    d = config.MEDIAPIPE_TO_CAMERA @ (np.asarray(kp, float)[8] - np.asarray(kp, float)[5])
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None, None
    return palm.deproject(kp_px[5, 0], kp_px[5, 1], z, k), d / n


class GestureDebounce:
    """A label that changes only after N consecutive frames agree."""

    def __init__(self, n: int = config.GESTURE_DEBOUNCE_FRAMES):
        self.n, self.state, self._cand, self._run = n, NONE, NONE, 0

    def update(self, raw: str) -> str:
        if raw == self.state:
            self._cand, self._run = raw, 0
        elif raw == self._cand:
            self._run += 1
            if self._run >= self.n:
                self.state, self._run = raw, 0
        else:
            self._cand, self._run = raw, 1
        return self.state
