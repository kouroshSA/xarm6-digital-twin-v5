# teleop_hand/testing.py
"""Synthetic HandFrames and a fake clock/source, for tests and --demo."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from teleop_hand.messages import HandFrame, make_frame

#: A comfortable palm pose: 0.6 m in front of the camera, palm facing it.
PALM0 = np.array([0.0, 0.05, 0.60])
ROT0 = np.eye(3)


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FrameSource:
    """`poll()` like UdpReceiver: newest queued frame, or None."""

    def __init__(self):
        self.q: deque = deque()

    def push(self, f: HandFrame) -> None:
        self.q.append(f)

    def poll(self) -> Optional[HandFrame]:
        f = None
        while self.q:
            f = self.q.popleft()
        return f


class Hand:
    """A scripted hand: set pos/rot/opening, `frame(t)` emits a HandFrame."""

    def __init__(self, pos=PALM0, rot=ROT0, opening: float = 0.9, quality: float = 0.9):
        self.pos = np.array(pos, float)
        self.rot = np.array(rot, float)
        self.opening = opening
        self.quality = quality
        self.detected = True
        self.fist = False
        self.approx = False
        self.seq = 0

    def frame(self, t: float) -> HandFrame:
        self.seq += 1
        if not self.detected:
            return make_frame(self.seq, t_mono=t, detected=False)
        return make_frame(self.seq, t_mono=t, detected=True, pos_valid=True,
                          palm_pos_cam_m=self.pos, palm_rot_cam=self.rot,
                          rot_source="fused", rot_quality=self.quality,
                          opening=self.opening, aux_hand_fist=self.fist,
                          confidence=0.95, approx=self.approx)


def rot_about(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * np.deg2rad(deg)).as_matrix()
