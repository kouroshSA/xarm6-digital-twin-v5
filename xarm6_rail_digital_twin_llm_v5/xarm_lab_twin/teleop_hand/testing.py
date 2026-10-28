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
        self.gesture = "none"
        self.point_origin = None      # camera frame, metres
        self.point_dir = None
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
                          gesture=self.gesture, point_origin_cam_m=self.point_origin,
                          point_dir_cam=self.point_dir,
                          confidence=0.95, approx=self.approx)


def rot_about(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * np.deg2rad(deg)).as_matrix()


# ---------------------------------------------------------------------------
# a synthetic hand skeleton (MediaPipe's 21 landmarks), metres, wrist at the
# origin, fingers along +y, palm normal along z
# ---------------------------------------------------------------------------
_FINGERS = {                      # name: (first landmark, mcp xy, segment lengths)
    "thumb": (1, (0.03, 0.02), (0.04, 0.032, 0.028)),
    "index": (5, (0.03, 0.09), (0.040, 0.025, 0.022)),
    "middle": (9, (0.01, 0.095), (0.044, 0.028, 0.024)),
    "ring": (13, (-0.01, 0.09), (0.041, 0.026, 0.023)),
    "pinky": (17, (-0.03, 0.08), (0.032, 0.020, 0.020)),
}


def synth_keypoints(extended=("index", "middle", "ring", "pinky"), curled=()) -> np.ndarray:
    """(21, 3) landmarks. Each finger is straight (listed in `extended`),
    curled into the palm (`curled`), or -- if unlisted -- half bent."""
    kp = np.zeros((21, 3))
    for name, (i0, (mx, my), seg) in _FINGERS.items():
        if name == "thumb":
            for k, a in enumerate((0, 1, 2)):
                kp[i0 + 1 + k] = [mx + 0.03 * (k + 1), my + 0.02 * (k + 1), 0.0]
            kp[i0] = [mx, my, 0.0]
            continue
        bend = (0.0, 0.0, 0.0) if name in extended else \
            ((80.0, 90.0, 60.0) if name in curled else (35.0, 35.0, 25.0))
        p = np.array([mx, my, 0.0])
        kp[i0] = p
        ang = 0.0
        for k in range(3):
            ang += bend[k]
            r = np.deg2rad(ang)
            p = p + seg[k] * np.array([0.0, np.cos(r), -np.sin(r)])
            kp[i0 + 1 + k] = p
    return kp


def synth_pose(kind: str) -> np.ndarray:
    return {
        "open": lambda: synth_keypoints(),
        "fist": lambda: synth_keypoints(extended=(), curled=("index", "middle", "ring", "pinky")),
        "point": lambda: synth_keypoints(extended=("index",), curled=("middle", "ring", "pinky")),
        "peace": lambda: synth_keypoints(extended=("index", "middle"), curled=("ring", "pinky")),
        "index_curled": lambda: synth_keypoints(extended=("middle", "ring", "pinky"),
                                                curled=("index",)),
    }[kind]()
