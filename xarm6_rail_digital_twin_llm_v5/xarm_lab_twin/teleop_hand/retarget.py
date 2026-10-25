# teleop_hand/retarget.py
"""Gripper opening from hand keypoints, via dex-retargeting (§3).

`GripperRetargeter.opening(joint_pos)` -> 0 (closed) .. 1 (open), where
`joint_pos` is the (21, 3) MANO-frame array `SingleHandDetector.detect()`
returns. One DoF today; a dexterous hand later is a new URDF + yml, not new
code here.

Two rules:
  * joints by NAME. `retargeting.joint_names` puts six dummy free-joint DoFs
    before `finger_joint` (see xarm_gripper.yml for why they are there), and
    any URDF edit can reorder them.
  * FINGER_JOINT_MAX is READ from the URDF through the retargeter, not
    copied into Python, so the two cannot drift.

`pinch_reference()` is the direct thumb-index distance mapped by calibrated
endpoints: the tracker publishes it as `opening_raw_m` and test 1 checks the
retargeter against it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from teleop_hand import config

ASSETS = Path(__file__).resolve().parent / "assets" / "retarget"
CONFIG_YML = ASSETS / "xarm_gripper.yml"
JOINT = "finger_joint"


def pinch_distance_m(joint_pos: np.ndarray) -> float:
    joint_pos = np.asarray(joint_pos, float)
    return float(np.linalg.norm(joint_pos[4] - joint_pos[8]))


def pinch_reference(joint_pos: np.ndarray,
                    closed_m: float = config.PINCH_CLOSED_M,
                    open_m: float = config.PINCH_OPEN_M) -> float:
    d = pinch_distance_m(joint_pos)
    return float(np.clip((d - closed_m) / (open_m - closed_m), 0.0, 1.0))


class GripperRetargeter:
    def __init__(self, config_path: Path = CONFIG_YML, urdf_dir: Optional[Path] = None,
                 low_pass_alpha: Optional[float] = None):
        from dex_retargeting.retargeting_config import RetargetingConfig
        RetargetingConfig.set_default_urdf_dir(str(urdf_dir or ASSETS))
        cfg = RetargetingConfig.load_from_file(str(config_path))
        if low_pass_alpha is not None:
            cfg.low_pass_alpha = low_pass_alpha
        self.retargeting = cfg.build()
        self.indices = self.retargeting.optimizer.target_link_human_indices
        names = list(self.retargeting.joint_names)
        self.ji = names.index(JOINT)                       # by name
        robot = self.retargeting.optimizer.robot
        lo, hi = robot.joint_limits[robot.dof_joint_names.index(JOINT)]
        self.finger_min, self.finger_max = float(lo), float(hi)
        self.scaling = float(cfg.scaling_factor)

    def reset(self) -> None:
        self.retargeting.reset()

    def finger_q(self, joint_pos: np.ndarray) -> float:
        joint_pos = np.asarray(joint_pos, float)
        ref = joint_pos[self.indices[1], :] - joint_pos[self.indices[0], :]
        return float(self.retargeting.retarget(ref)[self.ji])

    def opening(self, joint_pos: np.ndarray) -> float:
        q = self.finger_q(joint_pos)
        return float(np.clip((q - self.finger_min) / (self.finger_max - self.finger_min),
                             0.0, 1.0))


# ---------------------------------------------------------------------------
# receiver side: one opening -> what each effector does
# ---------------------------------------------------------------------------
class BinaryGrip:
    """Close below GRIP_CLOSE_BELOW, open above GRIP_OPEN_ABOVE, hold in
    between. `update()` returns "close" / "open" on a TRANSITION, else None."""

    def __init__(self, close_below: float = config.GRIP_CLOSE_BELOW,
                 open_above: float = config.GRIP_OPEN_ABOVE, closed: bool = False):
        assert close_below < open_above
        self.close_below, self.open_above = close_below, open_above
        self.closed = closed

    def update(self, opening: Optional[float]) -> Optional[str]:
        if opening is None:
            return None
        if not self.closed and opening < self.close_below:
            self.closed = True
            return "close"
        if self.closed and opening > self.open_above:
            self.closed = False
            return "open"
        return None


class ContinuousGrip:
    """Real standard gripper: send only on a change of at least
    REAL_GRIP_MIN_CHANGE, at most REAL_GRIP_MAX_HZ. `update()` returns the
    opening to send, or None."""

    def __init__(self, min_change: float = config.REAL_GRIP_MIN_CHANGE,
                 max_hz: float = config.REAL_GRIP_MAX_HZ):
        self.min_change, self.period = min_change, 1.0 / max_hz
        self.sent: Optional[float] = None
        self._t = -1e9

    def update(self, opening: Optional[float], now: float) -> Optional[float]:
        if opening is None or now - self._t < self.period:
            return None
        if self.sent is not None and abs(opening - self.sent) < self.min_change:
            return None
        self.sent, self._t = float(opening), now
        return self.sent


class PinchFreeze:
    """True while the opening changes faster than PINCH_FREEZE_RATE (plus a
    short hold): closing the hand drags the palm with it, and tracking that
    is "the arm jumps when I pinch"."""

    def __init__(self, rate: float = config.PINCH_FREEZE_RATE,
                 hold_s: float = config.PINCH_FREEZE_HOLD_S, enabled: bool = True):
        self.rate, self.hold_s, self.enabled = rate, hold_s, enabled
        self._last: Optional[tuple] = None
        self._until = -1e9

    def reset(self) -> None:
        self._last = None
        self._until = -1e9

    def update(self, opening: Optional[float], t: float) -> bool:
        if not self.enabled or opening is None:
            return False
        if self._last is not None:
            o0, t0 = self._last
            if t > t0 and abs(opening - o0) / (t - t0) > self.rate:
                self._until = t + self.hold_s
        self._last = (opening, t)
        return t < self._until
