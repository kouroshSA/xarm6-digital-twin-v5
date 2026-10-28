# teleop_hand/gesture_control.py
"""`GestureController`: HandFrames -> fist-follow targets and point commands.

Pure of MuJoCo and sockets (the scene and executor are injected), so the
state machine is testable with scripted frames.

    fist    held FOLLOW_ENGAGE_S  ->  FOLLOW: every new frame yields a target EE
            position at a standoff from the fist, until the fist opens, the
            hand is lost, or the clutch takes over.
    point   ray held steady POINT_DWELL_S  ->  one `PointTarget`; the gesture
            must end before it can fire again.

The arm never moves on a stale frame: no new frame for `DROPOUT_S` drops to
idle, and a re-engaged fist has to be held again.

Hand positions arrive in the camera frame; `HandCameraPose` carries them to the
world frame. With the nominal pose that mapping is a PLACEHOLDER (see
`camera_pose.py`) -- the controller reports `cam_source` in its status so the
HUD and the runner can say so.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from teleop_hand import config
from teleop_hand.camera_pose import HandCameraPose
from teleop_hand.follow import FistFollower
from teleop_hand.pointing import PointDwell, PointTarget, resolve_pointing


@dataclasses.dataclass
class GestureOutput:
    kind: str = "none"                       # none | follow | point
    follow_target_m: Optional[np.ndarray] = None
    point: Optional[PointTarget] = None
    status: dict = dataclasses.field(default_factory=dict)


class GestureController:
    def __init__(self, cam: HandCameraPose, scene, follower: Optional[FistFollower] = None,
                 dwell: Optional[PointDwell] = None):
        self.cam = cam
        self.scene = scene
        self.follower = follower or FistFollower()
        self.dwell = dwell or PointDwell()
        self.mode = "idle"                   # idle | arming | follow
        self._fist_since: Optional[float] = None
        self._last_seq: Optional[int] = None
        self._last_new_now: Optional[float] = None
        self._follow_target: Optional[np.ndarray] = None
        self.last_point: Optional[PointTarget] = None

    def pause(self) -> None:
        """The clutch took over (or the arm is busy): forget everything."""
        self.mode = "idle"
        self._fist_since = None
        self._follow_target = None
        self.follower.reset()
        self.dwell.reset()

    def update(self, frame, now: float, dt: float, ee_m, base_mm) -> GestureOutput:
        st = {"gesture": getattr(frame, "gesture", "none"), "mode": self.mode,
              "cam": self.cam.source}
        out = GestureOutput(status=st)
        if frame is None:
            self.pause()
            st["mode"] = self.mode
            return out
        new = frame.seq != self._last_seq
        if new:
            self._last_seq = frame.seq
            self._last_new_now = now
        elif self._last_new_now is None or now - self._last_new_now > config.DROPOUT_S:
            if self.mode != "idle":
                self.pause()
            st["mode"] = self.mode
            st["stale"] = True
            return out

        g = frame.gesture if frame.valid or frame.gesture == "point" else "none"

        # ---- fist ----------------------------------------------------------
        if g == "fist" and frame.valid:
            self.dwell.reset()
            if self._fist_since is None:
                self._fist_since = now
            if self.mode == "idle":
                self.mode = "arming"
            if self.mode == "arming" and now - self._fist_since >= config.FOLLOW_ENGAGE_S:
                self.mode = "follow"
                self.follower.reset()
            if self.mode == "follow":
                if new:
                    tgt = self.follower.update(self.cam.point(frame.pos()), ee_m, dt)
                    if tgt is not None:
                        self._follow_target = tgt
                if self._follow_target is not None:
                    # Re-derive from the CURRENT EE every tick so the standoff
                    # holds while the arm moves between camera frames.
                    self._follow_target = self.follower.target(ee_m)
                    out.kind, out.follow_target_m = "follow", self._follow_target
                    st.update(self.follower.status)
                    st["fist_world_mm"] = np.round(self.follower.fist_m * 1000, 0).tolist()
            st["mode"] = self.mode
            return out

        # ---- anything else ends a fist ---------------------------------------
        if self.mode != "idle":
            self.mode = "idle"
            self._fist_since = None
            self._follow_target = None
            self.follower.reset()

        # ---- point -----------------------------------------------------------
        if g == "point":
            ray = frame.point_ray
            if ray is None:
                st["point"] = "no depth on the knuckle"
            elif new:
                o_w = self.cam.point(ray[0])
                d_w = self.cam.direction(ray[1])
                fired = self.dwell.update(o_w, d_w, now)
                st["point_progress"] = round(self.dwell.progress, 2)
                if fired is not None:
                    tgt = resolve_pointing(fired[0], fired[1], self.scene, base_mm)
                    if tgt is None:
                        st["point"] = "nothing safe along the ray"
                    else:
                        self.last_point = tgt
                        out.kind, out.point = "point", tgt
            else:
                st["point_progress"] = round(self.dwell.progress, 2)
        else:
            self.dwell.reset()
        st["mode"] = self.mode
        return out
