# teleop_hand/real.py
"""Real xArm 6 side of hand teleop (§6): stream targets to the controller,
veto them with the twin's IK first, latch faults, and a fake controller.

CONTROL PATH. The mapper works in the TWIN's world frame, anchored on the
mirrored twin's EE (the twin's ~90 mm tool). The controller positions the
REAL TCP (217 mm tool, `set_tcp_offset`). The two differ by a constant tool
transform, which is MEASURED at every clutch engage rather than copied from a
constant:

    T_off = inv(T_twin_ee) @ T_real_tcp      (both from the same joint angles)
    real target = T_twin_target @ T_off

so a twin-frame target becomes the pose the real TCP must take for the real
arm to stand in the same joint configuration. Calibration error between twin
and cell is absorbed too, since both poses are read live.

GUARD BEFORE SEND. Each target is solved with the twin's IK seeded from the
MIRRORED real joints and must pass the §4.7 guards (branch flip, joint
limits, wrist band) before the controller sees it. The controller does its
own IK; this keeps it away from targets that would flip it.

HOLDS. Clutch released, dropout, tracker silence, a guard refusal or a fault:
nothing is sent. Never extrapolated. In mode 7 the controller then simply
finishes the last (already-vetted) target.

FAULTS LATCH. A controller error or warning stops streaming and stays
stopped until the operator presses the recovery key: clean_error,
motion_enable, set_mode(7), set_state(0) -- and then a FRESH clutch press.
Nothing recovers on its own.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np
from transforms3d.euler import euler2mat

from teleop_hand import config, guards
from teleop_kin.real_teach import (FakeBackend, FakeController, controller_mode,
                                   return_to_position_mode)
from vr import transforms


def pose_to_T(pos_m, rot) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(rot, float)
    T[:3, 3] = np.asarray(pos_m, float)
    return T


def rpy_pose_to_T(pose_mm_deg) -> np.ndarray:
    """[x, y, z, roll, pitch, yaw] world mm/deg (RealXArmAPI.get_position,
    sxyz like the twin's set_position) -> 4x4 in metres."""
    x, y, z, r, p, yw = (float(v) for v in pose_mm_deg[:6])
    return pose_to_T(np.array([x, y, z]) / 1000.0,
                     euler2mat(*np.deg2rad([r, p, yw]), axes="sxyz"))


def T_to_rpy_pose(T) -> tuple:
    roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(T[:3, :3])
    p = T[:3, 3] * 1000.0
    return float(p[0]), float(p[1]), float(p[2]), roll, pitch, yaw


def read_err_warn(ctl) -> tuple:
    """(error_code, warn_code) from the raw SDK handle's report properties."""
    return int(getattr(ctl, "error_code", 0) or 0), int(getattr(ctl, "warn_code", 0) or 0)


class RealHandStreamer:
    def __init__(self, arm, backend, *, limits: np.ndarray, check_wrist: bool,
                 hz: float = config.REAL_STREAM_HZ, speed_mm_s: float = config.MAX_POS_SPEED,
                 use_rail: bool = True, log: Callable[[str], None] = print,
                 on_vetted: Optional[Callable[[np.ndarray], None]] = None):
        self.arm = arm                     # the twin (mirror)
        self.backend = backend
        self.ctl = backend.arm             # raw SDK handle (or FakeController)
        self.limits = limits
        self.check_wrist = check_wrist
        self.period = 1.0 / hz
        self.rail_period = 1.0 / config.REAL_RAIL_MAX_HZ
        self.speed = float(speed_mm_s)
        self.use_rail = use_rail
        self.log = log
        self.on_vetted = on_vetted         # --fake: the fake arm "moves" there
        self.T_off: Optional[np.ndarray] = None
        self.faulted: Optional[tuple] = None
        self.blocked: Optional[str] = None
        self.streaming = False
        self.n_sent = 0
        self.n_vetoed = 0
        self.sent: list = []               # (t, pose) -- tests and latency logs
        self._t_send = -1e9
        self._t_rail = -1e9
        self._was_engaged = False
        self._qadr = [int(arm.model.jnt_qposadr[j]) for j in arm.ik_solver.joint_ids]

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> bool:
        """Safety settings, then the streaming mode. Returns False (and
        leaves mode 0) if the controller did not take it."""
        for what, fn in (("collision sensitivity",
                          lambda: self.backend.set_collision_sensitivity(
                              config.REAL_COLLISION_SENSITIVITY)),):
            try:
                rc = fn()
                if rc not in (0, None):
                    self.log(f"[HANDR] {what} returned {rc}")
            except Exception as e:                 # noqa: BLE001
                self.log(f"[HANDR] could not set {what}: {e}")
        self.ctl.motion_enable(True)
        self.ctl.set_mode(config.REAL_STREAM_MODE)
        self.ctl.set_state(0)
        mode = controller_mode(self.ctl)
        if mode is not None and mode != config.REAL_STREAM_MODE:
            self.log(f"[HANDR] controller reports mode {mode}, not "
                     f"{config.REAL_STREAM_MODE}: firmware without online Cartesian "
                     f"planning? Refusing to stream.")
            self.stop()
            return False
        self.streaming = True
        return True

    def stop(self) -> None:
        """Stop streaming and return to mode 0 / state 0. Never raises."""
        self.streaming = False
        return_to_position_mode(self.ctl, self.log, "[HANDR]")

    # ---- per tick -----------------------------------------------------------
    def poll_fault(self) -> Optional[tuple]:
        if self.faulted:
            return self.faulted
        err, warn = read_err_warn(self.ctl)
        if err or warn or getattr(self.ctl, "has_err_warn", False):
            self.faulted = (err, warn)
            self.log(f"[HANDR] CONTROLLER FAULT error={err} warn={warn} -- streaming "
                     f"stopped. Press the recovery key, then a fresh clutch press.")
        return self.faulted

    def recover(self) -> bool:
        """The explicit operator action (§6.6). Never called automatically."""
        try:
            self.backend.clean_error()
            self.ctl.motion_enable(True)
            self.ctl.set_mode(config.REAL_STREAM_MODE)
            self.ctl.set_state(0)
        except Exception as e:                     # noqa: BLE001
            self.log(f"[HANDR] recovery failed: {e}")
            return False
        err, warn = read_err_warn(self.ctl)
        if err or warn:
            self.log(f"[HANDR] still faulted after recovery (error={err} warn={warn})")
            return False
        self.faulted = None
        self.log("[HANDR] recovered -- release and press the clutch to continue")
        return True

    def mirrored_q(self) -> np.ndarray:
        with self.arm.lock:
            return np.array([float(self.arm.data.qpos[a]) for a in self._qadr])

    def twin_ee(self) -> tuple:
        with self.arm.lock:
            mujoco.mj_kinematics(self.arm.model, self.arm.data)
            return (self.arm.data.site_xpos[self.arm.ee_site].copy(),
                    self.arm.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy())

    def measure_tool_offset(self) -> np.ndarray:
        code, pose = self.backend.get_position()
        if code != 0 or pose is None:
            raise RuntimeError(f"get_position failed (code {code})")
        T_real = rpy_pose_to_T(pose)
        T_twin = pose_to_T(*self.twin_ee())
        self.T_off = np.linalg.inv(T_twin) @ T_real
        return self.T_off

    def step(self, out, now: float) -> None:
        """One control tick with the mapper's output (mirror already synced)."""
        engaged = bool(out.engaged)
        if engaged and not self._was_engaged:
            self.measure_tool_offset()
            d = self.T_off[:3, 3] * 1000.0
            self.log(f"[HANDR] engaged; twin->real tool offset ({d[0]:.0f}, {d[1]:.0f}, "
                     f"{d[2]:.0f}) mm")
        self._was_engaged = engaged
        if not engaged or not self.streaming or self.faulted:
            self.blocked = None
            return

        if self.use_rail and out.rail_mm is not None and now - self._t_rail >= self.rail_period:
            # The arm target below is in WORLD coordinates and RealXArmAPI
            # converts it with the MEASURED rail, so it never relies on this
            # rail command having completed.
            self.backend.set_rail_position(out.rail_mm, speed_mm_s=config.RAIL_MAX_SPEED,
                                           wait=False)
            self._t_rail = now

        if now - self._t_send < self.period:
            return
        ref = self.mirrored_q()
        q, why = guards.solve_guarded(self.arm, out.target_pos, out.target_rot, ref,
                                      self.limits, check_wrist=self.check_wrist)
        if q is None:
            self.blocked = why
            self.n_vetoed += 1
            return
        self.blocked = None
        T = pose_to_T(out.target_pos, out.target_rot) @ self.T_off
        pose = T_to_rpy_pose(T)
        self.backend.set_position(*pose, speed=self.speed, wait=False)
        self._t_send = now
        self.n_sent += 1
        self.sent.append((now, pose))
        if self.on_vetted is not None:
            self.on_vetted(q)


# ---------------------------------------------------------------------------
# fakes (--fake rehearsal and tests)
# ---------------------------------------------------------------------------
class HandFakeController(FakeController):
    """FakeController plus what streaming needs: error/warn report fields,
    motion_enable, and a log of calls."""

    def __init__(self, *a, **kw):
        self.error_code = 0
        self.warn_code = 0
        super().__init__(*a, **kw)

    def _dump(self) -> None:
        if self.state_file is not None:
            self.state_file.write_text(json.dumps(
                {"mode": self.mode, "state": self.state, "error": self.error_code}))

    @property
    def has_err_warn(self) -> bool:
        return bool(self.error_code or self.warn_code)

    def motion_enable(self, enable=True):
        self.calls.append(("motion_enable", bool(enable)))
        return 0

    def clean_error(self):
        self.calls.append(("clean_error",))
        self.error_code = self.warn_code = 0
        self._dump()
        return 0


class HandFakeBackend(FakeBackend):
    """FakeBackend plus Cartesian targets, rail commands and the gripper's
    continuous position. `get_position` is FK of the fake joints on the twin
    with a fixed fake tool (so T_off is a real, measurable transform)."""

    FAKE_TOOL_M = 0.127          # real 217 mm tool minus the twin's ~90 mm

    def __init__(self, ctl=None, twin=None, **kw):
        super().__init__(ctl or HandFakeController(), **kw)
        self.twin = twin
        self.targets: list = []
        self.rail_targets: list = []
        self.gripper_positions: list = []
        self.collision_sensitivity = None
        self._scratch = mujoco.MjData(twin.model) if twin is not None else None

    def set_collision_sensitivity(self, v):
        self.collision_sensitivity = int(v)
        return 0

    def clean_error(self):
        return self.arm.clean_error()

    def get_err_warn_code(self):
        return 0, [self.arm.error_code, self.arm.warn_code]

    def set_position(self, x, y, z, roll=0, pitch=0, yaw=0, speed=100, wait=True, **kw):
        if self.arm.mode != config.REAL_STREAM_MODE:
            return 1                  # the controller ignores it outside mode 7
        self.targets.append((x, y, z, roll, pitch, yaw))
        return 0

    def set_rail_position(self, position_mm, speed_mm_s=50.0, wait=True, **kw):
        self.rail_targets.append(float(position_mm))
        self.rail_mm = float(position_mm)
        return 0

    def set_gripper_opening(self, fraction, wait=False):
        self.gripper_positions.append(float(fraction))
        return 0

    def get_position(self):
        m, d = self.twin.model, self._scratch
        with self.twin.lock:
            d.qpos[:] = self.twin.data.qpos
        for j, q in zip(self.twin.ik_solver.joint_ids, np.deg2rad(self.joints_deg)):
            d.qpos[m.jnt_qposadr[j]] = q
        if self.rail_mm is not None:
            d.qpos[m.jnt_qposadr[self.twin.rail_jid]] = self.rail_mm / 1000.0
        mujoco.mj_kinematics(m, d)
        T = pose_to_T(d.site_xpos[self.twin.ee_site], d.site_xmat[self.twin.ee_site].reshape(3, 3))
        T = T @ pose_to_T([0, 0, self.FAKE_TOOL_M], np.eye(3))
        return 0, list(T_to_rpy_pose(T))


def write_state(path: Optional[str], **kv) -> None:
    if path:
        Path(path).write_text(json.dumps(kv))
