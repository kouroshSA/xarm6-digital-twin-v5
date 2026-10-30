# teleop_kin/servo.py
"""The twin-side servo core shared by kinesthetic teaching and hand teleop.

Owns the joint setpoint an interface has COMMANDED (`q_cmd`) and the
gravity-sag bias riding on it, so ctrl = q_cmd + bias. Three measured
problems are handled here and nowhere else (teleop_kin/README.md, findings
1, 2 and 4):

  * a load the scene does not compensate (a held object; the arm and tool
    carry gravcomp) sags the twin -> per-joint integral bias, gated on
    the joint being near its setpoint AND at rest (integrating during motion
    winds up on tracking lag);
  * the rail deflects a few mm while the arm moves -> IK and FK use the rail
    SETPOINT (`rail_at_setpoint`, `commanded_pose`);
  * a failed IK solve costs ~215 ms under arm.lock -> callers back off retries
    (that policy stays with the caller; it depends on what the target is).

All `arm.data` access is under `arm.lock`.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import mujoco
import numpy as np

from sim.mujoco_env import RAIL_ACT
from teleop_kin import config


class TwinServo:
    def __init__(self, arm):
        self.arm = arm
        self.gravity_comp = config.GRAVITY_COMP
        self.q_cmd: Optional[np.ndarray] = None   # rad, joints 1..6
        self.q_bias = np.zeros(6)
        self._scratch = mujoco.MjData(arm.model)
        self._jnt_qadr = [int(arm.model.jnt_qposadr[j]) for j in arm.ik_solver.joint_ids]
        self._jnt_dadr = [int(arm.model.jnt_dofadr[j]) for j in arm.ik_solver.joint_ids]
        self._rail_qadr = int(arm.model.jnt_qposadr[arm.rail_jid])

    # ---- reading ------------------------------------------------------------
    def read_q_ctrl(self) -> np.ndarray:
        with self.arm.lock:
            return np.array([float(self.arm.data.ctrl[a]) for a in self.arm.act_ids[1:7]])

    def read_q(self) -> np.ndarray:
        with self.arm.lock:
            return np.array([float(self.arm.data.qpos[a]) for a in self._jnt_qadr])

    def rail_setpoint_mm(self) -> float:
        with self.arm.lock:
            return float(self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]]) * 1000.0

    def has_live_command(self) -> bool:
        """Is ctrl still q_cmd + bias? False once a primitive (home, reset,
        rail step) has written ctrl itself."""
        if self.q_cmd is None:
            return False
        return bool(np.allclose(self.read_q_ctrl(), self.q_cmd + self.q_bias, atol=1e-6))

    def adopt_ctrl(self) -> None:
        """Take whatever ctrl holds as the command and restart the bias."""
        self.q_cmd = self.read_q_ctrl()
        self.q_bias[:] = 0.0

    def adopt_ctrl_if_foreign(self) -> None:
        if self.q_cmd is None or not self.has_live_command():
            self.adopt_ctrl()

    # ---- kinematics on the setpoint -------------------------------------------
    def load_commanded(self, d, q) -> None:
        """Scratch `d` <- live state with joints at q and the rail at its
        SETPOINT. Caller holds the lock. The rail deflects a few mm while the
        arm moves (reaction forces against a kp=2000 actuator) and springs
        back; a pose computed on the deflected rail is off by that much once
        it recovers."""
        d.qpos[:] = self.arm.data.qpos
        for adr, qq in zip(self._jnt_qadr, q):
            d.qpos[adr] = qq
        d.qpos[self._rail_qadr] = self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]]
        mujoco.mj_kinematics(self.arm.model, d)

    def commanded_pose(self, q: Optional[np.ndarray] = None) -> tuple:
        """(pos_m, rot) of the EE at joint setpoint q (default q_cmd, else
        ctrl), WITHOUT the sag bias: where the arm settles once the bias has
        absorbed gravity. Computed on scratch data; live state untouched."""
        if q is None:
            q = self.q_cmd if self.q_cmd is not None else self.read_q_ctrl()
        d = self._scratch
        with self.arm.lock:
            self.load_commanded(d, q)
            pos = d.site_xpos[self.arm.ee_site].copy()
            rot = d.site_xmat[self.arm.ee_site].reshape(3, 3).copy()
        return pos, rot

    @contextlib.contextmanager
    def rail_at_setpoint(self):
        """Solve IK as if the rail were at its setpoint (see load_commanded).
        IKSolver reads the rail from live qpos and only save/restores the six
        arm joints, so the live rail value is swapped in and out here, under
        the lock, around the solve."""
        with self.arm.lock:
            live = float(self.arm.data.qpos[self._rail_qadr])
            self.arm.data.qpos[self._rail_qadr] = float(
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]])
            try:
                yield
            finally:
                self.arm.data.qpos[self._rail_qadr] = live

    # ---- writing --------------------------------------------------------------
    def apply(self, dt: float) -> None:
        """ctrl = q_cmd + bias, integrating the bias on the joint error."""
        if self.q_cmd is None:
            return
        with self.arm.lock:
            q_now = np.array([float(self.arm.data.qpos[a]) for a in self._jnt_qadr])
            v_now = np.array([float(self.arm.data.qvel[a]) for a in self._jnt_dadr])
            if self.gravity_comp:
                err = self.q_cmd - q_now
                live = ((np.abs(err) < np.deg2rad(config.GRAVITY_COMP_WINDOW_DEG))
                        & (np.abs(v_now) < np.deg2rad(config.GRAVITY_COMP_MAX_VEL_DEG_S)))
                lim = np.deg2rad(config.GRAVITY_COMP_MAX_DEG)
                self.q_bias = np.clip(
                    self.q_bias + config.GRAVITY_COMP_KI * err * dt * live, -lim, lim)
            for i, aid in enumerate(self.arm.act_ids[1:7]):
                self.arm.data.ctrl[aid] = float(self.q_cmd[i] + self.q_bias[i])
