# teleop_kin/handle.py
"""The teach handle: a mocap body the operator drags in the MuJoCo viewer.

WHERE THE DRAG TARGET COMES FROM (Phase 0 finding, MuJoCo 3.8.1). The passive
viewer's `Simulate::Sync` ends with

    mjv_applyPerturbPose(m, d, &pert, 0);   // mocap bodies only

on every sync, so while the operator Ctrl-drags a selected MOCAP body the
viewer writes `perturb.refpos/refquat` into `data.mocap_pos/quat` itself. The
mocap pose therefore IS the drag target ("source 1" in the spec), and nothing
here has to read `viewer.perturb.refpos`. `drag_target()` is still the single
entry point, so if a future MuJoCo stops doing this, only that function changes.

All `data`/`model` access is under `arm.lock`.
"""
from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from teleop_kin import config

BODY_NAME = "teach_handle"
BALL_GEOM = "teach_handle_ball"


class TeachHandle:
    def __init__(self, arm, body_name: str = BODY_NAME):
        self.arm = arm
        model = arm.model
        try:
            self.body_id = int(model.body(body_name).id)
        except KeyError as exc:
            raise RuntimeError(
                f"scene has no '{body_name}' body -- regenerate envs/lab_scene.xml "
                f"with envs/build_mesh_scene.py") from exc
        self.mocap_id = int(model.body_mocapid[self.body_id])
        assert self.mocap_id >= 0, f"'{body_name}' is not a mocap body"
        self.ball_gid = int(model.geom(BALL_GEOM).id)
        #: Headless tests (no viewer, so no perturb state) set this to say
        #: whether the operator is "holding" the handle. None = ask the viewer.
        self.dragging_override: Optional[bool] = None

    # ---- pose -------------------------------------------------------------
    def read_pose(self) -> tuple:
        """(pos_m (3,), rot (3x3)) of the handle, world frame."""
        with self.arm.lock:
            pos = np.array(self.arm.data.mocap_pos[self.mocap_id], dtype=float)
            quat = np.array(self.arm.data.mocap_quat[self.mocap_id], dtype=float)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat / np.linalg.norm(quat))
        return pos, mat.reshape(3, 3)

    def write_pose(self, pos_m, rot) -> None:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
        with self.arm.lock:
            self.arm.data.mocap_pos[self.mocap_id] = np.asarray(pos_m, float)
            self.arm.data.mocap_quat[self.mocap_id] = quat
            # Refresh xpos now: the viewer seeds a drag from the body's xpos
            # (mjv_initPerturb), so a stale one would start the drag from the
            # handle's OLD position and the first mouse move would jump.
            mujoco.mj_kinematics(self.arm.model, self.arm.data)

    def ee_pose(self) -> tuple:
        with self.arm.lock:
            mujoco.mj_forward(self.arm.model, self.arm.data)
            pos = self.arm.data.site_xpos[self.arm.ee_site].copy()
            rot = self.arm.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy()
        return pos, rot

    def snap_to_ee(self) -> tuple:
        """Put the handle exactly on the EE site. Returns that pose."""
        pos, rot = self.ee_pose()
        self.write_pose(pos, rot)
        return pos, rot

    def drag_target(self) -> tuple:
        """The pose the operator is asking for. See the module docstring."""
        return self.read_pose()

    def is_dragging(self) -> bool:
        """Is the operator holding the handle right now?"""
        if self.dragging_override is not None:
            return bool(self.dragging_override)
        v = getattr(self.arm, "_viewer", None)
        if v is None:
            return False
        try:
            pert = v.perturb
            return int(pert.select) == self.body_id and int(pert.active) != 0
        except Exception:
            return False

    # ---- appearance -------------------------------------------------------
    def set_color(self, rgba) -> None:
        with self.arm.lock:
            self.arm.model.geom_rgba[self.ball_gid] = np.asarray(rgba, float)

    def set_blocked(self, blocked: bool) -> None:
        self.set_color(config.RGBA_BLOCKED if blocked else config.RGBA_TRACKING)

    def color(self) -> np.ndarray:
        with self.arm.lock:
            return np.array(self.arm.model.geom_rgba[self.ball_gid], dtype=float)
