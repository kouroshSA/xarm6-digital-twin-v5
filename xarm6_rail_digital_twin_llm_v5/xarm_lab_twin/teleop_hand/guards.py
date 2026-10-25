# teleop_hand/guards.py
"""IK with the §4.7 guards. Shared by the twin (whose ctrl it feeds) and the
real arm (where it only VETOES targets before the controller sees them).

`solve_guarded` never writes ctrl: the twin receiver writes the accepted
solution through TwinServo, the real runner sends the Cartesian target.

Joint limits come from the MODEL (`jnt_range`), not a copy of them.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np

from teleop_hand import config

#: Reasons, as shown on the operator view and in the log.
BRANCH_FLIP = "branch_flip"
JOINT_LIMIT = "joint_limit"
WRIST_SINGULAR = "wrist_singular"
IK_FAILED = "ik_failed"
POSE_MISSED = "pose_missed"

WRIST_JOINT = 4        # index of joint 5 among joints 1..6
#: Joint-limit margin: per-solve wobble tolerated, and where the hard stop is
#: (this far INSIDE the margin, i.e. margin - 1 = 2 deg from the limit).
IK_JITTER_DEG = 0.25
HARD_STOP_DEG = 1.0


def joint_limits_rad(arm) -> np.ndarray:
    m = arm.model
    lim = np.array([m.jnt_range[j] for j in arm.ik_solver.joint_ids], float)
    limited = np.array([bool(m.jnt_limited[j]) for j in arm.ik_solver.joint_ids])
    lim[~limited] = (-np.inf, np.inf)
    return lim


def in_wrist_band(q: np.ndarray) -> bool:
    return abs(float(np.rad2deg(q[WRIST_JOINT]))) < config.WRIST_SINGULAR_DEG


def check_solution(q: np.ndarray, ref_q: np.ndarray, limits: np.ndarray,
                   check_wrist: bool,
                   max_step_deg: float = config.BRANCH_FLIP_DEG,
                   margin_deg: float = config.JOINT_LIMIT_MARGIN_DEG) -> Optional[str]:
    """Reason to reject an IK solution, or None."""
    q, ref_q = np.asarray(q, float), np.asarray(ref_q, float)
    if float(np.max(np.abs(np.rad2deg(q - ref_q)))) > max_step_deg:
        return BRANCH_FLIP
    # Inside the margin is refused only when it is no further from the limit
    # than the reference: moving OUT must always be allowed. Measured on the
    # D435 axis test: refusing every in-margin solution froze an arm parked
    # at the margin edge, because IK results jitter by a fraction of a
    # degree, so even the return move read as "inside" and was refused.
    # IK_JITTER_DEG of "deeper" is tolerated so solver noise alone cannot
    # freeze it; a hard stop HARD_STOP_DEG from the limit bounds any creep
    # that tolerance could otherwise accumulate solve after solve.
    m = np.deg2rad(margin_deg)
    jit = np.deg2rad(IK_JITTER_DEG)
    hard = np.deg2rad(margin_deg - HARD_STOP_DEG)
    lo_in = q < limits[:, 0] + m
    hi_in = q > limits[:, 1] - m
    deeper_lo = lo_in & ((q < ref_q - jit) | (q < limits[:, 0] + hard))
    deeper_hi = hi_in & ((q > ref_q + jit) | (q > limits[:, 1] - hard))
    if np.any(deeper_lo | deeper_hi):
        return JOINT_LIMIT
    # Entering the band, or going deeper into it, is refused; moving OUT is
    # always allowed -- otherwise an arm already in the band could never leave.
    if (check_wrist and in_wrist_band(q)
            and abs(q[WRIST_JOINT]) < abs(ref_q[WRIST_JOINT]) - 1e-9):
        return WRIST_SINGULAR
    return None


_SCRATCH: dict = {}


def pose_error(arm, q, pos_m, rot) -> Tuple[float, float]:
    """(mm, deg) between the EE pose joint angles q ACTUALLY produce (FK on a
    scratch copy of the live state) and the requested pose."""
    d = _SCRATCH.get(id(arm.model))
    if d is None:
        d = _SCRATCH[id(arm.model)] = mujoco.MjData(arm.model)
    with arm.lock:
        d.qpos[:] = arm.data.qpos
    for j, qq in zip(arm.ik_solver.joint_ids, q):
        d.qpos[arm.model.jnt_qposadr[j]] = qq
    mujoco.mj_kinematics(arm.model, d)
    e_mm = float(np.linalg.norm(d.site_xpos[arm.ee_site] - np.asarray(pos_m, float))) * 1000.0
    r = d.site_xmat[arm.ee_site].reshape(3, 3)
    cos = (np.trace(r.T @ np.asarray(rot, float)) - 1.0) / 2.0
    return e_mm, float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def solve_guarded(arm, pos_m, rot, ref_q, limits: np.ndarray,
                  check_wrist: bool = False) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """IK seeded from live qpos; (q, None) if accepted, else (None, reason).
    `ref_q` is what a branch flip is measured against: the twin's current
    setpoint, or the mirrored real joints.

    THE SOLUTION IS CHECKED, NOT TRUSTED. `IKSolver.solve` treats orientation
    as best-effort: when it cannot meet the requested rotation it returns a
    position-only (or rotation-missed) answer and still reports success.
    Measured on the D435 full-orientation run: the gripper ended 9-18 deg away
    from the commanded orientation with no refusal anywhere. So FK of the
    returned joints is compared with the request here, and a miss beyond
    POSE_TOL_* is refused like any other guard."""
    with arm.lock:
        seed = np.array([float(arm.data.qpos[arm.model.jnt_qposadr[j]])
                         for j in arm.ik_solver.joint_ids])
        q = arm.ik_solver.solve(np.asarray(pos_m, float), target_rot=rot, seed_q=seed)
    if q is None:
        return None, IK_FAILED
    q = np.asarray(q, float)
    why = check_solution(q, ref_q, limits, check_wrist)
    if why:
        return None, why
    e_mm, e_deg = pose_error(arm, q, pos_m, rot)
    if e_mm > config.POSE_TOL_MM or e_deg > config.POSE_TOL_DEG:
        return None, POSE_MISSED
    return q, None
