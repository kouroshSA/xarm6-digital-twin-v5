# envs/scene_randomizer.py
"""
Perturb free-body positions and orientations in a MuJoCo scene before
a replay cycle. Generates spatial variations for data augmentation.
"""
import numpy as np
import mujoco
from typing import Optional


DEFAULT_POS_JITTER_MM = 20.0
DEFAULT_ROT_JITTER_DEG = 45.0
DEFAULT_INITIAL_JOINT_JITTER_DEG = 0.0

PERTURBABLE_BODIES = {"red_cube", "green_cube", "blue_cube"}


def randomize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos_jitter_mm: float = DEFAULT_POS_JITTER_MM,
    rot_jitter_deg: float = DEFAULT_ROT_JITTER_DEG,
    initial_joint_jitter_deg: float = DEFAULT_INITIAL_JOINT_JITTER_DEG,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Apply randomization to the scene. Returns a summary dict for logging.

    pos_jitter_mm: max absolute position perturbation (+/-N mm per axis)
    rot_jitter_deg: max absolute yaw rotation around vertical (+/-N deg)
    initial_joint_jitter_deg: per-joint angle perturbation (0 = off)
    rng: numpy RNG (pass np.random.default_rng(seed) for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    summary = {
        "perturbed_bodies": {},
        "joint_jitter_applied": [],
    }

    pos_jitter_m = pos_jitter_mm / 1000.0

    for body_name in PERTURBABLE_BODIES:
        try:
            body_id = model.body(body_name).id
        except KeyError:
            continue

        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr < 0:
            continue
        jnt_type = model.jnt_type[jnt_adr]
        if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_adr = model.jnt_qposadr[jnt_adr]

        dx = rng.uniform(-pos_jitter_m, pos_jitter_m)
        dy = rng.uniform(-pos_jitter_m, pos_jitter_m)
        # Z is not perturbed - keeps objects resting on the bench surface
        dyaw = rng.uniform(-rot_jitter_deg, rot_jitter_deg)

        data.qpos[qpos_adr + 0] += dx
        data.qpos[qpos_adr + 1] += dy

        # Yaw rotation as quaternion multiplication
        yaw_rad = np.deg2rad(dyaw)
        cos_h = np.cos(yaw_rad / 2)
        sin_h = np.sin(yaw_rad / 2)
        qw = data.qpos[qpos_adr + 3]
        qx = data.qpos[qpos_adr + 4]
        qy = data.qpos[qpos_adr + 5]
        qz = data.qpos[qpos_adr + 6]
        # Hamilton product (current) * (delta around Z axis)
        new_qw = qw * cos_h - qz * sin_h
        new_qx = qx * cos_h + qy * sin_h
        new_qy = qy * cos_h - qx * sin_h
        new_qz = qz * cos_h + qw * sin_h
        data.qpos[qpos_adr + 3] = new_qw
        data.qpos[qpos_adr + 4] = new_qx
        data.qpos[qpos_adr + 5] = new_qy
        data.qpos[qpos_adr + 6] = new_qz

        summary["perturbed_bodies"][body_name] = {
            "dx_mm": float(dx * 1000.0),
            "dy_mm": float(dy * 1000.0),
            "dyaw_deg": float(dyaw),
        }

    if initial_joint_jitter_deg > 0:
        joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6"]
        jitter_rad = np.deg2rad(initial_joint_jitter_deg)
        for name in joint_names:
            jid = model.joint(name).id
            qadr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            delta = rng.uniform(-jitter_rad, jitter_rad)
            new_val = np.clip(data.qpos[qadr] + delta, lo, hi)
            data.qpos[qadr] = new_val
            summary["joint_jitter_applied"].append({
                "joint": name,
                "delta_deg": float(np.rad2deg(delta)),
            })

    mujoco.mj_forward(model, data)
    return summary
