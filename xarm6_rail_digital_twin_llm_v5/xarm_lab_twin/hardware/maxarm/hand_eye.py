"""Where the MaxArm's wrist camera sits on the arm (eye-in-hand calibration).

The geometry
------------
The MaxArm's tool platform stays level (parallel linkage), so the wrist frame
is the wrist point plus ONE rotation: the base heading, about the vertical. A
camera bolted to the platform therefore sees the world through

    cam_to_arm = wrist_frame(wrist_xyz) @ cam_to_wrist

where `cam_to_wrist` is the constant this module solves for.

The arm frame is LEFT-HANDED (see kinematics.py: +x is the arm's right with the
reach along -y). Rotation matrices assume a right-handed frame, so everything
here works in the MIRRORED frame (x negated, "rh" below) and converts at the
edges. Built in the arm frame directly, the same correspondences fit no proper
rotation and the solver's reflection guard leaves a large residual
(`test_hand_eye.test_left_handed_frame_is_converted` pins that).

The procedure
-------------
A tag lies flat on the table at a KNOWN point: centred under the suction cup,
with its z the measured table contact (`kinematics.TABLE_CONTACT_Z_MM`). z is
in COMMAND coordinates -- the wrist z at which the cup lip reaches that point --
so no tool length is needed, and pixel_to_arm answers in the coordinates
set_position takes. Then for each arm pose the tag is seen from a different
place, and its position in the WRIST frame is known:

    w_i = wrist_frame(wrist_i)^-1 @ tag

while the camera measures it as q_i. N pairs (q_i, w_i) between two rigid
frames: the same problem `perception.extrinsic_calibration.solve_camera_pose`
already solves for a fixed camera, with the wrist playing the base. No second
solver (defect class #1).

Why a known tag point and not AX=XB: with every rotation about the same
(vertical) axis, AX=XB cannot see the camera's height; the table contact fixes it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from perception.extrinsic_calibration import (CalibrationResult, MIN_OBSERVATIONS,
                                              MIN_SPREAD_RATIO, solve_camera_pose)

CALIB_PATH = Path(__file__).with_name("hand_eye_calib.json")


def to_rh(xyz_arm) -> np.ndarray:
    """Arm frame (left-handed, mm) -> mirrored right-handed frame (metres)."""
    x, y, z = (float(v) for v in xyz_arm)
    return np.array([-x, y, z]) / 1000.0


def from_rh(p_rh) -> np.ndarray:
    """Mirrored right-handed frame (metres) -> arm frame (mm)."""
    x, y, z = (float(v) for v in p_rh)
    return np.array([-x, y, z]) * 1000.0


def wrist_frame(wrist_xyz_arm) -> np.ndarray:
    """4x4 pose of the level tool platform in the rh frame, metres.

    Rotation: the base heading about z (the platform turns with joint 1 and
    nothing else). Any constant offset of the camera from that heading is part
    of cam_to_wrist, so the zero of the heading does not matter.
    """
    t = to_rh(wrist_xyz_arm)
    h = math.atan2(t[1], t[0])
    c, s = math.cos(h), math.sin(h)
    f = np.eye(4)
    f[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    f[:3, 3] = t
    return f


def tag_in_wrist(tag_xyz_arm, wrist_xyz_arm) -> np.ndarray:
    """The tag's position in the wrist frame (metres)."""
    f = wrist_frame(wrist_xyz_arm)
    return f[:3, :3].T @ (to_rh(tag_xyz_arm) - f[:3, 3])


def solve(tag_xyz_arm, wrist_xyzs_arm, tag_cam_m) -> CalibrationResult:
    """cam_to_wrist from N views of one tag at a known arm-frame point.

    `tag_cam_m[i]` is the tag centre in the colour OPTICAL frame (metres) seen
    with the wrist at `wrist_xyzs_arm[i]` (mm, arm frame, read back). The result's
    `cam_to_base` IS cam_to_wrist; `spread_ratio` is that of the wrist-frame
    points, and below MIN_SPREAD_RATIO the rotation is poorly determined.
    """
    w = np.array([tag_in_wrist(tag_xyz_arm, p) for p in wrist_xyzs_arm])
    q = np.asarray(tag_cam_m, dtype=np.float64).reshape(-1, 3)
    if len(q) < MIN_OBSERVATIONS:
        raise ValueError(f"{len(q)} views; need at least {MIN_OBSERVATIONS}")
    return solve_camera_pose(q, w)


def cam_to_arm_rh(wrist_xyz_arm, cam_to_wrist) -> np.ndarray:
    """Camera optical frame -> rh arm frame (metres), for RGBDFrame.cam_to_world."""
    return wrist_frame(wrist_xyz_arm) @ np.asarray(cam_to_wrist, dtype=np.float64)


MIRROR = np.diag([-1.0, 1.0, 1.0, 1.0])


def cam_to_arm(wrist_xyz_arm, cam_to_wrist) -> np.ndarray:
    """Camera optical frame -> ARM frame (left-handed), metres: RGBDFrame.cam_to_world.

    With this on a frame, `pixel_to_world(u, v) * 1000` is a point set_position
    takes as it is (z in command coordinates), like the xArm cell's metres.
    The 3x3 block is a REFLECTION (det -1), not a rotation: right for points and
    clouds, wrong for anything that reads an orientation off it (a grasp angle
    comes out with its sign flipped). Use cam_to_arm_rh for orientations.
    """
    return MIRROR @ cam_to_arm_rh(wrist_xyz_arm, cam_to_wrist)


def camera_point_to_arm(p_cam_m, wrist_xyz_arm, cam_to_wrist) -> np.ndarray:
    """A point the camera measured (optical frame, metres) -> arm frame, mm.

    The answer is in COMMAND coordinates: z is the wrist z at which the cup lip
    reaches the point, so it can be passed to set_position as it is.
    """
    p = cam_to_arm_rh(wrist_xyz_arm, cam_to_wrist) @ np.append(np.asarray(p_cam_m, float), 1.0)
    return from_rh(p[:3])


@dataclass
class HandEye:
    cam_to_wrist: np.ndarray
    meta: dict


def save(result: CalibrationResult, meta: dict, path: Path = CALIB_PATH) -> None:
    out = dict(meta)
    out.update(cam_to_wrist=np.asarray(result.cam_to_base).round(6).tolist(),
               rms_mm=round(result.rms_mm, 2), max_mm=round(result.max_mm, 2),
               n=result.n, spread_ratio=round(result.spread_ratio, 3),
               min_spread_ratio=MIN_SPREAD_RATIO,
               frame="cam_to_wrist maps the colour OPTICAL frame to the wrist frame "
                     "of hand_eye.wrist_frame, in the right-handed MIRROR of the arm "
                     "frame (x negated), metres; z in command coordinates")
    path.write_text(json.dumps(out, indent=2) + "\n")


def load(path: Path = CALIB_PATH) -> HandEye:
    d = json.loads(Path(path).read_text())
    return HandEye(np.array(d["cam_to_wrist"], dtype=np.float64), d)
