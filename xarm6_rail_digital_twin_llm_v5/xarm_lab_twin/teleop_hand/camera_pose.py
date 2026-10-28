# teleop_hand/camera_pose.py
"""Where the hand-tracking camera is, in the robot WORLD frame.

The clutch-driven mapper is relative, so it only needs a rotation (the
`facing` / `behind` preset). The two gestures are ABSOLUTE: "the arm goes to
where my fist is" and "the arm goes to the thing my finger points at" both
need the camera's full pose, translation included.

World, not base: the arm base rides the rail, but the camera is fixed to the
lab, and IK / `set_position` work in the world frame.

`HandCameraPose.nominal()` is a PLACEHOLDER for the twin so the gestures can
be exercised before the real calibration exists. It is labelled `nominal` and
the runner prints a warning when it is in use. `from_correspondences` is the
real thing: hold the palm against the gripper at 6+ spread-out arm poses; the
palm position in the camera frame and the EE position from forward kinematics
are the point pairs, and `perception.extrinsic_calibration.solve_camera_pose`
(Kabsch, reflection-guarded, refuses coplanar sets) does the rest. This
module adds no second solver.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np

from teleop_hand import config


@dataclasses.dataclass
class HandCameraPose:
    cam_to_world: np.ndarray            # 4x4, metres; optical frame -> world
    source: str = "nominal"             # nominal | calibrated | file
    rms_mm: Optional[float] = None

    def __post_init__(self):
        self.cam_to_world = np.asarray(self.cam_to_world, float).reshape(4, 4)

    @property
    def is_nominal(self) -> bool:
        return self.source == "nominal"

    def point(self, p_cam) -> np.ndarray:
        return self.cam_to_world[:3, :3] @ np.asarray(p_cam, float) + self.cam_to_world[:3, 3]

    def direction(self, d_cam) -> np.ndarray:
        v = self.cam_to_world[:3, :3] @ np.asarray(d_cam, float)
        return v / np.linalg.norm(v)

    def world_to_cam_point(self, p_world) -> np.ndarray:
        r, t = self.cam_to_world[:3, :3], self.cam_to_world[:3, 3]
        return r.T @ (np.asarray(p_world, float) - t)

    def world_to_cam_dir(self, d_world) -> np.ndarray:
        return self.cam_to_world[:3, :3].T @ np.asarray(d_world, float)

    # ---- construction -------------------------------------------------------
    @classmethod
    def nominal(cls, preset: str = config.PRESET,
                position_mm=(350.0, 550.0, 1050.0)) -> "HandCameraPose":
        """Camera at `position_mm`, orientation from the same preset the mapper
        uses. For the `facing` preset the lens looks along world -y, so a palm
        0.65 m out lands near y = -100 mm, around the twin's home pose."""
        t = np.eye(4)
        t[:3, :3] = config.PRESETS[preset]
        t[:3, 3] = np.asarray(position_mm, float) / 1000.0
        return cls(t, "nominal")

    @classmethod
    def looking_at(cls, position_mm, target_mm, up=(0.0, 0.0, 1.0)) -> "HandCameraPose":
        """Camera at `position_mm` with its lens on `target_mm` (optical frame:
        z forward, x right, y down). For tests and rehearsals; the real pose
        comes from `from_correspondences`."""
        pos = np.asarray(position_mm, float)
        z = np.asarray(target_mm, float) - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        t = np.eye(4)
        t[:3, :3] = np.column_stack([x, y, z])
        t[:3, 3] = pos / 1000.0
        return cls(t, "nominal")

    @classmethod
    def from_correspondences(cls, palm_cam_m, ee_world_m) -> "HandCameraPose":
        from perception.extrinsic_calibration import solve_camera_pose
        res = solve_camera_pose(np.asarray(palm_cam_m), np.asarray(ee_world_m))
        if not res.ok:
            raise ValueError("camera calibration refused:\n" + res.report())
        return cls(res.cam_to_base, "calibrated", res.rms_mm)

    # ---- persistence --------------------------------------------------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "cam_to_world": self.cam_to_world.tolist(), "rms_mm": self.rms_mm,
            "source": "calibrated" if self.source == "calibrated" else "file"}, indent=1))

    @classmethod
    def load(cls, path) -> "HandCameraPose":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["cam_to_world"]), "file", d.get("rms_mm"))
