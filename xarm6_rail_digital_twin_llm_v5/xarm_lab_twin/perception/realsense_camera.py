"""The physical D435i, behind the same API as the simulated one.

``RealSenseWristCamera`` is the real-hardware twin of
``perception.sim_camera.SimWristCamera``: same ``capture()``, same
:class:`~perception.rgbd.RGBDFrame` out. Swapping one for the other is the only
change a perception routine needs when it moves from the twin to the cell.

Adapted from ``ufactory_vision/ggcnn_grasping_demo/camera/rs_camera.py``
(UFACTORY, BSD-3). What is added here beyond that original:

* frames carry their intrinsics and, when an arm is attached, their pose, so a
  caller never has to pair a frame with the right matrix by hand;
* invalid depth is ``NaN`` rather than 0 (see ``rgbd``);
* the intrinsics the device reports at connect time are checked against
  ``d435i_calib``, so a swapped camera body or a changed stream profile is
  caught at startup instead of showing up later as a systematic reprojection
  bias that looks like a calibration drift.

Requires ``pyrealsense2``.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from . import d435i_calib as calib
from .rgbd import RGBDFrame, pose_matrix

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - optional until real hardware is used
    rs = None

# How far the live device's intrinsics may drift from the recorded ones before
# we complain. Intel's factory calibration is stable to well under a pixel
# across power cycles, so a whole pixel already means something changed.
_INTRINSICS_TOL_PX = 1.0


class RealSenseWristCamera:
    """Streams colour + aligned depth from the wrist-mounted D435i.

    Parameters
    ----------
    arm:
        Optional. Any object exposing ``get_position()`` in the xArm SDK's
        convention (x, y, z in mm plus roll/pitch/yaw in degrees) -- either
        ``RealXArmAPI`` or the sim. When given, each frame carries a
        ``cam_to_world`` built from the live flange pose and the hand-eye
        calibration, which is what makes ``pixel_to_world`` usable. Without it
        frames come back pose-less, and asking for world coordinates raises
        rather than silently returning camera coordinates.
    serial:
        Bind to a specific device. Defaults to the serial recorded in
        ``d435i_calib``; pass ``None`` to take whatever is plugged in.
    verify_intrinsics:
        Compare the device's reported intrinsics against the recorded ones at
        connect time and warn on a mismatch.
    """

    def __init__(self, arm=None, serial: Optional[str] = calib.DEVICE_SERIAL,
                 width: int = calib.WIDTH, height: int = calib.HEIGHT,
                 fps: int = calib.FPS, verify_intrinsics: bool = True):
        if rs is None:
            raise ImportError(
                "pyrealsense2 is not installed in this environment. "
                "`pip install pyrealsense2` (it is already present in the "
                "ufactory_vision and ot2vision envs)."
            )
        self.arm = arm
        self.width, self.height, self.fps = width, height, fps

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # The device's own depth scale, rather than assuming the 1 mm default.
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        self._color_intr, self._depth_intr = self._read_intrinsics()
        if verify_intrinsics:
            self._verify_intrinsics()

    # -- intrinsics -------------------------------------------------------

    def _read_intrinsics(self) -> tuple[calib.Intrinsics, calib.Intrinsics]:
        frames = self.pipeline.wait_for_frames()
        c = frames.get_color_frame().profile.as_video_stream_profile().intrinsics
        d = frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
        return (
            calib.Intrinsics(c.width, c.height, c.fx, c.fy, c.ppx, c.ppy),
            calib.Intrinsics(d.width, d.height, d.fx, d.fy, d.ppx, d.ppy),
        )

    def _verify_intrinsics(self) -> None:
        """Warn if the live device disagrees with the recorded calibration.

        Worth doing at startup: every downstream coordinate depends on these,
        and a mismatch is otherwise invisible until grasps start missing by a
        consistent few millimetres.
        """
        for label, live, recorded in (
            ("colour", self._color_intr, calib.COLOR_INTRINSICS),
            ("depth", self._depth_intr, calib.DEPTH_INTRINSICS),
        ):
            if (live.width, live.height) != (recorded.width, recorded.height):
                print(f"[RealSenseWristCamera] {label} stream is "
                      f"{live.width}x{live.height} but d435i_calib records "
                      f"{recorded.width}x{recorded.height}. Intrinsics are "
                      f"per-resolution -- the sim camera no longer matches.")
                continue
            deltas = [abs(getattr(live, f) - getattr(recorded, f))
                      for f in ("fx", "fy", "cx", "cy")]
            if max(deltas) > _INTRINSICS_TOL_PX:
                print(f"[RealSenseWristCamera] {label} intrinsics differ from "
                      f"d435i_calib by up to {max(deltas):.2f} px "
                      f"(live fx={live.fx:.3f} fy={live.fy:.3f} "
                      f"cx={live.cx:.3f} cy={live.cy:.3f}). If this is a "
                      f"different camera body, update d435i_calib and rerun "
                      f"`python envs/build_mesh_scene.py`.")

    @property
    def color_intrinsics(self) -> calib.Intrinsics:
        """Live intrinsics, not the recorded ones -- what this device reports."""
        return self._color_intr

    @property
    def depth_intrinsics(self) -> calib.Intrinsics:
        return self._depth_intr

    # -- pose -------------------------------------------------------------

    def cam_to_world(self) -> Optional[np.ndarray]:
        """Camera optical frame -> robot base frame, from the live flange pose.

        ``None`` when no arm was supplied. Composed as
        ``base->flange`` (live, from the arm) then ``flange->colour optical``
        (fixed, from the hand-eye calibration).
        """
        if self.arm is None:
            return None
        code, pose = self.arm.get_position()
        if code != 0:
            print(f"[RealSenseWristCamera] get_position() returned {code}; "
                  f"frame will have no pose rather than a stale one")
            return None
        x_mm, y_mm, z_mm, roll_d, pitch_d, yaw_d = pose[:6]
        r_base_flange = calib._euler_to_mat(
            math.radians(roll_d), math.radians(pitch_d), math.radians(yaw_d))
        t_base_flange = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0

        r_fc, t_fc = calib.flange_to_color_optical()
        return pose_matrix(r_base_flange @ r_fc,
                           t_base_flange + r_base_flange @ t_fc)

    # -- capture ----------------------------------------------------------

    def capture(self, align: bool = True) -> RGBDFrame:
        """Grab one synchronised colour + depth frame.

        ``align=True`` reprojects depth into the colour frame via
        ``rs.align``, which is why the returned frame then carries the *colour*
        intrinsics -- deprojecting aligned depth with the depth imager's matrix
        is a ~15 mm lateral error that looks entirely reasonable.
        """
        frames = self.pipeline.wait_for_frames()
        if align:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # bgr8 off the wire; RGBDFrame is RGB, matching MuJoCo's output.
        color = np.asanyarray(color_frame.get_data())[:, :, ::-1].copy()

        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth *= self.depth_scale
        # 0 is librealsense's "no return" sentinel, not a reading at the origin.
        depth[depth == 0.0] = np.nan
        depth[(depth < calib.DEPTH_MIN_M) | (depth > calib.DEPTH_MAX_M)] = np.nan

        return RGBDFrame(
            color=color,
            depth=depth,
            intrinsics=self._color_intr if align else self._depth_intr,
            cam_to_world=self.cam_to_world(),
            t_wall=time.time(),
            source="real",
        )

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    # Kept for drop-in compatibility with ufactory_vision's RealSenseCamera.
    stop = close

    def __enter__(self) -> "RealSenseWristCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
