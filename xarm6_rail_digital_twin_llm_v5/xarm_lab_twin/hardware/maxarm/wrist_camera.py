"""The MaxArm's wrist D435, with every frame posed in the arm frame.

    from hardware.maxarm.real_maxarm import RealMaxArmAPI
    from hardware.maxarm.wrist_camera import MaxArmWristCamera
    arm = RealMaxArmAPI("/dev/ttyUSB0")
    cam = MaxArmWristCamera(arm)
    frame = cam.capture()
    xyz_mm = frame.pixel_to_world(u, v) * 1000      # arm frame: set_position(*xyz_mm)

`RealSenseWristCamera.cam_to_world` composes the xArm's flange pose with the
cell's hand-eye constant; neither exists here. This subclass replaces only that
pose with `hand_eye.cam_to_arm` (the measured mount in hand_eye_calib.json, the
wrist position read back from the servos). Streaming, alignment, depth scaling
and the xArm path are the parent's, unchanged.

Frames come back in the arm frame, METRES, left-handed: points and clouds are
right, the 3x3 block is a reflection (see hand_eye.cam_to_arm).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from perception.realsense_camera import RealSenseWristCamera

from . import hand_eye as he

# A dim room needs about this many frames before auto-exposure settles; the
# parent's 3-frame warm-up left the first captures black (2026-11-08).
WARMUP_FRAMES = 60


class MaxArmWristCamera(RealSenseWristCamera):
    """RealSenseWristCamera whose frames carry the MaxArm hand-eye pose.

    `arm` is required (RealMaxArmAPI, or anything whose get_position() returns
    the wrist in arm-frame mm). `serial` defaults to the camera the calibration
    was measured with, so a swapped camera fails to open instead of borrowing a
    mount it never had; pass None to take whatever is plugged in.
    """

    def __init__(self, arm, calib: Optional[he.HandEye] = None, serial: Optional[str] = "",
                 warmup_frames: int = WARMUP_FRAMES, **kwargs):
        if arm is None:
            raise ValueError("MaxArmWristCamera needs the arm: its pose is the frame's pose")
        self.hand_eye = calib if calib is not None else he.load()
        if serial == "":
            serial = self.hand_eye.meta.get("camera_serial")
        kwargs.setdefault("verify_intrinsics", False)   # d435i_calib holds another body's values
        super().__init__(arm=arm, serial=serial, **kwargs)
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

    def cam_to_world(self) -> Optional[np.ndarray]:
        """Camera optical frame -> MaxArm arm frame (metres), from the read-back wrist.

        None (and a message) when the read fails. The nozzle angle plays no
        part: the camera does not turn with the nozzle (measured 2026-11-09, the
        tag held within 0.1 px through nozzle 0 / +45 / -45 / 0).
        """
        code, pose = self.arm.get_position()
        if code != 0:
            print(f"[MaxArmWristCamera] get_position() returned {code}; "
                  f"frame will have no pose rather than a stale one")
            return None
        return he.cam_to_arm(pose[:3], self.hand_eye.cam_to_wrist)


def pixel_at_height(frame, u: float, v: float, z_mm: float) -> Optional[np.ndarray]:
    """Where pixel (u, v)'s ray meets the plane z = z_mm, arm frame mm; None if it never does.

    For surfaces the depth sensor cannot range: from the highest view the arm
    reaches (wrist z 225) a 40 mm cube's top is ~135 mm from the camera, inside
    this D435's minimum range at 640x480, and only 8% of it returned depth
    (2026-11-09). Its height is known (table contact + size), so the colour
    pixel and that height fix the point. z is in command coordinates, like
    everything from MaxArmWristCamera.
    """
    if frame.cam_to_world is None:
        raise ValueError("frame has no cam_to_world pose")
    k = frame.intrinsics
    t = np.asarray(frame.cam_to_world, dtype=np.float64)
    origin = t[:3, 3]
    ray = t[:3, :3] @ np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
    if abs(ray[2]) < 1e-9:
        return None
    s = (z_mm / 1000.0 - origin[2]) / ray[2]
    if s <= 0:
        return None
    return (origin + s * ray) * 1000.0
