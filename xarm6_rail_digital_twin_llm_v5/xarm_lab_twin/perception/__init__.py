"""Wrist-camera perception for the xArm6 twin — one API, two backends.

The Intel RealSense D435i is mounted eye-in-hand on the flange. It exists twice:
as MuJoCo cameras in the scene (``SimWristCamera``) and as the physical device
(``RealSenseWristCamera``). Both hand back the same :class:`RGBDFrame`, and
every number describing the camera lives once in :mod:`perception.d435i_calib`.

    from perception import SimWristCamera            # or RealSenseWristCamera

    cam = SimWristCamera(arm)
    frame = cam.capture()
    xyz_world = frame.pixel_to_world(320, 240)       # metres, base frame

``RealSenseWristCamera`` is imported lazily: it needs ``pyrealsense2``, which the
sim-only environment does not require.
"""
from .d435i_calib import COLOR_INTRINSICS, DEPTH_INTRINSICS, Intrinsics
from .rgbd import RGBDFrame, pose_matrix
from .sim_camera import SimWristCamera

__all__ = [
    "COLOR_INTRINSICS",
    "DEPTH_INTRINSICS",
    "Intrinsics",
    "RGBDFrame",
    "RealSenseWristCamera",
    "SimWristCamera",
    "pose_matrix",
]


def __getattr__(name: str):
    """Defer the pyrealsense2 import until someone actually asks for the real one."""
    if name == "RealSenseWristCamera":
        from .realsense_camera import RealSenseWristCamera
        return RealSenseWristCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
