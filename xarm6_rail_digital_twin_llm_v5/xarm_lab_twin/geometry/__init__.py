"""Pose geometry helpers: SE(3) interpolation, smoothing, dwell detection.

The contents are vendored from InternRobotics/Aether (MIT) — see NOTICE.md for
provenance and README.md for what this is for. Nothing here is wired into the
twin yet; it is importable and tested, and that is deliberate.
"""
from .pose_filters import (  # noqa: F401
    adaptive_pose_smoothing,
    detect_static_sequence,
    interpolate_poses,
    slerp,
    smooth_poses,
)

__all__ = [
    "slerp",
    "interpolate_poses",
    "smooth_poses",
    "detect_static_sequence",
    "adaptive_pose_smoothing",
]
