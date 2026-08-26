"""Solve where a fixed camera is, in robot base coordinates.

In the twin an observer's pose is an *input* -- ``observer_calib`` declares it and
MuJoCo obeys. On a bench it is an *unknown*: you bolt a D435 to the hood frame
and nobody knows where it is to the millimetre and tenth of a degree that a
survey needs. A 1 degree error at 1 m is 17 mm of position error, which is twice
the accuracy the overhead camera is worth having.

The procedure
-------------
Put a marker on the gripper -- an ArUco tag on a flat plate is the usual choice.
Then for each of N arm poses:

  * the marker's position in the ROBOT frame is known exactly, from forward
    kinematics plus wherever the tag sits on the flange;
  * the marker's position in the CAMERA frame is what the observer measures.

That is N point correspondences between two rigid frames, and the transform
between them is the camera's pose. :func:`solve_camera_pose` does it in closed
form (Kabsch/Umeyama) -- no initial guess, no iteration, no local minimum.

Marker on the gripper rather than on the bench, deliberately: a static marker
gives the same correspondence N times over, which is one constraint however long
you collect for. Moving the arm is what makes the observations independent.

What this module does NOT do
----------------------------
It does not find the marker in an image. On hardware that is an ArUco detector's
job and it needs the real camera; here the correspondences come in already
paired. So what the twin can validate is the solver and the *procedure* --
whether your poses are diverse enough, how the answer degrades with detector
noise, whether a degenerate set is caught rather than silently solved. It cannot
validate the detection, and it should not be read as having done so.

Which is still the useful half. On the bench you will be debugging a mounting, a
detector and a solver at once, with no ground truth to separate them; this takes
the solver out of that list before you start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Below this, the marker positions are too close to a line or a plane for the
# rotation to be well determined. It is the ratio of the smallest to the largest
# singular value of the centred point cloud -- i.e. how three-dimensional the
# poses actually were.
#
# The number to know on a real bench: jogging the arm around a tray at one
# height feels like a thorough sweep and is a PLANE. It solves, it reports a
# small residual, and the rotation about the plane's normal is barely
# constrained. Refusing is better than a confident answer that drifts.
MIN_SPREAD_RATIO = 0.05

# Fewer than this and the fit has no redundancy: three points always solve
# exactly, so the residual is zero and tells you nothing about whether the
# answer is right.
MIN_OBSERVATIONS = 6


@dataclass
class CalibrationResult:
    """A solved camera pose, and everything needed to judge whether to trust it."""

    cam_to_base: np.ndarray
    """4x4 taking a point in the camera's optical frame to the robot base frame."""

    rms_mm: float
    """Root-mean-square residual over the observations."""

    max_mm: float
    """Worst single residual. A large max with a small RMS is one bad detection,
    not a bad calibration -- find it and drop it rather than accepting the fit."""

    n: int
    spread_ratio: float
    """How three-dimensional the marker positions were. See MIN_SPREAD_RATIO."""

    residuals_mm: np.ndarray

    @property
    def ok(self) -> bool:
        return (self.n >= MIN_OBSERVATIONS
                and self.spread_ratio >= MIN_SPREAD_RATIO)

    @property
    def position_mm(self) -> np.ndarray:
        return self.cam_to_base[:3, 3] * 1000.0

    def report(self) -> str:
        lines = [
            f"camera at {np.round(self.position_mm, 1).tolist()} mm in the base frame",
            f"  {self.n} observations, RMS {self.rms_mm:.2f} mm, "
            f"worst {self.max_mm:.2f} mm",
            f"  pose spread ratio {self.spread_ratio:.3f} "
            f"(need >= {MIN_SPREAD_RATIO})",
        ]
        if self.n < MIN_OBSERVATIONS:
            lines.append(f"  REJECTED: fewer than {MIN_OBSERVATIONS} observations, "
                         f"so the residual cannot tell you anything")
        if self.spread_ratio < MIN_SPREAD_RATIO:
            lines.append("  REJECTED: the marker positions are nearly coplanar or "
                         "collinear. Re-collect with the arm at several heights "
                         "and reaches, not a sweep at one height.")
        if self.ok and self.max_mm > 3.0 * max(self.rms_mm, 1e-6):
            lines.append(f"  NOTE: worst residual is {self.max_mm / self.rms_mm:.1f}x "
                         f"the RMS -- probably one bad detection. Drop it and re-solve.")
        return "\n".join(lines)


def spread_ratio(points: np.ndarray) -> float:
    """How three-dimensional a set of points is: smallest/largest singular value."""
    centred = points - points.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if sv[0] <= 0:
        return 0.0
    return float(sv[-1] / sv[0])


def solve_camera_pose(points_cam: np.ndarray,
                      points_base: np.ndarray) -> CalibrationResult:
    """Closed-form rigid transform taking camera points to base points.

    ``points_cam[i]`` and ``points_base[i]`` are the same physical marker,
    measured by the camera and known from forward kinematics. Metres in, metres
    out. Kabsch with the reflection guard, so a noisy set can never produce a
    mirrored "rotation" that fits beautifully and is not a rotation.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    points_base = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if points_cam.shape != points_base.shape:
        raise ValueError(f"{len(points_cam)} camera points but "
                         f"{len(points_base)} base points")
    if len(points_cam) < 3:
        raise ValueError("at least 3 correspondences are needed to solve a pose")

    c_cam = points_cam.mean(axis=0)
    c_base = points_base.mean(axis=0)
    h = (points_cam - c_cam).T @ (points_base - c_base)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    # Without this, a noisy or degenerate set can be fitted by a reflection --
    # determinant -1 -- which minimises the residual and is not a pose.
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    trans = c_base - rot @ c_cam

    predicted = (rot @ points_cam.T).T + trans
    residuals = np.linalg.norm(predicted - points_base, axis=1) * 1000.0

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return CalibrationResult(
        cam_to_base=transform,
        rms_mm=float(np.sqrt(np.mean(residuals ** 2))),
        max_mm=float(residuals.max()),
        n=len(points_cam),
        spread_ratio=spread_ratio(points_base),
        residuals_mm=residuals,
    )


def drop_worst(points_cam: np.ndarray, points_base: np.ndarray,
               result: Optional[CalibrationResult] = None
               ) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove the single worst correspondence and return the trimmed sets.

    One misdetected marker pulls the whole fit, and on a real bench there will be
    one. Deliberately drops exactly one per call rather than everything past a
    threshold: trimming until the residual looks good is how a bad calibration
    gets talked into looking like a good one.
    """
    res = result or solve_camera_pose(points_cam, points_base)
    keep = np.ones(len(points_cam), dtype=bool)
    keep[int(np.argmax(res.residuals_mm))] = False
    return np.asarray(points_cam)[keep], np.asarray(points_base)[keep], int(
        np.argmax(res.residuals_mm))


def pose_error(solved: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """(position error mm, rotation error deg) between two 4x4 poses.

    Only used where the truth is known -- which in practice means the simulator.
    That is the whole reason to validate here first.
    """
    d_pos = float(np.linalg.norm(solved[:3, 3] - truth[:3, 3])) * 1000.0
    r = solved[:3, :3].T @ truth[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
    return d_pos, ang
