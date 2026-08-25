"""The RGB-D frame type and the geometry that turns pixels into robot coordinates.

Both camera backends -- ``perception.sim_camera`` (MuJoCo) and
``perception.realsense_camera`` (the physical D435i) -- return this same
``RGBDFrame``, and every consumer downstream works on it without knowing which
one produced it. That symmetry is the point: a perception routine developed
against the twin should run unchanged against the real cell.

Units, stated once and enforced everywhere below: **positions are metres,
depth is metres, and pixel coordinates are OpenCV-style** (origin top-left,
``+x`` right, ``+y`` down). Invalid or out-of-range depth is ``NaN``, never 0 --
0 is a legitimate reading only for a point at the sensor's optical centre, and
letting it stand in for "no data" is how a missing return quietly becomes a
grasp target at the camera itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .d435i_calib import Intrinsics


@dataclass
class RGBDFrame:
    """One synchronised colour + depth capture.

    Attributes
    ----------
    color:
        ``(H, W, 3)`` uint8 RGB.
    depth:
        ``(H, W)`` float32 metres. ``NaN`` where no valid return.
    intrinsics:
        The intrinsics that ``depth`` is expressed in. When frames are captured
        aligned (the default), depth has been reprojected into the colour frame,
        so this is the *colour* intrinsics -- carrying it on the frame means a
        caller can never pick the wrong one.
    cam_to_world:
        ``(4, 4)`` homogeneous transform from the camera's **optical** frame to
        the world/base frame, or ``None`` if the pose was not known at capture
        time (a bare real camera with no arm attached). Consumers that need
        world coordinates must check this rather than assume identity.
    t_wall:
        Capture timestamp, seconds. Sim frames carry sim time.
    source:
        ``"sim"`` or ``"real"``. Recorded so a dataset built from mixed sources
        stays auditable after the fact.
    """

    color: np.ndarray
    depth: np.ndarray
    intrinsics: Intrinsics
    cam_to_world: Optional[np.ndarray] = None
    t_wall: float = 0.0
    source: str = "sim"

    def __post_init__(self) -> None:
        if self.color.shape[:2] != self.depth.shape[:2]:
            raise ValueError(
                f"colour {self.color.shape[:2]} and depth {self.depth.shape[:2]} "
                "differ in size; they must be aligned before becoming a frame"
            )

    # -- geometry ---------------------------------------------------------

    def deproject_pixel(self, u: float, v: float,
                        depth_m: Optional[float] = None) -> Optional[np.ndarray]:
        """Pixel ``(u, v)`` -> 3D point in the **camera optical** frame, metres.

        ``depth_m`` defaults to the frame's own depth at that pixel. Returns
        ``None`` when there is no valid depth there -- callers must handle that
        rather than receive a plausible-looking wrong point.
        """
        if depth_m is None:
            iv, iu = int(round(v)), int(round(u))
            if not (0 <= iv < self.depth.shape[0] and 0 <= iu < self.depth.shape[1]):
                return None
            depth_m = float(self.depth[iv, iu])
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            return None
        k = self.intrinsics
        return np.array([
            (u - k.cx) * depth_m / k.fx,
            (v - k.cy) * depth_m / k.fy,
            depth_m,
        ], dtype=np.float64)

    def project_point(self, xyz_cam: np.ndarray) -> Optional[tuple[float, float]]:
        """3D point in the camera optical frame -> pixel ``(u, v)``.

        The inverse of :meth:`deproject_pixel`. ``None`` for points at or behind
        the optical centre, which do not project.
        """
        x, y, z = (float(c) for c in xyz_cam[:3])
        if z <= 1e-9:
            return None
        k = self.intrinsics
        return (k.fx * x / z + k.cx, k.fy * y / z + k.cy)

    def pixel_to_world(self, u: float, v: float,
                       depth_m: Optional[float] = None) -> Optional[np.ndarray]:
        """Pixel -> 3D point in the world/base frame, metres.

        This is the function that actually matters: it is what turns "the blue
        cube is at pixel (312, 208)" into a coordinate the arm can be commanded
        to. Returns ``None`` if depth is invalid there, and raises if the frame
        has no pose -- a silent identity transform would hand the caller camera
        coordinates dressed up as world ones.
        """
        if self.cam_to_world is None:
            raise ValueError(
                "frame has no cam_to_world pose; capture it from a camera bound "
                "to the arm, or pass the pose in explicitly"
            )
        p_cam = self.deproject_pixel(u, v, depth_m)
        if p_cam is None:
            return None
        return (self.cam_to_world @ np.append(p_cam, 1.0))[:3]

    def point_cloud(self, stride: int = 1,
                    in_world: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Dense cloud of the valid pixels.

        Returns ``(points, colors)`` -- ``(N, 3)`` float64 metres and ``(N, 3)``
        uint8 RGB. ``stride`` subsamples both axes. With ``in_world=True`` the
        points come back in the world frame, which needs ``cam_to_world``.
        """
        d = self.depth[::stride, ::stride]
        c = self.color[::stride, ::stride]
        h, w = d.shape
        k = self.intrinsics
        # Pixel centres of the subsampled grid, in full-resolution coordinates.
        us = (np.arange(w) * stride).astype(np.float64)
        vs = (np.arange(h) * stride).astype(np.float64)
        uu, vv = np.meshgrid(us, vs)

        valid = np.isfinite(d) & (d > 0.0)
        z = d[valid].astype(np.float64)
        x = (uu[valid] - k.cx) * z / k.fx
        y = (vv[valid] - k.cy) * z / k.fy
        pts = np.stack([x, y, z], axis=1)

        if in_world:
            if self.cam_to_world is None:
                raise ValueError("point_cloud(in_world=True) needs cam_to_world")
            pts = (self.cam_to_world[:3, :3] @ pts.T).T + self.cam_to_world[:3, 3]

        return pts, c[valid]

    # -- convenience ------------------------------------------------------

    def depth_stats(self) -> dict[str, float]:
        """Coverage and range, for sanity-checking a capture at a glance."""
        finite = self.depth[np.isfinite(self.depth)]
        total = float(self.depth.size)
        if finite.size == 0:
            return {"valid_frac": 0.0, "min_m": float("nan"),
                    "max_m": float("nan"), "median_m": float("nan")}
        return {
            "valid_frac": float(finite.size) / total,
            "min_m": float(finite.min()),
            "max_m": float(finite.max()),
            "median_m": float(np.median(finite)),
        }


def pose_matrix(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(R, t) -> 4x4 homogeneous transform."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m
