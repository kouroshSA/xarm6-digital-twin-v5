# teleop_hand/palm.py
"""Palm position and orientation in the CAMERA frame, from landmarks + depth.

Pure numpy: no MediaPipe, no camera. The tracker feeds it; tests feed it
synthetic points.

FRAME CONVENTION. Both palm frames -- the depth plane fit here and
MediaPipe's own (`detector.estimate_frame_from_hand_points`) -- use upstream
dex-retargeting's convention, columns [x, n, z]: x points from the middle MCP
toward the wrist (9 -> 0), n is the palm normal, z = x cross n, with n's sign
chosen so z points to the index side. They MUST agree, or fusing them (a
slerp between the two) would average two different axes.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from teleop_hand import config


def palm_frame(points: Dict[int, np.ndarray],
               extra: Optional[np.ndarray] = None) -> Optional[Tuple[np.ndarray, float]]:
    """Plane-fit palm frame from landmark points (any frame, metres).

    `points` maps landmark index -> 3-vector; needs 0, 5 and 9, uses any of
    the five palm landmarks present. `extra` (k, 3) are more points on the
    palm surface (depth samples inside the palm polygon, `palm_samples_px`)
    that join the plane fit only; the in-plane axis comes from the landmarks.
    Returns (R, noise_m), or None.

    Why extra points: five points leave two residual degrees of freedom, and
    a noise estimate from two is too wide to threshold -- at 20 mm of noise
    the quality spread 0.0-0.9 (10th-90th percentile). ~25 points pin it.

    noise_m is the UNBIASED off-plane noise estimate, rms * sqrt(n / (n-3)):
    a plane fit absorbs three of the n residual degrees of freedom, so the raw
    RMS of five points reads only ~63 % of the true noise (20 mm of noise
    looked like 12 mm and still scored quality ~0.5). Three points always fit
    a plane exactly, so with n < 4 the quality is unknown: noise_m = inf.
    """
    if not all(k in points for k in (0, 5, 9)):
        return None
    idx = [k for k in config.PALM_LANDMARKS if k in points]
    pts = np.array([points[k] for k in idx], float)
    if extra is not None and len(extra):
        pts = np.vstack([pts, np.asarray(extra, float).reshape(-1, 3)])
    centred = pts - pts.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred)
    normal = vt[2]
    n = len(pts)
    rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    noise = rms * float(np.sqrt(n / (n - 3))) if n > 3 else float("inf")

    x = np.asarray(points[0], float) - np.asarray(points[9], float)
    x = x - np.dot(x, normal) * normal
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x /= nx
    z = np.cross(x, normal)
    if np.dot(z, np.asarray(points[5], float) - np.asarray(points[9], float)) < 0:
        normal, z = -normal, -z
    return np.stack([x, normal, z], axis=1), noise


def rot_quality(noise_m: float) -> float:
    """Off-plane noise estimate -> 0..1 (<= GOOD mm is 1, >= BAD mm is 0)."""
    mm = noise_m * 1000.0
    g, b = config.ROT_QUALITY_GOOD_MM, config.ROT_QUALITY_BAD_MM
    return float(np.clip((b - mm) / (b - g), 0.0, 1.0))


def fuse(rot_mediapipe_cam: Optional[np.ndarray], rot_depth: Optional[np.ndarray],
         quality: float) -> Tuple[Optional[np.ndarray], str]:
    """§2.4 fusion: slerp MediaPipe -> depth by `quality` when quality is good
    enough, else MediaPipe alone. Returns (R, rot_source)."""
    if rot_depth is not None and quality >= config.FUSE_MIN_QUALITY:
        if rot_mediapipe_cam is None:
            return rot_depth, "depth"
        both = Rotation.from_matrix(np.stack([rot_mediapipe_cam, rot_depth]))
        return Slerp([0.0, 1.0], both)([quality])[0].as_matrix(), "fused"
    return rot_mediapipe_cam, "mediapipe"


# ---------------------------------------------------------------------------
# depth lookup
# ---------------------------------------------------------------------------
def depth_at(depth_m: np.ndarray, u: float, v: float, win: int = config.DEPTH_WINDOW_PX) -> float:
    """Median of the non-zero depths in a win x win window; 0.0 if none."""
    h, w = depth_m.shape[:2]
    r = win // 2
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(ui - r, 0), min(ui + r + 1, w)
    v0, v1 = max(vi - r, 0), min(vi + r + 1, h)
    if u0 >= u1 or v0 >= v1:
        return 0.0
    patch = depth_m[v0:v1, u0:u1].ravel()
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else 0.0


def deproject(u: float, v: float, z: float, k: np.ndarray) -> np.ndarray:
    """Pinhole deprojection with intrinsics matrix K (3x3)."""
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


#: Fractions along wrist->MCP and MCP->MCP segments sampled for the plane fit.
SAMPLE_FRACTIONS = (0.25, 0.5, 0.75)


def palm_samples_px(kp_px: np.ndarray) -> np.ndarray:
    """Pixels inside the palm polygon: along wrist->each MCP and between
    neighbouring MCPs. 4*3 + 3*3 = 21 samples."""
    kp_px = np.asarray(kp_px, float)
    out = []
    mcps = (5, 9, 13, 17)
    for m in mcps:
        for f in SAMPLE_FRACTIONS:
            out.append(kp_px[0, :2] + f * (kp_px[m, :2] - kp_px[0, :2]))
    for a, b in zip(mcps[:-1], mcps[1:]):
        for f in SAMPLE_FRACTIONS:
            out.append(kp_px[a, :2] + f * (kp_px[b, :2] - kp_px[a, :2]))
    return np.array(out)


def palm_from_depth(kp_px: np.ndarray, depth_m: np.ndarray, k: np.ndarray):
    """§2.3 + §2.4.1. `kp_px` is (21, 2) pixel landmarks in the depth image
    (aligned to colour). Returns (pos or None, rot or None, quality, points).

    Position: mean of the deprojected palm LANDMARKS (spec). Orientation:
    plane through the landmarks plus the interior samples."""
    pts: Dict[int, np.ndarray] = {}
    for i in config.PALM_LANDMARKS:
        z = depth_at(depth_m, kp_px[i, 0], kp_px[i, 1])
        if z > 0:
            pts[i] = deproject(kp_px[i, 0], kp_px[i, 1], z, k)
    if len(pts) < config.MIN_VALID_PALM_POINTS:
        return None, None, 0.0, pts
    pos = np.mean(list(pts.values()), axis=0)
    extra = []
    for u, v in palm_samples_px(kp_px):
        z = depth_at(depth_m, u, v, win=3)
        if z > 0:
            extra.append(deproject(u, v, z, k))
    fr = palm_frame(pts, np.array(extra) if extra else None)
    if fr is None:
        return pos, None, 0.0, pts
    rot, noise = fr
    return pos, rot, rot_quality(noise), pts


# ---------------------------------------------------------------------------
# left-hand clutch gesture (§2.5)
# ---------------------------------------------------------------------------
FINGER_TIPS = (8, 12, 16, 20)
FINGER_MCPS = (5, 9, 13, 17)


def is_fist(kp: np.ndarray) -> bool:
    """All four fingertips closer to the wrist than their MCPs. `kp` is
    (21, 3) in any metric frame (MediaPipe world landmarks)."""
    kp = np.asarray(kp, float)
    w = kp[0]
    return all(np.linalg.norm(kp[t] - w) < np.linalg.norm(kp[m] - w)
               for t, m in zip(FINGER_TIPS, FINGER_MCPS))


class Debounce:
    """A boolean that changes only after N consecutive frames agree."""

    def __init__(self, n: int = config.FIST_DEBOUNCE_FRAMES):
        self.n = n
        self.state = False
        self._run = 0

    def update(self, raw: bool) -> bool:
        if raw == self.state:
            self._run = 0
        else:
            self._run += 1
            if self._run >= self.n:
                self.state, self._run = raw, 0
        return self.state
