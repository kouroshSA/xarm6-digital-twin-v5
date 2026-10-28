# teleop_hand/hand_calib.py
"""Calibrating the hand-tracking camera's pose in the robot WORLD frame.

The gestures are absolute, so they need `HandCameraPose`. This module holds the
parts of getting it that can be tested without hardware; `scripts/
calibrate_hand_camera.py` is the thin runner that adds the real D435 and arm.

Method: an ArUco tag is fixed to the gripper. For each of N arm poses the tag's
position is known twice -- in the CAMERA frame (PnP on the tag's four corners,
whose size is known) and in the WORLD frame (forward kinematics of the pose the
real arm reports, plus the tag's fixed offset in the tool frame). N paired
points between two rigid frames is the camera pose, and
`HandCameraPose.from_correspondences` (-> `perception.extrinsic_calibration`)
solves it. There is no second solver here.

Why a tag and not the operator's palm: a palm centre is a statistical guess
from 21 landmarks, good to a centimetre or two and biased by hand shape; a
printed square of known size is good to a few millimetres. The calibration's
error is the floor on how well "go where my fist is" and "grasp what I point
at" can ever work.

PnP, not the depth image, for the camera-side point: depth at a tag centre
lands on the tag's edges or the background half the time. The depth reading is
kept only as a CROSS-CHECK (`depth_check_mm`), which catches a wrong tag size
(the usual mistake) because PnP scales with it and depth does not.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

from teleop_hand.camera_pose import HandCameraPose

DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_TAG_ID = 0


@dataclasses.dataclass
class TagObservation:
    pos_cam_m: np.ndarray               # tag centre, optical frame, metres
    reproj_px: float                    # RMS corner reprojection error
    depth_check_mm: Optional[float]     # PnP z minus measured depth, or None
    corners_px: np.ndarray


def _aruco(dict_name: str):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    prm = cv2.aruco.DetectorParameters()
    # Whole-pixel corners are the dominant error for a small tag at ~1 m: a
    # half-pixel miss on a 30 px tag is ~2% of range, i.e. centimetres in z.
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2, cv2.aruco.ArucoDetector(d, prm)


def detect_tag(rgb: np.ndarray, k: np.ndarray, size_m: float,
               depth_m: Optional[np.ndarray] = None, tag_id: int = DEFAULT_TAG_ID,
               dict_name: str = DEFAULT_DICT) -> Optional[TagObservation]:
    """Tag centre in the camera frame, or None if the tag `tag_id` is not seen."""
    cv2, det = _aruco(dict_name)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    hit = [c for c, i in zip(corners, ids.ravel()) if int(i) == tag_id]
    if not hit:
        return None
    img = hit[0].reshape(4, 2).astype(np.float64)
    h = size_m / 2.0
    # ArUco corner order: TL, TR, BR, BL, as seen on the tag, tag plane z = 0.
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
    pos = tvec.ravel()
    dcheck = None
    if depth_m is not None:
        u, v = np.round(img.mean(axis=0)).astype(int)
        patch = depth_m[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
        patch = patch[patch > 0.05]
        if patch.size >= 5:
            dcheck = float((pos[2] - np.median(patch)) * 1000.0)
    return TagObservation(pos, reproj, dcheck, img)


def tag_world_m(ee_pos_m, ee_rot, tag_offset_mm) -> np.ndarray:
    """Tag centre in the world frame: the EE point plus the tag's fixed offset
    expressed in the EE (tool) frame."""
    return np.asarray(ee_pos_m, float) + np.asarray(ee_rot, float) @ (
        np.asarray(tag_offset_mm, float) / 1000.0)


# ---------------------------------------------------------------------------
# collecting
# ---------------------------------------------------------------------------
class SampleSet:
    """Paired (camera, world) tag positions, with checks applied as they arrive
    so the operator learns a pose was useless while the arm is still there."""

    MIN_MOVE_MM = 60.0          # a new pose must be this far from every kept one

    def __init__(self):
        self.cam: list = []
        self.world: list = []

    def __len__(self):
        return len(self.cam)

    def try_add(self, cam_m, world_m) -> tuple:
        """(accepted, reason)."""
        w = np.asarray(world_m, float)
        for prev in self.world:
            if np.linalg.norm(w - prev) * 1000.0 < self.MIN_MOVE_MM:
                return False, (f"within {self.MIN_MOVE_MM:.0f} mm of a pose already "
                               "kept -- move the arm further")
        self.cam.append(np.asarray(cam_m, float))
        self.world.append(w)
        return True, "ok"

    def spread(self) -> float:
        from perception.extrinsic_calibration import spread_ratio
        return spread_ratio(np.array(self.world)) if len(self) >= 3 else 0.0

    def undo(self) -> None:
        if self.cam:
            self.cam.pop()
            self.world.pop()


def solve(samples: SampleSet, trim: int = 2) -> tuple:
    """-> (HandCameraPose, report text). Drops up to `trim` outliers (one bad
    PnP is the usual fault; a bad calibration shows as a high RMS instead)."""
    from perception.extrinsic_calibration import drop_worst, solve_camera_pose
    cam, world = np.array(samples.cam), np.array(samples.world)
    res = solve_camera_pose(cam, world)
    lines = [res.report()]
    for i in range(trim):
        if not res.ok or res.max_mm < 3.0 * max(res.rms_mm, 1e-9) or len(cam) <= 6:
            break
        cam, world, dropped = drop_worst(cam, world, res)
        res = solve_camera_pose(cam, world)
        lines.append(f"\ndropped observation {dropped} (outlier {i + 1}/{trim}):\n" + res.report())
    if not res.ok:
        raise ValueError("camera calibration refused:\n" + "\n".join(lines))
    return HandCameraPose(res.cam_to_base, "calibrated", res.rms_mm), "\n".join(lines)


def describe_pose(pose: HandCameraPose) -> str:
    """Human-checkable summary: where the camera is and which way it looks."""
    p = pose.cam_to_world[:3, 3] * 1000.0
    look = pose.direction([0, 0, 1])
    return (f"camera at x={p[0]:.0f} y={p[1]:.0f} z={p[2]:.0f} mm; lens looks along "
            f"({look[0]:+.2f}, {look[1]:+.2f}, {look[2]:+.2f}) world")


# ---------------------------------------------------------------------------
# synthetic camera (tests and `--fake`): what a D435 would see of the tag
# ---------------------------------------------------------------------------
def render_tag(pose: HandCameraPose, tag_world_m_, tag_normal_world, k: np.ndarray,
               size_m: float, shape=(480, 640), tag_id: int = DEFAULT_TAG_ID,
               dict_name: str = DEFAULT_DICT):
    """Draw the tag as `pose`'s camera would see it. -> (rgb, depth_m).

    The tag's plane faces `tag_normal_world`; its in-plane axes are an arbitrary
    but fixed completion. Warped by the exact pinhole homography, so the
    detector sees a perspective-correct marker, not a pasted square."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    px = 200
    marker = cv2.aruco.generateImageMarker(d, tag_id, px, borderBits=1)
    marker = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    n = np.asarray(tag_normal_world, float)
    n /= np.linalg.norm(n)
    a = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    h = size_m / 2.0 * (px + 40) / px
    c = np.asarray(tag_world_m_, float)
    corners_w = [c - a * h + b * h, c + a * h + b * h, c + a * h - b * h, c - a * h - b * h]
    cam_pts = np.array([pose.world_to_cam_point(p) for p in corners_w])
    uv = (k @ cam_pts.T).T
    uv = (uv[:, :2] / uv[:, 2:3])
    # Supersample: warping 240 px of marker down to ~40 px with bilinear
    # sampling aliases and moves the corners by ~half a pixel, which would
    # make the synthetic camera lie about the detector. Pixel centres sit at
    # +0.5 in a supersampled grid, hence the shift.
    ss = 4
    uv = ((uv + 0.5) * ss).astype(np.float32)
    s = px + 40
    src = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
    hm = cv2.getPerspectiveTransform(src, uv)
    h_px, w_px = shape
    big = (w_px * ss, h_px * ss)
    warped = cv2.warpPerspective(marker, hm, big, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    ink = cv2.warpPerspective(np.full_like(marker, 255), hm, big)
    warped = np.where(ink > 127, warped, 128).astype(np.uint8)
    gray = cv2.resize(warped, (w_px, h_px), interpolation=cv2.INTER_AREA)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mask = cv2.resize(ink, (w_px, h_px), interpolation=cv2.INTER_AREA) > 200
    depth = np.zeros((h_px, w_px), np.float32)
    depth[mask] = float(pose.world_to_cam_point(c)[2])
    return rgb, depth
