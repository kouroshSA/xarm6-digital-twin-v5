# Vendored from InternRobotics/Aether (MIT) — see NOTICE.md
"""SE(3) pose interpolation, smoothing and dwell detection.

Five functions lifted from Aether's ``aether/utils/postprocess_utils.py``.
The maths is unchanged; what differs is packaging — type hints, house-style
docstrings, and module-level SciPy imports instead of per-call ones.

**Why vendor rather than depend.** Upstream's module imports ``torch``,
``einops`` and ``plyfile`` at the top, so importing it to reach five pure-NumPy
functions would pull a deep-learning stack into a simulator that does not need
one. Aether's headline artefact is a CogVideoX-5B video-diffusion pipeline; none
of that is wanted here. Taking the five functions is ~150 lines and no new
dependency (SciPy is already required).

**What was deliberately NOT taken**, and why, so nobody re-imports them later:

* ``align_rigid`` / ``align_camera_extrinsics`` / ``apply_transformation`` /
  ``compute_scale`` — torch implementations of Procrustes alignment. This repo
  already aligns point sets in ``perception/extrinsic_calibration.py`` and
  ``perception/observer_calib.py``. A second alignment path is defect class #1
  (one fact, several copies, and the copies drift).
* ``smooth_trajectory`` — needs ``filterpy`` for a Kalman filter.
  ``smooth_poses`` covers the need without the dependency.
* ``depth_edge`` / ``project`` / ``save_ply`` / ``postprocess_pointmap`` —
  torch plus ``plyfile``. Revisit only if the D435i point-cloud work needs them.

Nothing here is wired into the twin yet; see README.md for intended consumers.

CONVENTIONS. A "pose" is a ``(4, 4)`` homogeneous matrix, and a pose *sequence*
is ``(N, 4, 4)``. Rotations are the upper-left ``3x3``; translation is
``[:3, 3]``. Quaternions follow SciPy's ``(x, y, z, w)`` ordering, which is NOT
the ``(w, x, y, z)`` MuJoCo uses — convert at the boundary if you mix them.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

__all__ = [
    "slerp",
    "interpolate_poses",
    "smooth_poses",
    "detect_static_sequence",
    "adaptive_pose_smoothing",
]

#: Above this |dot| two quaternions are close enough that slerp's trigonometry
#: loses precision (``sin_theta_0`` heads for zero), so we fall back to nlerp.
_DOT_THRESHOLD = 0.9995

SmoothMethod = Literal["gaussian", "savgol", "ma"]


# ---------------------------------------------------------------------------
# rotation interpolation
# ---------------------------------------------------------------------------
def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    ``t=0`` returns ``q1``, ``t=1`` returns ``q2``. Quaternions are
    ``(x, y, z, w)``, SciPy's ordering.

    Takes the shorter arc: ``q`` and ``-q`` are the same rotation, so a
    negative dot product means the raw pair would interpolate the long way
    round, and ``q2`` is negated to prevent it.
    """
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)

    dot = float(np.sum(q1 * q2))

    # If the dot product is negative, slerp would not take the shorter path.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    if dot > _DOT_THRESHOLD:
        # Too close for the trigonometry to be well conditioned: linearly
        # interpolate and renormalise instead.
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q1) + (s1 * q2)


def interpolate_poses(pose1: np.ndarray, pose2: np.ndarray,
                      weight: float) -> np.ndarray:
    """Interpolate between two ``(4, 4)`` poses.

    ``weight`` is the weight of **pose1**, so:

        weight = 1.0  ->  pose1
        weight = 0.0  ->  pose2

    That direction is worth reading twice, because it is the opposite of the
    ``t``-style convention used by :func:`slerp` immediately above, and the
    work order that commissioned this module stated it backwards. It is kept
    as upstream has it so vendored behaviour is bit-identical; the test suite
    pins the true direction.

    Rotation is slerped, translation is linear.
    """
    R1 = R.from_matrix(np.asarray(pose1, dtype=float)[:3, :3])
    R2 = R.from_matrix(np.asarray(pose2, dtype=float)[:3, :3])
    t1 = np.asarray(pose1, dtype=float)[:3, 3]
    t2 = np.asarray(pose2, dtype=float)[:3, 3]

    q1 = R1.as_quat()
    q2 = R2.as_quat()

    # 1-weight because `weight` is pose1's share, while slerp's t runs q1 -> q2.
    q_interp = slerp(q1, q2, 1.0 - weight)
    R_interp = R.from_quat(q_interp)

    t_interp = weight * t1 + (1.0 - weight) * t2

    pose_interp = np.eye(4)
    pose_interp[:3, :3] = R_interp.as_matrix()
    pose_interp[:3, 3] = t_interp
    return pose_interp


# ---------------------------------------------------------------------------
# sequence smoothing
# ---------------------------------------------------------------------------
def smooth_poses(poses: np.ndarray, window_size: int = 5,
                 method: SmoothMethod = "gaussian") -> np.ndarray:
    """Temporally smooth an ``(N, 4, 4)`` pose sequence.

    ``window_size`` must be odd. ``method`` is ``"gaussian"`` (sigma =
    window/6, so the window spans ~99.7% of the weight), ``"savgol"``
    (polynomial fit, order ``min(window-1, 3)``) or ``"ma"`` (box average).

    THE HEMISPHERE PRE-PASS IS LOad-BEARING. Translations and quaternions are
    both filtered component-wise, which is only valid for quaternions if
    consecutive samples share a hemisphere: ``q`` and ``-q`` are the same
    rotation, so a sign flip between neighbours makes the filter average across
    a discontinuity and emit garbage. The loop below walks the sequence
    negating any sample that opposes its predecessor. Upstream already does
    this (contrary to the work order, which asked for it to be added); the
    regression test in ``test_pose_filters.py`` exists to keep it that way —
    it feeds a deliberately sign-flipped sequence and fails if the pre-pass is
    removed.
    """
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4), got {poses.shape}")
    if window_size % 2 != 1:
        raise ValueError(f"window_size must be odd, got {window_size}")

    N = poses.shape[0]
    smoothed = np.zeros_like(poses)

    translations = poses[:, :3, 3]
    quats = R.from_matrix(poses[:, :3, :3]).as_quat()  # (N, 4), xyzw

    # Consistent quaternion signs — see the docstring.
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    if method == "gaussian":
        sigma = window_size / 6.0
        smoothed_trans = gaussian_filter1d(translations, sigma, axis=0, mode="nearest")
        smoothed_quats = gaussian_filter1d(quats, sigma, axis=0, mode="nearest")
    elif method == "savgol":
        poly_order = min(window_size - 1, 3)
        smoothed_trans = savgol_filter(translations, window_size, poly_order,
                                       axis=0, mode="nearest")
        smoothed_quats = savgol_filter(quats, window_size, poly_order,
                                       axis=0, mode="nearest")
    elif method == "ma":
        kernel = np.ones(window_size) / window_size
        smoothed_trans = np.array(
            [np.convolve(translations[:, i], kernel, mode="same") for i in range(3)]).T
        smoothed_quats = np.array(
            [np.convolve(quats[:, i], kernel, mode="same") for i in range(4)]).T
    else:
        # Upstream falls through here and dies on an UnboundLocalError naming
        # an internal variable, which tells the caller nothing. Deliberate
        # divergence: same failure, legible message.
        raise ValueError(
            f"unknown method {method!r}; expected 'gaussian', 'savgol' or 'ma'")

    # Filtering a unit quaternion component-wise does not preserve unit norm.
    smoothed_quats = smoothed_quats / np.linalg.norm(smoothed_quats, axis=1,
                                                     keepdims=True)

    smoothed_rots = R.from_quat(smoothed_quats).as_matrix()
    for i in range(N):
        smoothed[i] = np.eye(4)
        smoothed[i, :3, :3] = smoothed_rots[i]
        smoothed[i, :3, 3] = smoothed_trans[i]
    return smoothed


# ---------------------------------------------------------------------------
# dwell detection
# ---------------------------------------------------------------------------
def detect_static_sequence(poses: np.ndarray,
                           threshold: float = 0.01) -> tuple[bool, float, float]:
    """Is this pose sequence effectively stationary?

    Returns ``(is_static, trans_diff, rot_diff)`` where the two diffs are mean
    per-step magnitudes: L2 for translation, Frobenius for the rotation matrix.

    Both are compared against the SAME ``threshold`` despite carrying different
    units (metres vs. a dimensionless matrix norm) — upstream's choice, kept so
    behaviour matches. Pass a threshold tuned to whichever term dominates your
    data rather than assuming 0.01 means one centimetre.
    """
    poses = np.asarray(poses, dtype=float)
    translations = poses[:, :3, 3]
    rotations = poses[:, :3, :3]

    trans_diff = float(
        np.linalg.norm(translations[1:] - translations[:-1], axis=1).mean())
    rot_diff = float(
        np.linalg.norm(rotations[1:] - rotations[:-1], axis=(1, 2)).mean())

    return (trans_diff < threshold and rot_diff < threshold), trans_diff, rot_diff


def adaptive_pose_smoothing(poses: np.ndarray, trans_diff: float,
                            rot_diff: float, base_window: int = 5) -> np.ndarray:
    """Smooth with a window sized inversely to how much the sequence moves.

    Slow or dwelling sequences get a wider window (more smoothing, jitter is
    the dominant signal); fast ones stay near ``base_window`` so real motion is
    not flattened. Capped at 41 samples.

    ``trans_diff`` and ``rot_diff`` come from :func:`detect_static_sequence`,
    so the usual call is that function followed by this one.
    """
    motion_magnitude = trans_diff + rot_diff
    adaptive_window = min(
        41, max(base_window,
                int(base_window * (0.1 / max(motion_magnitude, 1e-6)))))

    # The window must stay odd for smooth_poses.
    if adaptive_window % 2 == 0:
        adaptive_window += 1

    return smooth_poses(poses, window_size=adaptive_window, method="gaussian")
