# vr/transforms.py
"""Coordinate math between the WebXR world frame and the MuJoCo twin frame,
plus the teleop clutch, an exponential smoother, and a workspace clamp.

Frames
------
WebXR ``local-floor`` reference space: right-handed, +X right, +Y up,
-Z forward, metres, origin on the floor where the session started.

MuJoCo twin: right-handed, Z-up, metres. Bench top ~ z=0.76, workspace in
front of the arm.

Basis change XR -> twin (the spec's explicit equations are authoritative):

    twin_x =  s * xr_x
    twin_y = -s * xr_z
    twin_z =  s * xr_y

i.e. ``twin = s * (M @ xr)`` with the orthonormal basis matrix

    M = [[1, 0,  0],
         [0, 0, -1],
         [0, 1,  0]]

which is a +90 deg rotation about X (XR-up +Y becomes twin-up +Z).

Orientation
-----------
A controller/head orientation R_xr (controller-local -> XR-world) is rebased
into the twin and aligned to the gripper frame by

    R_twin = M @ R_xr @ B          B = diag(1, -1, -1)  (180 deg about X)

The trailing B aligns the controller's "laser" axis (controller-local -Z,
the natural pointing direction) with the gripper's approach axis
(gripper-local +Z). Consequence, verified in tests:
  * controller pointing straight down  -> R_twin == Rx(180) -> roll=180,
    pitch=0, yaw=0  (gripper pointing straight down, the canonical grasp).
  * controller laser forward (R_xr = I) -> gripper points forward,
    horizontally at the bench.

Quaternion order: WebXR reports quaternions as [x, y, z, w]. The functions
here accept that order (what the client streams up) and convert internally.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat

from vr import config

# XR -> twin position basis (twin = M @ xr, before scale/offset).
_M = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0],
               [0.0, 1.0, 0.0]], dtype=float)

# Controller-local -> gripper-local alignment (180 deg about X): maps the
# controller's -Z laser axis onto the gripper's +Z approach axis.
_B = np.diag([1.0, -1.0, -1.0]).astype(float)


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------
def _quat_xyzw_to_mat(q_xyzw) -> np.ndarray:
    """3x3 rotation matrix from a WebXR-order [x, y, z, w] quaternion."""
    x, y, z, w = (float(q_xyzw[0]), float(q_xyzw[1]),
                  float(q_xyzw[2]), float(q_xyzw[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ], dtype=float)


# ---------------------------------------------------------------------------
# position / orientation mapping
# ---------------------------------------------------------------------------
def xr_to_twin_pos(p_xr) -> np.ndarray:
    """Map an XR-world position (m) to a twin-frame position (m).

    Applies the basis change, the uniform ``config.WORLD_SCALE``, then adds
    ``config.TWIN_ORIGIN_OFFSET_M`` so the operator's standing hand origin
    lands near the bench workspace.
    """
    p = np.asarray(p_xr, dtype=float).reshape(3)
    return config.WORLD_SCALE * (_M @ p) + config.TWIN_ORIGIN_OFFSET_M


def xr_to_twin_quat(q_xr) -> np.ndarray:
    """Map an XR-world orientation quaternion [x,y,z,w] to a twin-frame 3x3
    rotation matrix suitable for ``IKSolver.solve(target_rot=...)`` and (via
    :func:`twin_rot_to_rpy_deg`) for ``SimXArmAPI.set_position``."""
    R_xr = _quat_xyzw_to_mat(q_xr)
    return _M @ R_xr @ _B


def xr_delta_to_twin(d_xr) -> np.ndarray:
    """Map an XR-world *displacement* (m) to a twin-frame displacement (m):
    basis change + scale only, no origin offset. Used for relative head
    tracking (head motion from the session-start head position)."""
    d = np.asarray(d_xr, dtype=float).reshape(3)
    return config.WORLD_SCALE * (_M @ d)


def xr_to_twin_head_quat(q_xr) -> np.ndarray:
    """Map an XR-world head orientation [x,y,z,w] to a MuJoCo mocap
    quaternion (w,x,y,z) for the ``vr_head`` body.

    Pure rebasis (conjugation ``M @ R @ M.T``) — no gripper alignment — so an
    identity headset orientation leaves ``vr_head`` at identity (cameras face
    the bench, upright, per the baked camera quats), and head yaw about the
    XR up-axis becomes yaw about the twin up-axis.
    """
    R_xr = _quat_xyzw_to_mat(q_xr)
    R_twin = _M @ R_xr @ _M.T
    return mat2quat(R_twin)  # (w, x, y, z)


def twin_rot_to_rpy_deg(R: np.ndarray) -> tuple:
    """3x3 twin rotation -> (roll, pitch, yaw) degrees, extrinsic XYZ.

    Uses the same axis convention (``sxyz``) as
    ``SimXArmAPI.get_position`` / ``set_position`` so the angles round-trip
    through the public API.
    """
    roll, pitch, yaw = mat2euler(R, axes="sxyz")
    return (float(np.rad2deg(roll)),
            float(np.rad2deg(pitch)),
            float(np.rad2deg(yaw)))


# ---------------------------------------------------------------------------
# workspace clamp
# ---------------------------------------------------------------------------
def clamp_workspace_mm(p_twin_mm) -> np.ndarray:
    """Clamp a twin-frame position (mm) into ``config.WORKSPACE_AABB_MM``."""
    p = np.asarray(p_twin_mm, dtype=float).reshape(3).copy()
    aabb = config.WORKSPACE_AABB_MM
    p[0] = float(np.clip(p[0], *aabb["x"]))
    p[1] = float(np.clip(p[1], *aabb["y"]))
    p[2] = float(np.clip(p[2], *aabb["z"]))
    return p


# ---------------------------------------------------------------------------
# exponential smoother
# ---------------------------------------------------------------------------
class Smoother:
    """First-order exponential smoother for a 3-vector (the clutch target
    position, in twin-frame metres). Damps controller jitter before IK.

    ``alpha`` in (0, 1]: out = alpha*new + (1-alpha)*prev. Lower = smoother
    but laggier. Reset on clutch (re)engage so a stale value can't yank the
    first frame.
    """

    def __init__(self, alpha: float = config.SMOOTH_ALPHA):
        self.alpha = float(alpha)
        self._state: Optional[np.ndarray] = None

    def reset(self, value=None) -> None:
        self._state = None if value is None else np.asarray(value, float).copy()

    def update(self, value) -> np.ndarray:
        v = np.asarray(value, dtype=float).reshape(3)
        if self._state is None:
            self._state = v.copy()
        else:
            self._state = self.alpha * v + (1.0 - self.alpha) * self._state
        return self._state.copy()


# ---------------------------------------------------------------------------
# clutch
# ---------------------------------------------------------------------------
class Clutch:
    """Standard VR teleop clutch: the EE only follows the controller while the
    grip (squeeze) button is held.

    On engage, the offset between the current controller pose and the current
    EE pose is frozen, so engaging never jumps the arm. While held,

        target_pos = controller_pos + frozen_pos_offset
        target_rot = frozen_rot_offset @ controller_rot

    On release the arm freezes (the caller simply stops servoing).

    All poses are twin-frame (metres / 3x3 rotation).
    """

    def __init__(self):
        self.engaged: bool = False
        self._pos_offset = np.zeros(3, dtype=float)
        self._rot_offset = np.eye(3, dtype=float)

    def engage(self, controller_pos_twin, controller_rot_twin,
               ee_pos_twin, ee_rot_twin) -> None:
        cp = np.asarray(controller_pos_twin, float).reshape(3)
        ep = np.asarray(ee_pos_twin, float).reshape(3)
        self._pos_offset = ep - cp
        # rot_offset such that rot_offset @ controller_rot == ee_rot at engage.
        self._rot_offset = np.asarray(ee_rot_twin, float).reshape(3, 3) @ \
            np.asarray(controller_rot_twin, float).reshape(3, 3).T
        self.engaged = True

    def release(self) -> None:
        self.engaged = False

    def target(self, controller_pos_twin, controller_rot_twin) -> tuple:
        """-> (target_pos_twin_m (3,), target_rot_twin (3x3))."""
        cp = np.asarray(controller_pos_twin, float).reshape(3)
        cr = np.asarray(controller_rot_twin, float).reshape(3, 3)
        return cp + self._pos_offset, self._rot_offset @ cr
