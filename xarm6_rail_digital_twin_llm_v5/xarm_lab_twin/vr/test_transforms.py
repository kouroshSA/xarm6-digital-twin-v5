# vr/test_transforms.py
"""Unit tests for vr/transforms.py (acceptance test 2).

Run from the working directory (xarm_lab_twin/):
    python -m vr.test_transforms        # plain-python, prints PASS/FAIL
    pytest vr/test_transforms.py        # if pytest is installed
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vr import config, transforms


def _rx(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def test_pos_mapping_known_point():
    """twin = scale*(M@xr) + offset, with M the +90-about-X basis."""
    old_scale = config.WORLD_SCALE
    config.WORLD_SCALE = 1.0
    try:
        p = transforms.xr_to_twin_pos([1.0, 2.0, 3.0])
        # M@(1,2,3) = (1, -3, 2); + offset (0, 0.30, 0.95)
        expected = np.array([1.0, -3.0, 2.0]) + config.TWIN_ORIGIN_OFFSET_M
        assert np.allclose(p, expected, atol=1e-9), (p, expected)
    finally:
        config.WORLD_SCALE = old_scale


def test_pos_mapping_scale():
    old_scale = config.WORLD_SCALE
    config.WORLD_SCALE = 0.5
    try:
        p = transforms.xr_to_twin_pos([2.0, 0.0, 0.0])
        expected = np.array([1.0, 0.0, 0.0]) + config.TWIN_ORIGIN_OFFSET_M
        assert np.allclose(p, expected, atol=1e-9), (p, expected)
    finally:
        config.WORLD_SCALE = old_scale


def test_controller_pointing_down_is_gripper_down():
    """Controller laser (-Z local) pointing along world-down (XR -Y) must map
    to the canonical 'gripper straight down' pose: roll=180, pitch=0, yaw=0."""
    # R_xr = Rx(-90) sends controller-local -Z to XR-world -Y (down).
    R_xr = _rx(-90.0)
    assert np.allclose(R_xr @ np.array([0, 0, -1.0]), [0, -1, 0], atol=1e-9)
    # As an [x,y,z,w] quaternion: Rx(-90) -> (sin(-45), 0, 0, cos(-45)).
    q = [np.sin(np.deg2rad(-45.0)), 0.0, 0.0, np.cos(np.deg2rad(-45.0))]
    R_twin = transforms.xr_to_twin_quat(q)
    # Expect Rx(180) = diag(1, -1, -1).
    assert np.allclose(R_twin, np.diag([1.0, -1.0, -1.0]), atol=1e-6), R_twin
    roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(R_twin)
    assert abs(abs(roll) - 180.0) < 1e-3, (roll, pitch, yaw)
    assert abs(pitch) < 1e-3 and abs(yaw) < 1e-3, (roll, pitch, yaw)


def test_identity_controller_points_forward_horizontal():
    """Controller at rest (laser forward, R_xr=I) -> gripper approach axis
    (+Z) points along twin +Y (forward at the bench), horizontally."""
    R_twin = transforms.xr_to_twin_quat([0.0, 0.0, 0.0, 1.0])
    approach = R_twin @ np.array([0, 0, 1.0])  # gripper +Z in twin
    assert np.allclose(approach, [0, 1, 0], atol=1e-6), approach


def test_quat_returns_orthonormal_matrix():
    """A non-trivial controller quaternion yields a proper rotation matrix."""
    q = [0.18, -0.36, 0.54, 0.74]
    q = list(np.array(q) / np.linalg.norm(q))
    R = transforms.xr_to_twin_quat(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_clutch_no_jump_on_engage():
    """At engage, target() must reproduce the EE pose exactly (no jump)."""
    c = transforms.Clutch()
    ctrl_p = np.array([0.1, 0.2, 0.3])
    ctrl_R = _rx(30.0)
    ee_p = np.array([0.5, -0.1, 1.0])
    ee_R = _rx(180.0)
    c.engage(ctrl_p, ctrl_R, ee_p, ee_R)
    tp, tR = c.target(ctrl_p, ctrl_R)
    assert np.allclose(tp, ee_p, atol=1e-9), (tp, ee_p)
    assert np.allclose(tR, ee_R, atol=1e-9), (tR, ee_R)
    # Moving the controller +5cm in twin x moves the target +5cm in twin x.
    tp2, _ = c.target(ctrl_p + np.array([0.05, 0, 0]), ctrl_R)
    assert np.allclose(tp2, ee_p + np.array([0.05, 0, 0]), atol=1e-9)


def test_clamp_workspace():
    aabb = config.WORKSPACE_AABB_MM
    far = transforms.clamp_workspace_mm([1e6, -1e6, 1e6])
    assert far[0] == aabb["x"][1] and far[1] == aabb["y"][0] and far[2] == aabb["z"][1]
    inside = transforms.clamp_workspace_mm([0.0, 400.0, 1000.0])
    assert np.allclose(inside, [0.0, 400.0, 1000.0])


def test_smoother_converges():
    s = transforms.Smoother(alpha=0.5)
    s.reset()
    first = s.update([1.0, 1.0, 1.0])
    assert np.allclose(first, [1.0, 1.0, 1.0])  # first sample passes through
    second = s.update([3.0, 3.0, 3.0])
    assert np.allclose(second, [2.0, 2.0, 2.0])  # 0.5*3 + 0.5*1


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
