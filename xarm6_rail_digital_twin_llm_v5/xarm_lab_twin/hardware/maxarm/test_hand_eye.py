"""Hardware-free tests for the MaxArm eye-in-hand calibration.

    python -m hardware.maxarm.test_hand_eye

A known camera mount generates the views; the test grades where a camera point
ends up in ARM coordinates (the consumer), not only the solved matrix.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from perception.extrinsic_calibration import pose_error, solve_camera_pose

from . import hand_eye as he

PASS, FAIL = "PASS", "FAIL"

TAG = (0.0, -200.0, 52.0)          # arm frame, mm (the floor-contact spot)


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = (f(math.radians(a)) for a in (rx, ry, rz) for f in (math.cos, math.sin))
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


# A plausible mount: looking down (optical z ~ -z), tipped a few degrees,
# 35 mm beside and 20 mm above the wrist point.
TRUE = np.eye(4)
TRUE[:3, :3] = _rot(178.0, 4.0, 93.0)
TRUE[:3, 3] = [0.012, -0.035, 0.020]


def _poses():
    return [(TAG[0] + dx, TAG[1] + dy, z)
            for dx in (-50, -20, 0, 25, 50) for dy in (-40, 0, 40) for z in (100, 150, 200)]


def _views(noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    inv = np.linalg.inv(TRUE)
    q = []
    for p in _poses():
        w = he.tag_in_wrist(TAG, p)
        q.append((inv @ np.append(w, 1.0))[:3] + rng.normal(0, noise_mm / 1000.0, 3))
    return np.array(q)


def test_recovers_a_known_mount():
    res = he.solve(TAG, _poses(), _views(noise_mm=0.5))
    dt, da = pose_error(res.cam_to_base, TRUE)
    if dt > 1.0 or da > 0.5:
        return f"{FAIL} mount off by {dt:.2f} mm / {da:.2f} deg"
    return f"{PASS} known mount recovered within {dt:.2f} mm / {da:.2f} deg (0.5 mm noise, {res.n} views)"


def test_camera_point_lands_on_the_tag_in_arm_coordinates():
    res = he.solve(TAG, _poses(), _views(noise_mm=0.5))
    q = _views()
    worst = max(np.linalg.norm(he.camera_point_to_arm(qi, p, res.cam_to_base) - TAG)
                for qi, p in zip(q, _poses()))
    # and a NEW point: 30 mm right of the tag, seen from a pose not in the set
    new = np.array([TAG[0] + 30.0, TAG[1] - 10.0, TAG[2]])
    pose = (TAG[0] - 35.0, TAG[1] + 15.0, 170.0)
    f = he.wrist_frame(pose) @ TRUE
    q_new = (np.linalg.inv(f) @ np.append(he.to_rh(new), 1.0))[:3]
    miss = np.linalg.norm(he.camera_point_to_arm(q_new, pose, res.cam_to_base) - new)
    if worst > 2.0 or miss > 2.0:
        return f"{FAIL} camera points land {worst:.2f} mm (tag) / {miss:.2f} mm (new point) off"
    return f"{PASS} camera points land on the arm-frame target: tag {worst:.2f} mm, unseen point {miss:.2f} mm"


def test_left_handed_frame_is_converted():
    """Negative control: built in the arm frame as if it were right-handed."""
    q = _views()

    def naive(p):                       # no x negation anywhere
        t = np.array(p, float) / 1000.0
        h = math.atan2(t[1], t[0])
        r = np.array([[math.cos(h), -math.sin(h), 0], [math.sin(h), math.cos(h), 0], [0, 0, 1]])
        return r.T @ (np.array(TAG) / 1000.0 - t)

    bad = solve_camera_pose(q, np.array([naive(p) for p in _poses()]))
    good = he.solve(TAG, _poses(), q)
    if good.rms_mm > 0.1 or bad.rms_mm < 5.0:
        return f"{FAIL} rms converted {good.rms_mm:.2f} mm, naive {bad.rms_mm:.2f} mm"
    return f"{PASS} converted frame fits ({good.rms_mm:.3f} mm); the naive left-handed one does not ({bad.rms_mm:.1f} mm)"


def test_round_trip_frames():
    p = np.array([37.0, -181.0, 99.0])
    if not np.allclose(he.from_rh(he.to_rh(p)), p):
        return f"{FAIL} to_rh/from_rh do not round-trip"
    f = he.wrist_frame((0.0, -200.0, 100.0))
    if not np.allclose(f[:3, 3], [0.0, -0.2, 0.1]):
        return f"{FAIL} wrist_frame translation {f[:3, 3]}"
    return f"{PASS} frame conversions round-trip"


class _FakeArm:
    def __init__(self, xyz, yaw=None):
        self.xyz, self.yaw = xyz, yaw

    def get_position(self):
        return 0, [*self.xyz, 180.0, 0.0, self.yaw]


def _posed_camera(arm):
    """MaxArmWristCamera's pose half, without opening a device."""
    from .wrist_camera import MaxArmWristCamera
    cam = MaxArmWristCamera.__new__(MaxArmWristCamera)
    cam.arm, cam.hand_eye = arm, he.HandEye(TRUE, {})
    return cam


def test_frame_pixel_lands_on_the_arm_point():
    """The consumer: a captured frame's pixel_to_world, x1000, is a set_position target."""
    from perception.d435i_calib import COLOR_INTRINSICS
    from perception.rgbd import RGBDFrame
    target = np.array([TAG[0] + 25.0, TAG[1] - 15.0, TAG[2]])
    worst = 0.0
    for pose in [(TAG[0] + 50.0, TAG[1] + 40.0, 180.0), (TAG[0] + 20.0, TAG[1] + 60.0, 150.0)]:
        cam = _posed_camera(_FakeArm(pose, yaw=0.0))
        q = (np.linalg.inv(he.wrist_frame(pose) @ TRUE) @ np.append(he.to_rh(target), 1.0))[:3]
        k = COLOR_INTRINSICS
        u, v = k.fx * q[0] / q[2] + k.cx, k.fy * q[1] / q[2] + k.cy
        depth = np.full((k.height, k.width), np.nan, np.float32)
        frame = RGBDFrame(color=np.zeros((k.height, k.width, 3), np.uint8), depth=depth,
                          intrinsics=k, cam_to_world=cam.cam_to_world())
        got = frame.pixel_to_world(u, v, depth_m=q[2]) * 1000.0
        worst = max(worst, float(np.linalg.norm(got - target)))
    if worst > 0.01:
        return f"{FAIL} pixel_to_world*1000 lands {worst:.3f} mm from the arm-frame point"
    return f"{PASS} a frame pixel lands on its arm-frame point (pixel_to_world*1000, {worst:.4f} mm)"


def test_pixel_at_height_lands_on_the_point():
    """A pixel plus a known height (no depth) gives back the arm-frame point."""
    from perception.d435i_calib import COLOR_INTRINSICS as k
    from perception.rgbd import RGBDFrame
    from .wrist_camera import pixel_at_height
    pose = (TAG[0] + 70.0, TAG[1] + 40.0, 200.0)
    cam = _posed_camera(_FakeArm(pose, yaw=0.0))
    frame = RGBDFrame(color=np.zeros((k.height, k.width, 3), np.uint8),
                      depth=np.full((k.height, k.width), np.nan, np.float32),
                      intrinsics=k, cam_to_world=cam.cam_to_world())
    worst = 0.0
    for top in ([TAG[0] + 40.0, TAG[1], 92.0], [TAG[0] - 10.0, TAG[1] - 20.0, 72.0]):
        q = (np.linalg.inv(he.wrist_frame(pose) @ TRUE) @ np.append(he.to_rh(top), 1.0))[:3]
        got = pixel_at_height(frame, k.fx * q[0] / q[2] + k.cx, k.fy * q[1] / q[2] + k.cy, top[2])
        worst = max(worst, float(np.linalg.norm(got - np.array(top))))
    if worst > 0.01:
        return f"{FAIL} pixel_at_height lands {worst:.3f} mm off"
    return f"{PASS} pixel + known height lands on the arm-frame point ({worst:.4f} mm, no depth)"


def test_frame_pose_ignores_the_nozzle():
    """The camera does not ride the nozzle (measured): any nozzle angle, same pose."""
    poses = [_posed_camera(_FakeArm((50.0, -160.0, 200.0), yaw=a)).cam_to_world()
             for a in (None, 0.0, 45.0, -90.0)]
    if any(p is None for p in poses) or not all(np.allclose(p, poses[0]) for p in poses):
        return f"{FAIL} frame pose depends on the nozzle angle"
    return f"{PASS} frame pose is the same at every nozzle angle (camera fixed to the platform)"


TESTS = [test_round_trip_frames, test_recovers_a_known_mount,
         test_camera_point_lands_on_the_tag_in_arm_coordinates,
         test_left_handed_frame_is_converted,
         test_frame_pixel_lands_on_the_arm_point, test_frame_pose_ignores_the_nozzle,
         test_pixel_at_height_lands_on_the_point]


def main() -> int:
    bad = 0
    for t in TESTS:
        try:
            msg = t()
        except Exception as exc:  # noqa: BLE001
            msg = f"{FAIL} {t.__name__}: {type(exc).__name__}: {exc}"
        print(" ", msg)
        bad += msg.startswith(FAIL)
    print(f"\n  {len(TESTS) - bad} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
