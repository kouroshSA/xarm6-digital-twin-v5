# teleop_hand/test_hand_calib.py
"""No-hardware tests for the hand-camera calibration.

    python -m teleop_hand.test_hand_calib      (needs opencv-contrib: xarm6sim_hand)

Graded on what the CONSUMER gets: a saved calibration, loaded the way
`--cam-extrinsic` loads it, must put a camera-frame point where the world says
it is. "A pose was produced" would pass for any pose.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from teleop_hand import hand_calib as hc
from teleop_hand.camera_pose import HandCameraPose

K = np.array([[610.0, 0, 320], [0, 610.0, 240], [0, 0, 1]])
CAM = HandCameraPose.looking_at((-1100.0, -100.0, 1300.0), (0.0, -100.0, 950.0))


def _view(world_mm, size_mm=100.0, cam=CAM):
    w = np.asarray(world_mm, float) / 1000.0
    return hc.render_tag(cam, w, cam.cam_to_world[:3, 3] - w, K, size_mm / 1000.0)


def test_detect_recovers_tag_position():
    cam = HandCameraPose.looking_at((-600.0, 0.0, 1000.0), (0.0, 0.0, 950.0))
    w = np.array([-100.0, 20.0, 940.0]) / 1000.0
    rgb, depth = hc.render_tag(cam, w, cam.cam_to_world[:3, 3] - w, K, 0.1)
    o = hc.detect_tag(rgb, K, 0.1, depth)
    assert o is not None
    err = np.linalg.norm(cam.point(o.pos_cam_m) - w) * 1000
    assert err < 4.0, f"{err:.1f} mm"


def test_wrong_tag_size_is_flagged_by_the_depth_check():
    rgb, depth = _view([0, -100, 950], size_mm=100.0)
    good = hc.detect_tag(rgb, K, 0.100, depth)
    bad = hc.detect_tag(rgb, K, 0.060, depth)         # operator typed 60 for a 100 mm tag
    assert abs(good.depth_check_mm) < 15.0
    assert abs(bad.depth_check_mm) > 100.0, bad.depth_check_mm


def test_tag_absent_returns_none():
    assert hc.detect_tag(np.full((480, 640, 3), 128, np.uint8), K, 0.1) is None


def test_samples_reject_a_repeat_pose():
    s = hc.SampleSet()
    assert s.try_add([0, 0, 1], [0.0, 0.0, 0.9])[0]
    ok, why = s.try_add([0, 0, 1], [0.01, 0.0, 0.9])
    assert not ok and "further" in why


def test_coplanar_poses_are_refused():
    rng = np.random.default_rng(0)
    s = hc.SampleSet()
    for _ in range(12):
        w = np.array([rng.uniform(-.3, .3), rng.uniform(-.3, .3), 0.9])      # one height
        s.try_add(CAM.world_to_cam_point(w), w)
    try:
        hc.solve(s)
    except ValueError:
        return
    raise AssertionError("a one-height sweep was accepted")


def test_saved_calibration_places_points_in_the_world():
    rng = np.random.default_rng(1)
    s = hc.SampleSet()
    while len(s) < 10:
        w = np.array([rng.uniform(-.4, .3), rng.uniform(-.4, .3), rng.uniform(.85, 1.25)])
        rgb, depth = hc.render_tag(CAM, w, CAM.cam_to_world[:3, 3] - w, K, 0.1)
        o = hc.detect_tag(rgb, K, 0.1, depth)
        assert o is not None
        s.try_add(o.pos_cam_m, w)
    pose, _ = hc.solve(s)
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "cp.json"
        pose.save(f)
        got = HandCameraPose.load(f)                    # what --cam-extrinsic does
    assert got.source == "file" and not got.is_nominal
    fist_world = np.array([-0.2, 0.1, 1.0])
    seen = CAM.world_to_cam_point(fist_world)           # what the tracker would report
    err = np.linalg.norm(got.point(seen) - fist_world) * 1000
    assert err < 10.0, f"{err:.1f} mm"


def main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:                      # noqa: BLE001
                fails += 1
                print("FAIL", name, "-", repr(e))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
