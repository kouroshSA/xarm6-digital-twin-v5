# teleop_hand/test_gestures.py
"""Gesture classification, pointing ray, fist-follow, point resolution, and the
controller's state machine. No camera, no MuJoCo.

    python -m teleop_hand.test_gestures

What each test grades is the CONSUMER's behaviour (where the target ends up,
whether it is safe), not that a number was produced: a target at the wrong
place passes "a target exists".
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from teleop_hand import config, gestures  # noqa: E402
from teleop_hand.camera_pose import HandCameraPose  # noqa: E402
from teleop_hand.follow import FistFollower  # noqa: E402
from teleop_hand.gesture_control import GestureController  # noqa: E402
from teleop_hand.messages import HandFrame, make_frame  # noqa: E402
from teleop_hand.pointing import PointDwell, SphereScene, resolve_pointing  # noqa: E402
from teleop_hand.testing import synth_pose  # noqa: E402
from vr import transforms as vr_transforms  # noqa: E402

BASE_MM = np.array([0.0, 0.0, 800.0])
BENCH_M = 0.75
FLOOR_MM = BENCH_M * 1000 + config.POINT_FLOOR_CLEARANCE_MM


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _tilt(d, deg, axis=(0, 0, 1)):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.deg2rad(deg) * _unit(axis)).apply(d)


# ---- classification --------------------------------------------------------
def test_classify_gestures():
    got = {k: gestures.classify_gesture(synth_pose(k)) for k in
           ("open", "fist", "point", "peace", "index_curled")}
    assert got == {"open": "none", "fist": "fist", "point": "point",
                   "peace": "none", "index_curled": "none"}, got


def test_classify_survives_hand_rotation_and_noise():
    """The tests use curl geometry, not axes: any hand orientation must agree."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    for kind in ("fist", "point", "open"):
        for _ in range(20):
            r = Rotation.random(random_state=int(rng.integers(1e6))).as_matrix()
            kp = synth_pose(kind) @ r.T + rng.normal(0, 0.002, (21, 3))
            assert gestures.classify_gesture(kp) == (kind if kind != "open" else "none"), kind


def test_gesture_debounce():
    d = gestures.GestureDebounce(3)
    seq = ["fist", "fist", "none", "fist", "fist", "fist", "point", "point", "none"]
    out = [d.update(s) for s in seq]
    # a lone "none" between fists resets the run; a change needs 3 in a row
    assert out == ["none", "none", "none", "none", "none", "fist", "fist", "fist", "fist"], out
    for s in ("point", "point", "point"):
        d.update(s)
    assert d.state == "point"


# ---- pointing ray ------------------------------------------------------------
def test_pointing_ray_uses_knuckle_depth_and_landmark_direction():
    """Origin from the MCP's measured depth; direction from the landmarks. The
    fingertip's depth is deliberately WRONG (background, 0.9 m) and must not
    matter."""
    kp = synth_pose("point")
    k = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    wrist_cam = np.array([0.02, 0.05, 0.60])
    kp_cam = kp + wrist_cam
    uv = np.stack([k[0, 0] * kp_cam[:, 0] / kp_cam[:, 2] + k[0, 2],
                   k[1, 1] * kp_cam[:, 1] / kp_cam[:, 2] + k[1, 2]], axis=1)
    depth = np.full((480, 640), 0.9, np.float32)           # background
    u, v = np.round(uv[5]).astype(int)
    depth[v - 3:v + 4, u - 3:u + 4] = kp_cam[5, 2]          # the knuckle, on the hand
    o, d = gestures.pointing_ray(kp, uv, depth, k)
    assert o is not None
    assert np.linalg.norm(o - kp_cam[5]) < 0.003, (o, kp_cam[5])
    want = _unit(kp[8] - kp[5])
    assert np.degrees(np.arccos(np.clip(d @ want, -1, 1))) < 0.1
    # no depth on the knuckle -> no ray, never a guess
    assert gestures.pointing_ray(kp, uv, np.zeros_like(depth), k) == (None, None)
    assert gestures.pointing_ray(kp, uv, None, k) == (None, None)


# ---- camera pose -----------------------------------------------------------------
def test_camera_pose_roundtrip_and_calibration():
    cam = HandCameraPose.nominal()
    p = np.array([0.05, -0.1, 0.65])
    assert np.allclose(cam.world_to_cam_point(cam.point(p)), p)
    d = _unit([0.2, -0.3, 0.9])
    assert np.allclose(cam.world_to_cam_dir(cam.direction(d)), d, atol=1e-9)
    # recover a known pose from palm-at-gripper correspondences
    rng = np.random.default_rng(1)
    pts_cam = rng.uniform([-0.25, -0.2, 0.4], [0.25, 0.2, 0.9], (10, 3))
    pts_world = np.array([cam.point(q) for q in pts_cam]) + rng.normal(0, 0.001, (10, 3))
    got = HandCameraPose.from_correspondences(pts_cam, pts_world)
    assert got.source == "calibrated" and got.rms_mm < 3.0
    assert np.abs(got.cam_to_world - cam.cam_to_world).max() < 5e-3
    # a coplanar sweep must be refused, not solved
    flat = pts_cam.copy()
    flat[:, 2] = 0.6
    try:
        HandCameraPose.from_correspondences(flat, np.array([cam.point(q) for q in flat]))
    except ValueError:
        pass
    else:
        raise AssertionError("coplanar calibration was accepted")


def test_wire_format_roundtrip_and_old_frames():
    f = make_frame(7, detected=True, pos_valid=True, palm_pos_cam_m=[0, 0, .6],
                   gesture="point", point_origin_cam_m=np.array([0, 0, .6]),
                   point_dir_cam=np.array([0, 0, 1.0]))
    g = HandFrame.from_json(f.to_json())
    assert g.gesture == "point" and g.point_ray is not None
    old = make_frame(8, detected=True, pos_valid=True, palm_pos_cam_m=[0, 0, .6])
    assert HandFrame.from_json(old.to_json()).gesture == "none"
    # "point" without a ray is not a pointing gesture anyone can act on
    assert make_frame(9, gesture="point").point_ray is None


# ---- fist follow ---------------------------------------------------------------------
def _follow(fist_mm, ee_mm, n=40):
    f = FistFollower()
    t = None
    for _ in range(n):
        t = f.update(np.asarray(fist_mm) / 1000, np.asarray(ee_mm) / 1000, 1 / 30)
    return f, t * 1000


def test_follow_standoff_band():
    fist = np.array([400.0, -100.0, 1000.0])
    for ee_dist, want in ((400.0, 150.0), (60.0, 100.0), (125.0, 125.0)):
        ee = fist + _unit([1, 1, 0.5]) * ee_dist
        _, t = _follow(fist, ee)
        d = np.linalg.norm(t - fist)
        assert abs(d - want) < 1.0, (ee_dist, d)
        # approaches ALONG the line it is already on: no swinging round the hand
        assert np.allclose(_unit(t - fist), _unit(ee - fist), atol=1e-6)


def test_follow_never_inside_minimum_distance_or_below_floor():
    rng = np.random.default_rng(2)
    for _ in range(200):
        fist = np.array([rng.uniform(-300, 600), rng.uniform(-250, 500), rng.uniform(780, 1250)])
        ee = fist + _unit(rng.normal(size=3)) * rng.uniform(20, 500)
        _, t = _follow(fist, ee, n=3)
        assert t[2] >= config.FOLLOW_Z_MIN_MM - 1e-6, t
        inside = np.allclose(vr_transforms.clamp_workspace_mm(t), t, atol=1e-6)
        dist = np.linalg.norm(t - fist)
        # the workspace box may force it closer; otherwise the band holds
        if inside:
            assert dist >= config.FOLLOW_STANDOFF_MM[0] - 1.0, (fist, ee, t, dist)


def test_follow_rejects_single_frame_outliers_but_accepts_real_jumps():
    f = FistFollower()
    ee = np.array([0.3, -0.05, 1.06])
    for _ in range(10):
        f.update([0.4, -0.1, 1.0], ee, 1 / 30)
    before = f.fist_m.copy()
    assert f.update([0.9, 0.4, 0.8], ee, 1 / 30) is None          # one outlier frame
    assert np.allclose(f.fist_m, before)
    f.update([0.4, -0.1, 1.0], ee, 1 / 30)                        # back to normal
    got = [f.update([0.7, -0.1, 1.0], ee, 1 / 30) for _ in range(5)]   # the hand really moved
    assert got[-1] is not None and f.fist_m[0] > before[0] + 0.01


# ---- pointing -----------------------------------------------------------------------------
def _scene():
    return SphereScene({"red_cube": ([0.0, 0.15, 0.78], 0.03),
                        "blue_cube": ([0.22, -0.25, 0.78], 0.03)},
                       bench_top_m=BENCH_M,
                       obstacles={"heater": ([0.1, 0.13, 0.79], 0.05)})


def test_point_at_object_within_cone_is_a_grasp():
    s = _scene()
    o = np.array([0.0, -0.3, 1.1])
    d = _unit(np.array([0.0, 0.15, 0.78]) - o)
    t = resolve_pointing(o, _tilt(d, 3.0), s, BASE_MM)          # 3 deg off: finger noise
    assert t.kind == "grasp" and t.name == "red_cube", t
    assert np.allclose(t.position_mm, [0, 150, 780])
    t2 = resolve_pointing(o, _tilt(d, 25.0), s, BASE_MM)        # 25 deg off: not that cube
    assert t2 is None or t2.name != "red_cube"


def test_point_picks_the_smallest_angle_not_the_nearest_object():
    """From 50 mm to the side, the far cube is 4 deg off the ray and the near
    one 9.5 deg: the finger is pointing at the far one."""
    s = SphereScene({"near": ([0.0, -0.1, 0.9], 0.03), "far": ([0.0, 0.3, 0.9], 0.03)}, BENCH_M)
    t = resolve_pointing(np.array([0.05, -0.4, 0.9]), [0.0, 1.0, 0.0], s, BASE_MM)
    assert t.kind == "grasp" and t.name == "far" and t.angle_deg < 5.0, t
    t = resolve_pointing(np.array([-0.05, -0.4, 0.9]), _unit([0.1, 0.5, 0.0]), s, BASE_MM)
    assert t.name == "near", t


def test_point_at_bench_hovers_above_it_never_into_it():
    s = _scene()
    rng = np.random.default_rng(3)
    for _ in range(100):
        o = np.array([rng.uniform(-0.3, 0.5), rng.uniform(-0.45, -0.15), rng.uniform(0.95, 1.2)])
        # aim at a random bench point, away from the objects
        aim = np.array([rng.uniform(-0.4, 0.6), rng.uniform(-0.1, 0.4), BENCH_M])
        t = resolve_pointing(o, _unit(aim - o), s, BASE_MM)
        if t is None or t.kind == "grasp":
            continue
        assert t.position_mm[2] >= FLOOR_MM - 1e-6, t
        for p in [t.position_mm, *t.fallbacks]:
            assert p[2] >= FLOOR_MM - 1e-6
            assert np.allclose(vr_transforms.clamp_workspace_mm(p), p, atol=1e-6)
            assert np.linalg.norm((p - BASE_MM)[:2]) <= config.POINT_REACH_MM + 1e-6


def test_point_into_free_space_goes_that_way_and_stays_safe():
    s = _scene()
    o = np.array([0.1, -0.2, 1.0])
    d = _unit([0.3, 1.0, 0.25])                                # up and away: hits nothing
    assert s.cast(o, d) is None
    t = resolve_pointing(o, d, s, BASE_MM)
    assert t.kind == "free"
    # it is on the ray (within the sampling), and it went as far as the limits allow
    v = t.position_mm / 1000 - o
    assert np.degrees(np.arccos(_unit(v) @ d)) < 0.5
    assert t.position_mm[2] >= FLOOR_MM
    assert np.linalg.norm((t.position_mm - BASE_MM)[:2]) > 0.5 * config.POINT_REACH_MM
    # a ray through an obstacle stops short of it, by the standoff
    o3 = np.array([0.1, -0.2, 0.9])
    d3 = _unit(np.array([0.1, 0.13, 0.80]) - o3)
    hit = s.cast(o3, d3)
    assert hit is not None and hit.name == "heater"
    t3 = resolve_pointing(o3, d3, s, BASE_MM, cone_deg=0.0)                       # cone off: exercise the ray cast
    assert t3.kind == "hover" and t3.name == "heater"
    assert np.linalg.norm(t3.position_mm / 1000 - hit.point_m) >= config.POINT_STANDOFF_MM / 1000 - 1e-6


def test_point_with_nothing_safe_returns_none_not_a_guess():
    s = _scene()
    # pointing straight down at the bench from outside the workspace box
    assert resolve_pointing(np.array([2.0, 2.0, 1.0]), [0, 0, -1.0], s, BASE_MM) is None


def test_point_dwell():
    d = PointDwell(dwell_s=1.0, steady_deg=4.0)
    o, ray = np.zeros(3), _unit([0, 1, 0])
    fired = [d.update(o, ray, t * 0.1) for t in range(15)]
    assert sum(f is not None for f in fired) == 1 and fired[10] is not None      # once, at 1.0 s
    d.reset()
    for t in range(30):                       # a sweeping ray never settles
        assert d.update(o, _tilt(ray, 6.0 * (t % 2) + 0.0 * t), t * 0.1) is None


# ---- controller state machine -------------------------------------------------------------
def _frame(seq, cam, gesture, world_pos=None, ray=None, valid=True):
    kw = dict(detected=True, pos_valid=valid, gesture=gesture)
    if world_pos is not None:
        kw["palm_pos_cam_m"] = cam.world_to_cam_point(np.asarray(world_pos, float))
    if ray is not None:
        kw["point_origin_cam_m"] = cam.world_to_cam_point(ray[0])
        kw["point_dir_cam"] = cam.world_to_cam_dir(ray[1])
    return make_frame(seq, t_mono=0.0, **kw)


def test_controller_fist_needs_engage_time_then_follows_then_stops():
    cam = HandCameraPose.nominal()
    c = GestureController(cam, _scene())
    ee = np.array([0.3, -0.05, 1.06])
    fist = np.array([0.45, -0.2, 1.0])
    dt, kinds, seq, now = 1 / 30, [], 0, 0.0
    for _ in range(int(0.6 / dt)):                       # fist for 0.6 s
        seq += 1
        now += dt
        kinds.append(c.update(_frame(seq, cam, "fist", fist), now, dt, ee, BASE_MM).kind)
    first = kinds.index("follow")
    assert abs(first * dt - config.FOLLOW_ENGAGE_S) < 2 * dt, first * dt     # not instant
    assert kinds[-1] == "follow"
    seq += 1
    now += dt
    o = c.update(_frame(seq, cam, "none", fist), now, dt, ee, BASE_MM)       # hand opens
    assert o.kind == "none" and c.mode == "idle"
    for _ in range(3):                                   # a fresh fist has to be HELD again
        seq += 1
        now += dt
        o = c.update(_frame(seq, cam, "fist", fist), now, dt, ee, BASE_MM)
    assert o.kind == "none"


def test_controller_drops_to_idle_on_stale_frames():
    cam = HandCameraPose.nominal()
    c = GestureController(cam, _scene())
    ee = np.array([0.3, -0.05, 1.06])
    now = 0.0
    for seq in range(1, 30):
        now += 1 / 30
        o = c.update(_frame(seq, cam, "fist", [0.45, -0.2, 1.0]), now, 1 / 30, ee, BASE_MM)
    assert o.kind == "follow"
    stale = _frame(29, cam, "fist", [0.45, -0.2, 1.0])        # same seq: no new frame
    for _ in range(int((config.DROPOUT_S + 0.1) * 30)):
        now += 1 / 30
        o = c.update(stale, now, 1 / 30, ee, BASE_MM)
    assert o.kind == "none" and o.status.get("stale")


def test_controller_point_fires_once_per_gesture():
    cam = HandCameraPose.nominal()
    c = GestureController(cam, _scene())
    ee = np.array([0.3, -0.05, 1.06])
    o_w = np.array([0.0, -0.3, 1.1])
    d_w = _unit(np.array([0.0, 0.15, 0.78]) - o_w)
    fired, seq, now = [], 0, 0.0
    for _ in range(int(2.5 * 30)):                       # keep pointing for 2.5 s
        seq += 1
        now += 1 / 30
        o = c.update(_frame(seq, cam, "point", ray=(o_w, d_w)), now, 1 / 30, ee, BASE_MM)
        if o.kind == "point":
            fired.append((now, o.point))
    assert len(fired) == 1, len(fired)
    assert abs(fired[0][0] - config.POINT_DWELL_S) < 0.1
    assert fired[0][1].kind == "grasp" and fired[0][1].name == "red_cube"
    for _ in range(3):                                   # release, then point again
        seq += 1
        now += 1 / 30
        c.update(_frame(seq, cam, "none", [0, 0, 0.6]), now, 1 / 30, ee, BASE_MM)
    again = []
    for _ in range(int(1.5 * 30)):
        seq += 1
        now += 1 / 30
        o = c.update(_frame(seq, cam, "point", ray=(o_w, d_w)), now, 1 / 30, ee, BASE_MM)
        again.append(o.kind)
    assert again.count("point") == 1


def test_controller_wobbling_point_never_fires():
    cam = HandCameraPose.nominal()
    c = GestureController(cam, _scene())
    ee = np.array([0.3, -0.05, 1.06])
    o_w = np.array([0.0, -0.3, 1.1])
    d_w = _unit(np.array([0.0, 0.15, 0.78]) - o_w)
    now = 0.0
    for seq in range(1, 120):
        now += 1 / 30
        d = _tilt(d_w, 7.0 * np.sin(seq / 3.0))
        assert c.update(_frame(seq, cam, "point", ray=(o_w, d)), now, 1 / 30, ee, BASE_MM).kind != "point"


def _main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
