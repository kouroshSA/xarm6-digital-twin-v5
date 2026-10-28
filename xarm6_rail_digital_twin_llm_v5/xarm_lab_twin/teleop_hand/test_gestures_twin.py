# teleop_hand/test_gestures_twin.py
"""Fist-follow and point-to-grasp against the live twin. Headless, real wrist
camera, real GG-CNN.

    MUJOCO_GL=egl python -m teleop_hand.test_gestures_twin

Graded on the consumer side, never on return codes or flags: where the EE
physically is, and WHICH body the gripper holds and how far it rose. A confident
grasp of the wrong cube returns 0 and looks fine in every log.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import numpy as np  # noqa: E402

from teleop_hand import config  # noqa: E402
from teleop_hand.camera_pose import HandCameraPose  # noqa: E402
from teleop_hand.gesture_control import GestureController  # noqa: E402
from teleop_hand.point_grasp import PointGraspExecutor  # noqa: E402
from teleop_hand.pointing import TwinScene  # noqa: E402
from teleop_hand.test_twin import Twin  # noqa: E402

CAM = HandCameraPose.nominal()
BENCH_MM = 750.0


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class GTwin(Twin):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.clutch = False
        self.scene = TwinScene(self.arm)
        self.rx.gestures = GestureController(CAM, self.scene)
        self.rx.point_executor = PointGraspExecutor(self.arm)

    def fist_at(self, world_mm, seconds=1.0):
        self.hand.gesture = "fist"
        self.move(pos=CAM.world_to_cam_point(np.asarray(world_mm) / 1000.0), seconds=seconds)

    def pose_mm(self):
        return self.ee()[0] * 1000.0

    def body_mm(self, name):
        with self.arm.lock:
            return np.array(self.arm.data.body(name).xpos) * 1000.0

    def point(self, origin_m, direction, seconds):
        h = self.hand
        h.gesture = "point"
        h.point_origin = CAM.world_to_cam_point(origin_m)
        h.point_dir = CAM.world_to_cam_dir(_unit(direction))
        h.pos = h.point_origin.copy()
        self.run(seconds)


# ---------------------------------------------------------------------------
def test_fist_follow_keeps_standoff_and_orientation():
    tw = GTwin()
    ee0, rot0 = tw.ee()
    fist = np.array([470.0, -200.0, 1000.0])
    assert np.linalg.norm(ee0 * 1000 - fist) > 200
    speeds = []
    last = [tw.pose_mm(), time.time()]

    def watch():
        p, now = tw.pose_mm(), time.time()
        speeds.append(np.linalg.norm(p - last[0]) / max(now - last[1], 1e-3))
        last[0], last[1] = p, now

    tw.fist_at(fist, seconds=1.0)
    tw.run_sim(4.0, each=watch)
    d = np.linalg.norm(tw.pose_mm() - fist)
    lo, hi = config.FOLLOW_STANDOFF_MM
    assert lo - 12 <= d <= hi + 12, f"EE is {d:.0f} mm from the fist, want {lo:.0f}-{hi:.0f}"
    # it came TO the fist: it is now much nearer than it started
    assert d < np.linalg.norm(ee0 * 1000 - fist) - 60
    # orientation unchanged (locked at engage)
    _, rot1 = tw.ee()
    from teleop_kin.receiver import _rot_angle_deg
    assert _rot_angle_deg(rot0, rot1) < 4.0, _rot_angle_deg(rot0, rot1)
    assert tw.pose_mm()[2] >= config.FOLLOW_Z_MIN_MM - 15
    assert tw.body_mm("red_cube_back")[2] < 800            # nothing got knocked around
    print(f"      EE {d:.0f} mm from the fist; peak EE speed {np.percentile(speeds, 98):.0f} mm/s")


def test_fist_follow_tracks_a_moving_fist():
    tw = GTwin()
    tw.fist_at([470, -200, 1000], seconds=1.0)
    tw.run_sim(3.0)
    for target in ([470, -200, 1000], [560, -120, 1020], [330, -150, 960]):
        tw.fist_at(target, seconds=1.2)
        tw.run_sim(3.0)
        d = np.linalg.norm(tw.pose_mm() - np.array(target))
        lo, hi = config.FOLLOW_STANDOFF_MM
        assert lo - 15 <= d <= hi + 15, f"after moving to {target}: {d:.0f} mm"


def test_releasing_the_fist_stops_the_arm():
    tw = GTwin()
    tw.fist_at([470, -200, 1000], seconds=1.0)
    tw.run_sim(3.0)
    tw.hand.gesture = "none"
    tw.run_sim(0.5)
    p0 = tw.pose_mm()
    tw.move(pos=CAM.world_to_cam_point(np.array([0.2, 0.1, 1.0])), seconds=1.0)   # hand wanders off
    tw.run_sim(2.0)
    assert np.linalg.norm(tw.pose_mm() - p0) < 4.0, "the arm kept moving after the fist opened"


def test_clutch_takes_priority_over_the_gesture():
    tw = GTwin()
    tw.clutch = True                       # operator is teleoperating
    p0 = tw.pose_mm()
    tw.fist_at([470, -200, 1000], seconds=1.0)
    tw.run_sim(2.0)
    # relative teleop: a hand that moved ~0.2 m moves the arm by that, but it must
    # NOT have gone to the fist standoff -- that is the gesture's behaviour
    d = np.linalg.norm(tw.pose_mm() - np.array([470.0, -200.0, 1000.0]))
    assert tw.rx.gesture_out is None or tw.rx.gesture_out.kind == "none"
    assert not (config.FOLLOW_STANDOFF_MM[0] - 12 <= d <= config.FOLLOW_STANDOFF_MM[1] + 12) \
        or np.linalg.norm(tw.pose_mm() - p0) < 300


def test_point_at_a_cube_grasps_that_cube_and_lifts_it():
    tw = GTwin()
    cube = "red_cube_back"
    z0 = tw.body_mm(cube)[2]
    others = {n: tw.body_mm(n) for n in ("red_cube_front", "green_cube", "blue_cube")}
    o = np.array([0.0, -0.12, 1.05])
    d = _unit(tw.body_mm(cube) / 1000 - o)
    from scipy.spatial.transform import Rotation
    d = Rotation.from_rotvec(np.deg2rad(3.0) * np.array([0, 0, 1.0])).apply(d)   # finger noise
    tw.point(o, d, seconds=config.POINT_DWELL_S + 0.5)
    res = tw.rx.last_point_result
    assert res is not None, "the pointing gesture never fired"
    ok, detail = tw.arm.verify_grasp()
    assert res.ok and ok == "held" and cube in detail, (res, detail)
    assert tw.body_mm(cube)[2] > z0 + 80, f"cube rose {tw.body_mm(cube)[2] - z0:.0f} mm"
    for n, p in others.items():
        assert np.linalg.norm(tw.body_mm(n) - p) < 5.0, f"{n} moved"
    print(f"      {res.message}; cube rose {tw.body_mm(cube)[2] - z0:.0f} mm")


def test_point_refuses_when_gg_cnn_proposes_nothing_near_the_target():
    """The seen-is-not-graspable rule: with the detector replaced by one that
    finds nothing, the arm must not descend on the prior."""
    tw = GTwin()
    tw.rx.point_executor = PointGraspExecutor(tw.arm, detector_factory=lambda: type(
        "Nothing", (), {"detect": lambda self, f, top_k=3: []})())
    cube = "red_cube_back"
    o = np.array([0.0, -0.12, 1.05])
    tw.point(o, _unit(tw.body_mm(cube) / 1000 - o), seconds=config.POINT_DWELL_S + 0.5)
    res = tw.rx.last_point_result
    assert res is not None and not res.ok and res.stage == "detect", res
    assert tw.arm.verify_grasp()[0] == "empty"
    assert tw.pose_mm()[2] > tw.body_mm(cube)[2] + 200, "it descended without a grasp"


def test_point_at_empty_bench_hovers_above_it():
    tw = GTwin()
    o = np.array([0.0, -0.15, 1.10])
    aim = np.array([0.30, 0.05, BENCH_MM / 1000])          # bare bench, no object near
    tw.point(o, _unit(aim - o), seconds=config.POINT_DWELL_S + 0.5)
    res = tw.rx.last_point_result
    assert res is not None and res.ok, res
    tw.run_sim(0.5)
    assert tw.arm.verify_grasp()[0] == "empty"
    z = tw.pose_mm()[2]
    assert z >= BENCH_MM + config.POINT_FLOOR_CLEARANCE_MM - 15, f"EE at z={z:.0f}"
    print(f"      hovered at {np.round(tw.pose_mm()).astype(int).tolist()} mm")


def test_point_into_free_space_goes_that_way_safely():
    tw = GTwin()
    o = np.array([0.05, -0.15, 1.00])
    d = _unit([0.35, 1.0, 0.2])
    tw.point(o, d, seconds=config.POINT_DWELL_S + 0.5)
    res = tw.rx.last_point_result
    assert res is not None and res.ok, res
    tw.run_sim(0.5)
    p = tw.pose_mm()
    v = p / 1000 - o
    ang = np.degrees(np.arccos(np.clip(_unit(v) @ d, -1, 1)))
    assert ang < 6.0, f"EE is {ang:.1f} deg off the pointed direction"
    assert p[2] >= BENCH_MM + config.POINT_FLOOR_CLEARANCE_MM - 15
    assert tw.arm.verify_grasp()[0] == "empty"


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
