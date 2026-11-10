"""Tests for the MaxArm digital twin (robots/maxarm, rooms/maxarm_petri,
hardware/maxarm/sim_maxarm.py, scripts/maxarm_lid_twin.py).

    MUJOCO_GL=egl python -m hardware.maxarm.test_sim_maxarm            # all (~2 min)
    MUJOCO_GL=egl python -m hardware.maxarm.test_sim_maxarm --fast     # no room physics

Graded against things other than the twin itself: the generated files
against their generators, the compiled model's wrist against
`kinematics.forward`, the camera against the lab's calibration module, the
settle model against the bench numbers it was fitted to, and the lid task
by where the lid's BODY ends up, never by the commands returning.
"""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from . import kinematics as kin  # noqa: E402
from . import petri, repeatability  # noqa: E402
from .real_maxarm import ARRIVAL_TOL_MM  # noqa: E402
from .sim_maxarm import HOME_JOINTS_DEG, twin_to_vendor, vendor_to_twin, arm_to_local  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL_XML = ROOT / "robots" / "maxarm" / "model" / "maxarm.xml"
ROOM_DIR = ROOT / "rooms" / "maxarm_petri"
PASS, FAIL = "PASS", "FAIL"
FAST = "--fast" in sys.argv

_room = None


def room():
    """One SimXArmAPI for the slow tests (building the room takes a moment)."""
    global _room
    if _room is None:
        sys.path.insert(0, str(ROOT))
        import scripts.maxarm_lid_twin as T
        arm, ma = T.connect(False, seed=0)
        _room = (T, arm, ma, T.Dishes(arm, ma))
    return _room


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
def test_generated_model_is_current():
    """The checked-in MJCF equals what build_model.py renders from its sources."""
    bm = _load(ROOT / "robots" / "maxarm" / "build_model.py", "maxarm_build_model")
    assert MODEL_XML.read_text() == bm.render(), \
        "robots/maxarm/model/maxarm.xml is stale: run python robots/maxarm/build_model.py"


def test_generated_room_is_current():
    mr = _load(ROOM_DIR / "make_room.py", "maxarm_make_room")
    assert json.loads((ROOM_DIR / "room.json").read_text()) == mr.room(), \
        "rooms/maxarm_petri/room.json is stale: run python rooms/maxarm_petri/make_room.py"
    assert (ROOM_DIR / "petri_base.xml").read_text() == petri.base_mjcf()
    assert (ROOM_DIR / "petri_lid.xml").read_text() == petri.lid_mjcf()


def test_robot_type_and_room_validate():
    sys.path.insert(0, str(ROOT))
    from twin_rooms import load_robot_type, load_room
    rt = load_robot_type("maxarm")
    assert rt.value("home.joints_deg") == list(HOME_JOINTS_DEG)
    r = load_room("maxarm_petri")
    assert [x.instance for x in r.robots] == ["arm0", "arm1"]
    assert r.controllable_robot().type == "xarm6_rail"


def test_home_is_joint_zero():
    """q = 0 is the vendor home (pulses 500/500/500), so the position
    actuators hold home from the first step and the arm cannot sprawl."""
    assert HOME_JOINTS_DEG == (0.0, 0.0, 0.0, 0.0, 0.0), HOME_JOINTS_DEG
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p = d.xpos[m.body("nozzle").id] * 1000
    want = arm_to_local(kin.pulses_to_xyz(kin.HOME_PULSES)) * 1000
    assert np.allclose(p, want, atol=1e-6), (p, want)


def test_twin_fk_matches_kinematics():
    """The compiled model's wrist point equals kinematics.forward at every
    reachable pose on a grid; the joint mapping round-trips."""
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    n, worst = 0, 0.0
    for x in range(-200, 201, 40):
        for y in range(-260, -79, 30):
            for z in (56, 70, 100, 150, 200, 224):
                try:
                    sol = kin.solve((x, y, z), kin.TABLE_CONTACT_Z_MM + 3)
                except kin.Unreachable:
                    continue
                q = vendor_to_twin(sol.joints_deg)
                d.qpos[:5] = q
                mujoco.mj_forward(m, d)
                err = np.linalg.norm(d.xpos[m.body("nozzle").id] - arm_to_local(sol.xyz_mm)) * 1000
                worst = max(worst, err)
                n += 1
                back = twin_to_vendor(q)
                assert max(abs(a - b) for a, b in zip(back, sol.joints_deg)) < 1e-9
                for j in range(4):
                    lo, hi = m.jnt_range[j]
                    assert lo - 1e-6 <= q[j] <= hi + 1e-6, f"joint {j} {q[j]} outside {lo}..{hi} at {sol.xyz_mm}"
    assert n > 100, n
    assert worst < 1e-3, f"worst wrist error {worst:.4f} mm over {n} poses"
    # The cup lip is the measured table contact below the wrist point.
    lip = d.site_xpos[m.site("cup_lip").id]
    wrist = d.xpos[m.body("nozzle").id]
    assert abs((wrist[2] - lip[2]) * 1000 - kin.TABLE_CONTACT_Z_MM) < 1e-6


def test_wrist_camera_is_the_calibrated_mount():
    """The camera sits where hand_eye_calib.json puts it (35 mm above the
    wrist, ~59 mm beside it), looks down, and carries the D435's intrinsics
    exactly as the lab's wrist camera does (one source: d435i_calib)."""
    m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = m.camera("cam_maxarm_wrist").id
    rel = d.cam_xpos[cam] - d.xpos[m.body("nozzle").id]
    c2w = np.asarray(json.loads((ROOT / "hardware/maxarm/hand_eye_calib.json").read_text())["cam_to_wrist"])
    t = c2w[:3, 3] * 1000
    # Same height and same distance from the wrist as the calibration; the
    # nozzle axis is L4 ahead of the wrist point, so compare to the wrist.
    wrist_pt = d.xpos[m.body("nozzle").id] + np.array([0.0, kin.L4 / 1000.0, 0.0])
    rel = d.cam_xpos[cam] - wrist_pt
    assert abs(rel[2] * 1000 - t[2]) < 1e-3, (rel * 1000, t)
    assert abs(np.hypot(rel[0], rel[1]) * 1000 - np.hypot(t[0], t[1])) < 1e-3
    look = -d.cam_xmat[cam].reshape(3, 3)[:, 2]          # MuJoCo cameras look along -z
    assert look[2] < -0.99, f"camera does not look down: {look}"
    sys.path.insert(0, str(ROOT))
    from perception import d435i_calib as calib
    lab = mujoco.MjModel.from_xml_path(str(ROOT / "envs" / "lab_scene.xml"))
    lc = lab.camera(calib.COLOR_CAM_NAME).id
    assert np.array_equal(m.cam_resolution[cam], lab.cam_resolution[lc])
    assert np.allclose(m.cam_intrinsic[cam], lab.cam_intrinsic[lc])
    assert np.allclose(m.cam_sensorsize[cam], lab.cam_sensorsize[lc])


def test_settle_model_reproduces_the_bench():
    """The fitted error model gives back the numbers it was fitted to."""
    assert abs(repeatability.sag_z_mm(200, 223)) == repeatability.SAG_Z_MM_AT_VIEW
    assert repeatability.sag_z_mm(55, 200) == 0.0
    assert 4.5 < repeatability.sag_z_mm(150, 223, loaded=True) < 6.5       # measured 5.8
    rep = repeatability.Repeatability(seed=1)
    target = (75.0, -210.0, 200.0)
    errs = np.array([np.subtract(rep.settle(target), target) for _ in range(2000)])
    radial = np.linalg.norm(errs[:, :2], axis=1)
    assert 1.2 < np.median(radial) < 2.0, np.median(radial)
    assert np.percentile(np.linalg.norm(errs - errs.mean(0), axis=1), 95) < 4.5
    assert -6.6 < errs[:, 2].mean() < -5.4, errs[:, 2].mean()         # 6.0 short at the view pose
    assert np.allclose(repeatability.exact().settle(target), target)
    # Seeded: the same seed replays the same misses.
    a = repeatability.Repeatability(seed=7).settle(target)
    b = repeatability.Repeatability(seed=7).settle(target)
    assert a == b


def test_dishes_rest_and_lift_apart():
    """Physics: a seated lid stays put, and lifting the lid does not lift the
    base (the first dish model's radial plates interpenetrated and did)."""
    m = mujoco.MjModel.from_xml_path(str(ROOM_DIR / "build" / "scene.xml")) \
        if (ROOM_DIR / "build" / "scene.xml").is_file() else None
    if m is None:
        sys.path.insert(0, str(ROOT))
        from twin_rooms.builder import build_room
        m, _ = build_room("maxarm_petri")
    d = mujoco.MjData(m)
    lid, base = m.body("obj/dish2_lid").id, m.body("obj/dish2_base").id
    mujoco.mj_forward(m, d)
    p0 = d.xpos[lid].copy()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    drift = (d.xpos[lid] - p0) * 1000
    assert np.linalg.norm(drift) < 1.0, f"lid drifted {drift} mm at rest"
    assert abs((d.xpos[lid][2] - d.xpos[base][2]) * 1000 - petri.SEATED_LID_BOTTOM_MM) < 1.0
    for _ in range(500):
        d.xfrc_applied[lid, 2] = 0.3
        mujoco.mj_step(m, d)
    assert d.xpos[lid][2] > 0.9, "the lid did not lift"
    assert abs(d.xpos[base][2] - 0.75) * 1000 < 3.0, f"the base came up with the lid: {d.xpos[base]}"


# ---- slow: the room's physics with SimXArmAPI --------------------------------
def test_set_position_lands_like_the_bench():
    T, arm, ma, dishes = room()
    ma.snap_home()
    ma._wait_sim(0.3)
    target = (60.0, -200.0, 150.0)
    ma.set_position(*target, speed=150)
    rb = np.array(ma.get_position()[1][:3])
    truth = np.array(ma.world_to_arm(ma.wrist_world()))
    assert np.linalg.norm(rb - truth) < 1.5, f"read-back {rb} vs truth {truth} (pulse quantisation only)"
    miss = np.linalg.norm(rb - target)
    assert 0.3 < miss <= ARRIVAL_TOL_MM, f"a twin move missed by {miss:.2f} mm; the bench misses 0.4-6.4"
    assert rb[2] < target[2], "the settle model sags DOWN"
    # Refusals are the device's (kinematics gate), not the twin's.
    from .real_maxarm import MaxArmError
    for bad in ((0, -200, 40), (0, -30, 100), (0, -400, 100)):
        try:
            ma.set_position(*bad)
            assert False, f"{bad} was not refused"
        except MaxArmError:
            pass


def test_suction_takes_the_lid_only_when_it_touches():
    T, arm, ma, dishes = room()
    ma.snap_home()
    ma.open_gripper(wait=False)
    base_xy = dishes.base_xy_arm("dish3")
    lid0 = dishes.lid_xyz_arm("dish3")
    T.hover(ma, base_xy)
    # 10 mm above the lid: the pump runs on nothing (negative control).
    ma.set_position(base_xy[0], base_xy[1], T.Z_SEATED_TOP + 10, speed=100)
    ma.close_gripper()
    assert ma.held is None, ma.held
    ma.open_gripper(wait=False)
    # On the lid: the weld takes it, and the lid BODY rises with the cup.
    ma.set_position(base_xy[0], base_xy[1], T.Z_SEATED_TOP + T.TOUCH_MM, speed=30)
    ma.close_gripper()
    assert ma.held == "dish3_lid", ma.held
    T.hover(ma, base_xy, speed=100)
    lid1 = dishes.lid_xyz_arm("dish3")
    assert lid1[2] - lid0[2] > 60, f"lid rose only {lid1[2] - lid0[2]:.1f} mm"
    assert abs(dishes.base_xy_arm("dish3")[0] - base_xy[0]) < 3, "the base came up too"
    # Put it straight back (set down, never dropped) and let go.
    T.place(ma, base_xy, T.Z_SEATED_TOP, "dish3 lid")
    ma._wait_sim(0.5)
    st = dishes.lid_state("dish3")
    assert st["verdict"] in ("seated", "perched"), st


def test_lid_sequence_ends_seated():
    """The consumer test: lid off, aside and back on from memory, with the
    measured scatter, ends with the lid SEATED on its base -- re-seating if
    the first set-down perched it. Graded on the lid body's pose."""
    T, arm, ma, dishes = room()
    ma.snap_home()
    ma.open_gripper(wait=False)
    d = "dish1"
    memory = {d: {"base_xy": dishes.base_xy_arm(d), "aside_xy": petri.aside_spots_arm()[0]}}
    assert T.lid_off(ma, dishes, d, memory), "lid off / aside failed"
    flat = dishes.lid_flat_on_table(d)
    assert flat["flat"] and math.dist(flat["xy"], memory[d]["aside_xy"]) < T.ASIDE_TOL_MM, flat
    assert dishes.lid_state(d)["verdict"] == "off"
    verdict = T.lid_on(ma, dishes, d, memory)
    ma.go_home(speed=150)
    st = dishes.lid_state(d)
    assert verdict == "seated" and st["verdict"] == "seated", (verdict, st)
    assert ma.held is None


TESTS = [
    test_generated_model_is_current,
    test_generated_room_is_current,
    test_robot_type_and_room_validate,
    test_home_is_joint_zero,
    test_twin_fk_matches_kinematics,
    test_wrist_camera_is_the_calibrated_mount,
    test_settle_model_reproduces_the_bench,
    test_dishes_rest_and_lift_apart,
]
SLOW = [
    test_set_position_lands_like_the_bench,
    test_suction_takes_the_lid_only_when_it_touches,
    test_lid_sequence_ends_seated,
]


def main() -> int:
    tests = TESTS if FAST else TESTS + SLOW
    lines, failures = [], 0
    for t in tests:
        try:
            t()
            lines.append(f"{PASS}  {t.__name__}")
        except Exception:        # noqa: BLE001
            failures += 1
            lines.append(f"{FAIL}  {t.__name__}\n" + "".join("        " + l for l in
                                                                traceback.format_exc().splitlines(True)))
    for line in lines:
        print("  " + line)
    print(f"\n  {len(tests) - failures} passed, {failures} failed")
    if _room is not None:
        _room[1].disconnect()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
