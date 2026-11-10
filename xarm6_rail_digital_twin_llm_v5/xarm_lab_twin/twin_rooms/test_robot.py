"""robot.json: the real types load, every name is found in every model variant
inside the robot's own bodies, every ref resolves to the value code uses, and
each kind of mistake is refused with a message naming the field.

    python -m twin_rooms.test_robot

The ref tests compare against the CONSUMERS' constants (SimXArmAPI.HOME_*,
the scene's jnt_range), not just "the ref resolved": a ref pointing at the
wrong constant resolves fine.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import mujoco
import numpy as np

from twin_rooms import load_room, list_rooms
from twin_rooms.robot import (ROBOTS_DIR, RoomError, check_against_model,
                              list_robot_types, load_robot_type, parse_robot)

XARM = json.loads((ROBOTS_DIR / "xarm6_rail" / "robot.json").read_text())


def test_types_load_with_every_name_checked():
    assert list_robot_types() == ["maxarm", "xarm6_rail", "xarm6_rail_basic"], list_robot_types()
    for t in list_robot_types():
        load_robot_type(t)          # check_model=True: every variant compiled and checked


def test_every_room_robot_has_a_type_and_its_effector():
    for rid in list_rooms() + ["empty_bench"]:
        for r in load_room(rid).robots:
            rt = load_robot_type(r.type, check_model=False)
            assert (r.end_effector or rt.end_effectors["default"]) in rt.end_effector_names()


def test_home_and_ik_refs_are_the_values_the_sim_uses():
    from sim.mujoco_env import IK_POS_TOL_M, SimXArmAPI
    rt = load_robot_type("xarm6_rail", check_model=False)
    np.testing.assert_allclose(np.deg2rad(rt.value("home.joints_deg")), SimXArmAPI.HOME_JOINTS_RAD)
    assert rt.value("home.rail_mm") / 1000.0 == SimXArmAPI.HOME_RAIL_M
    assert rt.value("ik.pos_tol_m") == IK_POS_TOL_M
    pg = rt.end_effector("parallel_gripper")
    assert pg["closed_ctrl"] > pg["open_ctrl"] and pg["aperture_m"] < rt.end_effector("bio_gripper")["aperture_m"]


def test_rail_mount_matches_the_scene():
    """base_at_rail_zero_mm + rail axis must predict where the base actually is
    at three rail positions."""
    rt = load_robot_type("xarm6_rail", check_model=False)
    m = mujoco.MjModel.from_xml_path(str(rt.model_path()))
    d = mujoco.MjData(m)
    j = m.joint(rt.names["rail_joint"])
    base = m.body(rt.names["base_body"]).id
    lo, hi = m.jnt_range[j.id]
    z0 = np.array(rt.value("rail.base_at_rail_zero_mm")) / 1000.0
    for q in (lo, (lo + hi) / 2, hi):
        d.qpos[j.qposadr[0]] = q
        mujoco.mj_kinematics(m, d)
        want = z0 + q * np.array(rt.rail["axis"])
        assert np.allclose(d.xpos[base], want, atol=1e-6), (q, d.xpos[base], want)


def test_basic_type_is_not_the_lab_arm():
    """The basic scene shares the lab's names, so a name map alone cannot tell
    them apart. Their joint ranges differ; nothing may take one for the other."""
    lab = load_robot_type("xarm6_rail", check_model=False)
    basic = load_robot_type("xarm6_rail_basic", check_model=False)
    ml = mujoco.MjModel.from_xml_path(str(lab.model_path()))
    mb = mujoco.MjModel.from_xml_path(str(basic.model_path()))
    rl = ml.jnt_range[ml.joint("joint3").id]
    rb = mb.jnt_range[mb.joint("joint3").id]
    assert not np.allclose(rl, rb), (rl, rb)
    assert basic.end_effector_names() == ["none"]
    # and the lab's gripper map, applied to the basic scene, is refused
    errs = check_against_model(parse_robot(XARM, lab.dir), mb, "basic")
    assert any("no body 'led_strip_A'" in e for e in errs), errs


# ---------------------------------------------------------------------------
# bad definitions: (description, mutate(d) -> d, expected fragment)
# ---------------------------------------------------------------------------
def _names(**kw):
    return lambda d: {**d, "names": {**d["names"], **kw}}


BAD = [
    ("misspelt joint", _names(arm_joints=["joint1", "joint2", "joint3", "joint4", "joint5", "jiont6"]),
     "names.arm_joints: no joint 'jiont6'"),
    ("actuator/joint order swapped", _names(arm_actuators=["act2", "act1", "act3", "act4", "act5", "act6"]),
     "actuator 'act2' drives 'joint2', not 'joint1'"),
    ("name outside the robot", _names(ee_site="end_effector", gripper_body="bench"),
     "body 'bench' is outside the robot's bodies"),
    ("actuator left unnamed",
     lambda d: {**d, "end_effectors": {**d["end_effectors"], "parallel_gripper":
                {k: v for k, v in d["end_effectors"]["parallel_gripper"].items() if k != "finger_actuators"}}},
     "actuator 'act_finger_l' drives a joint of the robot but robot.json does not name it"),
    ("mount not top-level", lambda d: {**d, "model": {**d["model"], "bodies": ["xarm_base"]}},
     "'xarm_base' is not a top-level body"),
    ("ref to a missing constant", lambda d: {**d, "home": {**d["home"], "rail_mm": {"ref": "arm_backend:HOME_RAIL_CM"}}},
     "ref 'arm_backend:HOME_RAIL_CM' does not resolve"),
    ("default effector undefined", lambda d: {**d, "end_effectors": {**d["end_effectors"], "default": "suction"}},
     "end_effectors.default: 'suction' is not defined here"),
    ("rail joint without actuator", lambda d: {**d, "names": {k: v for k, v in d["names"].items() if k != "rail_actuator"}},
     "give both rail_joint and rail_actuator"),
    ("joint limits copied in", lambda d: {**d, "joint_limits_deg": [[-360, 360]] * 6},
     "unknown field 'joint_limits_deg'"),
    ("missing variant file", lambda d: {**d, "model": {**d["model"], "variants": {"meshes": "nope.xml"}}},
     "file not found: nope.xml"),
]


def _load_mutated(d: dict) -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        rd = tmp / "xarm6_rail"
        rd.mkdir()
        for k, rel in d["model"]["variants"].items():   # keep paths resolvable
            p = (ROBOTS_DIR / "xarm6_rail" / rel).resolve()
            if p.is_file():
                d["model"]["variants"][k] = str(p)
        (rd / "robot.json").write_text(json.dumps(d))
        load_robot_type("xarm6_rail", robots_dir=tmp)
    finally:
        shutil.rmtree(tmp)


def test_bad_definitions_are_refused_with_the_field_named():
    failures = []
    for desc, mutate, fragment in BAD:
        try:
            _load_mutated(mutate(copy.deepcopy(XARM)))
            failures.append(f"{desc}: accepted")
        except RoomError as e:
            if not any(fragment in p for p in e.problems):
                failures.append(f"{desc}: wanted {fragment!r}, got {e.problems}")
    assert not failures, "\n".join(failures)


def test_the_unmodified_definition_passes_the_same_harness():
    """Control: the harness itself accepts the real file."""
    _load_mutated(copy.deepcopy(XARM))


def test_unknown_type_lists_the_types_that_exist():
    try:
        load_robot_type("lite6_rail")
        raise AssertionError("loaded a type that does not exist")
    except RoomError as e:
        assert "xarm6_rail" in e.problems[0], e.problems


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
