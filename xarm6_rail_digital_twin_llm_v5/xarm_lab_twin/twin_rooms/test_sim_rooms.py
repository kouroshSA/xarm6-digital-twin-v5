"""SimXArmAPI in rooms: the lab is unchanged, and a built room can be driven
through the real primitives.

    MUJOCO_GL=egl python -m twin_rooms.test_sim_rooms

Graded by the consumers: a pick in a built room is judged by the weld that
actually holds the cube and how far the cube rose, and physical_outcome's
report must use logical names. A grasp of an object marked "grippable": false
is the negative control.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim.mujoco_env import (BIN_BODIES, CUBE_BODIES, GRIPPABLE_BODIES,  # noqa: E402
                            RACK_TUBE_GROUPS, TUBE_BODY_NAMES, SimXArmAPI)
from twin_rooms.room import ROOMS_DIR  # noqa: E402

LAB_XML = (ROOMS_DIR.parent / "envs" / "lab_scene.xml").resolve()


def _room(objects, robots=None):
    tmp = Path(tempfile.mkdtemp())
    rd = tmp / "t_room"
    rd.mkdir()
    (rd / "room.json").write_text(json.dumps({
        "id": "t_room", "name": "test", "description": "",
        "environment": {"extract_from": str(LAB_XML), "bodies": ["bench", "observers"]},
        "robots": robots or [{"instance": "arm0", "type": "xarm6_rail"}],
        "objects": objects}))
    return tmp, rd


def _pos(arm, name):
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        return arm.data.xpos[arm.model.body(arm._mj(name)).id].copy()


def _held(arm, name):
    with arm.lock:
        return bool(arm.data.eq_active[arm.weld_eqids[name]])


# ---------------------------------------------------------------------------
# the lab, unchanged
# ---------------------------------------------------------------------------
def test_lab_lists_equal_the_constants_in_order():
    for kw in (dict(scene_xml="envs/lab_scene.xml"), dict()):
        arm = SimXArmAPI(render=False, **kw)
        try:
            assert arm.room.room_id == "lab_a2s2"
            assert list(arm.grippable.items()) == list(GRIPPABLE_BODIES.items())
            assert arm.cube_bodies == CUBE_BODIES
            assert arm.bin_bodies == BIN_BODIES
            assert arm.tube_bodies == TUBE_BODY_NAMES
            assert arm.rack_tube_groups == RACK_TUBE_GROUPS
            assert arm._room_fixtures == ()
            assert (arm._rail_x0_mm, arm._rail_limits_mm) == (-350.0, (0.0, 700.0))
            assert arm.HOME_JOINTS_RAD == SimXArmAPI.HOME_JOINTS_RAD
            assert arm.HOME_RAIL_M == SimXArmAPI.HOME_RAIL_M
            assert arm._mj("joint1", "joint") == "joint1"
        finally:
            arm.disconnect()


def test_held_geom_table_covers_every_welded_body():
    """The validator's held-geom table is derived from the welds now. Every
    geom of every welded body must be in it -- including the cup's, which the
    hand-written table had missed."""
    arm = SimXArmAPI("envs/lab_scene.xml", render=False)
    try:
        held = arm.validator.HELD_CUBE_GEOMS
        for body, weld in GRIPPABLE_BODIES.items():
            b = arm.model.body(body).id
            geoms = [arm.model.geom(g).name for g in range(arm.model.ngeom)
                     if arm.model.geom_bodyid[g] == b]
            assert geoms and all(held.get(g) == weld for g in geoms), (body, geoms)
        assert held["translucent_cup_floor"] == "grip_translucent_cup"
        assert len(held) == 68, len(held)
    finally:
        arm.disconnect()


# ---------------------------------------------------------------------------
# a built room, through the real primitives
# ---------------------------------------------------------------------------
OBJECTS = [
    {"name": "red_cube", "source": "builtin:cube", "type": "cube",
     "params": {"rgba": [0.9, 0.1, 0.1, 1]}, "pose": {"pos": [0.0, -0.25, 0.765]}},
    {"name": "blue_bin", "source": "builtin:bin", "type": "bin",
     "pose": {"pos": [0.25, -0.30, 0.75]}},
    {"name": "glued_cube", "source": "builtin:cube", "type": "cube", "grippable": False,
     "pose": {"pos": [-0.25, -0.25, 0.765]}},
]


def _pick(arm, name, lift_mm=100.0):
    c = _pos(arm, name) * 1e3
    arm.set_rail_position(float(np.clip(c[0] + 350.0, 0, 700)), wait=True)
    assert arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True) == 0
    assert arm.set_position(c[0], c[1], c[2] + 30, roll=180, wait=True) == 0
    arm.close_lite6_gripper()
    arm.set_position(c[0], c[1], c[2] + lift_mm, roll=180, wait=True)
    return c, _pos(arm, name) * 1e3


def test_pick_and_place_in_a_built_room():
    tmp, rd = _room(OBJECTS)
    arm = SimXArmAPI(room=str(rd), render=False)
    try:
        assert arm.room.room_id == "t_room"
        assert arm._mj("joint1", "joint") == "arm0/joint1"
        assert set(arm.grippable) == {"red_cube", "blue_bin"}
        assert arm.cube_bodies == ("red_cube",) and arm.bin_bodies == ("blue_bin",)
        arm.reset_scene()
        c0, c1 = _pick(arm, "red_cube")
        assert _held(arm, "red_cube"), "the weld does not hold the cube"
        assert c1[2] - c0[2] > 50.0, f"cube rose only {c1[2] - c0[2]:.1f} mm"
        # carry it over the bin and drop it in
        b = _pos(arm, "blue_bin") * 1e3
        arm.set_rail_position(b[0] + 350.0, wait=True)
        assert arm.set_position(b[0], b[1], b[2] + 120, roll=180, wait=True) == 0
        arm.open_lite6_gripper()
        import time
        time.sleep(1.0)
        notes = arm.physical_outcome()
        assert "red_cube in blue_bin" in notes, notes
        assert "obj/" not in notes and "arm0/" not in notes, notes
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_non_grippable_object_is_not_picked():
    """Negative control for the pick: same motion on an object with no weld."""
    tmp, rd = _room(OBJECTS)
    arm = SimXArmAPI(room=str(rd), render=False)
    try:
        arm.reset_scene()
        assert "glued_cube" not in arm.grippable
        c0, c1 = _pick(arm, "glued_cube")
        assert abs(c1[2] - c0[2]) < 5.0, f"a cube with no weld rose {c1[2] - c0[2]:.1f} mm"
        assert not any(_held(arm, n) for n in arm.grippable)
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_rotated_robot_is_refused_by_name():
    tmp, rd = _room([], robots=[{"instance": "arm0", "type": "xarm6_rail",
                                 "base_pose": {"pos": [0, 0, 0], "euler": [0, 0, 90]}}])
    try:
        SimXArmAPI(room=str(rd), render=False)
        raise AssertionError("accepted a rail that does not run along +x")
    except ValueError as e:
        assert "not world +x" in str(e), e
    finally:
        shutil.rmtree(tmp)


def test_shifted_robot_moves_the_rail_origin():
    """A base_pose translated 200 mm along x must shift the rail origin by
    200 mm, so push_object's world-x -> rail conversion follows the robot."""
    tmp, rd = _room([], robots=[{"instance": "arm0", "type": "xarm6_rail",
                                 "base_pose": {"pos": [0.2, 0, 0]}}])
    arm = SimXArmAPI(room=str(rd), render=False)
    try:
        assert abs(arm._rail_x0_mm - (-150.0)) < 1e-6, arm._rail_x0_mm
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


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
    sys.stdout.flush()
    os._exit(1 if failed else 0)   # sim threads are daemons; skip their teardown
