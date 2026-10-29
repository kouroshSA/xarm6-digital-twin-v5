"""room.json validation: the real rooms load, and each kind of mistake is
refused with a message that names the field.

    python -m twin_rooms.test_schema

A bad-room test asserts on the MESSAGE, not just that loading raised: an
error that does not say which field is wrong is half a validator.
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

from twin_rooms import (DEFAULT_ROOM, ROOMS_DIR, RoomError, list_rooms, load_room,
                        load_room_dir)

LAB_XML = (ROOMS_DIR.parent / "envs" / "lab_scene.xml").resolve()
TEMPLATE = json.loads((ROOMS_DIR / "_templates" / "empty_bench" / "room.json").read_text())


# ---------------------------------------------------------------------------
# good rooms
# ---------------------------------------------------------------------------
def test_real_rooms_load():
    assert set(list_rooms()) >= {"lab_a2s2", "basic_cubes_bins"}, list_rooms()
    assert "_templates/empty_bench" in list_rooms(include_templates=True)
    for rid in ("lab_a2s2", "basic_cubes_bins", "empty_bench"):
        load_room(rid)


def test_default_room_is_the_lab_file_unchanged():
    """No --room must mean the very file every tool loaded before rooms."""
    room = load_room(DEFAULT_ROOM)
    assert room.is_passthrough
    assert room.scene_path() == LAB_XML, room.scene_path()
    assert room.scene_path("primitive").name == "lab_scene_primitive.xml"


def test_passthrough_objects_exist_in_their_scene():
    """Every 'scene' object must be a body in the scene it describes, and every
    free body of the scene must be listed (a room that silently omits an
    object would leave it out of the generated prompt)."""
    for rid in ("lab_a2s2", "basic_cubes_bins"):
        room = load_room(rid)
        m = mujoco.MjModel.from_xml_path(str(room.scene_path()))
        bodies = {m.body(i).name for i in range(m.nbody)}
        free = {m.body(m.jnt_bodyid[j]).name for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        named = {o.name for o in room.objects}
        assert named <= bodies, (rid, sorted(named - bodies))
        assert free <= named, (rid, "free bodies missing from room.json", sorted(free - named))
        for o in room.objects:
            assert o.dynamic == (o.name in free), (rid, o.name, "dynamic flag disagrees with the scene")


def test_hash_follows_the_scene_file():
    """A passthrough room's hash must change when the scene XML behind it does,
    not only when room.json does."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "envs").mkdir()
        shutil.copy(LAB_XML, tmp / "envs" / "lab_scene.xml")
        d = json.loads((ROOMS_DIR / "lab_a2s2" / "room.json").read_text())
        d["scene"]["variants"] = {"meshes": "../../envs/lab_scene.xml"}
        rd = tmp / "rooms" / "lab_a2s2"
        rd.mkdir(parents=True)
        (rd / "room.json").write_text(json.dumps(d))
        h0 = load_room_dir(rd).definition_hash()
        with open(tmp / "envs" / "lab_scene.xml", "a") as f:
            f.write("<!-- edited -->\n")
        assert load_room_dir(rd).definition_hash() != h0
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# bad rooms: (description, mutate(d) -> d, expected message fragment)
# ---------------------------------------------------------------------------
def _obj(**kw):
    o = {"name": "red_cube", "source": "builtin:cube", "pose": {"pos": [0, -0.2, 0.78]}}
    o.update(kw)
    return o


BAD = [
    ("missing required field", lambda d: d.pop("robots") and d, "missing required field 'robots'"),
    ("typo in a field name", lambda d: {**d, "objetcs": []}, "unknown field 'objetcs'"),
    ("id differs from folder", lambda d: {**d, "id": "other"}, "must equal the folder name"),
    ("bad id characters", lambda d: {**d, "id": "Empty-Bench"}, "'Empty-Bench' is not valid"),
    ("both scene and environment",
     lambda d: {**d, "scene": {"default": "a", "variants": {"a": "../../../envs/lab_scene.xml"}}},
     "exactly one of 'scene'"),
    ("environment file missing", lambda d: {**d, "environment": "nope.xml"}, "file not found: nope.xml"),
    ("pos with two numbers", lambda d: {**d, "objects": [_obj(pose={"pos": [0, 1]})]},
     "objects[0].pose.pos: expected at least 3 items, got 2"),
    ("unknown source kind", lambda d: {**d, "objects": [_obj(source="builtin:teapot")]},
     "objects[0].source"),
    ("object without a pose", lambda d: {**d, "objects": [{"name": "c", "source": "builtin:cube"}]},
     "missing 'pose'"),
    ("unknown builtin parameter",
     lambda d: {**d, "objects": [_obj(params={"colour": [1, 0, 0, 1]})]},
     "unknown parameter 'colour' for cube"),
    ("duplicate object names", lambda d: {**d, "objects": [_obj(), _obj()]},
     "name 'red_cube' is used more than once"),
    ("scene object in a composed room", lambda d: {**d, "objects": [{"name": "x", "source": "scene"}]},
     "only valid in a passthrough"),
    ("mjcf file missing", lambda d: {**d, "objects": [_obj(source="mjcf:parts/tray.xml")]},
     "file not found: parts/tray.xml"),
    ("two controllable robots",
     lambda d: {**d, "robots": d["robots"] + [{**d["robots"][0], "instance": "arm1"}]},
     "exactly one robot must be controllable"),
    ("two robots, one controllable, same instance",
     lambda d: {**d, "robots": d["robots"] + [{**d["robots"][0], "controllable": False}]},
     "instance 'arm0' is used more than once"),
    ("unknown end effector",
     lambda d: {**d, "robots": [{**d["robots"][0], "end_effector": "suction"}]},
     "'suction' is not an end effector of xarm6_rail"),
    ("inverted workspace", lambda d: {**d, "workspace_bounds": {"min": [0, 0, 1.0], "max": [1, 1, 0.8]}},
     "z min 1.0 is not below max 0.8"),
    ("wrong type", lambda d: {**d, "objects": {"red_cube": {}}}, "objects: expected array"),
]


def test_bad_rooms_are_refused_with_the_field_named():
    failures = []
    for desc, mutate, fragment in BAD:
        tmp = Path(tempfile.mkdtemp())
        try:
            rd = tmp / "_templates" / "empty_bench"
            rd.mkdir(parents=True)
            d = mutate(copy.deepcopy(TEMPLATE))
            # keep extract_from resolvable from the temp location
            env = d.get("environment")
            if isinstance(env, dict):
                env["extract_from"] = str(LAB_XML)
            (rd / "room.json").write_text(json.dumps(d))
            try:
                load_room_dir(rd)
                failures.append(f"{desc}: accepted")
            except RoomError as e:
                if not any(fragment in p for p in e.problems):
                    failures.append(f"{desc}: wanted {fragment!r}, got {e.problems}")
        finally:
            shutil.rmtree(tmp)
    assert not failures, "\n".join(failures)


def test_the_unmodified_template_passes_the_same_harness():
    """Control for the test above: with no mutation the harness accepts the
    template, so the refusals are caused by the mutations, not by the harness."""
    tmp = Path(tempfile.mkdtemp())
    try:
        rd = tmp / "_templates" / "empty_bench"
        rd.mkdir(parents=True)
        d = copy.deepcopy(TEMPLATE)
        d["environment"]["extract_from"] = str(LAB_XML)
        (rd / "room.json").write_text(json.dumps(d))
        load_room_dir(rd)
    finally:
        shutil.rmtree(tmp)


def test_invalid_json_is_reported_with_its_position():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "room.json").write_text('{"id": "x",}')
        try:
            load_room_dir(tmp)
            raise AssertionError("accepted invalid JSON")
        except RoomError as e:
            assert "not valid JSON" in e.problems[0] and "line 1" in e.problems[0], e.problems
    finally:
        shutil.rmtree(tmp)


def test_unknown_room_lists_the_rooms_that_exist():
    try:
        load_room("no_such_room")
        raise AssertionError("loaded a room that does not exist")
    except RoomError as e:
        assert "lab_a2s2" in e.problems[0], e.problems


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
