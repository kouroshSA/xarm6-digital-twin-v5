"""Recordings carry their room, keep the lab's layout, and replay in the room
they were made in.

    MUJOCO_GL=egl python -m twin_rooms.test_recordings

The replay test is graded by the consumer: the recorded commands are driven
into the model replay_room() rebuilds, and the arm must end where the
recording says it ended. "A model was loaded" would pass for the wrong room.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import recording  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from twin_rooms.recordings import replay_room  # noqa: E402
from twin_rooms.room import PROJECT_ROOT, ROOMS_DIR  # noqa: E402

LAB_XML = (ROOMS_DIR.parent / "envs" / "lab_scene.xml").resolve()
#: The lab's trajectory layout before rooms. Rooms may ADD to it, never change it.
LAB_DATASETS = {"t_wall", "t_sim", "rail_mm", "joints_deg", "ee_pos_mm", "ee_rpy_deg",
                "ctrl", "body_poses", "weld_active", "gripper"}
LAB_ATTRS = {"format_version", "state_hz", "n_samples", "joint_names",
             "actuator_names", "body_names", "eq_names"}


def _room(tmp: Path, objects):
    rd = tmp / "t_room"
    rd.mkdir()
    (rd / "room.json").write_text(json.dumps({
        "id": "t_room", "name": "test", "description": "",
        "environment": {"extract_from": str(LAB_XML), "bodies": ["bench", "observers"]},
        "robots": [{"instance": "arm0", "type": "xarm6_rail"}], "objects": objects}))
    return rd


def _record(arm, tmp: Path, moves, room=True):
    """Record `moves` (rail_mm, (x, y, z)) into tmp/recordings; return the
    session folder."""
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        rec = recording.Recorder(arm.model, arm.data, arm.lock, interface="test",
                                 scene_xml=arm.scene_xml,
                                 room=arm.room if room else None)
        rec.start()
        for rail, xyz in moves:
            arm.set_rail_position(rail, wait=True)
            arm.set_position(*xyz, roll=180, wait=True)
        time.sleep(0.3)
        return rec.stop(kept=True)
    finally:
        os.chdir(cwd)


CUBE = [{"name": "cube", "source": "builtin:cube", "pose": {"pos": [0.2, -0.25, 0.765]}}]
MOVES = [(300, (-50, -200, 950)), (450, (100, -250, 900))]


def test_recording_in_a_room_carries_it_and_replays_there():
    tmp = Path(tempfile.mkdtemp())
    rd = _room(tmp, CUBE)
    arm = SimXArmAPI(room=str(rd), render=False)
    try:
        arm.reset_scene()
        sess = Path(tmp) / _record(arm, tmp, MOVES)
        meta = json.loads((sess / "metadata.json").read_text())
        assert meta["room_id"] == "t_room" and meta["room_hash"] == arm.room.hash, meta
        assert meta["robots"] == [{"instance": "arm0", "type": "xarm6_rail",
                                   "end_effector": "parallel_gripper", "controllable": True}]
        with h5py.File(sess / "trajectory.h5") as f:
            assert f.attrs["room_id"] == "t_room"
            assert list(f.attrs["body_names"]) == ["cube"]        # logical, not obj/cube
            ctrl, t, joints = f["ctrl"][:], f["t_wall"][:], f["joints_deg"][:]
    finally:
        arm.disconnect()

    # Replay: drive the recorded ctrl into the rebuilt room; the arm must end
    # where it ended in the recording.
    meta["room_id"] = str(rd)          # the temp room is addressed by folder
    model, ctx, warnings = replay_room(meta)
    try:
        assert warnings == [], warnings
        d = mujoco.MjData(model)
        acts = [model.actuator(ctx.mjcf(n, "actuator")).id for n in recording.ACT_NAMES]
        jids = [model.joint(ctx.mjcf(n, "joint")).id for n in recording.JOINT_NAMES]
        steps_per_sample = max(1, int(round((t[1] - t[0]) / model.opt.timestep)))
        for k in range(len(ctrl)):
            d.ctrl[acts] = ctrl[k]
            for _ in range(steps_per_sample):
                mujoco.mj_step(model, d)
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, d)
        end = np.rad2deg([d.qpos[model.jnt_qposadr[j]] for j in jids])
        assert np.abs(end - joints[-1]).max() < 2.0, (end, joints[-1])
    finally:
        shutil.rmtree(tmp)


def test_lab_recording_keeps_its_layout_and_names_the_lab():
    """A Recorder given NO room (every pre-rooms caller) still records the
    lab's room, and the trajectory layout is the old one plus attributes."""
    tmp = Path(tempfile.mkdtemp())
    arm = SimXArmAPI("envs/lab_scene.xml", render=False)
    try:
        arm.reset_scene()
        sess = Path(tmp) / _record(arm, tmp, MOVES[:1], room=False)
        meta = json.loads((sess / "metadata.json").read_text())
        assert meta["room_id"] == "lab_a2s2" and meta["room_variant"] == "meshes", meta
        assert meta["scene_xml"] == "envs/lab_scene.xml"
        with h5py.File(sess / "trajectory.h5") as f:
            assert set(f.keys()) - {"frames"} == LAB_DATASETS, set(f.keys())
            assert LAB_ATTRS <= set(f.attrs), set(f.attrs)
            assert set(f.attrs) - LAB_ATTRS <= {"room_id", "room_hash"}
            assert "red_cube_front" in list(f.attrs["body_names"])
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_replay_room_rules():
    lab = {"room_id": "lab_a2s2", "room_hash": "x" * 12, "room_variant": "meshes"}
    _, ctx, w = replay_room(lab)
    assert ctx.room_id == "lab_a2s2" and any("has changed" in s for s in w), w
    _, ctx, w = replay_room({"scene_xml": "envs/lab_scene_primitive.xml"})
    assert ctx.room_id == "lab_a2s2" and ctx.variant == "primitive"
    assert any("made before rooms" in s for s in w), w
    _, ctx, w = replay_room({}, default_room="basic_cubes_bins")
    assert ctx.room_id == "basic_cubes_bins"
    _, ctx, w = replay_room(lab, override="empty_bench")
    assert ctx.room_id == "empty_bench" and w == []


def test_old_recordings_still_replay():
    """Every recording on this machine made before rooms loads its replay
    model, with its actuators resolved."""
    root = PROJECT_ROOT / "recordings"
    olds = [d for d in sorted(root.iterdir()) if (d / "metadata.json").is_file()
            and "room_id" not in json.loads((d / "metadata.json").read_text())] \
        if root.is_dir() else []
    for d in olds[:5]:
        model, ctx, w = replay_room(json.loads((d / "metadata.json").read_text()))
        assert ctx.room_id == "lab_a2s2"
        for n in recording.ACT_NAMES:
            model.actuator(ctx.mjcf(n, "actuator"))
    print(f"    ({len(olds)} pre-rooms recording(s) here; checked {min(5, len(olds))})")


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
    os._exit(1 if failed else 0)
