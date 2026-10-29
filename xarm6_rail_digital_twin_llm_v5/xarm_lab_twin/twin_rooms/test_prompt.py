"""The planner's prompt, registry and graders follow the room.

    MUJOCO_GL=egl python -m twin_rooms.test_prompt

The prompt is captured from LLMBrain.execute_task itself (the API call is
replaced by a function that records the system prompt), so what is tested is
what the planner would receive, not what a helper returns.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-no-network")

import mujoco  # noqa: E402

import agent.llm_brain as lb  # noqa: E402
from agent.outcome_checker import check_outcome, expected_outcome  # noqa: E402
from agent.room_prompt import (SceneView, object_names_text, prompt_parts,  # noqa: E402
                               read_prompt_file, registry_for)
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from twin_rooms.room import ROOMS_DIR  # noqa: E402

LAB_XML = (ROOMS_DIR.parent / "envs" / "lab_scene.xml").resolve()
LAB_SECTIONS = read_prompt_file(ROOMS_DIR / "lab_a2s2" / "agent_prompt.md")


class _Stop(Exception):
    pass


def _system_prompt(arm, task="put the green cube in the green bin"):
    lb.read_lessons_section = lambda **k: "STUB LESSONS"
    lb.read_world_model = lambda **k: types.SimpleNamespace(scene_changed=False)
    lb.render_for_system_prompt = lambda wm, **k: "STUB WORLD MODEL"
    brain = lb.LLMBrain(arm=arm, registry=registry_for(arm), model="haiku")
    seen = {}

    def create(**kw):
        seen["system"] = kw["system"]
        raise _Stop()
    brain.client.messages.create = create
    try:
        brain.execute_task(task)
    except _Stop:
        pass
    return seen["system"]


def _arm(**kw):
    arm = SimXArmAPI(render=False, **kw)
    arm._running = False                     # freeze physics: a stable prompt
    with arm.lock:
        mujoco.mj_resetData(arm.model, arm.data)
        mujoco.mj_forward(arm.model, arm.data)
    return arm


def _room(objects, context=""):
    tmp = Path(tempfile.mkdtemp())
    rd = tmp / "t_room"
    rd.mkdir()
    (rd / "room.json").write_text(json.dumps({
        "id": "t_room", "name": "Test bench", "description": "",
        "environment": {"extract_from": str(LAB_XML), "bodies": ["bench", "observers"]},
        "robots": [{"instance": "arm0", "type": "xarm6_rail"}],
        "objects": objects, "agent": {"context": context}}))
    return tmp, rd


OBJECTS = [
    {"name": "red_cube", "source": "builtin:cube", "type": "cube", "label": "red cube",
     "params": {"rgba": [0.9, 0.1, 0.1, 1]}, "pose": {"pos": [0.1, -0.25, 0.765]}},
    {"name": "blue_bin", "source": "builtin:bin", "type": "bin", "label": "blue bin",
     "pose": {"pos": [0.25, -0.30, 0.75]}},
]


# ---------------------------------------------------------------------------
def test_lab_prompt_carries_the_lab_sections_verbatim():
    arm = _arm(scene_xml="envs/lab_scene.xml")
    try:
        p = _system_prompt(arm)
        for name, text in LAB_SECTIONS.items():
            assert text in p, f"lab section {name!r} missing from the lab prompt"
        assert "mounted on a 700mm linear rail" in p and "- Rail: 0-700mm" in p
        assert "Bench extents: x ∈ [-750, +750] mm, y ∈ [-450, +450] mm." in p
        assert 'Example for "put green cube in green bin"' in p
        assert '"x": 0, "y": -150, "z": 830' in p          # green_cube in the scene
    finally:
        arm.disconnect()


def test_real_arm_prompt_parts_equal_the_lab_twins():
    """An arm with no twin model (RealXArmAPI) gets the lab's parts, as it
    always did -- through a scene view, not a second copy of the logic."""
    arm = _arm(scene_xml="envs/lab_scene.xml")
    try:
        real = types.SimpleNamespace(scene_xml="envs/lab_scene.xml")
        assert prompt_parts(real) == prompt_parts(arm)
        assert isinstance(SceneView(), SceneView)
    finally:
        arm.disconnect()


def test_empty_bench_prompt_has_no_lab_instruments():
    arm = _arm(room="empty_bench")
    try:
        p = _system_prompt(arm, "go home")
        for lab_only in ("place_tube_in_rack", "pcr_open", "Falcon tube heights",
                         "OT-2 deck", "left_tube_rack", "green_cube"):
            assert lab_only not in p, f"{lab_only!r} leaked into the empty bench prompt"
        assert "**This room (Empty bench):** An empty bench" in p
        assert "Movable objects (valid `push_object` targets, and graspable):\n  (none)" in p
    finally:
        arm.disconnect()


def test_room_objects_reach_prompt_registry_and_example():
    tmp, rd = _room(OBJECTS, context="Only a red cube and a blue bin.")
    arm = _arm(room=str(rd))
    try:
        p = _system_prompt(arm, "put the red cube in the blue bin")
        assert 'Example for "put red cube in blue bin"' in p
        assert '"x": 100, "y": -250, "z": 830' in p       # the cube's real xy
        assert '"position_mm": 450' in p                  # rail under x=100
        assert "  red_cube          (   100,   -250,   765)  red cube" in p
        reg = registry_for(arm)
        assert set(reg.objects) == {"red_cube", "blue_bin"}
        assert reg.objects["red_cube"].optimal_rail_mm == 450.0
        assert reg.find("red cube").name == "red_cube"
        names = object_names_text(arm)
        assert "Cubes: red_cube" in names and "Bins: blue_bin" in names, names
        assert "green_cube" not in names
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_grader_uses_the_rooms_names():
    """The regex grader grades a room's real outcome string; without the arm it
    would expect the lab's names. (The outcome string itself is produced by a
    real pick in twin_rooms.test_sim_rooms.)"""
    tmp, rd = _room(OBJECTS)
    arm = _arm(room=str(rd))
    try:
        task = "put the red cube in the blue bin"
        assert expected_outcome(task, arm) == ("all", ["red_cube in blue_bin"])
        assert check_outcome(task, "red_cube in blue_bin", arm=arm)[0] is True
        assert check_outcome(task, "red_cube moved (+12, -3)mm", arm=arm)[0] is False
        # control: the lab's names, which this room does not have
        assert expected_outcome(task) != ("all", ["red_cube in blue_bin"])
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_auto_play_tasks_are_asked_for_the_rooms_objects():
    from agent.prompt_variants import EPISODE_GEN_SYSTEM_PROMPT, episode_gen_system_prompt
    from twin_rooms import load_room, load_room_dir
    assert episode_gen_system_prompt() is EPISODE_GEN_SYSTEM_PROMPT
    assert episode_gen_system_prompt(load_room("lab_a2s2")) is EPISODE_GEN_SYSTEM_PROMPT
    empty = episode_gen_system_prompt(load_room("empty_bench"))
    assert "put the <object>" not in empty and "cube" not in empty.lower(), empty
    assert "- go home" in empty                       # its task_ideas
    tmp, rd = _room(OBJECTS)
    try:
        p = episode_gen_system_prompt(load_room_dir(rd))
        assert "- red cube (cube, movable)" in p and "- blue bin (bin, movable)" in p
        assert "objects: red cube; containers: blue bin" in p, p
        assert "green" not in p
    finally:
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
    os._exit(1 if failed else 0)
