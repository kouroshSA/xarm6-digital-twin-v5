"""Every entry point runs in the built rooms -- headless smoke test.

    MUJOCO_GL=egl python -m twin_rooms.test_entry_points

For each built room: run_task's sim setup (SimXArmAPI from --room, registry,
recorder, the planner's system prompt with the API call stubbed), auto_play's
task-generator prompt, random_play moving the arm, replay resolving the room,
and the manual-control tools (control_panel's SimController moving the rail,
realtime_keyboard's controller, terminal_control's module setup) -- none of
them opens a window. "Runs" means the arm actually moved where it was told.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
import types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-no-network")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from twin_rooms.room import PROJECT_ROOT  # noqa: E402

ROOMS = ("empty_bench", "simready_demo", "lab_a2s2_rebuilt")
BASIC = PROJECT_ROOT.parent.parent / "xarm6_rail_sim_interactive_basic_v5" / "xarm_lab_twin"


def test_llm_project_entry_points():
    import agent.llm_brain as lb
    from agent.prompt_variants import episode_gen_system_prompt
    from agent.room_prompt import registry_for
    from recording import Recorder
    from replay import load_replay_model
    from sim.mujoco_env import SimXArmAPI
    from twin_rooms import load_room
    from twin_rooms.args import add_room_arg, sim_kwargs
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from random_play import execute_random_episode

    lb.read_lessons_section = lambda *a, **k: ""
    lb.read_world_model = lambda *a, **k: types.SimpleNamespace(scene_changed=False)
    lb.render_for_system_prompt = lambda wm, **k: ""
    for room in ROOMS:
        p = argparse.ArgumentParser()
        add_room_arg(p)
        args = p.parse_args(["--room", room])
        arm = SimXArmAPI(**sim_kwargs(args, "envs/lab_scene.xml"), render=False)
        try:
            assert arm.room.room_id == room
            rec = Recorder(arm.model, arm.data, arm.lock, interface="smoke",
                           scene_xml=arm.scene_xml, room=arm.room)
            brain = lb.LLMBrain(arm=arm, registry=registry_for(arm), recorder=rec, model="haiku")
            seen = {}

            def create(**kw):
                seen["system"] = kw["system"]
                raise RuntimeError("stub")
            brain.client.messages.create = create
            try:
                brain.execute_task("go home")
            except RuntimeError:
                pass
            assert "## Scene geometry" in seen["system"], room
            for name in arm.room.objects:                         # every object is in it
                assert name in seen["system"], (room, name)
            assert "Generate N DIVERSE" in episode_gen_system_prompt(load_room(room))
            arm.reset_scene()
            stats = execute_random_episode(arm, _NullRecorder(), 2, np.random.default_rng(3))
            assert stats["n_succeeded"] >= 1, (room, stats)
            model, act_ids = load_replay_model(PROJECT_ROOT / "no_such_session", room)
            assert len(act_ids) == 7
        finally:
            arm.disconnect()


class _NullRecorder:
    def log_command(self, *a, **k):
        pass


def test_manual_control_tools():
    sys.path.insert(0, str(BASIC))                 # the basic project's own modules first
    cwd = os.getcwd()
    os.chdir(BASIC)
    try:
        for mod in ("recording", "rooms_bridge"):
            sys.modules.pop(mod, None)
        cp = importlib.import_module("control_panel")
        rk = importlib.import_module("realtime_keyboard")
        for room in ("basic_cubes_bins",) + ROOMS:
            ctl = cp.SimController(room)
            try:
                ctl.set_all_dofs([500, 0, 0, 0, 0, 0, 0])
                rail = ctl.model.joint(ctl.room.mjcf("rail", "joint"))
                time.sleep(2.0)
                with ctl.lock:
                    q = float(ctl.data.qpos[rail.qposadr[0]])
                assert abs(q - 0.5) < 0.01, (room, q)          # the rail went to 500 mm
            finally:
                ctl.stop()
            kb = rk.RealtimeKeyboardController(room)
            kb._running = False
        sys.argv = ["terminal_control.py", "--room", "empty_bench"]
        sys.modules.pop("terminal_control", None)
        tc = importlib.import_module("terminal_control")
        assert tc.ROOM.ctx.room_id == "empty_bench" and len(tc.act_ids) == 7
        assert tc.PRESETS["home"][0] == 350.0
    finally:
        os.chdir(cwd)


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
