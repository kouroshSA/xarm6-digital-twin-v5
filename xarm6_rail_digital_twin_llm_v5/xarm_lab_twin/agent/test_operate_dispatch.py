"""operate_part as a planner action: the planner can open a door.

Run: ``MUJOCO_GL=egl python -m agent.test_operate_dispatch``  (no model call)

Graded physically, through the planner's own dispatch (LLMBrain._run): which
way the hinge actually stands after the arm is home, not the return code.
operate_part used to report "1.00 of its travel" and return 0 while the arm,
going home, swept through the door and left it 2 deg open.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

from agent.llm_brain import LLMBrain  # noqa: E402
from agent.room_prompt import prompt_parts, registry_for  # noqa: E402
from scripts.validate_plan import DEFERRED  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402


def test_the_room_offers_its_parts(arm, brain):
    cmds = prompt_parts(arm)["room_commands"]
    assert "operate_part" in cmds, cmds
    assert "incubator.Door (hinged, 105 deg)" in cmds, cmds
    assert "incubator.LowerShelf (sliding, 80 mm)" in cmds, cmds
    assert "toaster.lever" not in cmds, cmds        # spring-return: cannot stay open


def test_the_lab_is_not_offered_it():
    lab = SimXArmAPI(render=False)
    try:
        assert "operate_part" not in prompt_parts(lab)["room_commands"]
    finally:
        lab.disconnect()


def test_the_gate_knows_it_is_deferred(arm, brain):
    assert "operate_part" in DEFERRED


def test_planner_opens_the_incubator_door(arm, brain):
    arm.reset_scene()
    results = brain._run([
        {"action": "operate_part", "params": {"part": "incubator.Door", "opening": 0.8}},
    ])
    assert [r["result"] for r in results] == [0], (results, arm.last_refusal)
    reached = arm.part_opening("incubator.Door")
    assert abs(reached - 0.8) < 0.08, f"door at {reached:.2f} of its travel, asked 0.8"
    outcome = arm.physical_outcome()
    assert "incubator Door open" in outcome, outcome
    print(f"      door at {reached:.2f}; outcome: {outcome}")


def test_an_unknown_part_halts_the_plan(arm, brain):
    arm.reset_scene()
    results = brain._run([
        {"action": "operate_part", "params": {"part": "incubator.Roof"}},
        {"action": "operate_part", "params": {"part": "incubator.Door"}},
    ])
    assert [r["result"] for r in results] == [1], results     # halted, door not tried
    assert "parts with handles" in arm.last_refusal, arm.last_refusal
    assert arm.part_opening("incubator.Door") < 0.05


def test_the_room_offers_put_in(arm, brain):
    cmds = prompt_parts(arm)["room_commands"]
    assert "- put_in" in cmds, cmds
    assert "incubator.UpperShelf" in cmds and "cabinet" in cmds, cmds
    assert "toaster" not in cmds.split("Containers in this room:")[1], cmds   # a lever is not a door
    assert "put_in" in DEFERRED


def test_put_in_refuses_behind_a_closed_door(arm, brain):
    """Nothing moves, and the refusal names the door to open."""
    arm.reset_scene()
    time.sleep(1.5)                 # the cucumber rolls ~13 mm to rest after a reset
    before = arm.get_body_pose("cucumber")[1][:3]
    results = brain._run([{"action": "put_in",
                           "params": {"object": "cucumber", "container": "incubator"}}])
    assert [r["result"] for r in results] == [1], (results, arm.last_refusal)
    assert "closed" in arm.last_refusal and "Door" in arm.last_refusal, arm.last_refusal
    after = arm.get_body_pose("cucumber")[1][:3]
    assert max(abs(a - b) for a, b in zip(before, after)) < 5.0, (before, after)


def test_planner_puts_the_cupcake_in_and_shuts_the_door(arm, brain):
    """Graded on where things END: the cupcake inside, standing (it went in on a
    side grasp turned only about the vertical), and the door pushed shut by the
    paw."""
    arm.reset_scene()
    results = brain._run([
        {"action": "operate_part", "params": {"part": "incubator.Door", "opening": 1.0}},
        {"action": "put_in", "params": {"object": "cupcake", "container": "incubator"}},
        {"action": "operate_part", "params": {"part": "incubator.Door", "opening": 0.0}},
    ])
    assert [r["result"] for r in results] == [0, 0, 0], (results, arm.last_refusal)
    outcome = arm.physical_outcome()
    assert "cupcake in incubator" in outcome, outcome
    assert "incubator Door open" not in outcome, outcome
    with arm.lock:
        up = arm.data.xmat[arm.cube_bids["cupcake"]].reshape(3, 3)[2, 2]
    assert up > 0.95, f"cupcake tipped (its z axis . world z = {up:.2f})"
    print(f"      outcome: {outcome}")


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    arm = SimXArmAPI(render=False, room="simready_demo")
    brain = LLMBrain(arm=arm, registry=registry_for(arm))
    failed = 0
    try:
        for t in TESTS:
            try:
                t() if t.__code__.co_argcount == 0 else t(arm, brain)
                print(f"PASS  {t.__name__}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {t.__name__}")
                traceback.print_exc()
    finally:
        arm.disconnect()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.stdout.flush()
    os._exit(1 if failed else 0)
