"""Do the vision actions actually do what the planner is told they do?

Run: ``MUJOCO_GL=egl python -m agent.test_vision_dispatch``

No LLM and no API key: plans are hand-written JSON and pushed through
``LLMBrain._run``, which is the same path ``execute_task`` uses once the model
has replied. That keeps the test about the dispatch layer rather than about
whether a model happens to emit the right JSON today.

The claim under test is the one a planner relies on: **"grasp_object with a
description picks up the object that description names."** Not "it returns 0",
and not "it moved somewhere" -- a confident move to the wrong cube returns 0 and
looks fine in the log. So success is checked physically, against the weld the
sim's magnetic gripper creates, and against which body actually rose off the
bench. That is the same standard ``--loop`` grades episodes by.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ANTHROPIC_API_KEY", "not-needed-for-this-test")

import mujoco  # noqa: E402

SCENE = "envs/lab_scene.xml"

# A flange pose that puts the cubes and bins in the wrist camera's view.
LOOKOUT = {"x": 60.0, "y": -200.0, "z": 1180.0,
           "roll": 180.0, "pitch": 0.0, "yaw": 0.0, "speed_mm_s": 100}


def _body_z_mm(arm, name: str) -> float:
    bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, name)
    with arm.lock:
        return float(arm.data.xpos[bid][2]) * 1000.0


def _welded_body(arm):
    """Which body the magnetic gripper currently holds, if any."""
    with arm.lock:
        for name, eqid in arm.weld_eqids.items():
            if arm.data.eq_active[eqid]:
                return name
    return None


def check_grasp_object_picks_the_named_cube(brain, arm) -> list[str]:
    """The headline claim, checked physically rather than by return code."""
    failures = []
    for description, expect in (("the blue cube", "blue_cube"),
                                ("the green cube", "green_cube")):
        arm.reset_scene()
        arm.open_lite6_gripper()
        z_before = _body_z_mm(arm, expect)

        results = brain._run([
            {"action": "move_to", "params": dict(LOOKOUT)},
            {"action": "grasp_object", "params": {"description": description}},
            {"action": "move_to", "params": {**LOOKOUT, "z": 1000.0}},
        ])
        codes = [r["result"] for r in results]
        if any(c != 0 for c in codes):
            failures.append(f"{description!r}: dispatch returned {codes} "
                            f"({getattr(arm, 'last_refusal', '')})")
            continue

        held = _welded_body(arm)
        z_after = _body_z_mm(arm, expect)
        if held != expect:
            failures.append(
                f"{description!r}: gripper is holding {held!r}, not {expect!r}")
        elif z_after - z_before < 20.0:
            failures.append(
                f"{description!r}: {expect} welded but only rose "
                f"{z_after - z_before:.1f} mm; it was not actually lifted")
        else:
            print(f"  {description!r:18s} -> holding {held}, lifted "
                  f"{z_after - z_before:.0f} mm")
    return failures


def check_missing_object_fails_loudly(brain, arm) -> list[str]:
    """An absent object must halt the plan, not return 0 and carry on.

    This is the silent-success shape CLAUDE.md lists first: a command that
    cannot do its job and reports rc=0 lets the rest of the plan run against an
    assumption that never held. ``_run`` halts on any non-zero rc, so returning
    1 here is what stops a "grasp the duck, now put it in the bin" plan from
    cheerfully dropping nothing into a bin.
    """
    arm.reset_scene()
    results = brain._run([
        {"action": "move_to", "params": dict(LOOKOUT)},
        {"action": "grasp_object", "params": {"description": "a rubber duck"}},
        {"action": "gripper_close", "params": {}},
    ])
    codes = [r["result"] for r in results]
    if len(codes) < 2 or codes[1] == 0:
        return [f"grasp_object on an absent object returned {codes}; "
                f"it must be non-zero"]
    if len(results) > 2:
        return [f"plan continued past the failed grasp: {codes}"]
    if not getattr(arm, "last_refusal", ""):
        return ["failed grasp left no refusal reason; the episode loop reads "
                "last_refusal to build its next constraint"]
    print(f"  absent object -> rc={codes[1]}, plan halted, reason recorded")
    return []


def check_locate_object_reaches_the_next_prompt(brain, arm) -> list[str]:
    """locate_object must deliver its finding somewhere a planner will read.

    Within the turn the planner only learns that the command succeeded -- the
    same limitation get_pose documents. The sighting's actual delivery path is
    the registry, which is rendered into the system prompt every turn. If that
    write-back breaks, locate_object becomes a print() and nothing else: a value
    computed correctly and handed to nobody.
    """
    arm.reset_scene()
    before = set(brain.registry.objects)
    results = brain._run([
        {"action": "move_to", "params": dict(LOOKOUT)},
        {"action": "locate_object", "params": {"description": "the red cube"}},
    ])
    if results[-1]["result"] != 0:
        return [f"locate_object failed: {getattr(arm, 'last_refusal', '')}"]

    new = set(brain.registry.objects) - before
    if not new:
        return ["locate_object succeeded but registered nothing; the sighting "
                "cannot reach the next turn's prompt"]

    context = brain.registry.to_llm_context()
    name = next(iter(new))
    if name not in context:
        return [f"registered {name!r} but it does not appear in the rendered "
                f"registry context, so the planner will never see it"]
    print(f"  locate_object registered {name!r} and it renders into the prompt")
    return []


def check_validator_flags_deferred_poses(brain) -> list[str]:
    """The pre-action gate must not certify camera-resolved poses silently."""
    from scripts.validate_plan import DEFERRED

    for action in ("locate_object", "move_to_object", "grasp_object"):
        if action not in DEFERRED:
            return [f"{action} is dispatchable but not in validate_plan.DEFERRED, "
                    f"so the gate will report it as an unrecognised action"]

    # Every vision action must also be a real dispatch entry -- the two lists
    # drifting apart is how the gate ends up describing an action nobody can run.
    handlers = brain._dispatch.__wrapped__ if hasattr(brain._dispatch, "__wrapped__") \
        else None
    del handlers  # the dict is built inside _dispatch; probe by dispatching instead
    for action in DEFERRED:
        rc = brain._dispatch(action, {})          # no description -> refusal
        if rc == -1:
            return [f"{action} is in DEFERRED but _dispatch does not know it"]
    print(f"  all {len(DEFERRED)} vision actions are dispatchable and flagged "
          f"as not pre-checkable")
    return []


def check_gate_classifies_and_announces_deferred() -> list[str]:
    """The gate's DEFERRED branch, exercised without a controller.

    ``scripts/validate_plan.py --self-test`` needs the xArm controller container
    on a socket, so on a workstation the branch that classifies vision actions
    never runs -- and it only matters in ``--mode real``, where nobody wants to
    discover it for the first time. ``PlanValidator.validate`` is called here as
    an unbound method against a stub: a DEFERRED action returns before touching
    ``self.arm`` or any limit, so nothing else is needed.

    Two things are asserted, and the second is the point: that the action is
    accepted, and that it is *reported* rather than silently waved through. A
    gate that prints "plan accepted" over poses it never saw is the failure its
    own module docstring warns about.
    """
    from scripts.validate_plan import PlanValidator

    stub = object.__new__(PlanValidator)
    # validate() resets rail_mm from _initial_rail_mm before the loop; a
    # DEFERRED action needs nothing beyond that.
    stub._initial_rail_mm = 0.0
    stub.rail_mm = 0.0
    plan = [{"action": "grasp_object", "params": {"description": "the blue cube"}},
            {"action": "done", "params": {}}]
    try:
        records = PlanValidator.validate(stub, plan)
    except Exception as exc:  # noqa: BLE001
        return [f"validate() raised on a vision action: {type(exc).__name__}: {exc}"]

    rec = records[0]
    if not rec["ok"]:
        return [f"gate rejected grasp_object: {rec['detail']}"]
    if not rec.get("deferred"):
        return ["grasp_object was accepted but not marked deferred, so the gate "
                "will report the plan as fully checked when it is not"]
    if "CANNOT be pre-checked" not in rec["detail"]:
        return [f"deferred detail does not say the pose was unchecked: "
                f"{rec['detail']!r}"]
    print("  gate accepts grasp_object and flags it as not pre-checked")
    return []


def main() -> int:
    from agent.object_registry import build_default_registry
    from agent.llm_brain import LLMBrain
    from sim.mujoco_env import SimXArmAPI

    arm = SimXArmAPI(SCENE, render=False)
    brain = LLMBrain(arm=arm, registry=build_default_registry(), recorder=None)

    if not brain.vision.available():
        print(f"SKIP  vision unavailable: {brain.vision.unavailable_reason}")
        arm.disconnect()
        return 0

    failures: list[str] = []
    for label, fn in (
        ("grasp_object picks the named cube",
         lambda: check_grasp_object_picks_the_named_cube(brain, arm)),
        ("absent object halts the plan",
         lambda: check_missing_object_fails_loudly(brain, arm)),
        ("locate_object reaches the next prompt",
         lambda: check_locate_object_reaches_the_next_prompt(brain, arm)),
        ("validator flags deferred poses",
         lambda: check_validator_flags_deferred_poses(brain)),
        ("gate classifies and announces deferred",
         check_gate_classifies_and_announces_deferred),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs
        print()

    brain.vision.close()
    arm.disconnect()
    print(f"{len(failures)} failure(s)" if failures
          else "all vision-dispatch checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
