"""Which gripper the arm actually ends up wearing.

This is a CONSUMER test, in the sense CLAUDE.md means: it does not ask
whether `_task_wants_bio_gripper` returns the right answer for a string --
that passed throughout the bug. It asks what attachment is on the arm after
`prepare_for_task` runs on the string an entry point really hands it.

The bug: Layer 1 appends its measurements to the task so the planner can see
grasp heights, and those measurements name every object Layer 1 resolved.
"put it back down at its original position on the bench" resolved
`well_plate_B` (its alias is "the plate on the bench"), so the task string
grew the word "plate", and the gripper keyword scan equipped the bio-gripper
for a task about two 30 mm cubes.

Run: python test_gripper_selection.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.task_validator import (append_scene_facts, strip_scene_facts,
                                  resolve_referents)
from agent.object_registry import build_default_registry
from agent.llm_brain import LLMBrain


class _FakeArm:
    def __init__(self): self.gripper = None
    def set_gripper(self, mode): self.gripper = mode


def _brain():
    b = LLMBrain.__new__(LLMBrain)          # no API key on this path
    b.arm = _FakeArm()
    b._speed_cap_task = None
    b.speed_tier, b.speed_cap_mm_s = "medium", 80
    return b


def _gripper_for(task, facts=()):
    b = _brain()
    b.prepare_for_task(append_scene_facts(task, list(facts)),
                       override_tier="medium")
    return b.arm.gripper


BENCH_TASK = ("Put the blue cube on top of red_cube_front. Then go to the "
              "home position. Then pick the blue cube back up and put it "
              "back down at its original position on the bench.")
PLATE_FACTS = ["well_plate_B: at (550, -200, 762) mm, 85 mm wide",
               "blue_cube: grasp at z=807 mm"]

CASES = [
    # (label, task, facts, expected gripper)
    ("cube task, Layer 1 resolved a plate", BENCH_TASK, PLATE_FACTS, "standard"),
    ("cube task, no facts",                 BENCH_TASK, (),          "standard"),
    ("plate task, with facts",   "move the 96-well plate to the OT-2 deck",
                                             PLATE_FACTS, "bio"),
    ("plate task, no facts",     "move the 96-well plate to the OT-2 deck",
                                             (),          "bio"),
    ("tip box task",             "pick up the tip box", (), "bio"),
]


def main() -> int:
    fails = 0
    print("gripper selection")
    for label, task, facts, want in CASES:
        got = _gripper_for(task, facts)
        ok = got == want
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: want {want}, got {got}")

    print("scene-fact marker round trip")
    for label, got, want in [
        ("strip(append(t, f)) == t",
         strip_scene_facts(append_scene_facts(BENCH_TASK, PLATE_FACTS)),
         BENCH_TASK),
        ("append(t, []) == t", append_scene_facts(BENCH_TASK, []), BENCH_TASK),
        ("strip is idempotent",
         strip_scene_facts(strip_scene_facts(
             append_scene_facts(BENCH_TASK, PLATE_FACTS))), BENCH_TASK),
    ]:
        ok = got == want
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    # The upstream half, recorded rather than asserted away: this resolution
    # is what put "plate" in the task. It is defensible -- "on the bench" IS
    # part of that plate's alias -- so the fix is at the consumer, not here.
    print("upstream: what 'on the bench' resolves to")
    reg = build_default_registry()
    hits = {r.phrase: [o.name for o in r.matches]
            for r in resolve_referents(BENCH_TASK, reg)}
    print(f"  [INFO] {hits}")

    fails += _sim_default_checks()

    print("\n" + ("ALL PASS" if not fails else f"{fails} FAILURE(S)"))
    return 1 if fails else 0


def _sim_default_checks() -> int:
    """What the arm is wearing in the sim, at construction and after a reset.

    set_gripper writes model materials and an instance attribute; a state
    reset touches neither. So a session that ran one plate task kept the bio
    attachment for every episode after it, and the graspability screen kept
    measuring with bio aperture and reach. Skipped (not failed) where MuJoCo
    cannot open a context.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from sim.mujoco_env import SimXArmAPI, DEFAULT_GRIPPER_MODE
        arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=False)
    except Exception as exc:                                   # noqa: BLE001
        print(f"sim default effector\n  [SKIP] {type(exc).__name__}: {exc}")
        return 0

    fails = 0
    print("sim default effector")
    try:
        for label, got, want in [
            ("the default IS the regular gripper", DEFAULT_GRIPPER_MODE,
             "standard"),
            ("a fresh sim wears the default", arm._gripper_mode,
             DEFAULT_GRIPPER_MODE),
        ]:
            ok = got == want
            fails += not ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}: "
                  f"want {want}, got {got}")

        arm.set_gripper("bio")
        ok = arm._gripper_mode == "bio"
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] bio still equips on request")

        arm.reset_scene()
        ok = arm._gripper_mode == DEFAULT_GRIPPER_MODE
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] reset_scene restores the "
              f"default: got {arm._gripper_mode}")

        hidden = all(float(arm.model.mat_rgba[m, 3]) == 0.0
                     for m in arm._bio_mat_ids)
        fails += not hidden
        print(f"  [{'PASS' if hidden else 'FAIL'}] bio geoms hidden again "
              f"after reset")
    finally:
        try: arm.disconnect()
        except Exception: pass          # noqa: BLE001
    return fails


if __name__ == "__main__":
    sys.exit(main())
