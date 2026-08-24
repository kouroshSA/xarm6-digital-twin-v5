"""The "put it back where it started" leg: facts, and grading.

Two defects, both of the shapes CLAUDE.md names, found by watching one task
fail 2 runs in 5 with an identical prompt.

1. Every placement fact this repo produced was "on top of object X". A leg
   that sets something down on OPEN BENCH had no fact at all, so the planner
   invented the height -- 830 three times (works), 807 twice. And 807 is only
   safe if the grasp that picked the object up happened where Layer 1 said;
   gripper_close welds whatever tool-to-object pose exists at that instant,
   so a grasp 18 mm high is frozen as a hold 18 mm low, and the next descent
   drives the cargo into the bench.

2. physical_outcome() had no word for "still holding it", so a run that
   halted mid-air clutching the cube returned "no objects displaced" -- the
   exact string a clean round trip returns. Three completions and two
   failures graded identically.

Run: python test_return_leg.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

from agent.outcome_checker import check_outcome, HELD_MARKER

TASK = ("Place blue_cube on top of red_cube_front. Return to home position. "
        "Pick blue_cube off red_cube_front and return it to its original "
        "position on the bench.")

_fails = 0


def chk(label, got, want=True):
    global _fails
    ok = (got == want)
    _fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + ("" if ok else f"  (got {got!r}, want {want!r})"))


def grader_checks():
    """The veto has to change the VERDICT, not just exist in a vocabulary."""
    print("grader: halting with cargo cannot score as success")
    # "put it back where it started" has no expected substring to look for --
    # the correct end state IS the starting state. That is exactly the case
    # where a missing veto lets a failure through.
    ok, why = check_outcome(TASK, "no objects displaced",
                            fallback_spec=("all", []))
    chk("a clean round trip still passes", ok, True)

    ok, why = check_outcome(TASK, f"blue_cube {HELD_MARKER}",
                            fallback_spec=("all", []))
    chk("halted holding the cube fails", ok, False)
    chk("and the reason names the object", "blue_cube" in why, True)

    # The veto must outrank a satisfied expectation, not just an empty one.
    ok, _ = check_outcome(TASK, f"blue_cube on red_cube_front; blue_cube {HELD_MARKER}",
                          fallback_spec=("all", ["blue_cube on red_cube_front"]))
    chk("veto outranks a matched expectation", ok, False)


def sim_checks():
    try:
        from sim.mujoco_env import SimXArmAPI
        from agent.task_validator import validate_task
        from agent.object_registry import build_default_registry
        arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=False)
    except Exception as exc:                                   # noqa: BLE001
        print(f"sim checks\n  [SKIP] {type(exc).__name__}: {exc}")
        return

    try:
        print("layer 1: the return leg gets measured facts")
        facts = validate_task(TASK, build_default_registry(), arm).facts
        blob = " || ".join(facts)
        chk("blue_cube gets a benchtop release height",
            "to set blue_cube back down on the open benchtop" in blob, True)
        chk("blue_cube gets a re-grasp height off red_cube_front",
            "to pick blue_cube back off red_cube_front" in blob, True)
        # The numbers, not just the sentences. Measured in the sim: re-grasp
        # at 867 leaves the next descent legal, 885 does not; a bench release
        # at 807 lands the cube resting at 780.
        bench = next((f for f in facts if f.startswith(
            "to set blue_cube back down")), "")
        regrasp = next((f for f in facts if f.startswith(
            "to pick blue_cube back off red_cube_front")), "")
        chk("bench release height is 807", "z=807 mm" in bench, True)
        chk("re-grasp height is 867", "z=867 mm" in regrasp, True)

        print("sim: physical_outcome has a word for cargo in the jaws")
        arm.set_rail_position(550, wait=True)
        arm.set_position(x=200, y=-250, z=892, roll=180, pitch=0, yaw=0,
                         speed=80, wait=True)
        arm.set_position(x=200, y=-250, z=807, roll=180, pitch=0, yaw=0,
                         speed=50, wait=True)
        arm.close_lite6_gripper()
        arm.set_position(x=200, y=-250, z=892, roll=180, pitch=0, yaw=0,
                         speed=80, wait=True)
        held = arm.physical_outcome()
        chk("holding it is reported", HELD_MARKER in held, True)
        chk("and it is not 'no objects displaced'",
            held.strip() != "no objects displaced", True)
        arm.open_lite6_gripper(); time.sleep(1.0)
        after = arm.physical_outcome()
        chk("released again, the marker is gone", HELD_MARKER in after, False)

        print("sim: a high re-grasp really does collide later")
        # The producer's claim, exercised through the consumer that refuses.
        arm.reset_scene()
        arm.set_rail_position(550, wait=True)
        for z, sp in ((892, 80), (807, 50)):
            arm.set_position(x=200, y=-250, z=z, roll=180, pitch=0, yaw=0,
                             speed=sp, wait=True)
        arm.close_lite6_gripper()
        arm.set_position(x=200, y=-250, z=892, roll=180, pitch=0, yaw=0,
                         speed=80, wait=True)
        arm.set_rail_position(350, wait=True)
        for z, sp in ((892, 80), (867, 50)):
            arm.set_position(x=0, y=-250, z=z, roll=180, pitch=0, yaw=0,
                             speed=sp, wait=True)
        arm.open_lite6_gripper(); time.sleep(0.6)

        def regrasp_then_descend(grasp_z):
            arm.set_position(x=0, y=-250, z=950, roll=180, pitch=0, yaw=0,
                             speed=80, wait=True)
            arm.set_position(x=0, y=-250, z=grasp_z, roll=180, pitch=0, yaw=0,
                             speed=50, wait=True)
            arm.close_lite6_gripper()
            arm.set_position(x=0, y=-250, z=950, roll=180, pitch=0, yaw=0,
                             speed=80, wait=True)
            arm.set_rail_position(550, wait=True)
            arm.set_position(x=200, y=-250, z=950, roll=180, pitch=0, yaw=0,
                             speed=80, wait=True)
            rc = arm.set_position(x=200, y=-250, z=807, roll=180, pitch=0,
                                  yaw=0, speed=50, wait=True)
            return rc

        chk("the reported 867 leaves the next descent legal",
            regrasp_then_descend(867.0) == 0, True)
    finally:
        try: arm.disconnect()
        except Exception: pass                                 # noqa: BLE001


def main() -> int:
    grader_checks()
    sim_checks()
    print("\n" + ("ALL PASS" if not _fails else f"{_fails} FAILURE(S)"))
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
