"""A drop is not a placement.

Run: ``MUJOCO_GL=egl python test_drop_grading.py``

A live --loop run of "put the cucumber down 150 mm to the right" scored three
successes by letting go 70 mm above the bench beside the cupcake; the loose
grader only saw "cucumber moved (153, -39)mm". At normal/strict stringency
physical_outcome() now reports "<name> dropped from N mm" when the gripper lets
go of an object above the surface under it (not over a bin), and check_outcome
vetoes it. Checked physically: real grasps and releases in the lab scene, and
the verdict the grader gives on the resulting text.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from agent.outcome_checker import check_outcome  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402

MOVE_SPEC = ("any", ["green_cube moved"])


def _pick_green(arm):
    arm.reset_scene()
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        c = arm.data.body("green_cube").xpos * 1000.0
    arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True)
    arm.set_position(c[0], c[1], c[2] + 30, roll=180, wait=True)
    assert arm.close_lite6_gripper() == 0, arm.last_refusal
    grasp_z = float(arm.get_position()[1][2])   # where the cube sat in the jaws
    assert arm.set_position(c[0], c[1], c[2] + 150, roll=180, wait=True) == 0, \
        arm.last_refusal
    return c, grasp_z


def _release_at(arm, x, y, z):
    # A refused move would release wherever the arm stopped and test nothing.
    for zz in (max(z, 900.0), z):
        assert arm.set_position(x, y, zz, roll=180, wait=True) == 0, arm.last_refusal
    arm.open_lite6_gripper()
    time.sleep(1.0)


def test_let_go_in_the_air_is_a_drop(arm):
    c, _ = _pick_green(arm)
    _release_at(arm, c[0] - 120, c[1], c[2] + 100)
    strict, loose = arm.physical_outcome("strict"), arm.physical_outcome("loose")
    assert "green_cube dropped from" in strict, strict
    assert "dropped from" not in loose, loose          # loose vocabulary unchanged
    ok, why = check_outcome("move the green cube", strict, fallback_spec=MOVE_SPEC)
    assert ok is False and "dropped, not placed" in why, why
    ok_loose, _ = check_outcome("move the green cube", loose, fallback_spec=MOVE_SPEC)
    assert ok_loose is True, loose                      # negative control
    print(f"      strict: {[s for s in strict.split('; ') if 'dropped' in s]}")


def test_set_down_then_released_is_a_placement(arm):
    c, grasp_z = _pick_green(arm)
    _release_at(arm, c[0] - 120, c[1], grasp_z)        # the height it was grasped at
    strict = arm.physical_outcome("strict")
    assert "dropped from" not in strict, strict
    ok, why = check_outcome("move the green cube", strict, fallback_spec=MOVE_SPEC)
    assert ok is True, why


def test_released_over_a_bin_is_a_placement(arm):
    _pick_green(arm)
    with arm.lock:
        b = arm.data.body("green_bin").xpos * 1000.0
    _release_at(arm, b[0], b[1], b[2] + 120)
    # "in" is judged at loose (strict's 20 mm bin tolerance is a separate
    # question); the drop fact is judged at strict, where it would appear.
    loose, strict = arm.physical_outcome("loose"), arm.physical_outcome("strict")
    assert "green_cube in green_bin" in loose, loose
    assert "dropped from" not in strict, strict


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    arm = SimXArmAPI(render=False)
    failed = 0
    try:
        for t in TESTS:
            try:
                t(arm)
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
