"""Consumer tests for pre-existing SimXArmAPI bugs found in the rooms recon.

1. push_object's grasp mode tested for "red_cube", which left the scene when
   the red cube was split into front/back, so both red cubes were pushed by
   the flyover path; and its absolute grasp height closed on air for any cube
   resting at 780 mm (blue too). Graded on whether the gripper HOLDS the cube
   after the close -- rc and final position are no evidence, the closing snap
   teleports the object either way -- with a bin as the flyover control.
2. _grasp_rejections read `arm_body_ids`, which was never assigned, so the
   "arm is penetrating something" guard fell back to an empty set and refused
   a grasp over ANY deep contact in the scene. Graded on whether the weld
   actually holds the cube: an unrelated overlap far from the arm must not
   block the grasp, and an object buried in the arm still must.

3. The grader's colour templates named red_cube / red_bin, neither of which
   exists, so a correct red-cube placement graded as a failure.

Run: MUJOCO_GL=egl python test_scene_fixes.py
"""
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim.mujoco_env import SimXArmAPI  # noqa: E402

SCENE = "envs/lab_scene.xml"


def _arm():
    arm = SimXArmAPI(SCENE, render=False)
    arm.reset_scene()
    return arm


def _pos(arm, name):
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        return arm.data.xpos[arm.model.body(name).id].copy()


def _put(arm, name, xyz_m):
    """Teleport a free body (no velocity) and settle the kinematics."""
    with arm.lock:
        j = arm.model.body(name).jntadr[0]
        a = arm.model.jnt_qposadr[j]
        arm.data.qpos[a:a + 3] = xyz_m
        arm.data.qvel[arm.model.jnt_dofadr[j]:arm.model.jnt_dofadr[j] + 6] = 0
        mujoco.mj_forward(arm.model, arm.data)


def _held(arm, name):
    with arm.lock:
        return bool(arm.data.eq_active[arm.weld_eqids[name]])


def _hover_over_green_cube(arm):
    c = _pos(arm, "green_cube") * 1e3
    arm.set_rail_position(350, wait=True)
    arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True)
    arm.set_position(c[0], c[1], c[2] + 15, roll=180, wait=True)
    return c


def _count_closes(arm, name=None):
    """Spy on close_lite6_gripper; records whether `name` was HELD after each
    close (a close that finds nothing in reach still counts as a call)."""
    calls = []
    real = arm.close_lite6_gripper

    def spy():
        rc = real()
        calls.append(_held(arm, name) if name else rc)
        return rc
    arm.close_lite6_gripper = spy
    return calls


def _push_grasps(name):
    """push_object on a cube must take the grasp path AND actually hold it:
    rc == 0 and the final position prove nothing, the snap teleports anyway."""
    arm = _arm()
    held = _count_closes(arm, name)
    p0 = _pos(arm, name) * 1e3
    rc = arm.push_object(name, to_x_mm=p0[0] - 100, to_y_mm=p0[1], speed_mm_s=80.0)
    assert rc == 0, rc
    assert held, f"push_object never closed the gripper on {name} (flyover path)"
    assert held[0], f"push_object closed on {name} but did not hold it"


def test_red_cube_push_grasps_the_cube():
    _push_grasps("red_cube_front")


def test_blue_cube_push_grasps_the_cube():
    # blue_cube was always on the grasp path; its 780 mm rest height made the
    # old absolute grasp z close on air.
    _push_grasps("blue_cube")


def test_green_cube_push_grasps_the_cube():
    _push_grasps("green_cube")


def test_bin_push_still_uses_flyover():
    arm = _arm()
    closes = _count_closes(arm)
    p0 = _pos(arm, "blue_bin") * 1e3
    arm.push_object("blue_bin", to_x_mm=p0[0], to_y_mm=p0[1] + 100, speed_mm_s=80.0)
    assert not closes, "bins are meant to take the flyover path"


def test_unrelated_overlap_does_not_block_a_grasp():
    arm = _arm()
    # blue_cube buried 10 mm into red_cube_back: a deep contact nowhere near
    # the arm. Before the fix this alone refused every grasp in the scene.
    _put(arm, "blue_cube", _pos(arm, "red_cube_back") + [0.02, 0.0, 0.0])
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        deep = [c for c in arm.data.contact[:arm.data.ncon] if c.dist < -0.004]
    assert deep, "test setup: expected a deep cube/cube contact"
    _hover_over_green_cube(arm)
    _put(arm, "blue_cube", _pos(arm, "red_cube_back") + [0.02, 0.0, 0.0])
    arm.close_lite6_gripper()
    assert _held(arm, "green_cube"), "grasp refused over a contact that does not involve the arm"


def test_object_buried_in_the_arm_still_blocks_a_grasp():
    arm = _arm()
    _hover_over_green_cube(arm)
    # Bury blue_cube in link6, well above the EE, so the nearest graspable is
    # still green_cube but the arm is penetrating something solid.
    _put(arm, "blue_cube", _pos(arm, "link6"))
    with arm.lock:
        arm_bids = set(arm.arm_body_ids)
        blue = arm.model.body("blue_cube").id
        hits = [c for c in arm.data.contact[:arm.data.ncon]
                if c.dist < -0.004 and blue in (arm.model.geom_bodyid[c.geom1],
                                                arm.model.geom_bodyid[c.geom2])
                and (arm.model.geom_bodyid[c.geom1] in arm_bids
                     or arm.model.geom_bodyid[c.geom2] in arm_bids)]
    assert hits, "test setup: blue_cube does not collide with the arm at link6"
    arm.close_lite6_gripper()
    assert not _held(arm, "green_cube"), "grasp accepted with an object buried in the arm"


def _pick_place(arm, name, dest):
    c = _pos(arm, name) * 1e3
    d = _pos(arm, dest) * 1e3
    arm.set_rail_position(350, wait=True)
    arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True)
    assert arm.set_position(c[0], c[1], c[2] + 30, roll=180, wait=True) == 0
    arm.close_lite6_gripper()
    assert _held(arm, name), f"test setup: {name} not grasped"
    arm.set_position(c[0], c[1], c[2] + 150, roll=180, wait=True)
    arm.set_position(d[0], d[1], d[2] + 150, roll=180, wait=True)
    arm.set_position(d[0], d[1], d[2] + 60, roll=180, wait=True)
    arm.open_lite6_gripper()
    for _ in range(300):
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)


def test_red_cube_task_grades_on_the_real_outcome():
    """The grader used to expect "red_cube in green_bin"; the scene's outcome
    says "red_cube_front in green_bin", so a correct red-cube placement graded
    as a FAILURE. Graded through check_outcome on a real physical_outcome()."""
    from agent.outcome_checker import check_outcome
    arm = _arm()
    _pick_place(arm, "red_cube_front", "green_bin")
    phys = arm.physical_outcome()
    assert "red_cube_front in green_bin" in phys, phys
    ok, why = check_outcome("put the red cube in the green bin", phys)
    assert ok is True, why
    ok, why = check_outcome("put the front red cube in the green bin", phys)
    assert ok is True, why
    ok, why = check_outcome("put the back red cube in the green bin", phys)
    assert ok is False, why            # the other red cube: still a failure
    ok, why = check_outcome("put the green cube in the red bin", phys)
    assert ok is None, why             # no red bin: unknowable, not "failed"


TESTS = [test_red_cube_task_grades_on_the_real_outcome, test_red_cube_push_grasps_the_cube, test_blue_cube_push_grasps_the_cube,
         test_green_cube_push_grasps_the_cube, test_bin_push_still_uses_flyover,
         test_unrelated_overlap_does_not_block_a_grasp,
         test_object_buried_in_the_arm_still_blocks_a_grasp]

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
