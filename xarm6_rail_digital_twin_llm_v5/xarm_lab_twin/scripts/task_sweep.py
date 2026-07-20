#!/usr/bin/env python3
"""Broad task sweep across graspable object classes, mesh vs primitive (no LLM).

SPIKE (branch: spike/xarm6-realistic-meshes). Extends the cube-only geometry
pass (scripts/pickplace_check.py) to every graspable object CLASS in the scene,
so we know the realistic mesh arm handles the full task suite, not just cubes.
Each task runs on a FRESH scene (deterministic start) and reports PASS/FAIL from
weld state + physical position deltas.

Tasks: cube pick/place, tube -> rack (place_tube_in_rack), well-plate with the
bio-gripper, and a bin push. Result (60 targets... n/a; single scripted run):
all four PASS on --scene meshes, identical-or-better vs primitive. See
envs/assets/xarm6/README.md.

Usage:
    MUJOCO_GL=egl python scripts/task_sweep.py            # both scenes
    MUJOCO_GL=egl python scripts/task_sweep.py --scene meshes
"""
import os, sys, argparse, traceback
import numpy as np, mujoco
sys.path.insert(0, os.getcwd())
from sim.mujoco_env import SimXArmAPI

SCENES = {"primitive": "envs/lab_scene.xml", "meshes": "envs/lab_scene_meshes.xml"}


def bpos(arm, n):
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        return arm.data.xpos[arm.model.body(n).id].copy() * 1000.0


def weld_active(arm, name):
    with arm.lock:
        return bool(arm.data.eq_active[arm.weld_eqids[name]])


def settle(arm, n=200):
    for _ in range(n):
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)


def task_cube_pickplace(arm):
    cube = bpos(arm, "green_cube"); binp = bpos(arm, "green_bin")
    arm.set_rail_position(350.0, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 100, roll=180, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 15, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = weld_active(arm, "green_cube")
    arm.set_position(cube[0], cube[1], cube[2] + 150, roll=180, wait=True)
    arm.set_position(binp[0], binp[1], binp[2] + 150, roll=180, wait=True)
    arm.set_position(binp[0], binp[1], binp[2] + 60, roll=180, wait=True)
    arm.open_lite6_gripper(); settle(arm)
    final = bpos(arm, "green_cube")
    d = np.linalg.norm(final[:2] - binp[:2])
    ok = grasped and d < 80 and "green_cube in green_bin" in arm.physical_outcome()
    return ok, f"grasped={grasped} final_xy_to_bin={d:.0f}mm"


def task_tube_to_rack(arm):
    tube = bpos(arm, "tube_L1")
    arm.set_rail_position(0.0, wait=True)
    r1 = arm.set_position(tube[0], tube[1], tube[2] + 120, roll=180, wait=True)
    r2 = arm.set_position(tube[0], tube[1], tube[2] + 20, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = any(weld_active(arm, f"tube_{s}") for s in ["L1","L2","L3","R1","R2","R3"])
    arm.set_position(tube[0], tube[1], tube[2] + 150, roll=180, wait=True)
    rc = arm.place_tube_in_rack(rack_name="right_tube_rack")
    settle(arm)
    return (grasped and rc == 0), f"reach_rc=({r1},{r2}) grasped={grasped} place_rc={rc}"


def task_wellplate_bio(arm):
    arm.set_gripper("bio")
    plate = bpos(arm, "well_plate_B")
    # rail toward the plate (bench, right side)
    arm.set_rail_position(700.0, wait=True)
    r1 = arm.set_position(plate[0], plate[1], plate[2] + 120, roll=180, wait=True)
    r2 = arm.set_position(plate[0], plate[1], plate[2] + 20, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = weld_active(arm, "well_plate_B")
    arm.set_position(plate[0], plate[1], plate[2] + 120, roll=180, wait=True)
    lifted = bpos(arm, "well_plate_B")[2]
    arm.open_lite6_gripper(); settle(arm)
    arm.set_gripper("standard")
    return (grasped and lifted > plate[2] + 40), \
        f"reach_rc=({r1},{r2}) grasped={grasped} lifted_z={lifted:.0f}(was {plate[2]:.0f})"


def task_bin_push(arm):
    b0 = bpos(arm, "blue_bin")
    rc = arm.push_object(target_name="blue_bin", to_x_mm=b0[0], to_y_mm=b0[1] - 120,
                         speed_mm_s=80.0)
    settle(arm)
    b1 = bpos(arm, "blue_bin")
    moved = np.linalg.norm(b1[:2] - b0[:2])
    return (rc == 0 and moved > 60), f"push_rc={rc} bin_moved={moved:.0f}mm"


TASKS = [
    ("cube pick/place", task_cube_pickplace),
    ("tube -> rack", task_tube_to_rack),
    ("well-plate (bio)", task_wellplate_bio),
    ("bin push", task_bin_push),
]


def run(scene):
    print(f"\n===== {scene} =====")
    for label, fn in TASKS:
        arm = SimXArmAPI(scene_xml=SCENES[scene], render=False)  # fresh scene per task
        try:
            ok, detail = fn(arm)
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:<18} {detail}")
        except Exception as e:
            print(f"  [ERR ] {label:<18} {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="both", choices=["primitive","meshes","both"])
    a = ap.parse_args()
    for s in (["primitive","meshes"] if a.scene=="both" else [a.scene]):
        run(s)
