#!/usr/bin/env python3
"""Deterministic pick/place + push geometry check (no LLM).

SPIKE (branch: spike/xarm6-realistic-meshes). Drives the two geometry-dependent
primitives that CLAUDE.md flags as sensitive to the arm's proportions -- the
grasp+lift pick/place and the fly-over ``push_object`` -- through a scripted
sequence, so the realistic mesh arm can be checked against the primitive one
without the LLM in the loop.

It reports, at the grasp pose, the z-stack (flange / gripper palm / fingertips /
EE site vs the cube) so any gripper-mount misalignment introduced by swapping
link geometry is visible as numbers, not just vibes. Optionally writes render
frames.

Findings on the mesh arm (see envs/assets/xarm6/README.md): both primitives
pass unchanged. Fingertips land inside the cube's z-span, the gripper stack
sits within ~6mm of where it sits on the primitive arm, and physical_outcome()
reports the cube in the bin.

Usage:
    MUJOCO_GL=egl python scripts/pickplace_check.py --scene meshes
    MUJOCO_GL=egl python scripts/pickplace_check.py --scene primitive
    MUJOCO_GL=egl python scripts/pickplace_check.py --scene meshes --frames /tmp/pp
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.mujoco_env import SimXArmAPI  # noqa: E402

SCENES = {"primitive": "envs/lab_scene.xml", "meshes": "envs/lab_scene_meshes.xml"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="meshes", choices=list(SCENES))
    ap.add_argument("--rail", type=float, default=350.0,
                    help="rail position (mm) that brings the base under the cube")
    ap.add_argument("--frames", default=None,
                    help="dir to write render frames into (skipped if omitted)")
    args = ap.parse_args()

    if args.frames:
        os.makedirs(args.frames, exist_ok=True)

    arm = SimXArmAPI(scene_xml=SCENES[args.scene], render=False)
    renderer = mujoco.Renderer(arm.model, height=720, width=1280) if args.frames else None

    def bpos(name):
        with arm.lock:
            mujoco.mj_forward(arm.model, arm.data)
            return arm.data.xpos[arm.model.body(name).id].copy()

    def snap(tag):
        if not renderer:
            return
        from PIL import Image
        cam = mujoco.MjvCamera()
        cam.lookat[:] = bpos("gripper")
        cam.distance, cam.azimuth, cam.elevation = 0.45, 140, -20
        with arm.lock:
            renderer.update_scene(arm.data, camera=cam)
            img = renderer.render()
        Image.fromarray(img).save(os.path.join(args.frames, f"{tag}.png"))

    def zstack():
        with arm.lock:
            mujoco.mj_forward(arm.model, arm.data)
            m, d = arm.model, arm.data
            def gz(n):
                return d.geom_xpos[m.geom(n).id][2] * 1e3
            rows = {
                "link6 origin": d.xpos[m.body("link6").id][2] * 1e3,
                "gripper palm": gz("gripper_geom"),
                "finger_left": gz("finger_left_geom"),
                "finger_right": gz("finger_right_geom"),
                "EE site": d.site_xpos[arm.ee_site][2] * 1e3,
                "cube center": d.xpos[m.body("green_cube").id][2] * 1e3,
            }
        print("  z-stack at grasp (mm):  " +
              "  ".join(f"{k}={v:.1f}" for k, v in rows.items()))

    cube = bpos("green_cube") * 1e3
    binp = bpos("green_bin") * 1e3
    print(f"\n[{args.scene}] green_cube={cube.round(1)}  green_bin={binp.round(1)}")

    # --- pick / place ---
    arm.set_rail_position(args.rail, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 100, roll=180, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 15, roll=180, wait=True)
    zstack(); snap("grasp")
    arm.close_lite6_gripper()
    arm.set_position(cube[0], cube[1], cube[2] + 150, roll=180, wait=True)
    lifted_z = bpos("green_cube")[2] * 1e3
    snap("lift")
    arm.set_position(binp[0], binp[1], binp[2] + 150, roll=180, wait=True)
    arm.set_position(binp[0], binp[1], binp[2] + 60, roll=180, wait=True)
    arm.open_lite6_gripper()
    for _ in range(200):
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)
    final = bpos("green_cube") * 1e3
    snap("released")
    print(f"  pick/place: cube lifted {cube[2]:.0f}->{lifted_z:.0f}mm; "
          f"final xy-dist to bin = {np.linalg.norm(final[:2]-binp[:2]):.1f}mm")

    # --- push (fly-over + weld) ---
    b0 = bpos("blue_cube") * 1e3
    rc = arm.push_object(target_name="blue_cube", to_x_mm=b0[0], to_y_mm=b0[1] + 150,
                         speed_mm_s=80.0)
    for _ in range(200):
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)
    b1 = bpos("blue_cube") * 1e3
    print(f"  push_object rc={rc}: blue_cube {b0[:2].round(0)} -> {b1[:2].round(0)}  "
          f"(target y {b0[1]+150:.0f})")
    print(f"  physical_outcome: {arm.physical_outcome().splitlines()[0][:80]}")


if __name__ == "__main__":
    main()
