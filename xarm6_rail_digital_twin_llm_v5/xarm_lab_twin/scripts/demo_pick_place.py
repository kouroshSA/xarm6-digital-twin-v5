#!/usr/bin/env python3
"""Scripted pick-and-place: cube -> cup, on either backend. No LLM.

The same sequence runs in the twin and on the arm, through the `ArmBackend`
contract, so what you watch in simulation is literally what would execute on
hardware — not an approximation of it. That is the point of rehearsing here.

    python scripts/demo_pick_place.py --mode sim              # with viewer
    python scripts/demo_pick_place.py --mode real --ip <ip> --i-am-at-the-estop

Heights are given in **mm above the benchtop**, which is how they were measured,
and converted to the world frame here. Both backends take world coordinates; the
real one converts to the arm base internally using the live rail position.

Sim caveat: the scene does not model this bench layout, so `--mode sim` first
moves the cube and cup to where they physically are. That is a rehearsal
convenience, not a scene edit — nothing is written to the XML.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from arm_backend import BENCH_TOP_Z_MM

# -- the cell, as measured -----------------------------------------------------
HOME_RAIL_MM = 350.0
CUBE_WORLD_XY = (300.0, -50.0)     # directly below the tip at home
CUP_WORLD_XY = (0.0, -50.0)        # 300 mm toward the rail origin (power end)
PICK_RAIL_MM = 350.0
PLACE_RAIL_MM = 50.0               # 300 mm toward the origin

GRASP_ABOVE_BENCH_MM = 40.0
LIFT_ABOVE_BENCH_MM = 200.0
RELEASE_ABOVE_BENCH_MM = 150.0

# Slow. A person is in the room and this is the first scripted task.
TRANSIT_SPEED = 60.0
APPROACH_SPEED = 30.0
RAIL_SPEED = 40.0


def w(above_bench_mm: float) -> float:
    """mm above the benchtop -> world z."""
    return BENCH_TOP_Z_MM + above_bench_mm


def build_steps():
    """The sequence, as data, so it can be printed before it is executed."""
    cx, cy = CUBE_WORLD_XY
    px, py = CUP_WORLD_XY
    return [
        ("rail to pick", "rail", PICK_RAIL_MM),
        ("above cube", "move", (cx, cy, w(LIFT_ABOVE_BENCH_MM)), TRANSIT_SPEED),
        ("lower to grasp", "move", (cx, cy, w(GRASP_ABOVE_BENCH_MM)), APPROACH_SPEED),
        ("close gripper", "close", None),
        ("lift", "move", (cx, cy, w(LIFT_ABOVE_BENCH_MM)), APPROACH_SPEED),
        ("rail toward the cup", "rail", PLACE_RAIL_MM),
        ("lower over cup", "move", (px, py, w(RELEASE_ABOVE_BENCH_MM)), APPROACH_SPEED),
        ("open gripper", "open", None),
        ("lift clear", "move", (px, py, w(LIFT_ABOVE_BENCH_MM)), APPROACH_SPEED),
        ("home", "home", None),
    ]


def setup_sim_objects(arm):
    """Put the sim's cube and cup where they physically are on the bench.

    The scene models a different layout, so a rehearsal against it would be
    rehearsing the wrong thing. Done with nudge_body at runtime — the scene XML
    is untouched.
    """
    for name, target in (("red_cube_front", CUBE_WORLD_XY),
                         ("translucent_cup", CUP_WORLD_XY)):
        rc, pose = arm.get_body_pose(name)
        if rc != 0 or pose is None:
            print(f"  ! {name} not in the scene; skipping")
            continue
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        arm.nudge_body(name, dx, dy, 0.0, 0.0, 0.0, 0.0)
        rc, now = arm.get_body_pose(name)
        print(f"  moved {name:16} -> ({now[0]:.0f}, {now[1]:.0f}, {now[2]:.0f})")


def run(arm, steps, dry: bool) -> int:
    for i, step in enumerate(steps, 1):
        label, kind = step[0], step[1]
        if kind == "move":
            (x, y, z), speed = step[2], step[3]
            desc = f"move_to ({x:.0f}, {y:.0f}, {z:.0f})  [{z - BENCH_TOP_Z_MM:.0f} mm above bench]"
        elif kind == "rail":
            desc = f"rail -> {step[2]:.0f} mm"
        else:
            desc = kind
        print(f"  {i:2}. {label:22} {desc}")
        if dry:
            continue

        if kind == "move":
            (x, y, z), speed = step[2], step[3]
            rc = arm.set_position(x, y, z, roll=180, pitch=0, yaw=0,
                                  speed=speed, wait=True)
        elif kind == "rail":
            rc = arm.set_rail_position(step[2], speed_mm_s=RAIL_SPEED, wait=True)
        elif kind == "close":
            rc = arm.close_gripper()
            state, detail = arm.verify_grasp()
            print(f"      verify_grasp -> {state} ({detail})")
        elif kind == "open":
            rc = arm.open_gripper()
        elif kind == "home":
            rc = arm.go_home(wait=True)
        else:
            raise ValueError(kind)

        if rc not in (0, None):
            print(f"      ABORT: step returned {rc}")
            return 1
        time.sleep(0.2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sim", "real"], default="sim")
    ap.add_argument("--ip", default="192.168.1.229")
    ap.add_argument("--no-render", action="store_true", help="sim only")
    ap.add_argument("--dry", action="store_true", help="print the sequence, move nothing")
    ap.add_argument("--i-am-at-the-estop", action="store_true",
                    help="required for --mode real")
    args = ap.parse_args()

    steps = build_steps()
    print(f"\nSequence ({args.mode}):")
    if args.mode == "real" and not args.i_am_at_the_estop and not args.dry:
        run(None, steps, dry=True)
        print("\n  Refusing to move a real arm without --i-am-at-the-estop.")
        return 2

    if args.mode == "sim":
        os.environ.setdefault("MUJOCO_GL", "" if not args.no_render else "egl")
        if args.no_render:
            os.environ["MUJOCO_GL"] = "egl"
        elif os.environ.get("MUJOCO_GL") == "egl":
            os.environ.pop("MUJOCO_GL")      # windowed viewer needs a real context
        from sim.mujoco_env import SimXArmAPI
        arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=not args.no_render)
        print("\n  placing objects to match the real bench:")
        setup_sim_objects(arm)
        arm.set_rail_position(HOME_RAIL_MM, wait=True)
        print()
    else:
        from hardware.real_arm import RealXArmAPI
        arm = RealXArmAPI(args.ip, effector="standard", ft_sensor=True)
        print()

    try:
        rc = run(arm, steps, dry=args.dry)
        if not args.dry and hasattr(arm, "physical_outcome"):
            try:
                print(f"\n  physical outcome: {arm.physical_outcome()}")
            except NotImplementedError:
                print("\n  (no physical outcome on hardware — needs perception)")
        return rc
    finally:
        if not args.dry:
            time.sleep(1.0)
            arm.disconnect()


if __name__ == "__main__":
    sys.exit(main())
