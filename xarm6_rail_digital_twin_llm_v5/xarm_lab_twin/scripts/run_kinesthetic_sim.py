#!/usr/bin/env python3
"""Kinesthetic teaching in the digital twin: drag the end effector with the mouse.

    python scripts/run_kinesthetic_sim.py
    python scripts/run_kinesthetic_sim.py --task-label "green cube to bin"
    python scripts/run_kinesthetic_sim.py --no-record --pos-speed 80

A teach handle (orange ball with an RGB axis tripod) sits on the end effector.
Double-click it, then Ctrl + right-drag to move it and Ctrl + left-drag to
rotate it; the arm follows through IK, rate-limited, the way a real arm in
manual mode follows the hand. Arrow keys work the gripper and the rail, Insert
starts and stops a take. Takes are ordinary `Recorder` sessions with
`interface == "kinesthetic_sim"`, so `export_lerobot.py` reads them unchanged.

Wires SimXArmAPI + Recorder + TeachHandle + KinestheticReceiver and runs the
control loop at `teleop_kin.config.FREQUENCY`; modelled on run_spacemouse.py.

THIS LAUNCHES THE INTERACTIVE VIEWER AND MUST NOT FORCE EGL -- same as
run_spacemouse.py. The Recorder's offscreen renders do not show the handle:
it lives in geom group 4, which only the interactive viewer turns on.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from recording import Recorder  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from teleop_kin import config, keys  # noqa: E402
from teleop_kin.handle import TeachHandle  # noqa: E402
from teleop_kin.receiver import KinestheticReceiver  # noqa: E402

#: The geom group the teach handle is drawn in (see envs/lab_scene_primitive.xml).
HANDLE_GROUP = 4


def main() -> int:
    ap = argparse.ArgumentParser(
        description="xArm6 digital-twin kinesthetic teaching (mouse-dragged handle)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servo", choices=["direct", "validated"],
                    default=config.SERVO_MODE,
                    help="direct = IK straight to ctrl (default); validated = "
                         "through set_position + FKValidator (paced, sluggish)")
    ap.add_argument("--no-record", action="store_true",
                    help="do not construct a Recorder at all")
    ap.add_argument("--scene", choices=["meshes", "primitive"], default="meshes",
                    help="which arm to load (default meshes)")
    ap.add_argument("--scene-xml", default=None,
                    help="explicit scene path, overrides --scene")
    ap.add_argument("--pos-speed", type=float, default=config.MAX_POS_SPEED,
                    help=f"max mm/s the arm chases the handle (default {config.MAX_POS_SPEED})")
    ap.add_argument("--rot-speed", type=float, default=config.MAX_ROT_SPEED,
                    help=f"max deg/s (default {config.MAX_ROT_SPEED})")
    ap.add_argument("--rail-speed", type=float, default=config.MAX_RAIL_SPEED,
                    help=f"rail step speed, mm/s (default {config.MAX_RAIL_SPEED})")
    ap.add_argument("--rail-step", type=float, default=config.RAIL_STEP_MM,
                    help=f"Left/Right rail step, mm (default {config.RAIL_STEP_MM})")
    ap.add_argument("--no-gravity-comp", action="store_true",
                    help="disable the sag bias (the tool then settles ~15 mm low)")
    ap.add_argument("--no-home", action="store_true",
                    help="start from the scene's initial pose instead of home")
    ap.add_argument("--task-label", default=config.TASK_LABEL_SIM)
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip the interactive viewer (for headless testing)")
    args = ap.parse_args()

    config.SERVO_MODE = args.servo
    config.RECORD = not args.no_record

    scene = args.scene_xml or (
        "envs/lab_scene.xml" if args.scene == "meshes"
        else "envs/lab_scene_primitive.xml")

    print(f"[KIN] building twin from {scene}")
    arm = SimXArmAPI(scene_xml=scene, render=not args.no_viewer)
    arm.set_viewer_geomgroup(HANDLE_GROUP, True)
    if not args.no_home:
        # The scene starts at all-zeros: the arm straight up, wrist axes
        # aligned -- a singularity, where even a small drag swings joints 4/6
        # past the branch-flip guard and the handle just turns red. Home is
        # the pose verified on the real arm and a sane place to start teaching.
        print("[KIN] moving to home…")
        arm.go_home(wait=True)

    # Created but NOT auto-started; Insert starts/stops takes. `interface` is
    # what tells these sessions apart in recordings/.
    def make_recorder():
        return Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                        interface="kinesthetic_sim", scene_xml=scene,
                        enable_frames=True)

    recorder = make_recorder() if config.RECORD else None

    handle = TeachHandle(arm)
    receiver = KinestheticReceiver(
        arm, handle=handle, recorder=recorder,
        recorder_factory=make_recorder if config.RECORD else None,
        task_label=args.task_label)       # constructor snaps the handle to the EE
    receiver.max_pos_speed = args.pos_speed
    receiver.max_rot_speed = args.rot_speed
    receiver.max_rail_speed = args.rail_speed
    receiver.rail_step_mm = args.rail_step
    receiver.servo_mode = args.servo
    receiver.gravity_comp = not args.no_gravity_comp

    key_handler = keys.make_handler(receiver)
    arm.register_key_handler(key_handler)

    print("\n" + "=" * 68)
    print(f"[KIN] servo={args.servo}  record={'on' if config.RECORD else 'off'}  "
          f"gravity-comp={'off' if args.no_gravity_comp else 'on'}")
    print(f"[KIN] arm chases the handle at <= {args.pos_speed:.0f} mm/s, "
          f"{args.rot_speed:.0f} deg/s   loop {config.FREQUENCY:.0f} Hz")
    print()
    print(keys.describe(args.rail_step))
    print()
    print("[KIN] The handle is hidden from recorded frames (geom group 4).")
    print("[KIN] Arrow keys drive the gripper/rail here, not the body nudge.")
    print("[KIN] Ctrl-C to stop; a take in progress is saved.")
    print("=" * 68 + "\n")

    rc = 0
    last = time.time()
    period = 1.0 / config.FREQUENCY
    seen_viewer = False
    try:
        while True:
            t0 = time.time()
            # The window appears asynchronously, so "no viewer" only means
            # "closed" once one has existed. Closing it ends the session.
            if not args.no_viewer:
                if arm._viewer is not None:
                    seen_viewer = True
                elif seen_viewer:
                    print("[KIN] viewer closed -- stopping.")
                    break
            # Measured dt, not the nominal one: the rate limits are in real
            # units per second and a slow tick must not slow the arm down.
            receiver.tick(max(t0 - last, 1e-4))
            last = t0
            slack = period - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[KIN] shutting down…")
    finally:
        arm.unregister_key_handler(key_handler)
        path = receiver.takes.stop(kept=True)
        if path:
            print(f"[KIN] saved in-progress take -> {path}")
        arm.disconnect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
