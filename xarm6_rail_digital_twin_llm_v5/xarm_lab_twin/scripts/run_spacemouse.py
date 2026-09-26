#!/usr/bin/env python3
"""SpaceMouse teleoperation of the xArm6 digital twin.

    python scripts/run_spacemouse.py --device pro
    python scripts/run_spacemouse.py --device compact --servo validated
    python scripts/run_spacemouse.py --no-record --pos-speed 80

Wires SimXArmAPI + Recorder + SpaceMouseDevice + SpaceMouseReceiver and runs
the control loop at `config.FREQUENCY`.

UNLIKE `run_vr.py`, THIS LAUNCHES THE INTERACTIVE VIEWER AND MUST NOT FORCE
EGL. The VR runner is headless on purpose -- the headset is the display, and
it needs `MUJOCO_GL=egl` for its offscreen renderer. Here the operator is
sitting at the monitor watching the arm, so the desktop viewer is the whole
point and an EGL context would fight it for the GL device.

Requires spacenavd and libspnav; see docs/spacemouse_setup.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from recording import Recorder  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from teleop_sm import buttons, config  # noqa: E402
from teleop_sm.device import SpaceMouseDevice, warn_if_multiple_devices  # noqa: E402
from teleop_sm.receiver import SpaceMouseReceiver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="xArm6 digital-twin SpaceMouse teleop",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=sorted(buttons.BUTTON_MAPS),
                    default="pro",
                    help="which SpaceMouse model is plugged in (button maps "
                         "differ and the protocol does not say)")
    ap.add_argument("--servo", choices=["direct", "validated"],
                    default=config.SERVO_MODE,
                    help="direct = IK straight to ctrl; validated = through "
                         "set_position + FKValidator")
    ap.add_argument("--no-record", action="store_true",
                    help="do not construct a Recorder at all")
    ap.add_argument("--scene", choices=["meshes", "primitive"], default="meshes",
                    help="which arm to load (default meshes)")
    ap.add_argument("--scene-xml", default=None,
                    help="explicit scene path, overrides --scene")
    ap.add_argument("--pos-speed", type=float, default=config.MAX_POS_SPEED,
                    help=f"mm/s at full deflection (default {config.MAX_POS_SPEED})")
    ap.add_argument("--rot-speed", type=float, default=config.MAX_ROT_SPEED,
                    help=f"deg/s at full deflection (default {config.MAX_ROT_SPEED})")
    ap.add_argument("--rail-speed", type=float, default=config.MAX_RAIL_SPEED,
                    help=f"mm/s at full deflection in rail mode "
                         f"(default {config.MAX_RAIL_SPEED})")
    ap.add_argument("--task-label", default=config.TASK_LABEL)
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip the interactive viewer (for headless testing)")
    args = ap.parse_args()

    config.SERVO_MODE = args.servo
    config.RECORD = not args.no_record

    scene = args.scene_xml or (
        "envs/lab_scene.xml" if args.scene == "meshes"
        else "envs/lab_scene_primitive.xml")

    warning = warn_if_multiple_devices()
    if warning:
        print(f"[SM] WARNING: {warning}")

    print(f"[SM] building twin from {scene}")
    arm = SimXArmAPI(scene_xml=scene, render=not args.no_viewer)

    # Created but NOT auto-started; RECORD_TOGGLE starts/stops takes.
    # `interface` is what distinguishes these sessions from vr_teleop ones in
    # recordings/; export_lerobot.py reads task_label generically, so nothing
    # downstream needs to change to consume them.
    def make_recorder():
        return Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                        interface="spacemouse_teleop", scene_xml=scene,
                        enable_frames=True)

    recorder = make_recorder() if config.RECORD else None

    receiver = SpaceMouseReceiver(
        arm, recorder=recorder,
        recorder_factory=make_recorder if config.RECORD else None,
        task_label=args.task_label, model=args.device)
    receiver.max_pos_speed = args.pos_speed
    receiver.max_rot_speed = args.rot_speed
    receiver.max_rail_speed = args.rail_speed

    device = SpaceMouseDevice()
    device.start()
    # Give the reader thread a moment to fail loudly if the daemon is absent,
    # rather than discovering it as silence once the loop is running.
    time.sleep(0.4)
    if not device.is_alive():
        print("[SM] the device thread died on startup — see the error above.")
        arm.disconnect()
        return 2

    print("\n" + "=" * 68)
    print(f"[SM] device={args.device}  servo={args.servo}  "
          f"record={'on' if config.RECORD else 'off'}")
    print(f"[SM] speeds: {args.pos_speed:.0f} mm/s  {args.rot_speed:.0f} deg/s  "
          f"rail {args.rail_speed:.0f} mm/s")
    print(f"[SM] loop {config.FREQUENCY:.0f} Hz   deadzone {config.DEADZONE:.2f}")
    print()
    print(buttons.describe(args.device))
    print()
    print("[SM] AXIS MAPPING IS PROVISIONAL: push the cap forward and confirm")
    print("[SM] the TCP moves +X. If it does not, fix AXIS_PERMUTATION /")
    print("[SM] AXIS_SIGNS in teleop_sm/config.py — see the probe script.")
    print("[SM] Ctrl-C to stop.")
    print("=" * 68 + "\n")

    dt = 1.0 / config.FREQUENCY
    warned_idle = False
    try:
        while True:
            t0 = time.time()

            state = device.state_twin()
            edges = device.pending_edges()
            held = {i for i in range(32) if device.button_held(i)}
            receiver.tick(state, edges, held, dt)

            # One-shot nudge if the device has never produced an event: the
            # loop otherwise looks healthy while doing nothing at all.
            if (not warned_idle and device.event_count == 0
                    and time.time() - t0 > 0 and device.seconds_since_event() > 8.0):
                print("[SM] no events yet — is the puck being moved? "
                      "Check: systemctl status spacenavd")
                warned_idle = True

            slack = dt - (time.time() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[SM] shutting down…")
    finally:
        device.stop()
        if receiver.rec is not None and receiver.rec.is_recording:
            path = receiver.rec.stop(kept=True, task_label=args.task_label)
            print(f"[SM] saved in-progress take -> {path}")
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
