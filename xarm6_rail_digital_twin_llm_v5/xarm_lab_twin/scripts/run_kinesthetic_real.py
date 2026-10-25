#!/usr/bin/env python3
"""Kinesthetic teaching on the REAL arm (manual / teach mode), mirrored into
the twin and recorded.

    python scripts/run_kinesthetic_real.py --ip 192.168.1.xxx
    python scripts/run_kinesthetic_real.py --ip 192.168.1.xxx --robot lite6
    python scripts/run_kinesthetic_real.py --fake          # rehearse, no robot

    HOLD  Space    manual mode: the arm releases and can be moved by hand.
                   Let go of Space and it locks where it is.
    Up / Down      open / close the gripper (real and twin)
    Insert         start / stop a take        Delete   discard take
    Ctrl-C         quit (always leaves manual mode first)

READ docs/kinesthetic_teaching.md BEFORE THE FIRST SESSION. In short: set the
payload and TCP for the fitted gripper first (manual mode compensates gravity
with them; wrong values make the arm sag or drift when released), keep the
e-stop in reach, one person at the arm.

The twin is a live mirror: every tick the controller's joint angles are
written into its ctrl (the exported action) and qpos (the state), and the
ordinary Recorder records the twin. Sessions carry
`interface == "kinesthetic_real"`. Frames are OFF -- the twin's cameras would
show the twin, not the bench; real D435 frames are a later phase.

SAFETY. Nothing here commands arm motion. Every exit path -- key release,
Ctrl-C, SIGTERM, an exception, a controller error -- runs set_mode(0) +
set_state(0).

KEYS COME FROM pynput, NOT THE VIEWER: hold-to-teach needs key RELEASE
events, and MuJoCo's key_callback delivers presses only. pynput listens
globally (whatever window has focus). Without pynput / an X display, pass
--teach-toggle: Space in the viewer window then toggles manual mode instead.
"""
from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import time

sys.path.insert(0, os.getcwd())

from recording import Recorder  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from teleop_kin import config  # noqa: E402
from teleop_kin.real_teach import (FakeBackend, FakeController, ManualMode,  # noqa: E402
                                   RealGripper, RealMirror, TeachHold,
                                   controller_mode)
from teleop_kin.takes import TakeControl  # noqa: E402

#: GLFW keycodes for the viewer fallback (--teach-toggle) and for swallowing
#: the keys pynput already handles, so they do not also nudge a body.
GLFW_SPACE = 32


def _build_backend(args):
    """RealXArmAPI (or the fake), plus open/close callables for the gripper."""
    if args.fake:
        ctl = FakeController(needs_detection_param=(args.robot == "lite6"),
                             state_file=args.fake_state_file)
        be = FakeBackend(ctl, rail_mm=None if args.no_rail else 350.0)
        return be, be.open_gripper, be.close_gripper
    # Imported here: it needs xarm-python-sdk, which the sim-only env lacks.
    from hardware.real_arm import RealXArmAPI
    if args.robot == "lite6":
        # The Lite 6's gripper is not one of RealXArmAPI's effectors, and it
        # has no rail and no F/T sensor. Drive its gripper through the SDK's
        # own Lite 6 calls on the raw handle.
        be = RealXArmAPI(args.ip, effector="none", ft_sensor=False,
                         assume_rail_mm=0.0)
        return be, be.arm.open_lite6_gripper, be.arm.close_lite6_gripper
    be = RealXArmAPI(args.ip, effector=args.effector, ft_sensor=False,
                     assume_rail_mm=0.0 if args.no_rail else None)
    if args.effector == "none":
        def _no_gripper():
            print("[TEACH] no effector fitted (--effector none)")
            return 1
        return be, _no_gripper, _no_gripper
    return (be, lambda: be.open_gripper(wait=True),
            lambda: be.close_gripper(wait=True))


def _start_keyboard(args, hold: TeachHold, actions: "queue.Queue[str]"):
    """pynput listener -> hold-to-teach + action queue. Returns the listener,
    or None if pynput / a display is unavailable."""
    try:
        from pynput import keyboard
    except Exception as e:                       # noqa: BLE001
        print(f"[TEACH] pynput unavailable ({e}); use --teach-toggle")
        return None
    teach_key = keyboard.Key.space
    keymap = {keyboard.Key.up: "open", keyboard.Key.down: "close",
              keyboard.Key.insert: "record", keyboard.Key.delete: "discard"}
    held: set = set()

    def on_press(key):
        if key == teach_key:
            hold.press()
            return
        action = keymap.get(key)
        # Autorepeat delivers a held key over and over; act on the first.
        if action is not None and key not in held:
            held.add(key)
            actions.put(action)

    def on_release(key):
        if key == teach_key:
            hold.release()
        held.discard(key)

    try:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        listener.wait()
    except Exception as e:                       # noqa: BLE001
        print(f"[TEACH] keyboard listener failed ({e}); use --teach-toggle")
        return None
    return listener


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kinesthetic teaching on the real xArm (manual mode + twin mirror)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", choices=["xarm6", "lite6"], default="xarm6")
    ap.add_argument("--ip", default=None, help="controller IP (required unless --fake)")
    ap.add_argument("--effector", choices=["standard", "bio", "vacuum", "none"],
                    default="standard", help="xArm 6 tool (default standard gripper)")
    ap.add_argument("--no-rail", action="store_true",
                    help="no linear rail fitted / do not read it")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--task-label", default=config.TASK_LABEL_REAL)
    ap.add_argument("--hz", type=float, default=config.FREQUENCY,
                    help=f"mirror rate (default {config.FREQUENCY:.0f}, the Recorder's)")
    ap.add_argument("--scene", choices=["meshes", "primitive"], default="meshes")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--teach-toggle", action="store_true",
                    help="Space in the VIEWER toggles manual mode (no pynput needed)")
    ap.add_argument("--fake", action="store_true",
                    help="simulated controller: rehearse the workflow, no robot")
    ap.add_argument("--fake-state-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fake-hold", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.fake and not args.ip:
        ap.error("--ip is required (or --fake to rehearse)")

    # Turn SIGTERM into an exception so the finally: below runs. Without this
    # `kill` would end the process with the arm still in manual mode.
    def _on_sigterm(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _on_sigterm)

    scene = "envs/lab_scene.xml" if args.scene == "meshes" else "envs/lab_scene_primitive.xml"
    print(f"[TEACH] building twin mirror from {scene}")
    arm = SimXArmAPI(scene_xml=scene, render=not args.no_viewer)

    backend = manual = gripper = listener = None
    takes = None
    rc = 0
    try:
        backend, open_fn, close_fn = _build_backend(args)
        ctl = backend.arm
        manual = ManualMode(ctl)
        manual.safe_exit()                   # start from a known mode 0

        use_rail = (not args.no_rail and args.robot == "xarm6"
                    and bool(getattr(backend, "_rail_trustworthy", False)))
        if not args.no_rail and args.robot == "xarm6" and not use_rail:
            print("[TEACH] rail not enabled/homed: the twin's rail stays where "
                  "it is (joint mirroring is unaffected)")
        mirror = RealMirror(arm, backend, use_rail=use_rail)
        mirror.sync()                        # twin onto the real pose before anything

        def _log_gripper(close: bool, rc_: int) -> None:
            if takes is not None:
                takes.log_command("gripper_done", {"closed": close, "rc": rc_})
        gripper = RealGripper(arm, open_fn, close_fn,
                              on_done=_log_gripper)

        def make_recorder():
            return Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                            interface="kinesthetic_real", scene_xml=scene,
                            enable_frames=False, gripper_fn=gripper.state)

        def _on_start(rec):
            # Context a reader of the session needs: which robot, and which
            # gripper source. A Lite 6 take mirrored into the xArm 6 twin has
            # valid JOINT data but meaningless twin EE/body poses.
            rec.log_command("kinesthetic_real", {
                "robot": args.robot, "effector": args.effector,
                "gripper_source": "real gripper command, flipped on completion",
                "rail_mirrored": use_rail, "fake": args.fake})

        takes = TakeControl(make_recorder() if not args.no_record else None,
                            make_recorder if not args.no_record else None,
                            args.task_label, log=lambda m: print(f"[TEACH] {m}"),
                            on_start=_on_start)

        hold = TeachHold()
        actions: "queue.Queue[str]" = queue.Queue()
        if args.teach_toggle:
            listener = None
        else:
            listener = _start_keyboard(args, hold, actions)
            if listener is None:
                raise SystemExit("[TEACH] no keyboard; rerun with --teach-toggle")

        viewer_map = {265: "open", 264: "close", 260: "record", 261: "discard"}

        def viewer_keys(keycode: int) -> bool:
            if args.teach_toggle:
                if keycode == GLFW_SPACE:
                    hold.toggle()
                    return True
                if keycode in viewer_map:
                    actions.put(viewer_map[keycode])
                    return True
                return False
            # pynput already has these; swallow them so they do not ALSO
            # nudge whatever body is selected in the viewer.
            return keycode in viewer_map or keycode == GLFW_SPACE
        arm.register_key_handler(viewer_keys)

        if args.fake_hold:
            hold.press()

        tcp_load = getattr(ctl, "tcp_load", None)
        print("\n" + "=" * 68)
        print(f"[TEACH] robot={args.robot}  {'FAKE controller' if args.fake else args.ip}  "
              f"rail={'mirrored' if use_rail else 'not mirrored'}  "
              f"record={'off' if args.no_record else 'on'}  {args.hz:.0f} Hz")
        if tcp_load is not None:
            print(f"[TEACH] controller payload/TCP load: {tcp_load}  "
                  f"<- must match the fitted gripper")
        if args.robot == "lite6":
            print("[TEACH] NOTE: the twin is an xArm 6. Lite 6 joint data is "
                  "recorded faithfully; the twin's EE pose is not the Lite 6's.")
        print("[TEACH]   " + ("Space (viewer) toggles manual mode" if args.teach_toggle
                              else "HOLD Space for manual mode; release to lock"))
        print("[TEACH]   Up/Down gripper   Insert take start/stop   Delete discard")
        print("[TEACH]   Ctrl-C quits (manual mode is always left first)")
        print("=" * 68 + "\n")

        period = 1.0 / args.hz
        last_err_check = 0.0
        while True:
            t0 = time.monotonic()
            ev = hold.poll()
            if ev == "enter":
                ok = manual.enter()
                takes.log_command("teach_mode", {"on": ok, "fallback": manual.used_fallback})
                print("[TEACH] MANUAL MODE -- arm released" if ok
                      else "[TEACH] could not enter manual mode")
                if not ok:
                    hold.engaged = False
            elif ev == "exit":
                manual.exit()
                takes.log_command("teach_mode", {"on": False})
                print(f"[TEACH] locked (mode {controller_mode(ctl)})")

            while True:
                try:
                    a = actions.get_nowait()
                except queue.Empty:
                    break
                if a in ("open", "close"):
                    takes.log_command("gripper", {"closed": a == "close"})
                    gripper.command(close=(a == "close"))
                elif a == "record":
                    takes.toggle()
                elif a == "discard":
                    takes.discard()

            mirror.sync()

            # A controller error while released: lock it. Checked at 5 Hz.
            if t0 - last_err_check > 0.2:
                last_err_check = t0
                if getattr(ctl, "has_err_warn", False) and manual.active:
                    print(f"[TEACH] controller error/warning "
                          f"(code {getattr(ctl, 'error_code', '?')}) -- leaving manual mode")
                    manual.safe_exit()
                    hold.engaged = False

            slack = period - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print("\n[TEACH] shutting down…")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        if isinstance(e.code, str):
            print(e.code)
        print(f"\n[TEACH] exiting ({rc})")
    except Exception as e:                       # noqa: BLE001
        rc = 1
        print(f"\n[TEACH] ERROR: {e!r}")
    finally:
        # A second Ctrl-C / SIGTERM must not interrupt the cleanup itself.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # FIRST, before anything that could fail: leave manual mode.
        if manual is not None:
            manual.safe_exit()
            print("[TEACH] controller returned to mode 0 / state 0")
        if listener is not None:
            listener.stop()
        if gripper is not None:
            gripper.wait_idle(timeout=3.0)
            gripper.stop()
        if takes is not None:
            path = takes.stop(kept=True)
            if path:
                print(f"[TEACH] saved in-progress take -> {path}")
        if backend is not None:
            try:
                backend.disconnect()
            except Exception:                    # noqa: BLE001
                pass
        arm.disconnect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
