#!/usr/bin/env python3
"""Hand-tracking teleop of the digital twin (arm + rail + gripper).

    # terminal 1 (xarm6sim_hand env)
    python scripts/run_hand_tracker.py --show
    # terminal 2 (xarm6sim env)
    python scripts/run_handteleop_sim.py
    python scripts/run_handteleop_sim.py --orient yaw --rail edge --operator-view
    python scripts/run_handteleop_sim.py --map absolute --abs-calib calib.json

HOLD Right Ctrl (the clutch) and move your hand: the arm follows. Pinch
thumb and index to close the gripper. Release the clutch and nothing moves.
Insert starts/stops a take (an ordinary Recorder session with
`interface == "handtrack_sim"`), Delete discards, Home re-homes, End resets.

The target is drawn with the teach-handle marker (geom group 4, never in
recorded frames): orange tracking, grey released, red refused, blue precision.

THIS LAUNCHES THE INTERACTIVE VIEWER AND MUST NOT FORCE EGL (same as
run_spacemouse.py / run_kinesthetic_sim.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from recording import Recorder  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from teleop_hand import config, keys  # noqa: E402
from teleop_hand.mapping import AbsoluteCalib, ArmTeleopMapper  # noqa: E402
from teleop_hand.messages import UdpReceiver  # noqa: E402
from teleop_hand.receiver import HandReceiver  # noqa: E402
from teleop_kin.handle import TeachHandle  # noqa: E402

HANDLE_GROUP = 4


def add_mapping_args(ap: argparse.ArgumentParser) -> None:
    """Flags shared by the sim and real runners."""
    ap.add_argument("--map", choices=["relative", "absolute"], default=config.MAP_MODE)
    ap.add_argument("--abs-calib", default=None,
                    help="JSON from run_hand_tracker.py --calibrate-absolute")
    ap.add_argument("--orient", choices=["locked", "yaw", "full"], default=config.ORIENT_MODE)
    ap.add_argument("--rail", choices=["edge", "keys"], default=config.RAIL_MODE)
    ap.add_argument("--clutch", choices=["key", "pedal", "gesture", "always"],
                    default=config.CLUTCH,
                    help="'always' is for headless smoke tests with --demo; never on hardware")
    ap.add_argument("--clutch-key", default=None,
                    help=f"pynput key name (default {config.CLUTCH_KEY}; pedal {config.PEDAL_KEY})")
    ap.add_argument("--precision-key", default=config.PRECISION_KEY)
    ap.add_argument("--scale", type=float, default=config.SCALE)
    ap.add_argument("--rot-scale", type=float, default=config.ROT_SCALE)
    ap.add_argument("--preset", choices=sorted(config.PRESETS), default=config.PRESET)
    ap.add_argument("--udp-port", type=int, default=config.UDP_PORT)
    ap.add_argument("--no-pinch-freeze", action="store_true")
    ap.add_argument("--operator-view", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--workspace-y-min", type=float, default=None, metavar="MM",
                    help="widen the shared workspace box's -y face FOR THIS RUN ONLY "
                         "(default -200 mm excludes the front-row cubes at y=-250)")


def build_mapper(args) -> ArmTeleopMapper:
    if args.workspace_y_min is not None:
        # vr.transforms.clamp_workspace_mm reads the dict at call time, so this
        # widens the box for this process only; the file (shared with
        # SpaceMouse and VR) is untouched.
        from vr import config as vr_config
        y = vr_config.WORKSPACE_AABB_MM["y"]
        vr_config.WORKSPACE_AABB_MM["y"] = (float(args.workspace_y_min), y[1])
        print(f"[HAND] workspace y >= {args.workspace_y_min:.0f} mm (this run only)")
    calib = None
    if args.map == "absolute":
        if not args.abs_calib:
            raise SystemExit("--map absolute needs --abs-calib (run_hand_tracker.py "
                             "--calibrate-absolute OUT.json)")
        d = json.load(open(args.abs_calib))
        calib = AbsoluteCalib(np.array(d["corner_a"]), np.array(d["corner_b"]))
    return ArmTeleopMapper(preset=args.preset, map_mode=args.map, orient_mode=args.orient,
                           rail_mode=args.rail, scale=args.scale, rot_scale=args.rot_scale,
                           abs_calib=calib, pinch_freeze=not args.no_pinch_freeze)


def start_held_keys(args, extra=()):
    if args.clutch == "always":
        return _AlwaysHeld(), "always"
    clutch_key = args.clutch_key or (config.PEDAL_KEY if args.clutch == "pedal"
                                     else config.CLUTCH_KEY)
    held = keys.HeldKeys([clutch_key, args.precision_key, *extra])
    ok = held.start()
    if not ok and args.clutch != "gesture":
        raise SystemExit("[HAND] the key/pedal clutch needs pynput and a display "
                         "(or use --clutch gesture)")
    return held, clutch_key


class _AlwaysHeld:
    """--clutch always: the clutch key reads held, nothing else does."""

    def pressed(self, name: str) -> bool:
        return name == "always"

    def on_tap(self, *a):
        pass

    def stop(self):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Hand-tracking teleop of the digital twin",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_mapping_args(ap)
    ap.add_argument("--scene", choices=["meshes", "primitive"], default="meshes")
    ap.add_argument("--no-gravity-comp", action="store_true")
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--start-j1-deg", type=float, default=None,
                    help="after homing, turn joint 1 to this angle (e.g. -70 faces the "
                         "operator side / front-row cubes)")
    ap.add_argument("--task-label", default=config.TASK_LABEL_SIM)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--trace", default=None,
                    help="write engaged/target/EE/rail at 10 Hz to this JSON-lines file")
    args = ap.parse_args()

    scene = "envs/lab_scene.xml" if args.scene == "meshes" else "envs/lab_scene_primitive.xml"
    mapper = build_mapper(args)
    held, clutch_key = start_held_keys(args)
    udp = UdpReceiver(port=args.udp_port)

    print(f"[HAND] building twin from {scene}")
    arm = SimXArmAPI(scene_xml=scene, render=not args.no_viewer)
    arm.set_viewer_geomgroup(HANDLE_GROUP, True)
    if not args.no_home:
        print("[HAND] moving to home…")     # all-zeros start is a wrist singularity
        arm.go_home(wait=True)
    if args.start_j1_deg is not None:
        q = list(np.rad2deg(arm.HOME_JOINTS_RAD))
        q[0] = args.start_j1_deg
        print(f"[HAND] turning joint 1 to {args.start_j1_deg:.0f} deg…")
        arm.set_servo_angle(q, speed=30, wait=True)

    def make_recorder():
        return Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                        interface="handtrack_sim", scene_xml=scene, enable_frames=True)

    rec_on = not args.no_record
    rx = HandReceiver(arm, mapper, udp,
                      clutch_fn=lambda: held.pressed(clutch_key),
                      precision_fn=lambda: held.pressed(args.precision_key),
                      gesture_clutch=args.clutch == "gesture",
                      marker=TeachHandle(arm),
                      recorder=make_recorder() if rec_on else None,
                      recorder_factory=make_recorder if rec_on else None,
                      task_label=args.task_label)
    rx.servo.gravity_comp = not args.no_gravity_comp
    handler = keys.make_viewer_handler(rx.post)
    arm.register_key_handler(handler)
    view = None
    if args.operator_view:
        from teleop_hand.operator_view import OperatorView
        view = OperatorView(arm, rx.status)
        view.start()

    print("\n" + "=" * 70)
    print(f"[HAND] map={args.map} orient={args.orient} rail={args.rail} preset={args.preset} "
          f"scale={args.scale} record={'on' if rec_on else 'off'}")
    print(f"[HAND] listening for HandFrames on udp/{udp.port} -- start run_hand_tracker.py")
    print(keys.describe(args.clutch, clutch_key, args.precision_key, args.rail))
    print("=" * 70 + "\n")

    period = 1.0 / config.FREQUENCY
    last = time.time()
    seen_viewer = False
    trace = open(args.trace, "w") if args.trace else None
    t_trace = 0.0
    last_banner = 0.0
    try:
        while True:
            t0 = time.time()
            if not args.no_viewer:
                if arm._viewer is not None:
                    seen_viewer = True
                elif seen_viewer:
                    print("[HAND] viewer closed -- stopping.")
                    break
            rx.tick(max(t0 - last, 1e-4))
            last = t0
            st = rx.status()
            if trace is not None and t0 - t_trace >= 0.1:
                t_trace = t0
                with arm.lock:
                    ee = arm.data.site_xpos[arm.ee_site] * 1000.0
                    ee = [round(float(v), 1) for v in ee]
                    ee_rot = arm.data.site_xmat[arm.ee_site].reshape(3, 3).copy()
                    sim_t = float(arm.data.time)
                    q_live = np.array([arm.data.qpos[arm.model.jnt_qposadr[j]]
                                       for j in arm.ik_solver.joint_ids])
                    contacts = sorted({
                        arm.model.geom(c.geom1).name + "|" + arm.model.geom(c.geom2).name
                        for c in arm.data.contact[:arm.data.ncon]
                        if any(n in (arm.model.geom(c.geom1).name + arm.model.geom(c.geom2).name)
                               for n in ("link", "gripper", "finger", "camera"))})
                q_err = (None if rx.servo.q_cmd is None else
                         round(float(np.rad2deg(np.max(np.abs(q_live - rx.servo.q_cmd)))), 2))
                pert = None
                v = getattr(arm, "_viewer", None)
                if v is not None:
                    try:
                        pert = [int(v.perturb.select), int(v.perturb.active)]
                    except Exception:
                        pert = None
                out = rx.out
                from vr.transforms import twin_rot_to_rpy_deg
                rpy = [round(v, 1) for v in twin_rot_to_rpy_deg(ee_rot)]
                tgt_rpy = (None if out is None or out.target_rot is None
                           else [round(v, 1) for v in twin_rot_to_rpy_deg(out.target_rot)])
                trace.write(json.dumps({
                    "t": round(t0, 3), "engaged": st.get("engaged"),
                    "target": None if out is None or out.target_pos is None
                    else [round(float(v) * 1000, 1) for v in out.target_pos],
                    "ee": ee, "rail": st.get("rail_mm"), "blocked": st.get("blocked"),
                    "opening": st.get("opening"), "grip": st.get("gripper_closed"),
                    "ee_rpy": rpy, "target_rpy": tgt_rpy, "rot_quality": st.get("rot_quality"),
                    "rot_held": st.get("rot_held"), "sim_t": round(sim_t, 3),
                    "q_err_deg": q_err, "contacts": contacts, "perturb": pert}) + "\n")
            if t0 - last_banner > 2.0 and (st.get("silent") or st.get("need_release")):
                last_banner = t0
                print("[HAND] tracker dead? no HandFrames" if st.get("silent")
                      else "[HAND] held -- release and press the clutch")
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[HAND] shutting down…")
    finally:
        arm.unregister_key_handler(handler)
        if trace is not None:
            trace.close()
        if view is not None:
            view.stop()
        held.stop()
        path = rx.takes.stop(kept=True)
        if path:
            print(f"[HAND] saved in-progress take -> {path}")
        lat = np.array(rx.latencies_s) * 1000.0
        if lat.size:
            print(f"[HAND] capture->command latency p50 {np.percentile(lat, 50):.0f} ms, "
                  f"p95 {np.percentile(lat, 95):.0f} ms ({lat.size} samples)")
        udp.close()
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
