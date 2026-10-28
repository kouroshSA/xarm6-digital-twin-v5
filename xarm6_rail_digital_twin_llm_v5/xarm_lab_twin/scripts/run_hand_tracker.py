#!/usr/bin/env python3
"""Hand tracker process: camera -> MediaPipe + dex-retargeting -> UDP.

    conda activate xarm6sim_hand
    python scripts/run_hand_tracker.py --show                  # D435, right hand
    python scripts/run_hand_tracker.py --webcam 0 --show       # no depth (approx)
    python scripts/run_hand_tracker.py --list                  # what cameras are there
    python scripts/run_hand_tracker.py --calibrate-absolute calib.json
    python scripts/run_hand_tracker.py --webcam 0 --calibrate-depth
    python scripts/run_hand_tracker.py --log-keypoints log.json --seconds 10

Publishes one HandFrame per camera frame to 127.0.0.1:47100 (--udp-port),
including frames with no hand. Start this first, then run_handteleop_sim.py
(or _real.py) in the xarm6sim env.

Place the D435 0.5-0.8 m in front of the operator, facing them, at chest
height -- the `facing` preset. See docs/hand_teleop.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from teleop_hand import config  # noqa: E402
from teleop_hand.messages import UdpPublisher  # noqa: E402


def list_cameras() -> None:
    try:
        import pyrealsense2 as rs
        for d in rs.context().query_devices():
            usb = d.get_info(rs.camera_info.usb_type_descriptor) \
                if d.supports(rs.camera_info.usb_type_descriptor) else "?"
            print(f"RealSense {d.get_info(rs.camera_info.name)}  serial "
                  f"{d.get_info(rs.camera_info.serial_number)}  USB {usb}")
    except Exception as e:                       # noqa: BLE001
        print(f"pyrealsense2: {e}")
    import glob
    print("video devices:", " ".join(sorted(glob.glob("/dev/video*"))))


def _source(args):
    from teleop_hand.tracker import RealSenseSource, WebcamSource
    if args.webcam is not None:
        return WebcamSource(args.webcam)
    src = RealSenseSource(serial=args.serial)
    print(f"[tracker] RealSense serial {src.serial}  USB {src.usb}")
    if str(src.usb).startswith("2"):
        print("[tracker] WARNING: USB 2 link -- use a USB 3 port and cable")
    return src


def _collect(tracker, seconds: float, show: bool, prompt: str):
    import cv2
    print(prompt)
    t_end = time.time() + seconds
    pos = []
    while time.time() < t_end:
        f = tracker.step()
        if f.valid:
            pos.append(f.pos())
        if show:
            cv2.imshow("hand tracker", tracker.render())
            cv2.waitKey(1)
    if len(pos) < 10:
        raise SystemExit("[tracker] too few valid frames; is the hand in view?")
    return np.median(np.array(pos), axis=0)


def run_demo(args) -> int:
    """A scripted hand at 30 Hz: lets the teleop side be tried with no camera."""
    from teleop_hand.testing import PALM0, Hand
    pub = UdpPublisher(port=args.udp_port)
    hand = Hand()
    t0 = time.monotonic()
    print(f"[tracker] DEMO hand -> {config.UDP_HOST}:{args.udp_port} (Ctrl-C to stop)")
    try:
        while args.seconds is None or time.monotonic() - t0 < args.seconds:
            t = time.monotonic() - t0
            hand.pos = PALM0 + 0.06 * np.array([np.sin(0.5 * t), 0.5 * np.sin(1.0 * t), 0.0])
            hand.opening = 0.15 if (t % 6.0) > 4.5 else 0.9
            pub.send(hand.frame(time.monotonic()))
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    pub.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Hand tracker -> HandFrame UDP",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=["Right", "Left"], default="Right")
    ap.add_argument("--swap-hands", action="store_true",
                    help="track the OTHER hand label. Not needed with a D435 or webcam "
                         "(MEASURED: MediaPipe's labels are correct on unmirrored images); "
                         "use it only if the skeleton lands on the wrong hand")
    ap.add_argument("--webcam", type=int, default=None, help="use webcam N (no depth)")
    ap.add_argument("--serial", default=None, help="RealSense serial (several connected)")
    ap.add_argument("--aux-gesture", action="store_true",
                    help="track the other hand too; a fist there is the gesture clutch")
    ap.add_argument("--gestures", action="store_true",
                    help="classify the main hand's fist / point and publish the pointing "
                         "ray (for run_handteleop_sim.py --gestures)")
    ap.add_argument("--udp-port", type=int, default=config.UDP_PORT)
    ap.add_argument("--show", action="store_true", help="debug window")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N s")
    ap.add_argument("--log-keypoints", default=None, help="write joint_pos per frame to JSON")
    ap.add_argument("--snapshot", default=None,
                    help="save the debug view (with overlay) to this PNG on exit")
    ap.add_argument("--calibrate-absolute", default=None, metavar="OUT.json",
                    help="record two opposite corners of your hand box and exit")
    ap.add_argument("--calibrate-depth", action="store_true",
                    help="webcam: hold the palm at 0.6 m to calibrate size->depth")
    ap.add_argument("--depth-k", type=float, default=config.WEBCAM_DEPTH_K)
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--demo", action="store_true",
                    help="no camera: publish a scripted hand (slow circle, a pinch every 6 s)")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return 0
    if args.demo:
        return run_demo(args)

    from teleop_hand.tracker import HandTracker, save_log
    src = _source(args)
    pub = UdpPublisher(port=args.udp_port)
    tr = HandTracker(src, hand=args.hand, aux_gesture=args.aux_gesture, gestures_on=args.gestures,
                     selfie=not args.swap_hands, publisher=pub, depth_k=args.depth_k)
    if args.log_keypoints:
        tr.log = []
    try:
        if args.calibrate_absolute:
            a = _collect(tr, 5.0, args.show, "[tracker] hold your palm at the LOWER-LEFT-NEAR "
                         "corner of a comfortable box (5 s)...")
            b = _collect(tr, 5.0, args.show, "[tracker] now the UPPER-RIGHT-FAR corner (5 s)...")
            json.dump({"corner_a": a.tolist(), "corner_b": b.tolist()},
                      open(args.calibrate_absolute, "w"), indent=1)
            print(f"[tracker] wrote {args.calibrate_absolute}: {a.round(3)} / {b.round(3)}")
            return 0
        if args.calibrate_depth:
            tr.depth_k = 1.0
            p = _collect(tr, 4.0, args.show, "[tracker] hold your open palm 0.6 m from the "
                         "camera (4 s)...")
            k = 0.6 * (1.0 / p[2])          # z = depth_k / size, measured with depth_k = 1
            print(f"[tracker] --depth-k {k:.1f}")
            return 0

        print(f"[tracker] publishing to {config.UDP_HOST}:{args.udp_port}  "
              f"hand={args.hand}  {'webcam (APPROX depth)' if args.webcam is not None else 'D435'}")
        import cv2
        t_end = time.time() + args.seconds if args.seconds else None
        t_report = time.time() + 5
        while t_end is None or time.time() < t_end:
            f = tr.step()
            if args.show:
                cv2.imshow("hand tracker", tr.render())
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if time.time() > t_report:
                t_report = time.time() + 5
                s = (f"open {f.opening:.2f} q {f.rot_quality:.2f}"
                     + (f" z {f.palm_pos_cam_m[2]:.3f}" if f.pos_valid else " no depth")
                     ) if f.detected else "no hand"
                print(f"[tracker] {tr.fps:4.1f} FPS  {s}")
    except KeyboardInterrupt:
        pass
    finally:
        if args.snapshot and tr.last is not None:
            import cv2
            cv2.imwrite(args.snapshot, tr.render())
            print(f"[tracker] snapshot -> {args.snapshot}")
        if tr.log is not None:
            save_log(tr.log, args.log_keypoints)
            print(f"[tracker] {len(tr.log)} keypoint frames -> {args.log_keypoints}")
        src.close()
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
