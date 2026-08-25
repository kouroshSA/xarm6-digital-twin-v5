#!/usr/bin/env python3
"""Run GG-CNN on the wrist camera and show what it would grasp.

    python scripts/grasp_check.py                        # sim, look at the bench
    python scripts/grasp_check.py --at 0 -150            # sim, look at green_cube
    python scripts/grasp_check.py --real                 # the physical D435i
    python scripts/grasp_check.py --at 0 -150 --out /tmp/g   # save an overlay

Prints each candidate as the arm pose it implies, so the numbers can be checked
against `set_position` before anything moves. Nothing here commands the arm --
`--at` only poses it to aim the camera.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from perception.grasp import GGCNNDetector  # noqa: E402


def _print_grasps(grasps, truth=None) -> None:
    if not grasps:
        print("  no grasp candidates above the quality threshold")
        return
    for i, g in enumerate(grasps):
        u, v = g.pixel
        print(f"  [{i}] quality {g.quality:.3f}  pixel ({u:.0f}, {v:.0f})  "
              f"depth {g.depth_m:.3f} m")
        print(f"      angle {np.degrees(g.angle_rad):+6.1f} deg   "
              f"jaw width {g.width_m * 1000:.1f} mm")
        if g.position_world is None:
            print(f"      camera frame {np.round(g.position_cam * 1000, 1).tolist()} mm "
                  f"(no arm attached, so no world pose)")
            continue
        pose = g.to_arm_pose()
        print(f"      arm pose  x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} mm  "
              f"roll={pose[3]:.0f} pitch={pose[4]:.0f} yaw={pose[5]:.1f} deg")
        if truth is not None:
            d = np.linalg.norm((g.position_world * 1000.0 - truth)[:2])
            print(f"      {d:.1f} mm (xy) from the target's true position")


def run_sim(args, det) -> int:
    import mujoco

    from perception.sim_camera import SimWristCamera
    from sim.mujoco_env import SimXArmAPI

    arm = SimXArmAPI("envs/lab_scene.xml", render=False)
    cam = SimWristCamera(arm, noise_std_m=args.depth_noise)

    truth = None
    if args.at is not None:
        x, y = args.at
        print(f"\nposing the flange above ({x:.0f}, {y:.0f}) at z={args.height:.0f} mm")
        code = arm.set_position(x, y, args.height, 180.0, 0.0, 0.0, wait=True)
        if code != 0:
            print(f"  set_position refused (rc={code}): {arm.last_refusal}")
            cam.close()
            arm.disconnect()
            return 1
        if args.target:
            bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, args.target)
            if bid >= 0:
                with arm.lock:
                    truth = np.array(arm.data.xpos[bid]) * 1000.0
                print(f"  {args.target} is really at "
                      f"{np.round(truth, 1).tolist()} mm")

    frame = cam.capture()
    grasps = det.detect(frame, top_k=args.top_k, min_quality=args.min_quality)
    print(f"\nSIM — {len(grasps)} candidate(s)")
    _print_grasps(grasps, truth)

    if args.out:
        _save_overlay(det, frame, grasps, args.out, "sim")

    cam.close()
    arm.disconnect()
    return 0


def run_real(args, det) -> int:
    from perception.realsense_camera import RealSenseWristCamera

    try:
        cam = RealSenseWristCamera(arm=None)
    except Exception as exc:  # noqa: BLE001
        print(f"\nREAL capture failed: {type(exc).__name__}: {exc}")
        print("  Is the D435i plugged in? `rs-enumerate-devices -s` lists it.")
        return 1
    for _ in range(10):     # let auto-exposure settle
        cam.capture()
    frame = cam.capture()
    cam.close()

    grasps = det.detect(frame, top_k=args.top_k, min_quality=args.min_quality)
    cov = frame.depth_stats()["valid_frac"] * 100.0
    print(f"\nREAL — {len(grasps)} candidate(s), depth {cov:.1f}% valid")
    _print_grasps(grasps)
    if args.out:
        _save_overlay(det, frame, grasps, args.out, "real")
    return 0


def _save_overlay(det, frame, grasps, out_dir: str, tag: str) -> None:
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_grasps.png")
    Image.fromarray(det.draw(frame, grasps)).save(path)
    print(f"\n  saved {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true", help="use the physical D435i")
    ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"),
                    help="pose the flange above this base-frame xy, mm (sim only)")
    ap.add_argument("--height", type=float, default=1080.0,
                    help="flange height for --at, mm (default 1080)")
    ap.add_argument("--target", default=None,
                    help="scene body to compare against, e.g. green_cube")
    ap.add_argument("--model", default="ggcnn", choices=("ggcnn", "ggcnn2"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-quality", type=float, default=0.1)
    ap.add_argument("--depth-noise", type=float, default=0.0, metavar="M",
                    help="sim depth noise std in metres, to approximate the "
                         "real sensor (default 0)")
    ap.add_argument("--out", metavar="DIR", help="save an overlay image here")
    args = ap.parse_args()

    det = GGCNNDetector(model=args.model)
    print(f"GG-CNN model={det.model_name}  input {det.out_size}x{det.out_size}")

    return run_real(args, det) if args.real else run_sim(args, det)


if __name__ == "__main__":
    sys.exit(main())
