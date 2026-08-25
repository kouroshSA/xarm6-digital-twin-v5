#!/usr/bin/env python3
"""Say what you want; see where the arm would go for it.

    python scripts/target_check.py "the blue cube"
    python scripts/target_check.py "a green cube" --max-size 0.08 --out /tmp/t
    python scripts/target_check.py "a green cube" --compare        # with/without depth
    python scripts/target_check.py "a bottle" --real               # the D435i

Prints the resolved object's measured size and the arm pose its best grasp
implies. Nothing here commands the arm -- `--at` only poses it to aim the camera.

`--compare` runs the query twice, with and without the physical size bound, and
shows both answers. On this bench "a green cube" resolves to the green *bin*
without it, which is the clearest demonstration of why the depth stage exists.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

DISTRACTORS = ("a bin", "a cup", "a test tube")


def _describe(label: str, target) -> None:
    if target is None:
        print(f"  {label}: no match")
        return
    size = target.max_size_m * 1000.0
    print(f"  {label}: score {target.score:.2f}, measured {size:.0f} mm across, "
          f"{target.n_pixels} depth px")
    if target.position_world is not None:
        p = target.position_world * 1000.0
        print(f"      centroid  x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f} mm (base frame)")
    else:
        p = target.position_cam * 1000.0
        print(f"      centroid  {np.round(p, 1).tolist()} mm (camera frame; no arm)")
    if not target.grasps:
        print("      no GG-CNN grasp landed on this object — "
              "position only, do not treat it as a grasp")
        return
    g = target.grasps[0]
    print(f"      grasp     quality {g.quality:.2f}, jaw {g.width_m * 1000:.0f} mm")
    if g.position_world is not None:
        pose = g.to_arm_pose()
        print(f"      arm pose  x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} mm  "
              f"roll={pose[3]:.0f} pitch={pose[4]:.0f} yaw={pose[5]:.1f} deg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phrase", help='what to look for, e.g. "the blue cube"')
    ap.add_argument("--real", action="store_true", help="use the physical D435i")
    ap.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"),
                    default=[60.0, -200.0],
                    help="pose the flange above this base xy, mm (sim only)")
    ap.add_argument("--height", type=float, default=1180.0)
    ap.add_argument("--max-size", type=float, default=None, metavar="M",
                    help="reject candidates wider than this, metres "
                         "(0.08 keeps cubes and drops bins)")
    ap.add_argument("--min-size", type=float, default=None, metavar="M")
    ap.add_argument("--compare", action="store_true",
                    help="also run without the size bound, to show its effect")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out", metavar="DIR", help="save an annotated image here")
    args = ap.parse_args()

    from perception.language import LanguageTargeter

    targeter = LanguageTargeter(device=args.device)
    print(f"grounding on {targeter.grounder.device}")

    arm = None
    if args.real:
        from perception.realsense_camera import RealSenseWristCamera
        try:
            cam = RealSenseWristCamera(arm=None)
        except Exception as exc:  # noqa: BLE001
            print(f"REAL capture failed: {type(exc).__name__}: {exc}")
            return 1
        for _ in range(10):
            cam.capture()
        frame = cam.capture()
        cam.close()
    else:
        from perception.sim_camera import SimWristCamera
        from sim.mujoco_env import SimXArmAPI
        arm = SimXArmAPI("envs/lab_scene.xml", render=False)
        cam = SimWristCamera(arm)
        x, y = args.at
        if arm.set_position(x, y, args.height, 180.0, 0.0, 0.0, wait=True) != 0:
            print(f"could not reach the viewing pose: {arm.last_refusal}")
            cam.close()
            arm.disconnect()
            return 1
        frame = cam.capture()
        cam.close()

    print(f'\nlooking for "{args.phrase}"')
    target = targeter.target(args.phrase, frame, max_size_m=args.max_size,
                             min_size_m=args.min_size, distractors=DISTRACTORS)
    label = f"max_size={args.max_size}" if args.max_size else "no size bound"
    _describe(label, target)

    if args.compare and args.max_size is not None:
        other = targeter.target(args.phrase, frame, distractors=DISTRACTORS)
        print()
        _describe("no size bound", other)

    if args.out:
        from PIL import Image
        os.makedirs(args.out, exist_ok=True)
        tag = args.phrase.replace(" ", "_")[:40]
        path = os.path.join(args.out, f"{tag}.png")
        Image.fromarray(targeter.draw(frame, target)).save(path)
        print(f"\n  saved {path}")

    if arm is not None:
        arm.disconnect()
    return 0 if target is not None else 1


if __name__ == "__main__":
    sys.exit(main())
