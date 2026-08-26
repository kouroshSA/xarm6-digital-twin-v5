#!/usr/bin/env python3
"""Solve where a fixed observer camera is, in robot base coordinates.

    python scripts/calibrate_observer.py --self-test          # prove the solver
    python scripts/calibrate_observer.py --observations f.json

An observer's coordinates are only worth what its pose is worth. In the twin the
pose is declared and exact; on a bench it has to be measured, and this is the
tool that measures it. See perception/extrinsic_calibration.py for the method.

## Collecting observations on hardware

Mount an ArUco tag on a flat plate on the gripper. Then for each of at least six
arm poses -- and vary HEIGHT and REACH, not just the sweep across the bench:

  1. jog the arm somewhere the observer can see the tag;
  2. read the tag's position in the ROBOT frame: forward kinematics for the
     flange, plus the fixed offset from flange to tag centre;
  3. read the tag's position in the CAMERA frame from the detector.

Write them out as JSON and pass it here:

    {"points_base_m": [[x, y, z], ...],
     "points_cam_m":  [[x, y, z], ...]}

Both in metres, index-aligned: entry i of each is the same physical tag position.

## What --self-test proves, and what it does not

It runs the solver against the twin's observers, whose poses are known exactly,
and checks that it recovers them: exactly from clean data, gracefully under
detector noise, and that it REFUSES a coplanar pose set rather than solving one.

It does not prove the tag detection, which needs the real camera. The value is
that when you calibrate the bench you will be debugging a mounting and a
detector, and not also a solver.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from perception.extrinsic_calibration import (drop_worst,  # noqa: E402
                                              solve_camera_pose)


def from_file(path: str, trim: int) -> int:
    with open(path) as fh:
        blob = json.load(fh)
    base = np.array(blob["points_base_m"], dtype=float)
    cam = np.array(blob["points_cam_m"], dtype=float)
    print(f"{len(base)} correspondences from {path}")

    res = solve_camera_pose(cam, base)
    print(res.report())

    for i in range(trim):
        if res.max_mm < 3.0 * max(res.rms_mm, 1e-9):
            break
        cam, base, dropped = drop_worst(cam, base, res)
        res = solve_camera_pose(cam, base)
        print(f"\ndropped observation {dropped} (trim {i + 1}/{trim}):")
        print(res.report())

    if res.ok:
        print("\ncam_to_base (metres):")
        print(np.array2string(res.cam_to_base, precision=6, suppress_small=True))
        print("\nPaste the position/orientation into perception/observer_calib.py "
              "and rerun `python -m perception.sync_scene`.")
    return 0 if res.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--observations", metavar="JSON",
                    help="paired base/camera marker positions, metres")
    ap.add_argument("--self-test", action="store_true",
                    help="validate the solver against the twin's known poses")
    ap.add_argument("--trim", type=int, default=0, metavar="N",
                    help="drop up to N outlying observations, one at a time")
    args = ap.parse_args()

    if args.self_test:
        from perception.test_extrinsic_calibration import main as selftest
        return selftest()
    if not args.observations:
        ap.error("give --observations FILE, or --self-test")
    return from_file(args.observations, args.trim)


if __name__ == "__main__":
    sys.exit(main())
