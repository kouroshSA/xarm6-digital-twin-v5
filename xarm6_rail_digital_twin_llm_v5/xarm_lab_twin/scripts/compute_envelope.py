#!/usr/bin/env python3
"""Compute the arm's swept volume — the region anything can be struck in (B0).

A rail-mounted arm's envelope is not the arm's reach. It is the reach *dragged*
along the rail, so it is far wider than a fixed-base arm of the same size and
wider than intuition suggests. B0 asks for this to be computed and then marked
on the floor and bench, with the operator station outside the line.

Sampled rather than derived: random joint configurations across the full joint
ranges and the full rail travel, tracking **every arm link**, not just the tool.
The elbow sweeps outside the wrist's path, so an end-effector-only envelope is
too small — which is the dangerous direction to be wrong in.

    python scripts/compute_envelope.py
    python scripts/compute_envelope.py --samples 40000 --margin 150
"""
from __future__ import annotations

import argparse
import os
import sys

import mujoco as mj
import numpy as np

sys.path.insert(0, os.getcwd())

# Bodies that physically sweep. The gripper and fingers are included because the
# fingertips are the part most likely to reach a person first.
ARM_BODIES = ["xarm_base", "link1", "link2", "link3", "link4", "link5", "link6",
              "gripper", "finger_left", "finger_right", "rail_carriage"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="envs/lab_scene.xml")
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--margin", type=float, default=100.0,
                    help="safety margin in mm added to every side of the marked "
                         "line (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = mj.MjModel.from_xml_path(args.scene)
    d = mj.MjData(m)
    rng = np.random.default_rng(args.seed)

    jids = [m.joint(f"joint{i}").id for i in range(1, 7)]
    qadr = [m.jnt_qposadr[j] for j in jids]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])

    rail = m.joint("rail")
    rail_adr = m.jnt_qposadr[rail.id]
    rail_lo, rail_hi = m.jnt_range[rail.id]

    bids = []
    for name in ARM_BODIES:
        try:
            bids.append((name, m.body(name).id))
        except Exception:
            pass  # scene variant without that body; reported below

    pts = []
    for _ in range(args.samples):
        d.qpos[qadr] = rng.uniform(lo, hi)
        d.qpos[rail_adr] = rng.uniform(rail_lo, rail_hi)
        mj.mj_kinematics(m, d)          # kinematics only: no physics needed
        for _n, bid in bids:
            pts.append(d.xpos[bid].copy())

    P = np.asarray(pts) * 1000.0        # mm
    x0, y0, z0 = P.min(axis=0)
    x1, y1, z1 = P.max(axis=0)
    mgn = args.margin

    print(f"scene   : {args.scene}")
    print(f"sampled : {args.samples} configurations x {len(bids)} bodies "
          f"= {len(P):,} points")
    print(f"tracked : {', '.join(n for n, _ in bids)}")
    print()
    print("SWEPT VOLUME (world mm, bench top is z=750, floor z=0)")
    print(f"  x: {x0:8.0f} .. {x1:8.0f}   (width  {x1-x0:6.0f} mm)")
    print(f"  y: {y0:8.0f} .. {y1:8.0f}   (depth  {y1-y0:6.0f} mm)")
    print(f"  z: {z0:8.0f} .. {z1:8.0f}   (height {z1-z0:6.0f} mm)")
    print()
    print(f"FLOOR MARKING with a {mgn:.0f} mm margin — tape this rectangle:")
    print(f"  x: {x0-mgn:8.0f} .. {x1+mgn:8.0f}   ({(x1-x0+2*mgn)/1000:.2f} m wide)")
    print(f"  y: {y0-mgn:8.0f} .. {y1+mgn:8.0f}   ({(y1-y0+2*mgn)/1000:.2f} m deep)")
    print(f"  highest reach: {z1:.0f} mm ({z1/1000:.2f} m) — check overhead clearance")
    print()
    print("  The operator station goes OUTSIDE this rectangle. Note the envelope")
    print("  is wider than the arm's reach because the rail drags it: a fixed-base")
    print("  xArm6 would sweep ~700 mm radius, this sweeps that plus rail travel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
