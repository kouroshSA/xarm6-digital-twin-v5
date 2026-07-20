#!/usr/bin/env python3
"""IK sanity harness — how well does the solver track the arm's kinematics?

SPIKE (branch: spike/xarm6-realistic-meshes). The realistic mesh scene
(``--scene meshes``) uses the true xArm6 joint frames, which differ from the
primitive scene the IK solver (``sim/ik_solver.py``) was tuned against. This
harness quantifies the gap so we know how much re-validation the mesh arm needs
before it can grade task execution.

Method (isolates *solver quality* from *reachability*):
  1. Sample N random joint configurations within the model's own joint limits.
  2. Forward-kinematics each one -> a target EE position that is, by
     construction, reachable for THIS arm with the rail held fixed.
  3. Reset to a neutral home pose, then ask ``IKSolver.solve(target)`` to
     recover a configuration reaching that position (position-only, matching
     the ``set_position`` path).
  4. Apply the returned joints, forward-kinematics, and measure the residual
     ||EE - target||. A solver faithful to the kinematics drives this to ~0;
     large residuals / None returns expose the frame mismatch.

Because the two scenes have different kinematics, each is sampled against its
OWN FK, so the comparison is apples-to-apples: "given targets reachable for
this arm, how well does the shared solver recover them?"

Usage:
    python scripts/ik_sanity.py                 # both scenes, 60 targets
    python scripts/ik_sanity.py --scene meshes  # just the mesh arm
    python scripts/ik_sanity.py --n 200 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.ik_solver import IKSolver, JOINT_NAMES  # noqa: E402

SCENES = {
    "primitive": "envs/lab_scene.xml",
    "meshes": "envs/lab_scene_meshes.xml",
}
# Position success bands (metres). 5mm is roughly a comfortable grasp; 1mm is
# "the solver really converged".
BANDS = (0.001, 0.005, 0.020)


def _load(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def _rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def run_scene(path: str, n: int, rng: np.random.Generator,
              orient: bool = False) -> dict:
    import threading

    model, data = _load(path)
    lock = threading.Lock()
    solver = IKSolver(model, data, lock)

    jids = [model.joint(nm).id for nm in JOINT_NAMES]
    qadr = [model.jnt_qposadr[j] for j in jids]
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])
    ee = model.site("end_effector").id
    rail_adr = model.jnt_qposadr[model.joint("rail").id]

    # Neutral home seed: mid-range of joints 2/3/5 folded slightly so the arm
    # starts in a generic elbow-down posture rather than the singular zero pose.
    home = np.zeros(6)
    home[1], home[2], home[4] = -0.5, -0.5, 1.0
    home = np.clip(home, lo, hi)

    rail_fixed = 0.0
    data.qpos[rail_adr] = rail_fixed

    def fk(q):
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        return (data.site_xpos[ee].copy(),
                data.site_xmat[ee].reshape(3, 3).copy())

    errors = []
    rot_errors = []
    fails = 0
    for _ in range(n):
        q_rand = rng.uniform(lo, hi)
        target, target_rot = fk(q_rand)

        # reset to home before solving so IK isn't warm-started at the answer
        for a, v in zip(qadr, home):
            data.qpos[a] = v
        data.qpos[rail_adr] = rail_fixed
        mujoco.mj_forward(model, data)

        with lock:
            q_sol = solver.solve(target,
                                 target_rot=target_rot if orient else None,
                                 seed_q=home)
        if q_sol is None:
            fails += 1
            continue
        reached, reached_rot = fk(q_sol)
        errors.append(float(np.linalg.norm(reached - target)))
        if orient:
            rot_errors.append(_rot_err_deg(reached_rot, target_rot))

    errors = np.array(errors)
    res = {
        "scene": path,
        "backend": solver.backend,
        "n": n,
        "none_fails": fails,
        "converged": len(errors),
        "orient": orient,
    }
    if len(errors):
        res.update(
            median_mm=float(np.median(errors) * 1e3),
            p90_mm=float(np.percentile(errors, 90) * 1e3),
            max_mm=float(errors.max() * 1e3),
        )
        for b in BANDS:
            # success counts a None-return as a miss (denominator = n)
            res[f"ok_{int(b*1000)}mm"] = int((errors < b).sum())
    if orient and rot_errors:
        re = np.array(rot_errors)
        res.update(
            rot_median_deg=float(np.median(re)),
            rot_p90_deg=float(np.percentile(re, 90)),
            rot_max_deg=float(re.max()),
            rot_ok_10deg=int((re < 10.0).sum()),
        )
    return res


def _fmt(r: dict) -> str:
    if not r.get("converged"):
        return (f"  {r['scene']:<26} backend={r['backend']:<9} "
                f"ALL {r['none_fails']}/{r['n']} returned None")
    n = r["n"]
    bands = "  ".join(
        f"<{int(b*1000)}mm {r[f'ok_{int(b*1000)}mm']:>3}/{n} "
        f"({100*r[f'ok_{int(b*1000)}mm']/n:4.0f}%)" for b in BANDS
    )
    out = (f"  {r['scene']:<26} backend={r['backend']:<9} "
           f"None={r['none_fails']:>2}/{n}  "
           f"median={r['median_mm']:6.2f}mm  p90={r['p90_mm']:7.2f}mm  "
           f"max={r['max_mm']:8.2f}mm\n"
           f"  {'':<26} pos: {bands}")
    if r.get("orient") and "rot_median_deg" in r:
        out += (f"\n  {'':<26} rot: median={r['rot_median_deg']:5.1f}deg  "
                f"p90={r['rot_p90_deg']:5.1f}deg  max={r['rot_max_deg']:5.1f}deg  "
                f"<10deg {r['rot_ok_10deg']:>3}/{n} "
                f"({100*r['rot_ok_10deg']/n:4.0f}%)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["primitive", "meshes", "both"],
                    default="both")
    ap.add_argument("--n", type=int, default=60,
                    help="number of FK-sampled targets per scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orient", action="store_true",
                    help="Also solve for orientation (6-DOF) and report the "
                         "residual rotation error. Exposes the known "
                         "orientation/yaw limitation of the iterative solver.")
    args = ap.parse_args()

    targets = (["primitive", "meshes"] if args.scene == "both"
               else [args.scene])

    mode = "6-DOF (pos+orient)" if args.orient else "position-only"
    print(f"\nIK sanity: {args.n} FK-sampled reachable targets/scene, "
          f"rail fixed, {mode}, seed={args.seed}\n"
          f"(residual = ||EE - target|| after solving from a neutral home "
          f"pose)\n")
    results = []
    for name in targets:
        # Independent RNG stream per scene, but derived from the same base seed
        # so a rerun is reproducible.
        rng = np.random.default_rng(args.seed)
        r = run_scene(SCENES[name], args.n, rng, orient=args.orient)
        results.append(r)
        print(_fmt(r))
        print()

    if len(results) == 2 and all(x.get("converged") for x in results):
        prim, mesh = results
        print("Delta (meshes vs primitive):")
        print(f"  median residual: {prim['median_mm']:.2f}mm -> "
              f"{mesh['median_mm']:.2f}mm")
        print(f"  <5mm success:    {100*prim['ok_5mm']/prim['n']:.0f}% -> "
              f"{100*mesh['ok_5mm']/mesh['n']:.0f}%")


if __name__ == "__main__":
    main()
