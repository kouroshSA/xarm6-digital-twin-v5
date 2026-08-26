"""Does the calibration solver recover a camera pose we already know?

Run: ``MUJOCO_GL=egl python -m perception.test_extrinsic_calibration``

This is the twin earning its keep. On the bench the calibration has no ground
truth -- that is the entire reason it exists -- so a wrong answer looks exactly
like a right one until grasps start missing. Here ``cam_overhead``'s pose is
known to the micrometre, so the solver can be handed simulated observations and
required to hand the pose back.

Four things, in increasing order of how much they would hurt to get wrong:

1. **Exactness.** With clean correspondences the answer must be the truth, not
   close to it. Any error here is a bug in the solver, not noise.
2. **Noise.** With detector error injected, the pose must degrade gracefully and
   proportionately -- and the numbers printed are the ones that tell you how good
   a marker detector the real rig needs.
3. **Degeneracy is refused.** Marker positions in a plane -- which is what a
   sweep at one working height produces, and it feels thorough -- must be
   rejected rather than solved. This is the failure that reaches hardware.
4. **A bad detection is visible.** One outlier must show up as a large max
   against a small RMS, so it can be dropped instead of absorbed.

What is NOT tested here is finding the marker in an image. That needs the real
camera; see the module docstring.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from . import observer_calib  # noqa: E402
from .extrinsic_calibration import (MIN_SPREAD_RATIO, drop_worst,  # noqa: E402
                                    pose_error, solve_camera_pose)

SCENE = os.path.join(os.path.dirname(__file__), "..", "envs", "lab_scene.xml")

# The marker sits on the gripper, so its base-frame position is FK plus a fixed
# offset. Realistic reach for this cell.
N_POSES = 14


def _true_pose(model, data, name: str) -> np.ndarray:
    """The observer's actual cam-optical -> world transform, from the scene."""
    from .sim_camera import _MUJOCO_TO_OPTICAL

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    t = np.eye(4)
    t[:3, :3] = np.array(data.cam_xmat[cid]).reshape(3, 3) @ _MUJOCO_TO_OPTICAL
    t[:3, 3] = np.array(data.cam_xpos[cid])
    return t


def _marker_positions(rng, n: int = N_POSES) -> np.ndarray:
    """Where the gripper-mounted marker gets held, in the base frame.

    Spread over the working volume in all three axes on purpose -- that is the
    procedure this test exists to prove out, and a flat set is checked
    separately.
    """
    return np.stack([
        rng.uniform(-0.30, 0.30, n),
        rng.uniform(-0.40, -0.05, n),
        rng.uniform(0.85, 1.20, n),
    ], axis=1)


def _observe(truth: np.ndarray, points_base: np.ndarray,
             noise_mm: float, rng) -> np.ndarray:
    """What the camera would report: base points seen from the camera, plus noise."""
    world_to_cam = np.linalg.inv(truth)
    pts = (world_to_cam[:3, :3] @ points_base.T).T + world_to_cam[:3, 3]
    if noise_mm > 0:
        pts = pts + rng.normal(0.0, noise_mm / 1000.0, pts.shape)
    return pts


def check_exact_recovery(model, data) -> list[str]:
    """Clean correspondences must give back the pose exactly."""
    rng = np.random.default_rng(0)
    failures = []
    for obs in observer_calib.OBSERVERS:
        truth = _true_pose(model, data, obs.name)
        base = _marker_positions(rng)
        cam = _observe(truth, base, 0.0, rng)
        res = solve_camera_pose(cam, base)
        d_pos, d_ang = pose_error(res.cam_to_base, truth)
        if not res.ok:
            failures.append(f"{obs.name}: clean solve rejected -- {res.report()}")
        elif d_pos > 0.01 or d_ang > 0.001:
            failures.append(f"{obs.name}: recovered pose is {d_pos:.4f} mm / "
                            f"{d_ang:.5f} deg off with noise-free data")
        else:
            print(f"  {obs.name:14s} recovered to {d_pos:.5f} mm / "
                  f"{d_ang:.6f} deg, RMS {res.rms_mm:.6f} mm")
    return failures


def check_noise_degrades_gracefully(model, data) -> list[str]:
    """Realistic detector noise must give a usable pose, and say how usable.

    The printed table is the practical output: it says what marker-detection
    accuracy the real rig needs to hit a given pose accuracy, which is a
    purchasing and procedure decision, not a code one.
    """
    rng = np.random.default_rng(1)
    truth = _true_pose(model, data, "cam_overhead")
    failures = []
    print("    detector noise -> resulting camera pose error")
    prev = -1.0
    for noise_mm in (0.5, 1.0, 2.0, 5.0):
        errs = []
        for trial in range(20):
            base = _marker_positions(np.random.default_rng(100 + trial))
            cam = _observe(truth, base, noise_mm, rng)
            res = solve_camera_pose(cam, base)
            errs.append(pose_error(res.cam_to_base, truth)[0])
        median = float(np.median(errs))
        print(f"      +/- {noise_mm:4.1f} mm  ->  {median:5.2f} mm "
              f"({N_POSES} poses, median of 20 trials)")
        if median > noise_mm * 3.0:
            failures.append(f"{noise_mm} mm of detector noise produced "
                            f"{median:.1f} mm of pose error; the solve is not "
                            f"averaging observations as it should")
        if median < prev:
            failures.append("pose error did not increase with noise; the trials "
                            "are not measuring what they claim to")
        prev = median
    return failures


def check_degenerate_poses_are_refused(model, data) -> list[str]:
    """A flat set of marker positions must be rejected, not solved.

    The one that reaches hardware: jogging around a tray at one height feels
    like a thorough sweep and is a plane. It fits, the residual is small, and
    the rotation about the plane normal is barely constrained.
    """
    rng = np.random.default_rng(2)
    truth = _true_pose(model, data, "cam_overhead")

    flat = _marker_positions(rng)
    flat[:, 2] = 1.05                       # every pose at one height
    res = solve_camera_pose(_observe(truth, flat, 0.5, rng), flat)

    if res.ok:
        return [f"a coplanar pose set was accepted (spread {res.spread_ratio:.4f} "
                f">= {MIN_SPREAD_RATIO}); the guard does not fire on the failure "
                f"it exists for"]
    if "REJECTED" not in res.report():
        return ["the result is not ok but the report does not say why"]

    # And the guard must not be so eager that a good set trips it. A check that
    # rejects everything passes the half above and is useless.
    varied = _marker_positions(rng)
    good = solve_camera_pose(_observe(truth, varied, 0.5, rng), varied)
    if not good.ok:
        return [f"a properly varied pose set was ALSO rejected "
                f"(spread {good.spread_ratio:.4f}); the guard is too tight to use"]

    print(f"  coplanar set refused (spread {res.spread_ratio:.4f}), "
          f"varied set accepted (spread {good.spread_ratio:.3f})")
    return []


def check_one_bad_detection_is_visible(model, data) -> list[str]:
    """A single outlier must stand out, and dropping it must recover the pose."""
    rng = np.random.default_rng(3)
    truth = _true_pose(model, data, "cam_overhead")
    base = _marker_positions(rng)
    cam = _observe(truth, base, 0.5, rng)
    cam[5] += np.array([0.05, -0.03, 0.02])      # one misdetection, ~60 mm

    dirty = solve_camera_pose(cam, base)
    if dirty.max_mm < 3.0 * dirty.rms_mm:
        return [f"the outlier does not stand out: max {dirty.max_mm:.1f} mm vs "
                f"RMS {dirty.rms_mm:.1f} mm, so it would be absorbed silently"]

    trimmed_cam, trimmed_base, dropped = drop_worst(cam, base, dirty)
    clean = solve_camera_pose(trimmed_cam, trimmed_base)
    before = pose_error(dirty.cam_to_base, truth)[0]
    after = pose_error(clean.cam_to_base, truth)[0]
    if dropped != 5:
        return [f"drop_worst removed observation {dropped}, not the corrupted 5"]
    if after >= before:
        return [f"dropping the outlier did not help: {before:.2f} -> {after:.2f} mm"]

    print(f"  outlier flagged (max {dirty.max_mm:.0f} mm vs RMS "
          f"{dirty.rms_mm:.1f} mm); dropping it: {before:.2f} -> {after:.2f} mm")
    return []


def main() -> int:
    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    failures: list[str] = []
    for label, fn in (
        ("exact recovery from clean data",
         lambda: check_exact_recovery(model, data)),
        ("noise degrades gracefully",
         lambda: check_noise_degrades_gracefully(model, data)),
        ("degenerate pose sets are refused",
         lambda: check_degenerate_poses_are_refused(model, data)),
        ("one bad detection is visible",
         lambda: check_one_bad_detection_is_visible(model, data)),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs
        print()

    print(f"{len(failures)} failure(s)" if failures
          else "all calibration-solver checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
