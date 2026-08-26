"""Does a detected grasp actually land on the object it is looking at?

Run: ``MUJOCO_GL=egl python -m perception.grasp.test_ggcnn``

Why this shape of test
----------------------
"The network produced a grasp" is not the claim that matters. A detector that
returns a confident, well-formed, entirely wrong pose passes that check every
time. The claim that matters is end-to-end: *point the wrist camera at a cube
whose position we already know, and the returned world coordinate should be that
cube* -- which exercises render, GG-CNN, deprojection, the hand-eye transform
and the world composition together. Any one of them being wrong moves the
answer, and no other test in the repo would notice.

This is CLAUDE.md's "test the CONSUMER's behaviour" rule applied to perception:
the consumer of a grasp is the arm, and what it needs is a coordinate that is
where the object is.

Because ground truth here is the cube's own body pose from MuJoCo, the
tolerances are tight -- a few centimetres, not "roughly the right half of the
table". GG-CNN grasps the visible top face of a 40 mm cube from a camera about
300 mm away, so anything beyond that is a transform bug, not model error.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from ..rgbd import RGBDFrame  # noqa: E402
from ..sim_camera import SimWristCamera  # noqa: E402
from .ggcnn import GGCNNDetector, Grasp  # noqa: E402

SCENE = os.path.join(os.path.dirname(__file__), "..", "..", "envs", "lab_scene.xml")

# Lateral tolerance: the grasp must land on the cube, which is 40 mm across.
XY_TOL_MM = 45.0
# Vertical: GG-CNN grasps the visible top face, so the point sits at the top of
# the cube rather than its centre. Generous, and not the discriminating axis.
Z_TOL_MM = 80.0


def _look_down_at(arm, x_mm: float, y_mm: float, height_mm: float) -> bool:
    """Put the flange above ``(x, y)`` pointing down, so the camera sees it."""
    code = arm.set_position(x_mm, y_mm, height_mm, 180.0, 0.0, 0.0, wait=True)
    return code == 0


def check_detects_a_known_cube(arm, cam, det) -> list[str]:
    """The headline test: look at a cube, get that cube's coordinates back."""
    failures = []
    targets = [
        ("green_cube", 0.0, -150.0),
        ("red_cube_front", 0.0, -250.0),
        ("blue_cube", 220.0, -250.0),
    ]

    for name, x, y in targets:
        bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, name)
        with arm.lock:
            truth = np.array(arm.data.xpos[bid]) * 1000.0

        if not _look_down_at(arm, x, y, 1080.0):
            failures.append(f"{name}: could not pose the arm above it; "
                            f"the test could not run, which is not a pass")
            continue

        frame = cam.capture()
        # Deep enough that a cluttered bench does not hide the target. The bench
        # holds three cubes, two bins and a cup, all genuinely graspable, so the
        # object we are pointing at is often not the single highest-scoring peak.
        grasps = det.detect(frame, top_k=8)
        if not grasps:
            failures.append(f"{name}: no grasp detected at all")
            continue

        # Nearest candidate among the top few: with several objects on the bench
        # the global best may legitimately be a different one. The test is that
        # the cube in view is found, not that it outranks everything else.
        best, best_d = None, 1e9
        for g in grasps:
            d = float(np.linalg.norm(g.position_world * 1000.0 - truth)[()])
            if d < best_d:
                best, best_d = g, d

        got = best.position_world * 1000.0
        dxy = float(np.linalg.norm((got - truth)[:2]))
        dz = abs(float(got[2] - truth[2]))
        if dxy > XY_TOL_MM or dz > Z_TOL_MM:
            failures.append(
                f"{name}: best grasp at {np.round(got, 1).tolist()} mm but the "
                f"cube is at {np.round(truth, 1).tolist()} mm "
                f"(dxy={dxy:.1f} mm, dz={dz:.1f} mm)")
        else:
            print(f"  {name:16s} grasp {np.round(got, 1).tolist()} vs truth "
                  f"{np.round(truth, 1).tolist()}  dxy={dxy:.1f} mm dz={dz:.1f} mm "
                  f"q={best.quality:.2f}")
    return failures


def check_no_pose_means_no_world_coords(det) -> list[str]:
    """A pose-less frame must refuse to produce arm coordinates, not invent them.

    The failure this guards is specific: a frame captured from a bare camera has
    no ``cam_to_world``, and if the detector quietly treated that as identity it
    would hand back camera coordinates in the shape of world ones. The arm would
    accept them and move somewhere wrong.
    """
    rng = np.random.default_rng(0)
    depth = np.full((480, 640), 0.4, dtype=np.float32)
    depth[200:280, 280:360] = 0.33          # a block to grasp
    frame = RGBDFrame(
        color=rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        depth=depth,
        intrinsics=__import__("perception.d435i_calib", fromlist=["x"]).COLOR_INTRINSICS,
        cam_to_world=None,
    )
    grasps = det.detect(frame, top_k=1)
    if not grasps:
        return ["synthetic block produced no grasp, so this check proved nothing"]
    g = grasps[0]
    if g.position_world is not None:
        return ["a pose-less frame produced a world position; it should be None"]
    try:
        g.to_arm_pose()
    except ValueError:
        print("  pose-less frame: world coords withheld and to_arm_pose() raises")
        return []
    return ["to_arm_pose() returned coordinates for a frame with no pose"]


def check_width_is_delivered(arm, cam, det) -> list[str]:
    """``width_m`` must be a plausible aperture, not a pixel count in disguise.

    Upstream computes this value and never reads it. Reviving it is only useful
    if it arrives in metres, so this asserts the magnitude rather than merely
    that the attribute exists.
    """
    _look_down_at(arm, 0.0, -150.0, 1080.0)
    grasps = det.detect(cam.capture(), top_k=1)
    if not grasps:
        return ["no grasp, cannot check width"]
    g = grasps[0]
    if not (0.005 < g.width_m < 0.30):
        return [f"width_m={g.width_m:.4f} m is not a plausible jaw opening "
                f"(width_px={g.width_px:.1f})"]
    if g.width_px <= 0:
        return ["width_px is non-positive"]
    print(f"  width delivered: {g.width_m * 1000:.1f} mm "
          f"(from {g.width_px:.1f} px at {g.depth_m:.3f} m)")
    return []


def main() -> int:
    from sim.mujoco_env import SimXArmAPI

    arm = SimXArmAPI(os.path.abspath(SCENE), render=False)
    cam = SimWristCamera(arm)
    det = GGCNNDetector()
    print(f"GG-CNN model={det.model_name} out_size={det.out_size}\n")

    failures: list[str] = []
    for label, fn in (
        ("detects a known cube", lambda: check_detects_a_known_cube(arm, cam, det)),
        ("width is delivered", lambda: check_width_is_delivered(arm, cam, det)),
        ("no pose => no world coords", lambda: check_no_pose_means_no_world_coords(det)),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs

    cam.close()
    arm.disconnect()
    print()
    print(f"{len(failures)} failure(s)" if failures else "all GG-CNN checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
