"""Do the observer cameras see the bench, and does the vision stack run through them?

Run: ``MUJOCO_GL=egl python -m perception.test_observers``

**They are where the calibration says.** A fixed camera's whole value is that its
pose is known; if the scene and ``observer_calib`` disagree, every world
coordinate it produces is wrong by that disagreement and nothing else notices.

**They measure position well**, checked without grounding at all -- project a
known object in, read the depth, deproject it back out. That isolates camera,
depth and extrinsics from whether a model matched the right box, and it is the
observer's actual job.

**The stack is camera-agnostic.** ``LanguageTargeter`` and ``GGCNNDetector``
work through a fixed camera unchanged, tested by running the real targeter
rather than by checking method names.

What is deliberately NOT asserted is that every observer identifies every
object. It does not, and the measurement is why the two cameras are placed
differently: overhead wins on position (7.7 mm vs 18.9 mm median X-Y) and loses
on identity (2/3 vs 3/3), because from above a cube shows only its flat top face
and green and blue separate by about 0.03 of grounding score. The oblique view
sees the sides. Requiring both cameras to get identity right would be testing a
claim that is false, and tuning until it passed would hide the finding that
justifies the layout.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from . import observer_calib  # noqa: E402
from .scene_camera import SceneCamera  # noqa: E402

SCENE = os.path.join(os.path.dirname(__file__), "..", "envs", "lab_scene.xml")

POSE_TOL_M = 1e-6
# An observer measures from a metre away, so it is held to a looser standard
# than the wrist camera's few millimetres. This is still tight enough that a
# wrong extrinsic or a mixed-up frame fails it.
XY_TOL_MM = 60.0

TARGETS = (("a blue cube", "blue_cube"),
           ("a red cube", "red_cube_front"),
           ("a green cube", "green_cube"))
DISTRACTORS = ("a bin", "a cup", "a test tube", "the bench")
CUBE_MAX_SIZE_M = 0.08


def _working_pose(model, data) -> None:
    """Arm up and out of the way, as it would be while surveying."""
    for name, deg in (("joint1", 0.0), ("joint2", -60.0), ("joint3", -30.0),
                      ("joint5", 90.0)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = np.radians(deg)
    mujoco.mj_forward(model, data)


def _truth(model, data, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid]) * 1000.0


def check_poses_match_the_calibration(model, data) -> list[str]:
    """The scene's cameras must sit exactly where observer_calib puts them."""
    failures = []
    for obs in observer_calib.OBSERVERS:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, obs.name)
        if cid < 0:
            failures.append(f"{obs.name} is not in the scene; run "
                            f"`python -m perception.sync_scene`")
            continue
        got = np.array(data.cam_xpos[cid])
        if not np.allclose(got, obs.pos, atol=POSE_TOL_M):
            failures.append(
                f"{obs.name} is at {np.round(got * 1000, 2).tolist()} mm, "
                f"calibration says {[round(v * 1000, 2) for v in obs.pos]} mm")
            continue

        # And it must be looking where it was told to. cam_xmat's third column
        # is the camera's +z, and a MuJoCo camera looks down -z.
        rot = np.array(data.cam_xmat[cid]).reshape(3, 3)
        view = -rot[:, 2]
        want = np.array(obs.look_at) - np.array(obs.pos)
        want /= np.linalg.norm(want)
        off = np.degrees(np.arccos(np.clip(float(view @ want), -1.0, 1.0)))
        if off > 0.01:
            failures.append(f"{obs.name} view axis is {off:.3f} deg off its "
                            f"look_at target")
        else:
            print(f"  {obs.name:14s} in place, aimed correctly, "
                  f"{obs.tilt_deg:.0f} deg off vertical")
    return failures


def check_observers_see_the_objects(model, data) -> list[str]:
    """Every cube must be visible, with valid depth, from every observer."""
    failures = []
    for obs in observer_calib.OBSERVERS:
        cam = SceneCamera.from_model(model, data, name=obs.name)
        frame = cam.capture()
        stats = frame.depth_stats()
        c2w = frame.cam_to_world
        w2c = np.linalg.inv(c2w)

        missed = []
        for _phrase, body in TARGETS:
            p_world = _truth(model, data, body) / 1000.0
            p_cam = (w2c @ np.append(p_world, 1.0))[:3]
            uv = frame.project_point(p_cam)
            if uv is None or not (0 <= uv[0] < frame.intrinsics.width
                                  and 0 <= uv[1] < frame.intrinsics.height):
                missed.append(body)
        cam.close()

        if missed:
            failures.append(f"{obs.name} cannot see {missed}; its placement "
                            f"does not cover the working strip")
        elif stats["valid_frac"] < 0.5:
            failures.append(f"{obs.name} depth is only "
                            f"{stats['valid_frac'] * 100:.0f}% valid")
        else:
            print(f"  {obs.name:14s} all {len(TARGETS)} cubes in frame, depth "
                  f"{stats['valid_frac'] * 100:.0f}% valid, median "
                  f"{stats['median_m']:.2f} m")
    return failures


def check_position_accuracy(model, data) -> list[str]:
    """How well does an observer measure a position it is TOLD to look at?

    Grounding is deliberately not involved. This projects each cube's true world
    position into the camera, reads the rendered depth there, deprojects it back
    out, and compares -- so it measures the camera, the depth and the extrinsics
    on their own. That is the observer's actual job, and testing it through the
    grounding model would confuse a bad pose with a bad match.
    """
    failures = []
    for obs in observer_calib.OBSERVERS:
        cam = SceneCamera.from_model(model, data, name=obs.name)
        frame = cam.capture()
        w2c = np.linalg.inv(frame.cam_to_world)
        errs = []
        for _phrase, body in TARGETS:
            truth = _truth(model, data, body)
            p_cam = (w2c @ np.append(truth / 1000.0, 1.0))[:3]
            uv = frame.project_point(p_cam)
            if uv is None:
                failures.append(f"{obs.name}: {body} does not project")
                continue
            got = frame.pixel_to_world(*uv)
            if got is None:
                failures.append(f"{obs.name}: no depth at {body}'s pixel")
                continue
            # The surface is seen, so the reading lands on the cube's near face
            # rather than its centre. X-Y is the axis under test.
            errs.append(float(np.linalg.norm((got * 1000.0 - truth)[:2])))
        cam.close()
        if errs:
            worst = max(errs)
            if worst > XY_TOL_MM:
                failures.append(f"{obs.name}: X-Y error up to {worst:.0f} mm")
            else:
                print(f"  {obs.name:14s} X-Y error median {np.median(errs):4.1f} mm, "
                      f"max {worst:4.1f} mm  ({obs.tilt_deg:.0f} deg off vertical)")
    return failures


def check_stack_runs_through_an_observer(model, data) -> list[str]:
    """The interface claim: LanguageTargeter works through a fixed camera.

    Requires each observer to return a plausible object -- cube-sized, on the
    bench, with a world pose -- for every phrase, which is what "the stack does
    not care which camera" means.

    It does NOT require the right cube from every camera, because that is false
    and known: overhead sees only top faces and separates green from blue by
    about 0.03 of score. See observer_calib's docstring. Identity is the wrist
    camera's job. So the assertion is that at least ONE observer resolves each
    phrase correctly -- the pair working together, which is how they are meant
    to be used -- and the per-camera outcome is printed either way.
    """
    from .language import LanguageTargeter

    targeter = LanguageTargeter()
    bodies = [b for _p, b in TARGETS] + ["green_bin", "blue_bin", "translucent_cup"]
    failures = []
    correct: dict[str, set[str]] = {}

    for obs in observer_calib.OBSERVERS:
        cam = SceneCamera.from_model(model, data, name=obs.name)
        frame = cam.capture()
        correct[obs.name] = set()
        for phrase, body in TARGETS:
            t = targeter.target(phrase, frame, max_size_m=CUBE_MAX_SIZE_M,
                                distractors=DISTRACTORS, attach_grasps=False)
            if t is None or t.position_world is None:
                failures.append(f"{obs.name}: {phrase!r} returned nothing")
                continue
            p = t.position_world * 1000.0
            if not (-800 < p[0] < 1300 and -500 < p[1] < 500 and 700 < p[2] < 1000):
                failures.append(f"{obs.name}: {phrase!r} resolved off the bench "
                                f"at {np.round(p, 0).tolist()} mm")
                continue
            hit = min(bodies, key=lambda b: np.linalg.norm(_truth(model, data, b)[:2] - p[:2]))
            if hit == body:
                correct[obs.name].add(phrase)
        cam.close()

    for name, got in correct.items():
        print(f"  {name:14s} identified {len(got)}/{len(TARGETS)} correctly: "
              f"{sorted(got) if got else 'none'}")

    for phrase, _body in TARGETS:
        if not any(phrase in got for got in correct.values()):
            failures.append(f"no observer identified {phrase!r}; the pair cannot "
                            f"survey this scene even between them")
    if not failures:
        print("  every phrase identified by at least one observer "
              "(overhead measures, oblique disambiguates)")
    return failures


def main() -> int:
    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    data = mujoco.MjData(model)
    _working_pose(model, data)

    print(observer_calib.summary())
    print()

    failures: list[str] = []
    for label, fn in (
        ("poses match the calibration",
         lambda: check_poses_match_the_calibration(model, data)),
        ("observers see the working strip",
         lambda: check_observers_see_the_objects(model, data)),
        ("position accuracy, without grounding",
         lambda: check_position_accuracy(model, data)),
        ("the stack runs through an observer",
         lambda: check_stack_runs_through_an_observer(model, data)),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs
        print()

    print(f"{len(failures)} failure(s)" if failures
          else "all observer checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
