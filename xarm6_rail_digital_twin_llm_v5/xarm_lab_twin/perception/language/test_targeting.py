"""Does "the blue cube" actually resolve to the blue cube?

Run: ``MUJOCO_GL=egl python -m perception.language.test_targeting``

Ground truth is the scene itself: MuJoCo knows where every body is, so a phrase
can be checked against the object it is supposed to name. That makes this the
only place in the stack where language targeting is *verified* rather than
demonstrated -- a targeter that confidently returns the wrong object produces
exactly as pretty a picture as one that works.

The suite deliberately includes the case that fails without depth. "a green
cube" grounds to the green **bin** as readily as to the green cube, because a
bin is a green box and a single image carries no scale. ``check_size_filter_
is_what_fixes_it`` asserts both halves of that: unfiltered targeting picks the
bin, and the size bound picks the cube. If someone later removes the depth stage
believing grounding alone is enough, the first half starts passing for the wrong
reason and the second half fails outright.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from ..sim_camera import SimWristCamera  # noqa: E402
from .targeter import LanguageTargeter  # noqa: E402

SCENE = os.path.join(os.path.dirname(__file__), "..", "..", "envs", "lab_scene.xml")

# A cube is 40 mm; the bins are ~150 mm. 80 mm sits well clear of both, so the
# bound is a statement about the objects, not a tuned threshold.
CUBE_MAX_SIZE_M = 0.08

# The target must land on the cube, which is 40 mm across.
XY_TOL_MM = 45.0

# Naming the alternatives sharpens Grounding DINO's scores for the thing we want.
DISTRACTORS = ("a bin", "a cup", "a test tube")

# Flange pose that puts the three cubes and both bins in one frame.
VIEW = (60.0, -200.0, 1180.0)


def _truth(arm, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"no body {name!r} in the scene")
    with arm.lock:
        return np.array(arm.data.xpos[bid]) * 1000.0


def _nearest_body(arm, xy_mm: np.ndarray, names) -> str:
    return min(names, key=lambda n: float(np.linalg.norm(_truth(arm, n)[:2] - xy_mm[:2])))


CANDIDATE_BODIES = ("red_cube_front", "red_cube_back", "green_cube", "blue_cube",
                    "green_bin", "blue_bin", "translucent_cup")


def check_phrases_hit_the_right_object(arm, frame, targeter) -> list[str]:
    """Each colour phrase must resolve to that colour's cube, not a neighbour."""
    failures = []
    for phrase, expect in (("a blue cube", "blue_cube"),
                           ("a red cube", "red_cube_front"),
                           ("a green cube", "green_cube")):
        t = targeter.target(phrase, frame, max_size_m=CUBE_MAX_SIZE_M,
                            distractors=DISTRACTORS)
        if t is None:
            failures.append(f"{phrase!r}: nothing grounded")
            continue
        if t.position_world is None:
            failures.append(f"{phrase!r}: no world position")
            continue

        got = t.position_world * 1000.0
        hit = _nearest_body(arm, got, CANDIDATE_BODIES)
        err = float(np.linalg.norm(_truth(arm, expect)[:2] - got[:2]))
        if hit != expect or err > XY_TOL_MM:
            failures.append(
                f"{phrase!r}: resolved to {hit} ({err:.1f} mm from {expect}), "
                f"size {t.max_size_m * 1000:.0f} mm, score {t.score:.2f}")
        else:
            print(f"  {phrase!r:16s} -> {expect:15s} {err:5.1f} mm  "
                  f"size {t.max_size_m * 1000:3.0f} mm  score {t.score:.2f}  "
                  f"{len(t.grasps)} grasp(s)")
    return failures


def check_size_filter_is_what_fixes_it(arm, frame, targeter) -> list[str]:
    """The load-bearing claim: depth, not grounding, separates cube from bin.

    Asserts BOTH halves. If grounding alone were sufficient the first assertion
    would fail and this check would tell us the depth stage is now dead weight --
    which is worth knowing too.
    """
    unfiltered = targeter.target("a green cube", frame, distractors=DISTRACTORS,
                                 attach_grasps=False)
    filtered = targeter.target("a green cube", frame, max_size_m=CUBE_MAX_SIZE_M,
                               distractors=DISTRACTORS, attach_grasps=False)

    if unfiltered is None or filtered is None:
        return ["'a green cube' did not ground; the comparison cannot be made"]

    un_hit = _nearest_body(arm, unfiltered.position_world * 1000.0, CANDIDATE_BODIES)
    fi_hit = _nearest_body(arm, filtered.position_world * 1000.0, CANDIDATE_BODIES)

    failures = []
    if un_hit == "green_cube":
        failures.append(
            "grounding alone already picked green_cube, so this check no longer "
            "demonstrates anything. Verify the size filter still matters before "
            "trusting it -- or drop the depth stage if it has become redundant.")
    if fi_hit != "green_cube":
        failures.append(
            f"with max_size_m={CUBE_MAX_SIZE_M} 'a green cube' still resolved to "
            f"{fi_hit}; the size filter is not doing its job")
    if not failures:
        print(f"  without size filter: {un_hit} "
              f"({unfiltered.max_size_m * 1000:.0f} mm)")
        print(f"  with    size filter: {fi_hit} "
              f"({filtered.max_size_m * 1000:.0f} mm)  <- depth resolves it")
    return failures


def check_mask_beats_the_box(frame, targeter) -> list[str]:
    """The depth mask must be a real subset of the box, not the whole thing.

    A box around a 40 mm cube from 300 mm away is mostly bench. If the mask were
    the box, the centroid would sit beside the cube rather than on it, and the
    size estimate would measure the bench.
    """
    t = targeter.target("a blue cube", frame, max_size_m=CUBE_MAX_SIZE_M,
                        distractors=DISTRACTORS, attach_grasps=False)
    if t is None:
        return ["'a blue cube' did not ground"]

    x0, y0, x1, y1 = t.box
    box_px = max(1.0, (x1 - x0) * (y1 - y0))
    frac = t.n_pixels / box_px
    if frac > 0.95:
        return [f"the mask is {frac * 100:.0f}% of the box -- depth is not "
                f"separating the object from the surface behind it"]
    print(f"  mask is {frac * 100:.0f}% of the grounding box "
          f"({t.n_pixels} px), so the centroid sits on the object")
    return []


def check_no_match_returns_none(frame, targeter) -> list[str]:
    """An absent object must return None, not the closest thing available."""
    t = targeter.target("a rubber duck", frame, max_size_m=CUBE_MAX_SIZE_M,
                        distractors=DISTRACTORS, attach_grasps=False)
    if t is not None:
        return [f"'a rubber duck' resolved to something at "
                f"{np.round(t.position_world * 1000, 1).tolist()} mm "
                f"(score {t.score:.2f}); it should have returned None"]
    print("  'a rubber duck' -> None, as it should")
    return []


def main() -> int:
    from sim.mujoco_env import SimXArmAPI

    arm = SimXArmAPI(os.path.abspath(SCENE), render=False)
    cam = SimWristCamera(arm)
    targeter = LanguageTargeter()
    print(f"grounder device: {targeter.grounder.device}\n")

    if arm.set_position(*VIEW, 180.0, 0.0, 0.0, wait=True) != 0:
        print(f"could not reach the viewing pose: {arm.last_refusal}")
        return 1
    frame = cam.capture()

    failures: list[str] = []
    for label, fn in (
        ("phrases hit the right object",
         lambda: check_phrases_hit_the_right_object(arm, frame, targeter)),
        ("depth is what separates cube from bin",
         lambda: check_size_filter_is_what_fixes_it(arm, frame, targeter)),
        ("mask is tighter than the box",
         lambda: check_mask_beats_the_box(frame, targeter)),
        ("absent object returns None",
         lambda: check_no_match_returns_none(frame, targeter)),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs
        print()

    cam.close()
    arm.disconnect()
    print(f"{len(failures)} failure(s)" if failures
          else "all language-targeting checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
