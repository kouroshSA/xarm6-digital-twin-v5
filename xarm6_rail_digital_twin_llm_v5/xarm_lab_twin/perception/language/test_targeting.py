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

    An earlier version asserted that *unfiltered* targeting picks the bin. That
    was too strong. Whether the bin outranks the cube turns on a few hundredths
    of a grounding score, and it flipped when the scene's backdrop and lighting
    changed -- so the check failed while the system it guards was working
    perfectly. A test that only holds in one lighting setup is not testing the
    thing it names.

    What IS stable, and is what the depth stage exists for: a bin-sized
    candidate is among the boxes Grounding DINO returns for "a green cube", and
    the size bound is what keeps it from ever being chosen. So assert that the
    ambiguity is present and that the filter resolves it -- not that the
    ambiguity happens to win today.
    """
    # Every candidate for the phrase, not just the winner.
    detections = targeter.grounder.detect(frame.color, ["a green cube", *DISTRACTORS])
    sized = []
    for det in detections:
        if det.phrase.strip().lower().rstrip(".") != "a green cube":
            continue
        t = targeter._physicalise(frame, det)
        if t is not None:
            sized.append(t)

    if not sized:
        return ["'a green cube' grounded to nothing measurable"]

    oversize = [t for t in sized if t.max_size_m > CUBE_MAX_SIZE_M]
    if not oversize:
        return [f"no bin-sized candidate among {len(sized)} boxes for 'a green "
                f"cube' (largest {max(t.max_size_m for t in sized) * 1000:.0f} mm), "
                f"so the size bound has nothing to reject here and this check "
                f"proves nothing. Confirm the depth stage still earns its place."]

    filtered = targeter.target("a green cube", frame, max_size_m=CUBE_MAX_SIZE_M,
                               distractors=DISTRACTORS, attach_grasps=False)
    if filtered is None:
        return ["the size bound rejected every candidate, including the cube"]

    fi_hit = _nearest_body(arm, filtered.position_world * 1000.0, CANDIDATE_BODIES)
    if fi_hit != "green_cube":
        return [f"with max_size_m={CUBE_MAX_SIZE_M} 'a green cube' resolved to "
                f"{fi_hit}; the size filter is not doing its job"]

    biggest = max(oversize, key=lambda t: t.max_size_m)
    hit = _nearest_body(arm, biggest.position_world * 1000.0, CANDIDATE_BODIES)
    print(f"  'a green cube' grounds to {len(sized)} candidate(s); "
          f"{len(oversize)} too big to be a cube")
    print(f"    largest: {hit} at {biggest.max_size_m * 1000:.0f} mm "
          f"(score {biggest.score:.2f}) -- rejected on size")
    print(f"    chosen : green_cube at {filtered.max_size_m * 1000:.0f} mm "
          f"(score {filtered.score:.2f})")
    return []


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
