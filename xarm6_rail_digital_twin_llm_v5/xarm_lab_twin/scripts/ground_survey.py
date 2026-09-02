#!/usr/bin/env python3
"""Ground named objects in a saved wrist survey, rejecting the robot's own parts.

Offline companion to ``scripts/wrist_survey.py``: point it at a survey
directory and it reports where each phrase grounds, across every frame.

NO WINNER-TAKE-ALL. An earlier version consolidated overlapping detections by
awarding each location to the highest-scoring phrase. It is deleted because it
is unsound: Grounding DINO's scores are NOT comparable between prompts. On this
bench "a blue bottle cap" scored 0.85 on a blue CUBE, beating "a blue cube" on
the same object (0.65-0.72), so the consolidation confidently mislabelled it.
Compare a phrase against ITSELF across views -- that is what the per-phrase
medians below do -- and disambiguate similar objects by grounding both phrases
and requiring them to resolve far apart, not by comparing their scores.

No arm, no camera. Frames are rebuilt from the survey directory exactly as
captured (colour, depth, intrinsics, cam_to_world).

THE FILTER. Grounding DINO never returns nothing: ask for an object that is not
there and it grounds to whatever is in frame, which over this bench is the rail
and the gripper mount at the edge of every view. Those cannot be told apart from
a real object by score alone (decoys ran 0.27-0.38, real objects 0.64-0.87 --
overlapping ranges), and the physical size filter does not catch them either,
because a piece of rail measures a plausible 117 mm.

What DOES tell them apart is a physical invariant:

    a bench object holds still in WORLD coordinates as the rail moves;
    anything bolted to the robot or the rail holds still in BASE coordinates.

The nadir pass sweeps the rail 0 -> 700, so it can measure both. Any cluster
whose base-frame position is constant while its world position slides with the
carriage is the robot looking at itself. Those zones are learned here and then
applied to every pass -- including the oblique frames, which are all taken from
one rail position and so could never run the test on their own.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

from arm_backend import world_to_base_mm
from perception.d435i_calib import COLOR_INTRINSICS
from perception.rgbd import RGBDFrame
from perception.language import LanguageTargeter

SURVEY = sys.argv[1] if len(sys.argv) > 1 else "/tmp/wrist_survey"

#: (phrase, min lateral size m, max lateral size m).
QUERIES = [
    ("a blue cube",                 0.010, 0.060),
    ("a red cube",                  0.010, 0.060),
    ("a test tube rack",            0.060, 0.170),
    ("a long test tube rack",       0.200, 0.340),
    ("a tube with a blue cap",      0.008, 0.075),
    ("a tube with an orange cap",   0.008, 0.075),
    ("a blue bottle cap",           0.008, 0.075),
    ("an orange bottle cap",        0.008, 0.075),
    ("a cup",                       0.030, 0.150),
    ("a rubber duck",               0.010, 0.150),   # NEGATIVE CONTROL
]

#: How close two detections must be to count as the same thing, mm.
CLUSTER_TOL_MM = 70.0
#: How far a detection must sit from a learned rail-fixed zone to survive, mm.
EXCLUDE_RADIUS_MM = 90.0


def load_frames(survey_dir):
    import imageio.v2 as imageio
    man = json.load(open(os.path.join(survey_dir, "manifest.json")))
    out = []
    for e in man:
        if not e["cam_to_world"]:
            continue
        color = imageio.imread(os.path.join(survey_dir, e["files"] + "_color.png"))
        depth = np.load(os.path.join(survey_dir, e["files"] + "_depth.npy"))
        out.append((e, RGBDFrame(color=np.asarray(color)[:, :, :3], depth=depth,
                                 intrinsics=COLOR_INTRINSICS,
                                 cam_to_world=np.array(e["cam_to_world"]),
                                 source="real")))
    return out


def detect_all(targeter, frames):
    """Every (phrase -> list of detections) with both world and base positions."""
    out = {}
    for phrase, lo, hi in QUERIES:
        hits = []
        for e, frame in frames:
            try:
                t = targeter.target(phrase, frame, max_size_m=hi, min_size_m=lo)
            except Exception:                                # noqa: BLE001
                continue
            if t is None:
                continue
            p = np.asarray(t.position_world, dtype=float)
            if np.max(np.abs(p)) < 10:
                p = p * 1000.0
            rail = float(e["rail_mm"])
            hits.append(dict(frame=e["files"], pass_name=e["pass_name"], rail=rail,
                             world=p, base=np.array(world_to_base_mm(p, rail)),
                             size=float(t.max_size_m) * 1000.0,
                             score=float(getattr(t, "score", np.nan))))
        out[phrase] = hits
    return out


def learn_rail_fixed_zones(all_hits):
    """Base-frame points where detections sit still while the world slides.

    Uses every phrase's detections together: a piece of rail is a piece of rail
    whatever word summoned it, and pooling them gives the test more evidence.
    """
    pool = [h for hits in all_hits.values() for h in hits
            if h["pass_name"] == "nadir"]
    zones, used = [], set()
    for i, a in enumerate(pool):
        if i in used:
            continue
        group = [j for j, b in enumerate(pool)
                 if np.linalg.norm(a["base"][:2] - b["base"][:2]) < CLUSTER_TOL_MM]
        if len(group) < 3:
            continue
        members = [pool[j] for j in group]
        world_spread = float(np.ptp([m["world"][0] for m in members]))
        base_spread = float(np.ptp([m["base"][0] for m in members]))
        rails = {m["rail"] for m in members}
        # Constant in base, sliding in world, seen from several rail positions.
        if len(rails) >= 3 and world_spread > 150 and base_spread < 40:
            used.update(group)
            zones.append(dict(base=np.median([m["base"] for m in members], axis=0),
                              n=len(members), world_spread=world_spread,
                              base_spread=base_spread,
                              phrases=sorted({m["_phrase"] for m in members
                                              if "_phrase" in m})))
    return zones


def main():
    frames = load_frames(SURVEY)
    by_pass = {}
    for e, _ in frames:
        by_pass[e["pass_name"]] = by_pass.get(e["pass_name"], 0) + 1
    print(f"loaded {len(frames)} frames: {by_pass}")

    targeter = LanguageTargeter(device="auto")
    all_hits = detect_all(targeter, frames)
    for phrase, hits in all_hits.items():
        for h in hits:
            h["_phrase"] = phrase

    zones = learn_rail_fixed_zones(all_hits)
    print(f"\n=== rail-fixed zones learned from the nadir sweep: {len(zones)} ===")
    for z in zones:
        print(f"  base ({z['base'][0]:6.1f}, {z['base'][1]:6.1f}, {z['base'][2]:6.1f}) "
              f"from {z['n']} detections | world x moved {z['world_spread']:.0f} mm, "
              f"base x moved {z['base_spread']:.0f} mm")
        print(f"     summoned by: {', '.join(z['phrases'])}")
    if not zones:
        print("  none -- nothing tracked the carriage; no exclusion applied")

    def is_robot(h):
        return any(np.linalg.norm(h["base"][:2] - z["base"][:2]) < EXCLUDE_RADIUS_MM
                   for z in zones)

    print("\n=== detections, robot hardware removed ===")
    for phrase, hits in all_hits.items():
        kept = [h for h in hits if not is_robot(h)]
        drop = len(hits) - len(kept)
        tag = "   [negative control]" if "duck" in phrase else ""
        print(f"\n{phrase!r}: {len(kept)} kept, {drop} rejected as robot{tag}")
        for h in sorted(kept, key=lambda h: -h["world"][0]):
            print(f"    {h['frame']:14s} {h['pass_name']:8s} world "
                  f"({h['world'][0]:7.1f}, {h['world'][1]:7.1f}, {h['world'][2]:7.1f}) "
                  f"size {h['size']:5.1f}  score {h['score']:.2f}")
        if len(kept) >= 2:
            P = np.array([h["world"] for h in kept])
            med = np.median(P, axis=0)
            dev = np.max(np.abs(P - med), axis=0)
            print(f"      median ({med[0]:7.1f}, {med[1]:7.1f}, {med[2]:7.1f})  "
                  f"max deviation ({dev[0]:.0f}, {dev[1]:.0f}, {dev[2]:.0f}) mm")


if __name__ == "__main__":
    main()
