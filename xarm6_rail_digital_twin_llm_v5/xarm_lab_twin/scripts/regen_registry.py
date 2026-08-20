#!/usr/bin/env python3
"""Regenerate `agent/objects.json` positions from the scene XML.

`position_xyz_m` is a *seed*: `ObjectRegistry.refresh_from_sim()` overwrites it from
the live sim before every LLM call, so a stale value on disk is usually masked at
runtime. That masking is exactly why it drifted unnoticed — 9 of 19 entries were
still describing the pre-`fde3d22` layout.

Anything reading the registry without a live sim (tooling, tests, a future real-arm
backend that has no `get_body_pose`) gets the seed, so the seed has to be right.

    python scripts/regen_registry.py            # rewrite in place
    python scripts/regen_registry.py --check    # report drift, change nothing
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.getcwd())

from agent.scene_geometry import DEFAULT_SCENE, load

REGISTRY = "agent/objects.json"
TOL_M = 0.001  # 1 mm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; do not write")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--registry", default=REGISTRY)
    args = ap.parse_args()

    scene = load(args.scene)
    raw = json.load(open(args.registry))

    drifted, missing = [], []
    for name, obj in raw.items():
        info = scene.get(name)
        if info is None:
            missing.append(name)
            continue
        want = [round(c / 1000.0, 6) for c in info.pos_mm]
        have = obj.get("position_xyz_m")
        if have is None or any(abs(a - b) > TOL_M for a, b in zip(have, want)):
            drifted.append((name, have, want))
            obj["position_xyz_m"] = want

    for name, have, want in drifted:
        hs = "none" if have is None else f"({have[0]*1000:.0f},{have[1]*1000:.0f})"
        print(f"  {name:<18} {hs:>16} -> ({want[0]*1000:.0f},{want[1]*1000:.0f})")
    if missing:
        print(f"  NOT IN SCENE (left untouched): {missing}")

    if args.check:
        print(f"\n{len(drifted)} drifted, {len(missing)} absent from scene")
        return 1 if drifted else 0

    if drifted:
        with open(args.registry, "w") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")
        print(f"\nrewrote {args.registry}: {len(drifted)} position(s) updated")
    else:
        print("\nno drift; nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
