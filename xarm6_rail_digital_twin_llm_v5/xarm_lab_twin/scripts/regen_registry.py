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


def check_build_default_registry(scene) -> list:
    """The THIRD copy of every object position, and the one that reaches the arm.

    `objects.json` is a seed; `build_default_registry()` in
    agent/object_registry.py hardcodes the same numbers in Python, and that is
    what run_task actually calls. Nothing compared it to anything.

    It mattered on hardware. `refresh_from_sim()` masks a stale value in the
    sim by overwriting it from live bodies every episode, but RealXArmAPI has
    no `get_body_pose` -- object poses on a real cell come from perception --
    so the real arm is driven straight from these hardcoded numbers. Two of
    them described a layout the cell has never had: the cup was 100 mm further
    from the rail than it is, and the cubes 20 mm closer together.

    Reported, not rewritten: these live in Python next to comments explaining
    where each object sits, and a script that edited them would leave the prose
    asserting the old value -- swapping a wrong number for a wrong sentence.
    """
    try:
        sys.path.insert(0, os.getcwd())
        from agent.object_registry import build_default_registry
        reg = build_default_registry()
    except Exception as exc:                                   # noqa: BLE001
        print(f"  build_default_registry() not checked ({type(exc).__name__}: {exc})")
        return []

    drift = []
    for name, obj in reg.objects.items():
        info = scene.get(name)
        if info is None:
            continue
        want = [c / 1000.0 for c in info.pos_mm]
        have = list(obj.position_xyz_m or [])
        if len(have) != 3 or any(abs(a - b) > TOL_M for a, b in zip(have, want)):
            drift.append((name, have, want))
            print(f"  [code] {name:<18} "
                  f"({have[0]*1000:.0f},{have[1]*1000:.0f}) -> "
                  f"({want[0]*1000:.0f},{want[1]*1000:.0f})")
    return drift


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; do not write")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--registry", default=REGISTRY)
    args = ap.parse_args()

    scene = load(args.scene)

    # agent/objects.json is gitignored -- a derived artifact, absent on a fresh
    # clone. It used to be created as a side effect of ObjectRegistry.register()
    # saving on every call; that side effect is gone (it was silently clobbering
    # this file from stale Python), so seed it here instead of dying on a
    # checkout that has simply never built one.
    if not os.path.exists(args.registry):
        print(f"{args.registry} absent -- seeding it from "
              f"build_default_registry()")
        sys.path.insert(0, os.getcwd())
        from agent.object_registry import build_default_registry
        build_default_registry().save()

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

    code_drift = check_build_default_registry(scene)

    if args.check:
        print(f"\n{len(drifted)} drifted, {len(missing)} absent from scene")
        if code_drift:
            print(f"{len(code_drift)} drifted in build_default_registry() "
                  f"-- edit agent/object_registry.py by hand; this script "
                  f"cannot rewrite Python")
        return 1 if (drifted or code_drift) else 0

    if drifted:
        with open(args.registry, "w") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")
        print(f"\nrewrote {args.registry}: {len(drifted)} position(s) updated")
    else:
        print("\nno drift; nothing written")
    if code_drift:
        print(f"{len(code_drift)} position(s) in build_default_registry() still "
              f"disagree with the scene -- fix agent/object_registry.py by hand")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
