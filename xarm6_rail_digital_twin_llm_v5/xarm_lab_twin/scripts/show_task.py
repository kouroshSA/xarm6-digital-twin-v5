#!/usr/bin/env python3
"""Watch the behavioural sweep tasks run in the MuJoCo viewer.

The sweep runs headless and prints a verdict. This runs the **same task
functions**, imported from `scripts/task_sweep.py`, with a window open and each
command narrated as it executes. Use it to see *why* a task passes or fails
rather than inferring it from one line of output.

    python scripts/show_task.py                    # all four, in order
    python scripts/show_task.py --task cube
    python scripts/show_task.py --task wellplate --scene primitive
    python scripts/show_task.py --list

The task bodies are NOT copied here. Duplicating them is how a "demo" quietly
drifts from the test it claims to demonstrate, which is the defect class this
repo keeps hitting. Narration comes from wrapping the arm's methods on the
instance, so the sequence itself stays in exactly one place.

Two things differ from the headless sweep, both deliberate:

- Each task gets a **fresh scene**, same as the sweep, so tasks cannot
  contaminate each other.
- `task_sweep._settle()` is viewer-aware: with a window open the sim thread is
  already stepping physics, and stepping it again by hand races the viewer's
  mjData copy (MuJoCo aborts, and the GL context goes down with it).

`pacing is real` is excluded by default -- it is a timing assertion with
nothing to look at. Pass `--task pacing` if you want to watch it anyway.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

# A windowed viewer needs a real GL context; EGL is for the headless path.
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("MUJOCO_GL")

from scripts.task_sweep import BEHAVIOURAL_TASKS, SCENES

#: CLI name -> the label used in BEHAVIOURAL_TASKS / the sweep output.
TASK_KEYS = {
    "cube": "cube pick/place",
    "tube": "tube -> rack",
    "wellplate": "well-plate (bio)",
    "binpush": "bin push",
    "pacing": "pacing is real",
}
DEFAULT_ORDER = ["cube", "tube", "wellplate", "binpush"]

#: Arm methods worth narrating. Anything not listed runs silently.
NARRATE = (
    "set_position", "set_rail_position", "set_servo_angle",
    "close_lite6_gripper", "open_lite6_gripper", "close_gripper", "open_gripper",
    "push_object", "place_tube_in_rack", "set_gripper", "go_home",
)


def _fmt(name, args, kwargs):
    if name == "set_position" and len(args) >= 3:
        x, y, z = args[0], args[1], args[2]
        return f"move to ({x:7.1f}, {y:7.1f}, {z:7.1f})"
    if name == "set_rail_position" and args:
        return f"rail -> {args[0]:.0f} mm"
    if name == "set_gripper" and args:
        return f"fit the {args[0]} gripper"
    if name == "push_object":
        return (f"push {kwargs.get('target_name', '?')} to "
                f"({kwargs.get('to_x_mm', 0):.0f}, {kwargs.get('to_y_mm', 0):.0f})")
    if name == "place_tube_in_rack":
        return f"place tube in {kwargs.get('rack_name', '?')}"
    return name.replace("_", " ")


def narrate(arm):
    """Wrap the interesting methods on this instance so each call prints.

    Instance attributes shadow the class methods, so nothing global is patched
    and no other sim in this process is affected.
    """
    counter = {"n": 0}

    def wrap(name, fn):
        def inner(*args, **kwargs):
            counter["n"] += 1
            t0 = time.time()
            rc = fn(*args, **kwargs)
            dt = time.time() - t0
            flag = "" if rc in (0, None, True) else f"  <-- rc={rc}"
            print(f"    {counter['n']:>2}. {_fmt(name, args, kwargs):<34} "
                  f"{dt:5.1f}s{flag}", flush=True)
            return rc
        return inner

    for name in NARRATE:
        fn = getattr(arm, name, None)
        if callable(fn):
            setattr(arm, name, wrap(name, fn))
    return arm


def run_one(key: str, scene_key: str, hold: float) -> bool:
    from sim.mujoco_env import SimXArmAPI

    label = TASK_KEYS[key]
    fn = dict(BEHAVIOURAL_TASKS)[label]
    print(f"\n{'=' * 66}\n  {label}   [scene: {scene_key}]\n{'=' * 66}")

    arm = SimXArmAPI(scene_xml=SCENES[scene_key], render=True)
    try:
        time.sleep(2.0)          # let the window map before anything moves
        narrate(arm)
        ok, detail = fn(arm)
        print(f"\n  {'PASS' if ok else 'FAIL'}  {detail}")
        try:
            print(f"  outcome: {arm.physical_outcome().split(';')[0]}")
        except Exception:        # noqa: BLE001 - narration only
            pass
        time.sleep(hold)
        return ok
    finally:
        try:
            arm.disconnect()
        except Exception:        # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASK_KEYS) + ["all"], default="all")
    ap.add_argument("--scene", choices=sorted(SCENES), default="meshes")
    ap.add_argument("--hold", type=float, default=3.0,
                    help="seconds to keep the window open after each task")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k in sorted(TASK_KEYS):
            mark = "" if k in DEFAULT_ORDER else "   (not in --task all)"
            print(f"  {k:<10} {TASK_KEYS[k]}{mark}")
        return 0

    keys = DEFAULT_ORDER if args.task == "all" else [args.task]
    results = [(k, run_one(k, args.scene, args.hold)) for k in keys]

    if len(results) > 1:
        print(f"\n{'=' * 66}")
        for k, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {TASK_KEYS[k]}")
        n_fail = sum(1 for _, ok in results if not ok)
        print(f"  {len(results) - n_fail} PASS  {n_fail} FAIL")
    return 1 if any(not ok for _, ok in results) else 0


if __name__ == "__main__":
    sys.exit(main())
