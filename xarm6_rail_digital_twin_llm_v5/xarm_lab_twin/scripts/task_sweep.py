#!/usr/bin/env python3
"""Regression harness for the sim: static invariants + behavioural task sweep.

Two layers, deliberately separated by cost:

- **static** (`scripts/sim_checks.py`) — scene/prompt/registry consistency. No physics,
  runs in ~1s. This is the layer an edit-verify loop should gate on.
- **behavioural** — actually drives the sim through a pick/place, tube-to-rack,
  well-plate and bin-push task on a fresh scene each time. Seconds to minutes.

**Exit code is the contract**: non-zero if any check FAILs, so this can gate a loop or
a pre-commit hook. The previous version of this script always exited 0, which meant a
FAIL was invisible to anything but a human reading the output.

Usage:
    python scripts/task_sweep.py                      # static only (fast, default)
    python scripts/task_sweep.py --json               # machine-readable
    python scripts/task_sweep.py --behavioural        # + drive the sim (needs MUJOCO_GL)
    MUJOCO_GL=egl python scripts/task_sweep.py --behavioural --scene both
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.getcwd())

from scripts.sim_checks import FAIL, PASS, SKIP, CheckResult, run_static_checks

SCENES = {"primitive": "envs/lab_scene_primitive.xml", "meshes": "envs/lab_scene.xml"}


# ---------------------------------------------------------------------------
# Behavioural tasks (unchanged in intent from the original sweep)
# ---------------------------------------------------------------------------

def _bpos(arm, n):
    """Body position in mm. Viewer-aware, for the same reason as _settle.

    mj_forward mutates mjData. With a viewer live the sim thread is already
    forwarding every step, so calling it from here is both unnecessary and a
    race against the viewer's copy -- the abort is
    "attempting to copy mjData while stack is in use". Read-only when a window
    is open; forward explicitly when headless, where nothing else is stepping.
    """
    import mujoco
    with arm.lock:
        if getattr(arm, "_viewer", None) is None:
            mujoco.mj_forward(arm.model, arm.data)
        return arm.data.xpos[arm.model.body(n).id].copy() * 1000.0


def _weld_active(arm, name):
    with arm.lock:
        return bool(arm.data.eq_active[arm.weld_eqids[name]])


def _settle(arm, n=200):
    """Advance physics so objects come to rest before measuring.

    Viewer-aware. With `render=True` the sim already has a thread stepping
    physics, and stepping it a second time from here races the viewer's mjData
    copy -- MuJoCo aborts with "attempting to copy mjData while stack is in
    use", taking the GL context down with it (GLXBadDrawable). When a viewer is
    live we just wait instead and let the sim thread do the stepping. Headless,
    which is how the sweep runs, the manual loop is still what advances time.
    """
    import mujoco
    import time as _time
    if getattr(arm, "_viewer", None) is not None:
        _time.sleep(max(n, 1) / 200.0)
        return
    for _ in range(n):
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)


def task_cube_pickplace(arm):
    cube = _bpos(arm, "green_cube"); binp = _bpos(arm, "green_bin")
    arm.set_rail_position(350.0, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 100, roll=180, wait=True)
    arm.set_position(cube[0], cube[1], cube[2] + 15, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = _weld_active(arm, "green_cube")
    arm.set_position(cube[0], cube[1], cube[2] + 150, roll=180, wait=True)
    arm.set_position(binp[0], binp[1], binp[2] + 150, roll=180, wait=True)
    arm.set_position(binp[0], binp[1], binp[2] + 60, roll=180, wait=True)
    arm.open_lite6_gripper(); _settle(arm)
    final = _bpos(arm, "green_cube")
    d = np.linalg.norm(final[:2] - binp[:2])
    ok = grasped and d < 80 and "green_cube in green_bin" in arm.physical_outcome()
    return ok, f"grasped={grasped} final_xy_to_bin={d:.0f}mm"


def task_tube_to_rack(arm):
    tube = _bpos(arm, "tube_L1")
    arm.set_rail_position(0.0, wait=True)
    r1 = arm.set_position(tube[0], tube[1], tube[2] + 120, roll=180, wait=True)
    r2 = arm.set_position(tube[0], tube[1], tube[2] + 20, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = any(_weld_active(arm, f"tube_{s}") for s in ["L1", "L2", "L3", "R1", "R2", "R3"])
    arm.set_position(tube[0], tube[1], tube[2] + 150, roll=180, wait=True)
    rc = arm.place_tube_in_rack(rack_name="right_tube_rack")
    _settle(arm)
    return (grasped and rc == 0), f"reach_rc=({r1},{r2}) grasped={grasped} place_rc={rc}"


def task_wellplate_bio(arm):
    arm.set_gripper("bio")
    plate = _bpos(arm, "well_plate_B")
    arm.set_rail_position(700.0, wait=True)
    r1 = arm.set_position(plate[0], plate[1], plate[2] + 120, roll=180, wait=True)
    r2 = arm.set_position(plate[0], plate[1], plate[2] + 20, roll=180, wait=True)
    arm.close_lite6_gripper()
    grasped = _weld_active(arm, "well_plate_B")
    arm.set_position(plate[0], plate[1], plate[2] + 120, roll=180, wait=True)
    lifted = _bpos(arm, "well_plate_B")[2]
    arm.open_lite6_gripper(); _settle(arm)
    arm.set_gripper("standard")
    return (grasped and lifted > plate[2] + 40), \
        f"reach_rc=({r1},{r2}) grasped={grasped} lifted_z={lifted:.0f}(was {plate[2]:.0f})"


def task_bin_push(arm):
    b0 = _bpos(arm, "blue_bin")
    rc = arm.push_object(target_name="blue_bin", to_x_mm=b0[0], to_y_mm=b0[1] - 120,
                         speed_mm_s=80.0)
    _settle(arm)
    b1 = _bpos(arm, "blue_bin")
    moved = np.linalg.norm(b1[:2] - b0[:2])
    return (rc == 0 and moved > 60), f"push_rc={rc} bin_moved={moved:.0f}mm"


def task_pacing_is_real(arm):
    """A commanded speed must cost wall-clock time, not just appear in a log.

    MuJoCo position actuators have no velocity limit, so speed only exists
    because `_execute_paced_arm` interpolates over `distance / speed` seconds.
    Anything that makes set_position think it has already arrived -- notably a
    diagnostic that writes qpos without restoring it -- silently reduces every
    move to a teleport while still returning 0 and still logging the speed.

    This check has caught that twice: once when the speed kwarg was accepted and
    ignored outright, and once when `_ik_pos_error` left the arm sitting on the
    target so the paced distance came out zero. Both looked correct in logs.

    Asserts a slow move takes materially longer than a fast one over the same
    path, and that a slow move is not instantaneous. Ratios rather than absolute
    times, so it does not turn into a flaky benchmark on a loaded machine.
    """
    import time
    arm.set_rail_position(350.0, wait=True)
    arm.set_position(292, -250, 950, roll=180, pitch=0, yaw=0, speed=120, wait=True)

    def timed(z, speed):
        t0 = time.time()
        rc = arm.set_position(292, -250, z, roll=180, pitch=0, yaw=0,
                              speed=speed, wait=True)
        return rc, time.time() - t0

    rc_slow, t_slow = timed(800, 20)          # ~150 mm at 20 mm/s -> ~7.5 s
    rc_back, _ = timed(950, 120)
    rc_fast, t_fast = timed(800, 120)         # same path at 120 mm/s -> ~1.3 s

    if rc_slow != 0 or rc_fast != 0 or rc_back != 0:
        return False, f"moves failed: slow={rc_slow} back={rc_back} fast={rc_fast}"
    if t_slow < 2.0:
        return False, (f"20 mm/s move took {t_slow:.2f}s -- not paced at all "
                       f"(pacing skipped or bypassed)")
    ratio = t_slow / max(t_fast, 1e-3)
    return ratio > 2.0, (f"slow={t_slow:.2f}s fast={t_fast:.2f}s ratio={ratio:.1f}x "
                         f"(need >2x and slow >2s)")


BEHAVIOURAL_TASKS = [
    ("cube pick/place", task_cube_pickplace),
    ("tube -> rack", task_tube_to_rack),
    ("well-plate (bio)", task_wellplate_bio),
    ("bin push", task_bin_push),
    ("pacing is real", task_pacing_is_real),
]


def run_behavioural(scene_key: str) -> list[CheckResult]:
    from sim.mujoco_env import SimXArmAPI
    results = []
    for label, fn in BEHAVIOURAL_TASKS:
        name = f"behaviour[{scene_key}].{label}"
        arm = None
        try:
            arm = SimXArmAPI(scene_xml=SCENES[scene_key], render=False)  # fresh scene per task
            ok, detail = fn(arm)
            results.append(CheckResult(name, PASS if ok else FAIL, detail))
        except Exception as exc:  # noqa: BLE001 - reported as FAIL, never silently passed
            results.append(CheckResult(name, FAIL, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(file=sys.stderr)
        finally:
            if arm is not None:
                try:
                    arm.disconnect()
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: list[CheckResult], as_json: bool) -> int:
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_pass = sum(1 for r in results if r.status == PASS)
    n_skip = sum(1 for r in results if r.status == SKIP)

    if as_json:
        json.dump({
            "summary": {"pass": n_pass, "fail": n_fail, "skip": n_skip, "total": len(results)},
            "results": [r.as_dict() for r in results],
        }, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            print(f"  [{r.status:<4}] {r.name:<40} {r.detail}")
        print(f"\n  {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP")

    return 1 if n_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--behavioural", action="store_true",
                    help="also drive the sim through the task suite (slow)")
    ap.add_argument("--scene", default="meshes", choices=["primitive", "meshes", "both"],
                    help="scene(s) for the behavioural pass (default: meshes)")
    args = ap.parse_args()

    results = run_static_checks()

    if args.behavioural:
        for key in (["primitive", "meshes"] if args.scene == "both" else [args.scene]):
            results.extend(run_behavioural(key))

    return report(results, args.json)


if __name__ == "__main__":
    sys.exit(main())
