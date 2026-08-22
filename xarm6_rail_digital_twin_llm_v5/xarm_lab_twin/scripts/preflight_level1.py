#!/usr/bin/env python3
"""Level 1 pre-flight: ask the controller whether a plan is legal, without moving.

Per A8.1 of the Stream A/B instructions, pre-flight has two levels:

- **Level 1 (this file)** -- pure query. Inverse/forward kinematics, joint-limit
  and TCP-limit checks. No motion, nothing enabled, no mode or state change. It
  therefore runs against the **real control box** as safely as against the
  container, which is the point: it can gate every plan cheaply, including
  during live operation.
- **Level 2 (`scripts/validate_plan.py`)** -- actually execute the plan against
  the containerised controller. Slower, catches ordering and state-machine
  problems Level 1 cannot see.

Use Level 1 always. Use Level 2 before any first-time or changed plan.

## What this does NOT know

The controller knows its own kinematics. It does not know your bench, the cube,
the cup, or the rail track. It will happily approve a pose that puts the gripper
inside the benchtop. **World geometry is the MuJoCo twin's job.** A plan should
pass both; neither substitutes for the other.

## Why is_tcp_limit is advisory, not a gate

Measured against the container (firmware v2.4.0, 2026-08-22):

    is_tcp_limit([207, 0, 112, -180, 0, 0])   -> False   # the current pose
    is_tcp_limit([200, 0, 300, 180, 0, 0])    -> True    # plainly reachable
    is_tcp_limit([9999, 0, 300, 180, 0, 0])   -> False   # plainly unreachable

It flags a reachable pose and passes one a metre out of range, so as a
reachability test it is worse than useless -- gating on it would reject good
plans and approve impossible ones. On a later run it hung outright. It is
recorded per waypoint and reported, but it never decides the verdict.

The checks that DO decide are, in order:

1. **IK solvable** -- `get_inverse_kinematics` returns code 0 and a solution.
2. **Joint limits** -- `is_joint_limit(joints)`. Verified to behave: all-zeros
   is False, a 999 deg wrist is True.
3. **FK round-trip** -- feed the IK solution back through
   `get_forward_kinematics` and compare with what was asked for. This is the
   check that catches IK returning a pose-shaped answer that is not the pose you
   asked for, which is the same under-convergence defect the sim's IK had.
4. **Local joint limits** -- cross-checked against `XARM6_JOINT_LIMITS_DEG` in
   `arm_backend`, so a controller that lies about its own limits is still caught.

Usage:
    python scripts/preflight_level1.py --self-test
    python scripts/preflight_level1.py --demo --host 127.0.0.1
    python scripts/preflight_level1.py --plan plan.json --host 192.168.1.229
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

sys.path.insert(0, os.getcwd())

from arm_backend import (
    XARM6_JOINT_LIMITS_DEG,
    RAIL_LIMITS_MM,
    world_to_base_mm,
)

#: How far the FK round-trip may disagree with the requested pose (mm).
#: Matches the sim's IK_POS_TOL_M (5 mm) so both layers hold one standard.
FK_ROUNDTRIP_TOL_MM = 5.0

#: Any single SDK query that takes longer than this is treated as UNKNOWN
#: rather than allowed to wedge the gate. The container has hung on
#: is_tcp_limit in practice.
QUERY_TIMEOUT_S = 8.0

#: How many times to sample the controller's IK per waypoint. Its IK is not
#: deterministic -- see check_waypoint. Every distinct solution must be legal.
IK_SAMPLES = 4


@dataclass
class WaypointVerdict:
    index: int
    label: str
    pose_base_mm: list
    ok: bool = False
    reasons: list = field(default_factory=list)
    advisories: list = field(default_factory=list)
    ik_code: Optional[int] = None
    joints_deg: Optional[list] = None
    fk_error_mm: Optional[float] = None
    tcp_limit: Optional[object] = None

    def render(self) -> str:
        mark = "OK  " if self.ok else "FAIL"
        p = ", ".join(f"{v:7.1f}" for v in self.pose_base_mm[:3])
        line = f"  [{mark}] {self.index:>2}. {self.label:<22} ({p})"
        for r in self.reasons:
            line += f"\n           - {r}"
        for a in self.advisories:
            line += f"\n           ~ {a}"
        return line


def _call(fn, *args, **kwargs):
    """Run one SDK query under a timeout. Returns (ok, value_or_error)."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return True, fut.result(timeout=QUERY_TIMEOUT_S)
        except FutureTimeout:
            return False, f"timed out after {QUERY_TIMEOUT_S:.0f}s"
        except Exception as exc:  # noqa: BLE001 - surfaced as a reason, never swallowed
            return False, f"{type(exc).__name__}: {exc}"


def controller_joint_limits(arm):
    """The controller's own joint limits, or None if it will not say.

    Preferred over the local constant. Measured 2026-08-22, the controller
    reports a WIDER envelope than either `XARM6_JOINT_LIMITS_DEG` or the MuJoCo
    scene -- J1/J4/J6 are +/-360 rather than +/-178.2, and J3 reaches -225
    rather than -178.2. Gating on the narrow copy would reject poses the arm can
    legally reach, so ask the machine and fall back to the constant.
    """
    ok, val = _call(getattr, arm, "reduced_joint_limits")
    if not ok or not val:
        return None
    try:
        lim = [(float(lo), float(hi)) for lo, hi in list(val)[:6]]
        return lim if len(lim) == 6 else None
    except Exception:  # noqa: BLE001
        return None


def _local_joint_limit_violations(joints_deg: Sequence[float], limits=None) -> list:
    out = []
    for i, (ang, (lo, hi)) in enumerate(zip(joints_deg, limits or XARM6_JOINT_LIMITS_DEG), start=1):
        if not (lo <= ang <= hi):
            out.append(f"joint{i} = {ang:.1f} deg outside [{lo}, {hi}]")
    return out


def check_waypoint(arm, index: int, label: str, pose_base_mm: Sequence[float],
                   limits=None) -> WaypointVerdict:
    """Query-only legality check for one pose, in the ARM BASE frame, mm + degrees.

    The controller's IK is **not deterministic**. Measured against the container
    (firmware v2.4.0, 2026-08-22), the identical pose queried six times returned
    two different branches, alternating:

        [-34.41, 17.68, -28.52,    0.0,  10.84,  -34.41]
        [-34.41, 34.70, -61.58, -180.0, -26.88,  145.59]

    The second has joint4 at +/-180.0 deg, which is outside the published limit
    of +/-178.2 -- and `is_joint_limit` did not flag it. So a plan checked once
    can pass and then execute a different, illegal solution.

    This function therefore samples IK `IK_SAMPLES` times and requires **every**
    distinct solution to be legal. A waypoint whose branches disagree is
    reported, because that instability is itself worth knowing before motion.
    """
    v = WaypointVerdict(index=index, label=label, pose_base_mm=list(pose_base_mm))

    solutions = []
    for _ in range(IK_SAMPLES):
        ok, res = _call(arm.get_inverse_kinematics, list(pose_base_mm),
                        input_is_radian=False, return_is_radian=False)
        if not ok:
            v.reasons.append(f"get_inverse_kinematics {res}")
            return v
        code, joints = res
        v.ik_code = code
        if code != 0 or not joints:
            v.reasons.append(f"no IK solution (code={code})")
            return v
        sol = tuple(round(float(a), 2) for a in joints[:6])
        if sol not in solutions:
            solutions.append(sol)

    v.joints_deg = [list(s) for s in solutions]
    if len(solutions) > 1:
        v.advisories.append(
            f"controller IK returned {len(solutions)} different solutions for this "
            f"pose; all are checked and all must be legal")

    for sol in solutions:
        joints = list(sol)

        for msg in _local_joint_limit_violations(joints, limits):
            v.reasons.append(msg)

        # ADVISORY. is_joint_limit is as unreliable as is_tcp_limit on this
        # firmware: measured 2026-08-22, joint4 = 178 deg reports violated while
        # 359 deg reports fine, inside a published range of +/-360. Recorded,
        # never decisive.
        ok, res = _call(arm.is_joint_limit, joints)
        if not ok:
            v.advisories.append(f"is_joint_limit unavailable ({res})")
        else:
            jcode, over = res
            if jcode == 0 and over:
                v.advisories.append(
                    f"is_joint_limit reports a violation (ADVISORY -- known unreliable)")

        ok, res = _call(arm.get_forward_kinematics, joints,
                        input_is_radian=False, return_is_radian=False)
        if not ok:
            v.advisories.append(f"get_forward_kinematics unavailable ({res})")
            continue
        fcode, back = res
        if fcode == 0 and back:
            err = sum((float(back[i]) - float(pose_base_mm[i])) ** 2 for i in range(3)) ** 0.5
            v.fk_error_mm = round(err, 2) if v.fk_error_mm is None else max(v.fk_error_mm, round(err, 2))
            if err > FK_ROUNDTRIP_TOL_MM:
                v.reasons.append(
                    f"FK round-trip is {err:.1f} mm from the requested pose "
                    f"(tolerance {FK_ROUNDTRIP_TOL_MM:.0f} mm) -- the IK solution "
                    f"does not reach where it claims")
        else:
            v.advisories.append(f"get_forward_kinematics returned code {fcode}")

    # Advisory only. See the module docstring for why this cannot be a gate.
    ok, res = _call(arm.is_tcp_limit, list(pose_base_mm))
    v.tcp_limit = res if ok else f"unavailable ({res})"
    if ok and isinstance(res, (list, tuple)) and len(res) == 2 and res[0] == 0 and res[1]:
        v.advisories.append("is_tcp_limit reports a limit (ADVISORY -- known unreliable)")

    # De-duplicate: the same branch can trip the same reason more than once.
    seen = set()
    v.reasons = [r for r in v.reasons if not (r in seen or seen.add(r))]
    v.ok = not v.reasons
    return v


def check_plan(arm, waypoints, limits=None) -> list:
    """waypoints: sequence of (label, [x, y, z, roll, pitch, yaw]) in the BASE frame."""
    if limits is None:
        limits = controller_joint_limits(arm)
    return [check_waypoint(arm, i, label, pose, limits)
            for i, (label, pose) in enumerate(waypoints, start=1)]


def world_plan_to_base(waypoints_world, rail_mm: float):
    """Convert (label, [x,y,z,r,p,y]) world-frame waypoints to the arm base frame.

    The base frame moves with the rail, so this is rail-dependent -- not a
    constant offset. Reuses arm_backend.world_to_base_mm rather than repeating
    the arithmetic, because a second copy of that conversion is exactly how the
    frames drifted apart the first time.
    """
    if not (RAIL_LIMITS_MM[0] <= rail_mm <= RAIL_LIMITS_MM[1]):
        raise ValueError(f"rail {rail_mm} mm outside {RAIL_LIMITS_MM}")
    out = []
    for label, pose in waypoints_world:
        x, y, z = world_to_base_mm(pose[:3], rail_mm)
        out.append((label, [x, y, z, *pose[3:]]))
    return out


DEMO_WORLD = [
    ("above cube",     [292.0, -250.0, 950.0, 180.0, 0.0, 0.0]),
    ("lower to grasp", [292.0, -250.0, 790.0, 180.0, 0.0, 0.0]),
    ("lift",           [292.0, -250.0, 950.0, 180.0, 0.0, 0.0]),
    ("over cup",       [ -8.0, -250.0, 900.0, 180.0, 0.0, 0.0]),
]


# --------------------------------------------------------------------------
# Self-test -- runs with no controller, so it can gate a commit.
# --------------------------------------------------------------------------

class _FakeArm:
    """Controller stub with known-good and known-bad behaviour.

    Deliberately mimics the container's observed misbehaviour: is_tcp_limit
    lies. The self-test asserts the gate's verdict does not depend on it.
    """

    def __init__(self, mode="good"):
        self.mode = mode

    def get_inverse_kinematics(self, pose, **kw):
        if self.mode == "no_ik":
            return 1, None
        return 0, [10.0, -20.0, -30.0, 0.0, 50.0, 5.0, 0.0]

    def is_joint_limit(self, joints):
        return (0, self.mode == "joint_limit")

    def get_forward_kinematics(self, joints, **kw):
        if self.mode == "bad_fk":
            return 0, [9999.0, 0.0, 0.0, 180.0, 0.0, 0.0]
        return 0, [292.0, -200.0, 103.0, 180.0, 0.0, 0.0]

    def is_tcp_limit(self, pose):
        return (0, True)          # always lies


def self_test() -> int:
    pose = [292.0, -200.0, 103.0, 180.0, 0.0, 0.0]
    cases = [
        ("good plan passes despite is_tcp_limit lying", "good", True),
        ("no IK solution is rejected", "no_ik", False),
        ("controller joint-limit claim is ADVISORY, not a rejection", "joint_limit", True),
        ("FK round-trip mismatch is rejected", "bad_fk", False),
    ]
    failures = 0
    for label, mode, expect_ok in cases:
        v = check_waypoint(_FakeArm(mode), 1, label, pose)
        status = "PASS" if v.ok == expect_ok else "FAIL"
        if v.ok != expect_ok:
            failures += 1
        print(f"  [{status}] {label:48} ok={v.ok} reasons={v.reasons}")

    # The advisory must still be recorded even though it does not reject.
    v = check_waypoint(_FakeArm("joint_limit"), 1, "advisory recorded", pose)
    adv_ok = any("ADVISORY" in a for a in v.advisories)
    failures += 0 if adv_ok else 1
    print(f"  [{'PASS' if adv_ok else 'FAIL'}] {'is_joint_limit claim is recorded as an advisory':48} "
          f"advisories={v.advisories}")

    # A local-limit violation must be caught even if the controller says fine.
    class _Lying(_FakeArm):
        def get_inverse_kinematics(self, pose, **kw):
            return 0, [0.0, 0.0, 0.0, 0.0, 0.0, 999.0, 0.0]
    v = check_waypoint(_Lying("good"), 1, "local limit cross-check", pose)
    ok = (not v.ok) and any("joint6" in r for r in v.reasons)
    failures += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {'local limits catch a lying controller':48} "
          f"ok={v.ok} reasons={v.reasons}")

    print(f"\n  {6 - failures} PASS  {failures} FAIL")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="controller address: 127.0.0.1 = container, or the real control box")
    ap.add_argument("--plan", help="JSON: [[label, [x,y,z,r,p,y]], ...]")
    ap.add_argument("--demo", action="store_true", help="use the built-in cube->cup plan")
    ap.add_argument("--frame", choices=["world", "base"], default="world")
    ap.add_argument("--rail", type=float, default=350.0,
                    help="rail position (mm) the plan is executed at; world frame only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.demo:
        waypoints = DEMO_WORLD
    elif args.plan:
        with open(args.plan) as fh:
            waypoints = [(lbl, pose) for lbl, pose in json.load(fh)]
    else:
        ap.error("need --demo, --plan or --self-test")

    if args.frame == "world":
        waypoints = world_plan_to_base(waypoints, args.rail)

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.host)
    try:
        limits_used = controller_joint_limits(arm)
        verdicts = check_plan(arm, waypoints, limits_used)
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass

    if args.json:
        print(json.dumps([asdict(v) for v in verdicts], indent=2))
    else:
        print(f"\nLevel 1 pre-flight against {args.host} "
              f"(rail={args.rail:.0f} mm, {args.frame} frame) -- NO MOTION")
        if limits_used:
            drift = [f"J{i+1}" for i, (a_, b_) in enumerate(zip(limits_used, XARM6_JOINT_LIMITS_DEG))
                     if abs(a_[0] - b_[0]) > 0.5 or abs(a_[1] - b_[1]) > 0.5]
            print(f"  joint limits: from the controller"
                  + (f"; DIFFER from XARM6_JOINT_LIMITS_DEG at {', '.join(drift)}" if drift else ""))
        else:
            print("  joint limits: controller would not say; using XARM6_JOINT_LIMITS_DEG")
        print()
        for v in verdicts:
            print(v.render())
        bad = [v for v in verdicts if not v.ok]
        print(f"\n  {len(verdicts) - len(bad)} OK  {len(bad)} FAIL")
        if bad:
            print("\n  Plan REFUSED. This checks the controller only -- passing it "
                  "still does not mean the arm clears the bench.")
    return 1 if any(not v.ok for v in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
