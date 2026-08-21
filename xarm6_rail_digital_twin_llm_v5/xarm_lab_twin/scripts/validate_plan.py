#!/usr/bin/env python3
"""Replay a planned command sequence against real xArm firmware, with no robot.

UFACTORY ship a Docker image that runs the actual xArm controller firmware plus
Studio with no hardware attached. Point `XArmAPI` at 127.0.0.1 and the same code
that drives the bench runs against a simulated controller, so a plan can be
checked for joint limits, unreachable poses, singularities and mode/state errors
*before* anything moves.

    docker run -dit --name uf-sim \\
      -p 18333:18333 -p 502:502 -p 503:503 -p 504:504 \\
      -p 30000:30000 -p 30001:30001 -p 30002:30002 -p 30003:30003 \\
      danielwang123321/uf-ubuntu-docker /bin/bash
    docker exec uf-sim /xarm_scripts/xarm_start.sh 6 6     # axis=6 type=6

    python scripts/validate_plan.py plan.json
    python scripts/validate_plan.py --self-test

## The division of labour

This validates the **controller**, not the world. There is no bench, no OT-2, no
cube and no camera in the container: it will tell you a pose is unreachable or a
joint is out of range; it will not tell you the arm would hit the bench. The
MuJoCo twin validates world geometry and reachability. Neither replaces the
other, and a plan should pass both.

## Two measured limitations (see docs/a8_controller_validator.md)

1. **The container does not simulate the 7th-axis rail.** `set_linear_motor_pos`
   returns success and the reported position stays at 0 regardless. Rail moves
   are therefore range-checked locally here and NOT sent to the controller —
   sending them would produce a meaningless pass.

2. **The container's serial number is blank**, and the SDK's joint-limit check
   does `int(self.sn[2:6])`, which raises ValueError. UFACTORY's documented
   workaround is `check_joint_limit=False`, which we deliberately do NOT use:
   it disables the very check this script exists to perform. Joint limits are
   instead validated locally against the scene's `jnt_range`, which came from
   the xArm6 URDF, and the controller is still asked to execute the motion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())

DEFAULT_IP = "127.0.0.1"

# Actions this validator understands. Anything else is reported as unvalidated
# rather than silently skipped -- an unrecognised action passing quietly is how
# a validator ends up certifying a plan it never checked.
CARTESIAN = {"move_to"}
JOINT = {"set_joints"}
RAIL = {"set_rail"}
IGNORED = {"gripper_open", "gripper_close", "done", "wait", "get_pose",
           "get_body_pose", "search_workspace", "pcr_open", "pcr_close"}


def scene_limits(scene_xml: str = "envs/lab_scene.xml"):
    """Joint ranges (deg) and rail range (mm), read from the scene.

    The scene's joint ranges came from the xArm6 URDF, so this keeps the
    validator's notion of a legal joint angle tied to the same source as the
    simulator rather than to a table typed in here.
    """
    import mujoco as mj
    import numpy as np
    m = mj.MjModel.from_xml_path(scene_xml)
    joints = []
    for i in range(1, 7):
        lo, hi = np.degrees(m.jnt_range[m.joint(f"joint{i}").id])
        joints.append((float(lo), float(hi)))
    r = m.jnt_range[m.joint("rail").id]
    return joints, (float(r[0]) * 1000.0, float(r[1]) * 1000.0)


class PlanValidator:
    def __init__(self, ip: str = DEFAULT_IP, scene_xml: str = "envs/lab_scene.xml"):
        from xarm.wrapper import XArmAPI
        self.joint_limits, self.rail_limits = scene_limits(scene_xml)
        # NOTE: check_joint_limit is left at its default on purpose. See module docstring.
        self.arm = XArmAPI(ip, is_radian=False)
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        code, ver = self.arm.get_version()
        print(f"[validator] controller {ver!r} (code {code}) at {ip}")

    def _recover(self):
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def _check_joints_local(self, angles):
        bad = []
        for i, (a, (lo, hi)) in enumerate(zip(angles, self.joint_limits), start=1):
            if not (lo <= a <= hi):
                bad.append(f"joint{i}={a:.1f} outside [{lo:.1f}, {hi:.1f}]")
        return bad

    def validate(self, commands: list[dict]) -> list[dict]:
        results = []
        for idx, cmd in enumerate(commands):
            action = cmd.get("action")
            p = cmd.get("params", {}) or {}
            rec = {"index": idx, "action": action, "ok": True, "detail": ""}

            if action in IGNORED:
                rec["detail"] = "not a controller motion; nothing to validate"

            elif action in CARTESIAN:
                rc = self.arm.set_position(
                    x=p.get("x"), y=p.get("y"), z=p.get("z"),
                    roll=p.get("roll", 180), pitch=p.get("pitch", 0),
                    yaw=p.get("yaw", 0), speed=p.get("speed_mm_s", 50), wait=True)
                err = self.arm.get_err_warn_code()[1]
                if rc != 0 or err[0]:
                    rec["ok"] = False
                    rec["detail"] = (f"controller rejected pose "
                                     f"({p.get('x')}, {p.get('y')}, {p.get('z')}): "
                                     f"rc={rc}, error={err[0]}")
                    self._recover()
                else:
                    rec["detail"] = "pose reachable"

            elif action in JOINT:
                angles = p.get("angles_deg") or p.get("angles") or []
                bad = self._check_joints_local(angles)
                if bad:
                    rec["ok"] = False
                    rec["detail"] = "; ".join(bad)
                else:
                    try:
                        rc = self.arm.set_servo_angle(
                            angle=angles, speed=p.get("speed_deg_s", 20), wait=True)
                        err = self.arm.get_err_warn_code()[1]
                        if rc != 0 or err[0]:
                            rec["ok"] = False
                            rec["detail"] = f"controller rejected joints: rc={rc}, error={err[0]}"
                            self._recover()
                        else:
                            rec["detail"] = "joints within limits and accepted"
                    except ValueError as exc:
                        # Blank serial number in the container; see module docstring.
                        rec["detail"] = (f"joints within limits (local check); controller "
                                         f"check unavailable in this container: {exc}")

            elif action in RAIL:
                pos = p.get("position_mm")
                lo, hi = self.rail_limits
                if pos is None or not (lo <= pos <= hi):
                    rec["ok"] = False
                    rec["detail"] = f"rail {pos} outside [{lo:.0f}, {hi:.0f}] mm"
                else:
                    # Deliberately not sent to the controller: the container has no
                    # linear motor, so it returns success and never moves.
                    rec["detail"] = (f"rail {pos:.0f} mm within range "
                                     f"(range-checked locally; container has no rail)")

            else:
                rec["ok"] = False
                rec["detail"] = f"unrecognised action {action!r} -- NOT validated"

            results.append(rec)
        return results

    def close(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


def make_gate(ip: str = DEFAULT_IP, scene_xml: str = "envs/lab_scene.xml"):
    """Build a pre-action gate for LLMBrain.plan_validator.

    Returns callable(commands) -> (ok, report_lines), or None if the controller
    container is not reachable.

    Returning None rather than a permissive gate is deliberate: a gate that
    silently passes everything when its backend is down is worse than no gate,
    because the operator believes a check happened. The caller decides whether an
    unreachable validator should block the run.
    """
    try:
        validator = PlanValidator(ip, scene_xml)
    except Exception as exc:  # noqa: BLE001
        print(f"[validator] controller at {ip} unreachable ({type(exc).__name__}: {exc})")
        return None

    def gate(commands):
        results = validator.validate(commands)
        bad = [r for r in results if not r["ok"]]
        report = [f"#{r['index']} {r['action']}: {r['detail']}" for r in bad]
        return (not bad), report

    gate.validator = validator
    return gate


SELF_TEST_PLAN = [
    {"action": "set_rail", "params": {"position_mm": 350, "speed_mm_s": 100}},
    {"action": "move_to", "params": {"x": 300, "y": 0, "z": 300, "roll": 180,
                                     "pitch": 0, "yaw": 0, "speed_mm_s": 50}},
    {"action": "set_joints", "params": {"angles_deg": [0, -30, -30, 0, 60, 0]}},
    {"action": "gripper_close", "params": {}},
    {"action": "done", "params": {"message": "ok"}},
]

SELF_TEST_BAD = [
    {"action": "move_to", "params": {"x": 2000, "y": 0, "z": 400, "roll": 180,
                                     "pitch": 0, "yaw": 0, "speed_mm_s": 50}},
    {"action": "set_joints", "params": {"angles_deg": [400, 0, 0, 0, 0, 0]}},
    {"action": "set_rail", "params": {"position_mm": 1200}},
]


def report(results, as_json: bool) -> int:
    bad = [r for r in results if not r["ok"]]
    if as_json:
        json.dump({"summary": {"commands": len(results), "rejected": len(bad)},
                   "results": results}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for r in results:
            mark = "ok  " if r["ok"] else "FAIL"
            print(f"  [{mark}] {r['index']:>2} {r['action']:<16} {r['detail']}")
        print(f"\n  {len(results) - len(bad)} accepted, {len(bad)} rejected")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="JSON file: a list of {action, params}")
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the validator accepts a good plan and rejects a bad one")
    args = ap.parse_args()

    v = PlanValidator(args.ip)
    try:
        if args.self_test:
            print("\n--- a plan that should be ACCEPTED ---")
            good = report(v.validate(SELF_TEST_PLAN), False)
            print("\n--- a plan that should be REJECTED ---")
            bad = report(v.validate(SELF_TEST_BAD), False)
            ok = (good == 0 and bad == 1)
            print(f"\n  self-test {'PASSED' if ok else 'FAILED'}: "
                  f"good plan {'accepted' if good == 0 else 'wrongly rejected'}, "
                  f"bad plan {'rejected' if bad == 1 else 'wrongly accepted'}")
            return 0 if ok else 1

        if not args.plan:
            ap.error("give a plan file, or --self-test")
        commands = json.load(open(args.plan))
        return report(v.validate(commands), args.json)
    finally:
        v.close()


if __name__ == "__main__":
    sys.exit(main())
