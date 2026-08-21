"""The contract every arm backend must satisfy, and a way to declare the gaps.

`SimXArmAPI` and `RealXArmAPI` are both driven by the same planner, episode loop,
grader and recorder. Until now that was an informal arrangement: the sim had 23
public methods, the real backend 22, and 7 of the sim's were in the LLM dispatch
table but absent from hardware — so a `--mode real` run would die with
`AttributeError` partway through a plan, after some of it had already executed.

This module makes the contract explicit and, just as importantly, makes the
*absences* explicit. A backend may either implement a method or declare it
unsupported with a reason. What it may not do is quietly lack it.

Declaring a gap:

    @unsupported("there is no physical equivalent; move the object by hand")
    def nudge_body(self, *a, **k): ...

`scripts/sim_checks.py` reads these declarations and fails if any contract method
is neither implemented nor declared, so a method added to the contract cannot be
forgotten on one side.
"""
from __future__ import annotations

import functools
import inspect

#: name -> what the method is for. This is the contract: every backend must
#: either implement each entry or declare it unsupported.
#:
#: Deliberately *not* everything either class happens to expose — only what the
#: shared stack (llm_brain dispatch, episode_loop, outcome_checker, recording)
#: actually calls. Extra backend-specific helpers are fine and are ignored here.
ARM_BACKEND_METHODS: dict[str, str] = {
    # -- motion ------------------------------------------------------------
    "set_position":       "Cartesian move, mm/deg, returns 0 on success",
    "set_servo_angle":    "joint move, degrees",
    "set_rail_position":  "linear rail move, mm",
    "get_position":       "(code, [x,y,z,roll,pitch,yaw]) in mm/deg",
    "get_servo_angle":    "(code, [j1..j6]) in degrees",
    "get_rail_position":  "(code, position_mm)",
    "go_home":            "drive to the canonical home pose",
    "rail_home":          "return the rail to its origin",

    # -- end effector ------------------------------------------------------
    "open_gripper":       "open the fitted effector",
    "close_gripper":      "close the fitted effector",
    "set_gripper":        "select/confirm the fitted effector ('standard'|'bio')",
    "verify_grasp":       "(state, detail) where state is held|empty|unknown",

    # -- world state -------------------------------------------------------
    "get_body_pose":      "(code, [x,y,z,r,p,y]) for a named object",
    "physical_outcome":   "human-readable description of what changed in the scene",
    "reset_scene":        "restore the starting state between episodes",

    # -- scene manipulation (sim conveniences with no direct hardware analogue)
    "push_object":        "slide/carry an object to a target xy",
    "place_tube_in_rack": "seat a held tube in a free rack slot",
    "nudge_body":         "teleport an object (operator adjustment)",
    "set_pcr_lid":        "open/close the thermocycler lid",

    # -- diagnostics -------------------------------------------------------
    "get_ft_data":        "(code, [fx,fy,fz,tx,ty,tz]) from the force/torque sensor",

    # -- lifecycle ---------------------------------------------------------
    "disconnect":         "release the arm / stop the sim",
}

#: xArm6 joint limits in degrees, and the rail's travel in mm.
#:
#: These duplicate the scene's `jnt_range`, which is the kind of duplication that
#: caused the drift A1 fixed — so `scripts/sim_checks.py` asserts they still match
#: (`arm.joint_limits_match_scene`). The duplication is accepted because
#: `hardware/real_arm.py` must not depend on MuJoCo: it has to work on a machine
#: driving the arm with no simulator installed.
XARM6_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-178.2, 178.2),   # joint1
    (-118.0, 120.0),   # joint2
    (-178.2, 11.0),    # joint3
    (-178.2, 178.2),   # joint4
    (-97.0, 178.2),    # joint5
    (-178.2, 178.2),   # joint6
)
RAIL_LIMITS_MM: tuple[float, float] = (0.0, 700.0)


def check_joint_limits_deg(angles) -> list[str]:
    """Return a description of every joint outside its limit; empty if all fine."""
    bad = []
    for i, (a, (lo, hi)) in enumerate(zip(angles, XARM6_JOINT_LIMITS_DEG), start=1):
        if a is None or not (lo <= a <= hi):
            bad.append(f"joint{i}={a} outside [{lo}, {hi}] deg")
    return bad


_UNSUPPORTED_ATTR = "__backend_unsupported__"


def unsupported(reason: str):
    """Mark a contract method as deliberately unavailable on this backend.

    The wrapped method raises `NotImplementedError` carrying `reason`. It must
    raise rather than return a benign value: a scene-manipulation call that
    quietly does nothing on hardware would let a plan "succeed" while the bench
    was untouched, which is the failure mode this whole contract exists to stop.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def raiser(self, *args, **kwargs):
            raise NotImplementedError(f"{type(self).__name__}.{fn.__name__}: {reason}")
        setattr(raiser, _UNSUPPORTED_ATTR, reason)
        return raiser
    return decorate


def unsupported_reason(cls, name: str) -> str | None:
    """The declared reason `name` is unsupported on `cls`, or None."""
    fn = getattr(cls, name, None)
    return getattr(fn, _UNSUPPORTED_ATTR, None) if fn is not None else None


def backend_report(cls) -> dict[str, tuple[str, str]]:
    """name -> (status, detail) for every contract method.

    status is 'implemented', 'unsupported' or 'MISSING'. MISSING is the one that
    matters: it means the shared stack can call something this backend does not
    have and has not declared, i.e. an AttributeError waiting to happen mid-plan.
    """
    out = {}
    for name, purpose in ARM_BACKEND_METHODS.items():
        reason = unsupported_reason(cls, name)
        if reason is not None:
            out[name] = ("unsupported", reason)
        elif callable(getattr(cls, name, None)):
            out[name] = ("implemented", purpose)
        else:
            out[name] = ("MISSING", purpose)
    return out


def missing_methods(cls) -> list[str]:
    return [n for n, (s, _) in backend_report(cls).items() if s == "MISSING"]


def render_parity_table(classes) -> str:
    """Contract method x backend, for the commit message and the sweep output."""
    reports = {c.__name__: backend_report(c) for c in classes}
    names = list(ARM_BACKEND_METHODS)
    w = max(len(n) for n in names) + 2
    head = "method".ljust(w) + "  ".join(f"{c:<14}" for c in reports)
    lines = [head, "-" * len(head)]
    mark = {"implemented": "impl", "unsupported": "unsup", "MISSING": "MISSING"}
    for n in names:
        row = n.ljust(w) + "  ".join(f"{mark[reports[c][n][0]]:<14}" for c in reports)
        lines.append(row)
    return "\n".join(lines)
