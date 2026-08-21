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


#: Where the arm's base sits in world coordinates when the rail is at 0 mm, and
#: how it moves with the rail. Measured from the scene:
#:
#:     rail=  0mm -> xarm_base world (-350.0, -50.0, 790.0)
#:     rail=350mm -> xarm_base world (   0.0, -50.0, 790.0)
#:     rail=700mm -> xarm_base world ( 350.0, -50.0, 790.0)
#:
#: i.e. exactly linear in x, fixed in y and z.
#:
#: **These are nominal, from the scene.** The real cell's numbers must be
#: measured (B4, sim-to-real pose delta) and overridden per deployment. Until
#: then a real-hardware Cartesian move is approximate, not accurate.
BASE_AT_RAIL_ZERO_MM: tuple[float, float, float] = (-350.0, -50.0, 790.0)

#: World z of the benchtop. The floor constraint is expressed relative to this.
BENCH_TOP_Z_MM: float = 750.0

#: Minimum world z for a commanded tool pose. Defends the benchtop against a
#: plan that drives the tool down into it.
#:
#: Only 5 mm above the bench because grasping a 30 mm cube resting on it
#: genuinely requires coming that close — a larger clearance would forbid the
#: task. This is a guard against gross errors (z=0, z=300), not a fine-grained
#: collision check, and it is only meaningful if the TCP offset is set correctly
#: (see docs/hardware_preflight.md §1.2): with no TCP set, z refers to the
#: flange and the tool hangs below it.
WORKSPACE_FLOOR_Z_MM: float = BENCH_TOP_Z_MM + 5.0

#: Commanded-pose bounds in world mm. Wider than the bench so reaching past an
#: edge is allowed, but a plan that asks for metres away is refused.
WORKSPACE_AABB_MM: dict[str, tuple[float, float]] = {
    "x": (-900.0, 1250.0),
    "y": (-1000.0, 700.0),
    "z": (WORKSPACE_FLOOR_Z_MM, 1500.0),
}


def world_to_base_mm(xyz_world, rail_mm: float,
                     base_at_rail_zero=BASE_AT_RAIL_ZERO_MM):
    """World coordinates -> arm-base coordinates, given the live rail position.

    The twin works in world coordinates (benchtop at z=750); the xArm controller
    works relative to its own base. The base slides with the rail, so this is not
    a constant offset and the rail position must be read, not assumed.
    """
    bx, by, bz = base_at_rail_zero
    return (xyz_world[0] - (bx + rail_mm),
            xyz_world[1] - by,
            xyz_world[2] - bz)


def base_to_world_mm(xyz_base, rail_mm: float,
                     base_at_rail_zero=BASE_AT_RAIL_ZERO_MM):
    """Arm-base coordinates -> world coordinates. Inverse of `world_to_base_mm`."""
    bx, by, bz = base_at_rail_zero
    return (xyz_base[0] + bx + rail_mm,
            xyz_base[1] + by,
            xyz_base[2] + bz)


def check_workspace_world(xyz_world, aabb=None, floor_z=None) -> list[str]:
    """Describe every way a commanded world pose is out of bounds; empty if fine."""
    aabb = aabb or WORKSPACE_AABB_MM
    floor_z = WORKSPACE_FLOOR_Z_MM if floor_z is None else floor_z
    x, y, z = xyz_world
    bad = []
    if z < floor_z:
        bad.append(f"z={z:.0f} below the {floor_z:.0f} mm floor "
                   f"(benchtop is {BENCH_TOP_Z_MM:.0f})")
    for axis, val in (("x", x), ("y", y), ("z", z)):
        lo, hi = aabb[axis]
        if not (lo <= val <= hi):
            bad.append(f"{axis}={val:.0f} outside [{lo:.0f}, {hi:.0f}] mm")
    return bad


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
