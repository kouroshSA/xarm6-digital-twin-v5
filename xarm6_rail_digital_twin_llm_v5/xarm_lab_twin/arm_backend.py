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
    "wave_goodbye":       "sweep the arm side to side N times (demo / attract motion)",
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
# MEASURED from the controller (2026-08-22) via `XArmAPI.reduced_joint_limits`,
# and identical to UFACTORY's published xArm6 ranges -- two independent sources.
#
# These were previously (-178.2, 178.2) on J1/J4/J6, (-178.2, 11) on J3 and
# (-97, 178.2) on J5. That 178.2 is 0.99*pi: it came from the xacro "limited"
# variant of the URDF, which is a conservative build option, NOT the hardware
# range. Gating on it rejected poses the arm can legally reach.
#
# This tuple is the ONE source. envs/build_mesh_scene.py imports it rather than
# repeating the numbers, and scripts/sim_checks.py ties the scene back to it.
XARM6_JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-360.0, 360.0),   # joint1
    (-117.97, 120.0),  # joint2
    (-225.0, 11.0),    # joint3
    (-360.0, 360.0),   # joint4
    (-97.0, 180.0),    # joint5
    (-360.0, 360.0),   # joint6
)
RAIL_LIMITS_MM: tuple[float, float] = (0.0, 700.0)

# The home pose, VERIFIED on the real arm 2026-08-21, shared by both backends.
# It lived in two places with two different values: hardware/real_arm.py held
# these joint angles, while sim/mujoco_env.py had its own "looks tidy" pose
# with joint1 at +90 deg -- base rotated to face the wall, which put the
# gripper inside the PCR module. Nothing noticed until the rail gained
# swept-path validation, at which point every episode died on its first
# command because the arm was already in contact and could not move at all.
#
# Two homes is two robots. This is the one.
HOME_JOINTS_DEG: tuple[float, ...] = (-0.6, -31.8, -25.4, 0.0, 57.2, -0.4)
HOME_RAIL_MM: float = 350.0

# How fast the arm goes home. Shared, because the twin used to SNAP home --
# go_home wrote ctrl directly with no interpolation, so the arm returned at
# whatever the PD gains allowed -- while the real backend crept there at
# 20 deg/s. That is both a parity defect and an unsafe habit to rehearse: a
# motion nobody would command on hardware should not look normal in the twin.
HOME_JOINT_SPEED_DEG_S: float = 20.0
HOME_RAIL_SPEED_MM_S: float = 50.0


#: Where the arm's base sits in world coordinates when the rail is at 0 mm, and
#: how it moves with the rail. Measured from the scene:
#:
#:     rail=  0mm -> xarm_base world (-350.0, -50.0, 790.0)
#:     rail=350mm -> xarm_base world (   0.0, -50.0, 790.0)
#:     rail=700mm -> xarm_base world ( 350.0, -50.0, 790.0)
#:
#: i.e. exactly linear in x, fixed in y and z.
#:
#: What the SCENE models. The sweep asserts this still matches lab_scene.xml.
#: Corrected 2026-08-21 from 790 to 847 by raising the rail assembly, so the
#: scene now agrees with the measured cell.
SCENE_BASE_AT_RAIL_ZERO_MM: tuple[float, float, float] = (-350.0, -50.0, 847.0)

#: What the REAL CELL measures. Used by RealXArmAPI for the world<->base
#: conversion, because the physical arm is what it is regardless of the model.
#:
#: z is MEASURED (lab, 2026-08-21): the arm base sits 97 mm above the benchtop,
#: so with the benchtop at world z=750 the base is at 847. x and y are still
#: nominal and need measuring.
#:
#: **The 57 mm difference from the scene is a real sim-to-real error**, not a
#: bookkeeping mismatch: the scene models the rail assembly too short. Recorded
#: as MEASURED_BASE_Z_DELTA_MM below so the sweep fails if it changes
#: unnoticed, and so it stays visible until the scene is corrected.
BASE_AT_RAIL_ZERO_MM: tuple[float, float, float] = (-350.0, -50.0, 847.0)

#: Gap between the modelled and measured base. Now 0. A non-zero value means the
#: twin and the cell disagree about where the arm is, which produces false
#: collisions in one direction and false confidence in the other.
MEASURED_BASE_Z_DELTA_MM: float = (BASE_AT_RAIL_ZERO_MM[2]
                                   - SCENE_BASE_AT_RAIL_ZERO_MM[2])

#: Distance from the tool flange to the gripper tip, mm.
#:
#: CONFIRMED on the real arm (2026-08-21). With `tcp_offset` verified to be
#: [0,0,0], the flange read z=144.95 in the base frame at the pose where the
#: gripper tip sat 25 mm above a benchtop 97 mm below the base:
#:
#:     tip in base frame = 25 - 97 = -72 mm
#:     tool length       = 145 - (-72) = 217 mm
#:
#: Written to the controller as `set_tcp_offset([0, 0, 217, 0, 0, 0])`, so the
#: arm now positions the gripper tip and its coordinates mean what the twin's
#: mean. Note the tool was pitched -8.5 deg when measured; a tilted tool drops
#: LESS vertically, so 217 is the conservative figure.
TOOL_LENGTH_MM: float = 217.0

#: Distance the Gripper G2 FINGERTIPS sit ABOVE the configured TCP, mm.
#:
#: MEASURED on the real arm 2026-09-18, with `tcp_offset` = [0,0,217,...]: the
#: jaws were closed to a single probe, the F/T zeroed, and the tool descended in
#: 1 mm steps onto a plate whose top was at base z = 34 (white opaque plate, so
#: depth is trustworthy -- clear labware returns ~1% valid depth and must not be
#: used for this). Contact came at TCP z = 4.0, the force stepping 0.11 N ->
#: 18.52 N in a single millimetre:
#:
#:     fingertip_z = TCP_z + 30
#:
#: **So TOOL_LENGTH_MM above is the ORIGINAL standard gripper's value and does
#: NOT describe the Gripper G2 fitted since 2026-09-14.** With the G2, a TCP set
#: to 217 is positioned 30 mm BEYOND the fingertips; the G2's real flange-to-tip
#: is about 187 mm. `TOOL_LENGTH_MM` is deliberately left at 217 because it only
#: feeds `floor_z_for_tcp`, where over-stating the tool length errs SAFE: the
#: floor holds the tips 30 mm further from the bench than asked. Correcting it
#: would LOWER the floor, which should not happen on one measurement -- re-measure
#: with `scripts/calibrate_tool.py` and this probe before changing it.
#:
#: What it is NOT safe to keep guessing is grasp height. Assuming the tips hang
#: *below* the TCP (they do not) puts a grasp ~38 mm too high, which closes the
#: jaws in clear air while every call still returns rc=0. Use
#: `grasp_z_for_surface()` rather than re-deriving this per script.
GRIPPER_G2_TIP_ABOVE_TCP_MM: float = 30.0


def grasp_z_for_surface(surface_z_mm: float,
                        depth_mm: float = 10.0,
                        tip_above_tcp: float = GRIPPER_G2_TIP_ABOVE_TCP_MM) -> float:
    """Commanded TCP z that puts the fingertips `depth_mm` BELOW a surface.

    `surface_z_mm` is the measured top of the thing being grasped, in the same
    frame the controller reports (base z, mm). `depth_mm` is how far down the
    object's side wall the tips should sit -- 10 mm is a reasonable default for a
    ~15 mm SBS microplate, which leaves the tips clear of the bay floor.

    With `depth_mm = 0` this returns the z at which the tips just touch, which is
    what the 2026-09-18 probe measured directly (surface 34 -> touch at 4.0).
    """
    return surface_z_mm - depth_mm - tip_above_tcp

# ---------------------------------------------------------------------------
# Measured cell configuration
#
# Read from the physical arm at 192.168.1.229 on 2026-08-21, not estimated.
# Recorded so a future session can tell what the controller was configured with,
# and so a drift can be spotted. These are FYI constants: the controller holds
# the authoritative copy (persisted with save_conf), and
# scripts/calibrate_tool.py re-derives them.
# ---------------------------------------------------------------------------

#: Controller identity at the time of measurement.
CELL_FIRMWARE = "6,6,XI1305,MC1305,v2.7.1"
CELL_SERIAL = "XI1305"

#: Payload the ARM carries: F/T sensor + gripper + fingertips. Drives gravity
#: compensation and collision detection. [mass_kg, (cx, cy, cz) mm from flange]
CELL_TCP_LOAD = (1.247, (-3.681, 1.950, 27.675))

#: Payload the SENSOR carries: everything BELOW it, i.e. gripper and down but
#: not the sensor itself. Drives the force-reading zero.
#:
#: The ~0.27 kg difference from CELL_TCP_LOAD is the sensor's own mass, and that
#: difference is the check that these were really measured: identifying with the
#: sensor disabled returned 0.0 kg while reporting success.
CELL_FT_LOAD = (0.977, (-0.169, 1.200, 62.371))

#: The rail reported `get_linear_motor_is_enabled() -> 0` at measurement time,
#: same as the container that has no rail at all. UNRESOLVED: either the rail is
#: not powered/connected or it needs enabling. The world<->base conversion reads
#: the rail position, so a rail stuck reporting 0 silently offsets every
#: Cartesian target by however far it has actually travelled.
CELL_RAIL_ENABLED_AT_MEASUREMENT = False

#: World z of the benchtop the rail is mounted on.
BENCH_TOP_Z_MM: float = 750.0

#: How far above that benchtop the gripper is allowed to descend.
#:
#: "Floor" here means the **benchtop**, not the literal ground — the arm has no
#: reason to reach the ground, and what we are protecting against is the gripper
#: pressing down onto the surface the rail is bolted to.
#:
#: PLACEHOLDER: 25 mm, roughly the rail height less clearance. **Measure this at
#: the bench and replace it.** It is the one number here that is a guess rather
#: than a measurement, and it is the number that decides whether the gripper
#: touches the benchtop.
#:
#: For reference, the sim's own grasp guidance puts a cube grasp at z=795 with
#: cube tops at z=780, so a 25 mm clearance (floor at 775) still permits picking
#: a cube off the bench.
BENCHTOP_CLEARANCE_MM: float = 25.0

#: Minimum world z for a commanded tool pose.
#:
#: A guard against gross errors — z=0, z=300 — not a fine-grained collision
#: check. It is only meaningful if the TCP offset is set correctly (see
#: docs/hardware_preflight.md §1.2): with no TCP set, z refers to the flange and
#: the tool hangs below it, making this optimistic by the tool length.
def floor_z_for_tcp(tcp_offset_z_mm: float,
                    bench_top_z=BENCH_TOP_Z_MM,
                    clearance=BENCHTOP_CLEARANCE_MM,
                    tool_length=TOOL_LENGTH_MM) -> float:
    """Minimum commandable world z, given the TCP offset actually configured.

    This has to depend on the TCP because the controller positions whatever the
    TCP offset points at. With no offset it positions the **flange**, and the
    gripper hangs `tool_length` below that -- so commanding the z you want the
    *tip* at drives the tip a full tool-length into the bench.

    Worked through:

        controlled point sits (tool_length - tcp_z) above the tip
        want:  tip_z >= bench_top + clearance
        so: commanded_z >= bench_top + clearance + (tool_length - tcp_z)

    tcp_z = 0   (no offset)      -> floor 992 mm, controlling the flange
    tcp_z = 217 (tip configured) -> floor 775 mm, controlling the tip
    """
    return bench_top_z + clearance + (tool_length - tcp_offset_z_mm)


#: Default floor assumes the TCP is configured at the gripper tip, which is what
#: makes real-arm coordinates mean the same thing as the twin's. RealXArmAPI
#: reads the live TCP offset at connect and recomputes rather than trusting this.
WORKSPACE_FLOOR_Z_MM: float = floor_z_for_tcp(TOOL_LENGTH_MM)

#: Commanded-pose bounds in world mm. Wider than the bench so reaching past an
#: edge is allowed, but a plan that asks for metres away is refused.
WORKSPACE_AABB_MM: dict[str, tuple[float, float]] = {
    "x": (-900.0, 1250.0),
    "y": (-1000.0, 700.0),
    "z": (WORKSPACE_FLOOR_Z_MM, 1500.0),
}


#: Yaw of the arm's base frame relative to world, in degrees. MEASURED on the
#: cell 2026-08-24, and the measurement cost a collision to get.
#:
#: The conversion used to be a pure translation, which silently asserted that
#: the base frame and world frame share their x and y axes. They do not. On
#: hardware the arm's +x points ALONG THE BENCH (world -y) and its +y runs down
#: the rail (world +x) -- a quarter turn from what the twin assumed. A plan to
#: pick a cube 200 mm in front of the base was therefore converted into a reach
#: 200 mm sideways, the arm swung joint1 to -90 deg, descended over the rail
#: it travels on, and stopped itself on a collision.
#:
#: Measured by grasping known objects and reading the controller's own pose,
#: at rail 350 mm (base then at world x=0):
#:
#:     object      world (twin)      base (measured)
#:     red cube    (   0, -250)      (200,    0)
#:     blue cube   ( 220, -250)      (200,  220)
#:     cup         (-245, -250)      (200, -245)
#:
#: All three sit 200 mm out from the rail axis, which is base +x, and differ
#: only along the rail, which is base +y. That is a -90 deg base yaw and
#: nothing else: with it, all three land exactly.
#:
#: `arm_backend` previously carried "x and y are still nominal and need
#: measuring" against BASE_AT_RAIL_ZERO_MM. This is that measurement. What was
#: missing was not an offset -- it was the orientation.
BASE_YAW_DEG: float = -90.0


def _yaw_cos_sin(yaw_deg: float) -> tuple[float, float]:
    import math
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def world_to_base_mm(xyz_world, rail_mm: float,
                     base_at_rail_zero=BASE_AT_RAIL_ZERO_MM,
                     yaw_deg: float = None):
    """World coordinates -> arm-base coordinates, given the live rail position.

    The twin works in world coordinates (benchtop at z=750); the xArm controller
    works relative to its own base. The base slides with the rail, so this is not
    a constant offset and the rail position must be read, not assumed -- and it
    is not a pure translation either, because the base frame is yawed relative
    to world. See BASE_YAW_DEG.
    """
    bx, by, bz = base_at_rail_zero
    c, s = _yaw_cos_sin(BASE_YAW_DEG if yaw_deg is None else yaw_deg)
    dx = xyz_world[0] - (bx + rail_mm)
    dy = xyz_world[1] - by
    return (dx * c + dy * s,
            -dx * s + dy * c,
            xyz_world[2] - bz)


def base_to_world_mm(xyz_base, rail_mm: float,
                     base_at_rail_zero=BASE_AT_RAIL_ZERO_MM,
                     yaw_deg: float = None):
    """Arm-base coordinates -> world coordinates. Inverse of `world_to_base_mm`."""
    bx, by, bz = base_at_rail_zero
    c, s = _yaw_cos_sin(BASE_YAW_DEG if yaw_deg is None else yaw_deg)
    ax, ay = xyz_base[0], xyz_base[1]
    return (ax * c - ay * s + bx + rail_mm,
            ax * s + ay * c + by,
            xyz_base[2] + bz)


#: The orientation half of the same frame relationship.
#:
#: Both the twin and the xArm controller report orientation as RPY degrees under
#: the convention ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` -- confirmed in three
#: independent places: ``perception.d435i_calib._euler_to_mat``, MuJoCo's
#: ``mat2euler(axes='sxyz')`` in ``sim.mujoco_env.get_position``, and UFACTORY's
#: own ``rpy_to_rot`` (``Rot = Rz*Ry*Rx``) in ``ufactory_vision``.
#:
#: Because the base->world rotation is itself a yaw, it composes on the LEFT of
#: that Rz and collapses to a scalar addition:
#:
#:     R_world = Rz(BASE_YAW) @ Rz(yaw) Ry(pitch) Rx(roll)
#:             = Rz(yaw + BASE_YAW) Ry(pitch) Rx(roll)
#:
#: so only yaw changes, exactly, with no gimbal caveat and no matrix round trip.
#: That identity is a shortcut, not an assumption: `test_rpy_matches_matrix_
#: composition` in hardware/test_real_arm.py checks it against the full matrix
#: product at a spread of angles, including pitched poses where a careless
#: euler shortcut would break.
def base_to_world_rpy_deg(rpy_deg, yaw_deg: float = None):
    """Arm-base orientation -> world orientation, as (roll, pitch, yaw) degrees.

    The translation half is `base_to_world_mm`. Converting one without the other
    is how `get_position` came to return a world position stapled to a base
    orientation -- a pose in no frame at all, which the wrist camera then
    composed into its hand-eye chain 90 deg out.
    """
    roll, pitch, yaw = (float(v) for v in rpy_deg[:3])
    return (roll, pitch,
            _wrap_deg(yaw + (BASE_YAW_DEG if yaw_deg is None else yaw_deg)))


def world_to_base_rpy_deg(rpy_deg, yaw_deg: float = None):
    """World orientation -> arm-base orientation. Inverse of `base_to_world_rpy_deg`."""
    roll, pitch, yaw = (float(v) for v in rpy_deg[:3])
    return (roll, pitch,
            _wrap_deg(yaw - (BASE_YAW_DEG if yaw_deg is None else yaw_deg)))


def world_to_base_pose(pose_world, rail_mm: float,
                       base_at_rail_zero=BASE_AT_RAIL_ZERO_MM,
                       yaw_deg: float = None):
    """A whole world pose [x,y,z,roll,pitch,yaw] -> the arm base frame.

    Prefer this over calling `world_to_base_mm` and forgetting the orientation.
    That exact omission happened independently in THREE places -- the hardware
    wrapper, the preflight waypoint converter, and the plan-validation gate --
    each of which converted the position and passed `*pose[3:]` straight
    through. Two of them carry a comment about not duplicating the arithmetic;
    duplicating it was never the problem, duplicating it INCOMPLETELY was.

    A pose is six numbers in one frame. Converting three of them is not a
    partial conversion, it is a pose in no frame at all.
    """
    x, y, z = world_to_base_mm(pose_world[:3], rail_mm, base_at_rail_zero, yaw_deg)
    roll, pitch, yaw = world_to_base_rpy_deg(pose_world[3:6], yaw_deg)
    return [x, y, z, roll, pitch, yaw]


def base_to_world_pose(pose_base, rail_mm: float,
                       base_at_rail_zero=BASE_AT_RAIL_ZERO_MM,
                       yaw_deg: float = None):
    """A whole base pose -> world. Inverse of `world_to_base_pose`."""
    x, y, z = base_to_world_mm(pose_base[:3], rail_mm, base_at_rail_zero, yaw_deg)
    roll, pitch, yaw = base_to_world_rpy_deg(pose_base[3:6], yaw_deg)
    return [x, y, z, roll, pitch, yaw]


def _wrap_deg(a: float) -> float:
    """Fold an angle into (-180, 180]. The controller accepts either, but a pose
    printed as yaw=270 next to a twin pose of yaw=-90 reads as a disagreement."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def check_workspace_world(xyz_world, aabb=None, floor_z=None) -> list[str]:
    """Describe every way a commanded world pose is out of bounds; empty if fine."""
    aabb = aabb or WORKSPACE_AABB_MM
    floor_z = WORKSPACE_FLOOR_Z_MM if floor_z is None else floor_z
    x, y, z = xyz_world
    bad = []
    if z < floor_z:
        bad.append(f"z={z:.0f} is below the {floor_z:.0f} mm benchtop clearance "
                   f"plane (benchtop {BENCH_TOP_Z_MM:.0f} + "
                   f"{floor_z - BENCH_TOP_Z_MM:.0f} mm)")
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
