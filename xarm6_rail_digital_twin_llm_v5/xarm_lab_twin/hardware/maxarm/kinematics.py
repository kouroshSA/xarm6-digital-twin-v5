"""Hiwonder MaxArm kinematics, ported from the vendor's own solver.

Runs on the PC, so every pose can be checked BEFORE anything is sent to the
arm. The board solves IK itself; this copy exists to refuse a pose up front,
and `scripts/maxarm_probe.py --kinematics` compares it with the board's.

Provenance
----------
The board runs `__espmax.mpy` (compiled MicroPython, mpy v5). Its constant
table holds L0=84.4, L1=8.14, L2=128.4, L3=138.0, and L4 is passed in from
`espmax.py` as 16.8. Those constants, and the algorithm below, match
`_espmax.cpp` / `_espmax.h` in Hiwonder's Arduino sources
(docs.hiwonder.com/projects/MaxArm, `Inverse_Kinematics_Basic_and_Application.zip`),
and the port follows that C source line by line.

The vendor ships THREE copies of these numbers, and they disagree:

* `espmax.py` has L0=84.0, L1=8.2, L2=128.0. It uses them ONLY to compute
  `ESPMax.ORIGIN` (the home point, (0, -163, 212)). The solver never reads them.
* The Arduino `ESPMax.cpp` caps z at 255; the MicroPython `espmax.py` caps it at
  225.
* The x sign. The published `_espmax.cpp` negates x inside `inverse()` and
  `forward()`; the board's compiled `__espmax.mpy` does NOT (measured on the lab
  unit, 2026-11-07: `__espmax.inverse(16.8, (120, -180, 85))` gives joint 1
  153.69, the C source 86.31). `espmax.py` negates x going in and coming out, so
  on this firmware the MicroPython `ESPMax` frame and the C-source frame agree.
  An earlier version of this module assumed the solver negates as the source
  does, and worked in the mirror image of `ESPMax`; `--kinematics` caught it.

This module uses the board's own convention, the MicroPython `ESPMax` frame
(called the ARM FRAME below), because the backend drives that class. In this
frame the home point is (0, -163, 212): the arm reaches along -y.
`_solver_inverse` / `_solver_forward` are the board's `__espmax` functions;
`inverse` / `forward` wrap them exactly as `espmax.py` does.

PROVISIONAL until measured on the device: which physical direction +x points.
The vendor frame is self-consistent (set_position and read_position agree), but
nothing in the docs says which way it faces. A wrist-camera hand-eye chain built
on the wrong sign would mirror every target -- the same shape as the xArm's
unmeasured 90 deg base yaw. `maxarm_probe.py --frame` settles it.

Silent vendor clamps this module REFUSES instead
------------------------------------------------
`ESPMax.set_position` quietly changes a pose instead of rejecting it:

* z > 225 is lowered to 225.
* servo 3 below pulse 470 is raised to 470, and servo 2 above 700 is lowered to
  700 (`set_servo_in_range`).
* a reach under 50 mm returns None -- neither True nor False -- and does nothing.

Each of these returns success while the tool ends up somewhere else, which is
defect class #2. `check_pose` rejects all of them, so the board's clamps never
fire on a pose that passed here.

Units: mm and degrees throughout; servo positions are bus-servo pulses (0-1000).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# -- geometry, from the board's compiled solver (see module docstring) --------
L0 = 84.4    # base plate to shoulder axis, vertical
L1 = 8.14    # shoulder axis horizontal offset from the base axis
L2 = 128.4   # upper arm
L3 = 138.0   # forearm
L4 = 16.8    # wrist to nozzle axis, horizontal (passed in by espmax.py)

#: Home, as the vendor defines it (`ESPMax.ORIGIN`). Computed from espmax.py's
#: OTHER link values, so it is not exactly FK(500,500,500), which gives
#: (0, -162.94, 212.8). Kept verbatim because the board's go_home() uses it.
HOME_XYZ_MM: tuple[float, float, float] = (0.0, -(8.2 + 138.0 + 16.8), 84.0 + 128.0)

#: Pulses at the vendor home posture (`ESPMax.servos` initial value).
HOME_PULSES: tuple[int, int, int] = (500, 500, 500)

# -- limits ------------------------------------------------------------------
PULSE_MIN, PULSE_MAX = 0, 1000
#: Vendor clamps in `ESPMax.set_servo_in_range`, enforced here as refusals.
SERVO2_PULSE_MAX = 700
SERVO3_PULSE_MIN = 470
#: Vendor clamp in `ESPMax.set_position` (MicroPython; the Arduino one is 255).
Z_MAX_MM = 225.0
#: Vendor refusal in `ESPMax.set_position`.
MIN_REACH_MM = 50.0

#: Lowest z this module accepts by default. **PROVISIONAL, NOT MEASURED.**
#: FK z is the WRIST height. The vendor model has no vertical tool length, so
#: the suction cup's lip sits some unmeasured distance below the z reported
#: here, and the table plane in this frame is unknown. 85 mm is the lowest z the
#: vendor's own examples command (Chapter 10 example: (120, -180, 85)), so it is
#: a height the vendor considers clear of the table. Lower it only after
#: measuring where the cup touches the table (`maxarm_probe.py --floor`).
DEFAULT_FLOOR_Z_MM = 85.0

#: Nozzle (PWM servo) rotation range, degrees, and its pulse mapping
#: (`SuctionNozzle.set_angle`: -90..90 -> 500..2500 us).
NOZZLE_ANGLE_RANGE_DEG = (-90.0, 90.0)


class Unreachable(ValueError):
    """The pose cannot be reached, or only by a pose the vendor would alter.

    `reasons` lists every problem found, not just the first.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


@dataclass(frozen=True)
class Solution:
    xyz_mm: tuple[float, float, float]
    joints_deg: tuple[float, float, float]   # vendor joint angles (see deg_to_pulse)
    pulses: tuple[float, float, float]       # bus-servo targets, unrounded


# -- the vendor solver, ported ------------------------------------------------

def _base_heading_deg(x: float, y: float) -> float:
    """Heading used by the C solver: degrees in (30, 390].

    The C code builds this from atan(y/x) with explicit quadrant cases; atan2
    reduced mod 360 is equivalent for every case it handles (x=0, y=0, both
    signs). The `<= 30 -> +360` wrap then makes the 240 deg servo range one
    continuous interval, centred on -y (heading 270).
    """
    if x == 0.0 and y == 0.0:
        raise Unreachable(["x and y are both 0: the base heading is undefined"])
    h = math.degrees(math.atan2(y, x)) % 360.0
    if h <= 30.0:
        h += 360.0
    return h


def inverse(xyz_mm) -> tuple[float, float, float]:
    """Arm-frame xyz (mm) -> vendor joint angles (deg). Raises Unreachable.

    As `ESPMax.set_position` does: negate x, then call the solver.
    """
    x, y, z = (float(v) for v in xyz_mm)
    return _solver_inverse((-x, y, z))


def forward(joints_deg) -> tuple[float, float, float]:
    """Vendor joint angles (deg) -> arm-frame xyz (mm).

    As `ESPMax.pulses_to_position` does: call the solver, negate x.
    """
    x, y, z = _solver_forward(joints_deg)
    return (-x, y, z)


def _solver_inverse(xyz_mm) -> tuple[float, float, float]:
    """The board's `__espmax.inverse`, in the solver's own frame.

    Port of `inverse()` in `_espmax.cpp` without its x negation, which the
    board's compiled solver does not have (module docstring). The C version
    prints to serial and returns NaN for an out-of-reach target, and the board
    raises "Unreachable position". This raises too, before anything is sent.
    """
    x, y, z = (float(v) for v in xyz_mm)
    heading = _base_heading_deg(x, y)
    r = math.hypot(x, y) - L1 - L4
    zz = z - L0
    if r <= 0.0:
        raise Unreachable([f"reach {math.hypot(x, y):.1f} mm is inside the "
                           f"wrist offset L1+L4={L1 + L4:.2f} mm"])
    d2 = r * r + zz * zz
    d = math.sqrt(d2)
    cos_beta = (L2 * L2 + L3 * L3 - d2) / (2.0 * L2 * L3)
    cos_gamma = (L2 * L2 + d2 - L3 * L3) / (2.0 * L2 * d)
    if not (-1.0 <= cos_beta <= 1.0 and -1.0 <= cos_gamma <= 1.0):
        raise Unreachable([f"({x:.1f}, {y:.1f}, {z:.1f}) is out of reach: wrist "
                           f"distance {d:.1f} mm, links span "
                           f"{abs(L3 - L2):.1f}..{L2 + L3:.1f} mm"])
    alpha = math.atan(zz / r)
    beta = math.acos(cos_beta)
    gamma = math.acos(cos_gamma)
    theta2 = math.pi - (alpha + gamma)
    theta3 = math.pi - (alpha + beta + gamma)
    return (heading - 150.0, math.degrees(theta2), math.degrees(theta3))


def _solver_forward(joints_deg) -> tuple[float, float, float]:
    """The board's `__espmax.forward`, in the solver's own frame."""
    a1, a2, a3 = (math.radians(float(v)) for v in joints_deg)
    a1 += math.radians(150.0)
    if a1 > 2.0 * math.pi:
        a1 -= 2.0 * math.pi
    beta = a2 - a3
    side = math.sqrt(L2 * L2 + L3 * L3 - 2.0 * L2 * L3 * math.cos(beta))
    cos_gamma = (side * side + L2 * L2 - L3 * L3) / (2.0 * side * L2)
    gamma = math.acos(min(cos_gamma, 1.0))
    alpha = (math.pi - a2) - gamma
    z = side * math.sin(alpha)
    r = math.sqrt(max(side * side - z * z, 0.0))
    z += L0
    r += L1 + L4
    # The C source returns (-r cos, r sin, z); the board's solver returns
    # (+r cos, r sin, z) (measured), and `forward` negates x for the arm frame.
    return (r * math.cos(a1), r * math.sin(a1), z)


def deg_to_pulse(joints_deg) -> tuple[float, float, float]:
    """Vendor joint angles -> bus-servo pulses. Port of `deg_to_pulse()`.

    Joint conventions (from the C comments): j1 0..240 -> 0..1000;
    j2 210..-30 -> 0..1000 (reversed); j3 -120..120 -> 0..1000.
    """
    a1, a2, a3 = (float(v) for v in joints_deg)
    return (a1 * 1000.0 / 240.0,
            (a2 - 210.0) * 1000.0 / -240.0,
            (a3 + 120.0) * 1000.0 / 240.0)


def pulse_to_deg(pulses) -> tuple[float, float, float]:
    """Bus-servo pulses -> vendor joint angles. Port of `pulse_to_deg()`."""
    p1, p2, p3 = (float(v) for v in pulses)
    return (p1 * 240.0 / 1000.0,
            p2 * -240.0 / 1000.0 + 210.0,
            p3 * 240.0 / 1000.0 - 120.0)


def pulses_to_xyz(pulses) -> tuple[float, float, float]:
    return forward(pulse_to_deg(pulses))


# -- the gate -----------------------------------------------------------------

def check_pulses(pulses) -> list[str]:
    """Every reason these pulses would be refused or clamped. Empty = fine."""
    p1, p2, p3 = pulses
    bad = []
    for i, p in enumerate(pulses, start=1):
        if not (PULSE_MIN <= p <= PULSE_MAX):
            bad.append(f"servo {i} pulse {p:.0f} outside {PULSE_MIN}..{PULSE_MAX}")
    if p2 > SERVO2_PULSE_MAX:
        bad.append(f"servo 2 pulse {p2:.0f} > {SERVO2_PULSE_MAX} "
                   f"(the vendor would silently clamp it)")
    if p3 < SERVO3_PULSE_MIN:
        bad.append(f"servo 3 pulse {p3:.0f} < {SERVO3_PULSE_MIN} "
                   f"(the vendor would silently clamp it)")
    return bad


def solve(xyz_mm, floor_z_mm: float = DEFAULT_FLOOR_Z_MM) -> Solution:
    """Validated IK. Raises Unreachable listing every problem.

    A pose that passes here is one `ESPMax.set_position` will execute exactly:
    none of its silent clamps applies.
    """
    x, y, z = (float(v) for v in xyz_mm)
    bad = []
    if z > Z_MAX_MM:
        bad.append(f"z {z:.1f} > {Z_MAX_MM:.0f} (the vendor would silently "
                   f"lower it to {Z_MAX_MM:.0f})")
    if floor_z_mm is not None and z < floor_z_mm:
        bad.append(f"z {z:.1f} is below the floor {floor_z_mm:.1f}")
    reach = math.hypot(x, y)
    if reach < MIN_REACH_MM:
        bad.append(f"reach {reach:.1f} < {MIN_REACH_MM:.0f} mm (the vendor "
                   f"returns None and does not move)")
    try:
        joints = inverse((x, y, z))
    except Unreachable as exc:
        raise Unreachable(bad + exc.reasons) from None
    pulses = deg_to_pulse(joints)
    bad += check_pulses(pulses)
    if bad:
        raise Unreachable(bad)
    return Solution((x, y, z), joints, pulses)


def check_pose(xyz_mm, floor_z_mm: float = DEFAULT_FLOOR_Z_MM) -> list[str]:
    """Like `solve`, but returns the reasons instead of raising (gate style)."""
    try:
        solve(xyz_mm, floor_z_mm)
    except Unreachable as exc:
        return exc.reasons
    return []


def nozzle_angle_to_pulse(angle_deg: float) -> float:
    lo, hi = NOZZLE_ANGLE_RANGE_DEG
    return (angle_deg - lo) * (2500.0 - 500.0) / (hi - lo) + 500.0
