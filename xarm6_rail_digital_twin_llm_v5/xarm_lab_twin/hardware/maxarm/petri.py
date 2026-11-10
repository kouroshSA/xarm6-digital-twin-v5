"""A 90 mm Petri dish as the twin models it: base and lid, two loose bodies.

ASSUMED standard polystyrene dimensions (mm). `NOTES_NEXT.md` says to MEASURE
them on the real dishes; until then the twin's dish is a typical one, and the
one number that matters most for the lid task -- the radial clearance between
the lid's skirt and the base's wall -- is 1.0 mm. The bench session of
2026-11-09 found the lid perched on the rim often enough that the real script
grew a re-seat step, which is consistent with a clearance of that order.

Heights are what the twin's suction cup meets; the real cup's bellows compress
~4-5 mm when it seals (`scripts/maxarm_lid.py` PRESS_MM), which is why the real
"lid top" read back near wrist z 60 while a rigid 52 mm tool over a 15 mm dish
would read 67.

Both parts are a disc plus a ring of `RING_SEGMENTS` thin boxes (MuJoCo has no
hollow cylinder primitive); the builder adds the free joint.
"""
from __future__ import annotations

import math

BASE_OD_MM = 88.0          # the fit: the base wall's OUTER face
BASE_HEIGHT_MM = 14.0
BASE_FLOOR_MM = 1.0
LID_ID_MM = 90.0           # the fit: the lid skirt's INNER face
LID_HEIGHT_MM = 9.0        # skirt bottom to top surface
LID_TOP_MM = 1.0
#: Wall thickness of the collision plates. Real polystyrene is ~1 mm; 1 mm
#: plates let a lid set down 6 mm off-centre pass its skirt THROUGH the base
#: wall (MuJoCo's box-box contact on two thin plates under a soft 2 mm drop),
#: so it read as seated at the right height when a real lid would perch,
#: tilted, on the rim. Thicker plates keep the two faces that define the fit
#: where they are (the lid grows outward, the base inward) and collide reliably.
WALL_MM = 2.5
LID_OD_MM = LID_ID_MM + 2 * WALL_MM
RING_SEGMENTS = 24

BASE_MASS_KG = 0.009
LID_MASS_KG = 0.007

#: Radial clearance between the lid skirt and the base wall when seated.
CLEARANCE_MM = (LID_ID_MM - BASE_OD_MM) / 2.0
#: The lid rests by its top plate on the base rim, so its top surface sits
#: BASE_HEIGHT + LID_TOP above the table, and its skirt bottom this far up:
SEATED_LID_BOTTOM_MM = BASE_HEIGHT_MM + LID_TOP_MM - LID_HEIGHT_MM
SEATED_LID_TOP_MM = BASE_HEIGHT_MM + LID_TOP_MM
#: A lid lying on the table by itself.
LID_ALONE_TOP_MM = LID_HEIGHT_MM

assert CLEARANCE_MM > 0, "the lid must fit over the base"
assert SEATED_LID_BOTTOM_MM > 0, "the lid skirt must not reach the table when seated"


def _ring(prefix: str, radius_mm: float, z0_mm: float, z1_mm: float, mass_kg: float,
          rgba: str) -> list[str]:
    """`RING_SEGMENTS` boxes tangent to a circle of `radius_mm` (the wall's mid
    radius), spanning z0..z1, as MJCF geom lines (metres). Each box's long
    side is turned 90 deg past its bearing so it lies ALONG the circle. The
    angle is in DEGREES: these files carry no <compiler angle="radian">, and
    a radian value here was read as a degree, so every plate stood radial,
    6 mm thick, the skirt and the wall interpenetrated 0.9 mm, and lifting
    the lid lifted the dish with it (found on the twin, 2026-11-10)."""
    n = RING_SEGMENTS
    half_w = math.pi * radius_mm / n * 1.02 / 1000.0     # 2% overlap, no gaps
    half_t = WALL_MM / 2.0 / 1000.0
    half_h = (z1_mm - z0_mm) / 2.0 / 1000.0
    zc = (z0_mm + z1_mm) / 2.0 / 1000.0
    r = radius_mm / 1000.0
    out = []
    for k in range(n):
        a = 2.0 * math.pi * k / n
        out.append(f'<geom name="{prefix}_wall{k}" type="box" size="{half_w:.6f} {half_t:.6f} {half_h:.6f}" '
                   f'pos="{r * math.cos(a):.6f} {r * math.sin(a):.6f} {zc:.6f}" '
                   f'axisangle="0 0 1 {math.degrees(a) + 90.0:.4f}" mass="{mass_kg / n:.6f}" rgba="{rgba}"/>')
    return out


def base_mjcf(name: str = "petri_base") -> str:
    """One top-level body, origin at the centre of its underside."""
    rgba = "0.85 0.9 0.95 0.45"
    r = (BASE_OD_MM - 2 * WALL_MM) / 2.0 / 1000.0
    floor = (f'<geom name="{name}_floor" type="cylinder" size="{r:.6f} {BASE_FLOOR_MM / 2000.0:.6f}" '
             f'pos="0 0 {BASE_FLOOR_MM / 2000.0:.6f}" mass="{BASE_MASS_KG * 0.6:.6f}" rgba="{rgba}"/>')
    walls = _ring(name, (BASE_OD_MM - WALL_MM) / 2.0, 0.0, BASE_HEIGHT_MM, BASE_MASS_KG * 0.4, rgba)
    return _wrap(name, [floor] + walls)


def lid_mjcf(name: str = "petri_lid") -> str:
    """One top-level body, origin at the centre of the skirt's bottom edge."""
    rgba = "0.85 0.9 0.95 0.45"
    r = LID_OD_MM / 2.0 / 1000.0
    top = (f'<geom name="{name}_top" type="cylinder" size="{r:.6f} {LID_TOP_MM / 2000.0:.6f}" '
           f'pos="0 0 {(LID_HEIGHT_MM - LID_TOP_MM / 2.0) / 1000.0:.6f}" '
           f'mass="{LID_MASS_KG * 0.6:.6f}" rgba="{rgba}"/>')
    walls = _ring(name, (LID_ID_MM + WALL_MM) / 2.0, 0.0, LID_HEIGHT_MM - LID_TOP_MM,
                  LID_MASS_KG * 0.4, rgba)
    return _wrap(name, [top] + walls)


def _wrap(name: str, geoms: list[str]) -> str:
    body = "\n".join("      " + g for g in geoms)
    return (f'<mujoco model="{name}">\n'
            f'  <!-- GENERATED by hardware/maxarm/petri.py -- DO NOT EDIT; edit the dimensions there. -->\n'
            f'  <worldbody>\n'
            f'    <body name="{name}">\n{body}\n'
            f'    </body>\n'
            f'  </worldbody>\n'
            f'</mujoco>\n')


# -- the maxarm_petri room's layout, in the ARM FRAME (mm) --------------------
# Dishes on an arc in front of the arm and a set-aside spot for each lid. Polar
# (angle from straight ahead, positive toward the arm's RIGHT = +x; radius).
# Chords between neighbours are >= 106 mm, so a 92 mm lid clears the next dish
# by at least 14 mm even with the measured ~3 mm placement scatter.
#: The MaxArm's base plate centre on the lab bench, world metres, facing -y.
ROOM_BASE_POS_M = (0.30, -0.60, 0.75)
DISH_RADIUS_MM = 205.0
DISH_ANGLES_DEG = (-55.0, 0.0, 55.0)
ASIDE_POLAR = ((-85.0, 205.0), (27.5, 235.0), (85.0, 205.0))


def polar_arm(angle_deg: float, radius_mm: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return (radius_mm * math.sin(a), -radius_mm * math.cos(a))


def dish_centres_arm() -> list[tuple[float, float]]:
    return [polar_arm(a, DISH_RADIUS_MM) for a in DISH_ANGLES_DEG]


def aside_spots_arm() -> list[tuple[float, float]]:
    return [polar_arm(a, r) for a, r in ASIDE_POLAR]
