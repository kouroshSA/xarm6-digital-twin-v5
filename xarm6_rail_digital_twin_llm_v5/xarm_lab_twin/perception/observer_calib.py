"""Fixed observer cameras watching the bench — where they are, and why there.

The wrist camera has a structural blind spot: it sees only what the gripper is
pointed at, it is occluded by its own gripper during the final approach, and it
cannot answer "did the cube land in the cup?" without the arm going to look.
On real hardware that last one has no other answer at all -- ``physical_outcome()``
reads MuJoCo's ground truth, which does not exist on a bench.

These are the second pair of eyes. Same model of camera as the wrist one, run at
720p rather than 640x480 because they watch from three to four times further
away; the profile is the same device's, read at that resolution.

**What an observer is good at, and what it is not.** Measured through the full
stack: an overhead observer locates a cube to ~5 mm in X-Y, and the oblique one
to ~9 mm -- good enough to survey a bench and to answer "did it land in the cup".

But overhead it confuses one colour of cube for another. From directly above you
see only a cube's top face, flat and top-lit, which is weak colour evidence; the
grounding model separates green from blue there by about 0.03 of score, which is
noise. Raising the resolution to 720p did NOT fix it -- that was the first guess
and it was wrong, so the cause is the viewpoint, not the pixel count.

Which is the architecture, not a defect to design around: survey and verify with
the observers, then let the WRIST camera -- closer, seeing the sides, target
centred -- decide identity and commit the grasp.

Where they are, and the measurement behind it
---------------------------------------------
Candidate placements were scored against the scene -- frustum coverage,
ray-cast occlusion with the arm in a working pose, and view tilt:

    placement                       sees   occluded   depth error into XY
    on the OT-2, looking NW          1/9       7             71%
    tripod, photo position (SW)      6/9       2             89%
    west / operator side, oblique    8/9       1             90%
    north-west, high                 8/9       1             84%
    near-top-down over the strip     7/9       0             14%

Two results decided the layout.

**The OT-2 is the worst mount available.** It is the most convenient flat
surface in the cell and it looks *along* the bench, so the bins hide the cubes
and the arm hides the rest: one object of nine.

**Tilt matters more than position.** A depth camera's error lies along its
viewing axis, so an oblique camera puts ~90% of that error into X and Y -- which
is exactly the coordinate handed to the arm. A near-vertical view puts 86% of it
into Z, where a top-down grasp barely cares. Same sensor, same noise, six times
the useful accuracy, bought with mounting height rather than money.

So OVERHEAD is the primary and TRIPOD is the complement: overhead measures
position well but cannot see object height or into a cup, and is blocked by the
arm during the approach it is watching. The oblique view fills exactly those
gaps. They are not redundant, and neither is a substitute for the other.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .d435i_calib import COLOR_INTRINSICS_720P, Intrinsics
from .d435i_calib import mujoco_camera_attrs

# The region that actually matters: the cubes, bins and cup the arm works with.
# Everything here is aimed at its centre rather than at the whole bench, because
# a D435 at a sensible height covers about a metre and the full bench is 1.5.
WORKING_STRIP_CENTRE = np.array([0.0, -0.25, 0.80], dtype=np.float64)


@dataclass(frozen=True)
class Observer:
    """One fixed camera: where it sits, what it looks at, what it is for."""

    name: str
    pos: tuple[float, float, float]
    look_at: tuple[float, float, float]
    purpose: str

    @property
    def tilt_deg(self) -> float:
        """Angle between the view axis and straight down.

        The number that decides how much depth error lands in X-Y: at 0 degrees
        none of it does, at 90 all of it. Worth reading before trusting an
        observer's xy to a few millimetres.
        """
        fwd = np.array(self.look_at) - np.array(self.pos)
        fwd /= np.linalg.norm(fwd)
        return math.degrees(math.acos(abs(float(fwd @ np.array([0.0, 0.0, -1.0])))))

    @property
    def xy_error_fraction(self) -> float:
        """Fraction of a depth error that shows up in X-Y at this tilt."""
        return math.sin(math.radians(self.tilt_deg))

    def xyaxes(self) -> tuple[np.ndarray, np.ndarray]:
        """MuJoCo ``xyaxes``: the camera's right and up vectors, in world.

        Used instead of a quaternion because a fixed camera is naturally
        specified as "sit here, look there", and xyaxes is that with no
        hand-rolled rotation in between. A MuJoCo camera looks down its own -z,
        so +z is backwards along the view.
        """
        z = np.array(self.pos, dtype=np.float64) - np.array(self.look_at, dtype=np.float64)
        z /= np.linalg.norm(z)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(z @ world_up)) > 0.98:
            # Looking (almost) straight down: world up is degenerate as a
            # reference, so pick the bench's long axis instead. Without this the
            # overhead camera's roll is numerically arbitrary and flips between
            # regenerations, silently rotating every image it produces.
            world_up = np.array([0.0, 1.0, 0.0])
        x = np.cross(world_up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return x, y


OBSERVERS: tuple[Observer, ...] = (
    Observer(
        name="cam_overhead",
        # ~0.75 m above the objects, offset to the operator side so the view is
        # 15 degrees off vertical rather than dead-on: enough to keep the X-Y
        # accuracy that makes this the primary camera, enough to see that an
        # object has height at all.
        pos=(0.0, -0.45, 1.55),
        look_at=tuple(WORKING_STRIP_CENTRE),
        purpose="primary: object positions, and did-it-land verification",
    ),
    Observer(
        name="cam_tripod",
        # The tripod already standing in the cell (see docs/real-lab photo).
        pos=(0.75, -1.05, 1.35),
        look_at=tuple(WORKING_STRIP_CENTRE),
        purpose="complement: object height, into-the-cup views, occlusion backup",
    ),
)

# Observers run at 720p while the wrist camera stays at 640x480: they watch from
# three to four times further away, so the same object covers a quarter of the
# pixels, and these are static cameras where the bandwidth and latency arguments
# for keeping the wrist small do not apply.
#
# It buys sharper masks and better size estimates. It does NOT fix the colour
# confusion described above -- that was the reason for trying it, and the
# confusion survived unchanged at both resolutions. Kept because more pixels at
# this range is right anyway, not because it solved that.
OBSERVER_INTRINSICS: Intrinsics = COLOR_INTRINSICS_720P

OBSERVER_BODY_NAME = "observers"

# Housing half-extents, metres. Cosmetic -- non-colliding, massless -- but drawn
# so the render shows where a real camera would have to be bolted. A viewpoint
# that looks fine as a disembodied frustum and impossible as a physical mount is
# not a viewpoint.
HOUSING_HALF = (0.045, 0.0125, 0.0125)


def by_name(name: str) -> Observer:
    for obs in OBSERVERS:
        if obs.name == name:
            return obs
    raise KeyError(f"no observer {name!r}; have {[o.name for o in OBSERVERS]}")


def emit_scene_xml(indent: str = " " * 4) -> str:
    """The ``<body name="observers">`` block, generated from OBSERVERS.

    Written by ``perception/sync_scene.py``; do not hand-edit the result. The
    poses here and the poses in the scene are the same fact, and this is the
    copy that gets to be authoritative.
    """
    attrs = mujoco_camera_attrs(OBSERVER_INTRINSICS)
    hx, hy, hz = HOUSING_HALF
    lines = [
        f'{indent}<!-- Fixed observer cameras. GENERATED by',
        f'{indent}     perception/observer_calib.py::emit_scene_xml() -- regenerate with',
        f'{indent}     `python -m perception.sync_scene`, do not retype.',
        f'{indent}     Placements were measured, not chosen: see that module for the',
        f'{indent}     coverage/occlusion/tilt table. Housings are massless and',
        f'{indent}     non-colliding; they exist so a render shows where a real camera',
        f'{indent}     would have to be mounted. -->',
        f'{indent}<body name="{OBSERVER_BODY_NAME}" pos="0 0 0">',
    ]
    for obs in OBSERVERS:
        x, y = obs.xyaxes()
        px, py, pz = obs.pos
        lines += [
            f'{indent}  <!-- {obs.name}: {obs.purpose}',
            f'{indent}       {obs.tilt_deg:.0f} deg off vertical -> '
            f'{obs.xy_error_fraction * 100:.0f}% of depth error lands in X-Y -->',
            f'{indent}  <geom name="{obs.name}_housing" type="box"',
            f'{indent}        size="{hx:.4f} {hy:.4f} {hz:.4f}"',
            f'{indent}        pos="{px:.4f} {py:.4f} {pz:.4f}"',
            f'{indent}        rgba="0.12 0.12 0.14 1" mass="0"',
            f'{indent}        contype="0" conaffinity="0"/>',
            f'{indent}  <camera name="{obs.name}"',
            f'{indent}          pos="{px:.6f} {py:.6f} {pz:.6f}"',
            f'{indent}          xyaxes="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} '
            f'{y[0]:.6f} {y[1]:.6f} {y[2]:.6f}"',
            f'{indent}          resolution="{attrs["resolution"]}"',
            f'{indent}          sensorsize="{attrs["sensorsize"]}"',
            f'{indent}          focalpixel="{attrs["focalpixel"]}"',
            f'{indent}          principalpixel="{attrs["principalpixel"]}"/>',
        ]
    lines.append(f'{indent}</body>')
    return "\n".join(lines)


def summary() -> str:
    rows = [f"{len(OBSERVERS)} observer camera(s), "
            f"{OBSERVER_INTRINSICS.width}x{OBSERVER_INTRINSICS.height} D435 optics:"]
    for obs in OBSERVERS:
        dist = np.linalg.norm(np.array(obs.pos) - np.array(obs.look_at))
        rows.append(
            f"  {obs.name:14s} at {tuple(round(v, 2) for v in obs.pos)} m, "
            f"{dist:.2f} m from the strip, {obs.tilt_deg:4.0f} deg off vertical "
            f"({obs.xy_error_fraction * 100:.0f}% of depth error -> X-Y)")
        rows.append(f"                 {obs.purpose}")
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit-xml", action="store_true")
    args = ap.parse_args()
    print(emit_scene_xml() if args.emit_xml else summary())
