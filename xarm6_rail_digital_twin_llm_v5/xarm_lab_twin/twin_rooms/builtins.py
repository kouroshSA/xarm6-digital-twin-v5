# twin_rooms/builtins.py
"""Parametric objects for `"source": "builtin:<kind>"`.

Each generator adds one body to an MjSpec body (normally the worldbody) and
names every element `<prefix><name>_<part>`, following the lab scene's own
convention (green_cube_geom, green_bin_w_front, tube_L1_cap...), so code that
knows the lab's names knows these.

The DEFAULTS reproduce the lab's objects: builtin:cube is green_cube,
builtin:bin is green_bin, builtin:tube_rack is left_tube_rack and builtin:tube
is tube_L1. That is a copy of numbers that live in lab_scene_primitive.xml, so
twin_rooms.test_builder builds each default and compares it, geom by geom,
with the lab body. Change a lab object and that test fails until this file
follows.

Body origins (what a room's `pose.pos` places):
  cube       the cube's centre
  bin        the centre of the floor's underside (rests on a surface at that z)
  tube_rack  the centre of the base plate (base underside is 5 mm below)
  tube       the tube's centre
  platform   the centre of its footprint at floor level (pose z = 0 stands it
             on the floor); a solid static box for raising furniture. Not a
             lab object, so it has no lab counterpart to match.
"""
from __future__ import annotations

from typing import Callable, Dict

import mujoco
import numpy as np

# Contact parameters shared by the lab's small rigid objects (class "cube",
# "tube_body", "well_plate_body" in the lab scene).
FRICTION = [1.0, 0.05, 0.0001]
SOLREF = [0.005, 1.0]
SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

DEFAULTS: Dict[str, dict] = {
    "cube": {"size": 0.03, "rgba": [0.2, 0.8, 0.3, 1.0], "mass": 0.05},
    "bin": {"size": [0.08, 0.08, 0.06], "wall": 0.004,
            "rgba": [0.2, 0.8, 0.3, 0.6], "mass": 0.5},
    "tube_rack": {"rows": 2, "cols": 4, "pitch": 0.04,
                  "rgba": [0.85, 0.85, 0.85, 0.7], "mass": 0.25},
    "tube": {"radius": 0.015, "length": 0.115, "rgba": [0.95, 0.95, 0.95, 0.5],
             "cap_rgba": [0.95, 0.55, 0.10, 1.0], "mass": 0.03},
    "platform": {"size": [0.8, 0.8, 0.78], "rgba": [0.45, 0.47, 0.5, 1.0], "mass": 40.0},
}

# The lab's hand-set inertias at its default dimensions, scaled with mass and
# size for other parameters (a hollow bin or rack is not a solid box, so the
# lab's numbers are a better base than a solid-box formula).
_BIN_INERTIA = np.array([6e-4, 6e-4, 9e-4])
_RACK_INERTIA = np.array([3e-4, 7e-4, 8e-4])
_TUBE_INERTIA = np.array([3e-5, 3e-5, 5e-6])


def _scaled(base_inertia, mass, base_mass, extent, base_extent):
    s = (np.asarray(extent, float) / np.asarray(base_extent, float))
    return list(base_inertia * (mass / base_mass) * float(np.mean(s)) ** 2)


def _params(kind: str, params: dict) -> dict:
    return {**DEFAULTS[kind], **(params or {})}


def _solid(geom):
    geom.friction = FRICTION
    geom.solref = SOLREF
    geom.solimp = SOLIMP
    return geom


def cube(parent, name: str, prefix: str, params: dict):
    p = _params("cube", params)
    size = p["size"] if isinstance(p["size"], list) else [p["size"]] * 3
    b = parent.add_body(name=prefix + name)
    _solid(b.add_geom(name=f"{prefix}{name}_geom", type=BOX,
                      size=[s / 2 for s in size], mass=p["mass"], rgba=p["rgba"]))
    return b


def bin(parent, name: str, prefix: str, params: dict):
    p = _params("bin", params)
    sx, sy, h = p["size"]
    t = p["wall"]
    b = parent.add_body(name=prefix + name)
    b.explicitinertial = True
    b.mass = p["mass"]
    b.ipos = [0, 0, h / 3]
    b.inertia = _scaled(_BIN_INERTIA, p["mass"], 0.5, p["size"], [0.08, 0.08, 0.06])
    b.add_geom(name=f"{prefix}{name}_floor", type=BOX, size=[sx / 2, sy / 2, 0.001],
               pos=[0, 0, 0.001], rgba=p["rgba"], mass=0)
    wall_rgba = [0.8, 0.8, 0.8, 0.5]
    for side, size, pos in (
            ("front", [sx / 2, t / 2, h / 2], [0, -sy / 2, h / 2]),
            ("back", [sx / 2, t / 2, h / 2], [0, sy / 2, h / 2]),
            ("left", [t / 2, sy / 2, h / 2], [-sx / 2, 0, h / 2]),
            ("right", [t / 2, sy / 2, h / 2], [sx / 2, 0, h / 2])):
        b.add_geom(name=f"{prefix}{name}_w_{side}", type=BOX, size=size, pos=pos,
                   rgba=wall_rgba, mass=0)
    return b


def tube_rack(parent, name: str, prefix: str, params: dict):
    p = _params("tube_rack", params)
    rows, cols, pitch = p["rows"], p["cols"], p["pitch"]
    hx, hy = cols * pitch / 2, rows * pitch / 2           # half extents of the grid
    wall_h, t = 0.025, 0.001                               # half height, half thickness
    b = parent.add_body(name=prefix + name)
    b.explicitinertial = True
    b.mass = p["mass"]
    b.ipos = [0, 0, 0.020]
    b.inertia = _scaled(_RACK_INERTIA, p["mass"], 0.25, [hx, hy], [0.08, 0.04])
    b.add_geom(name=f"{prefix}{name}_base", type=BOX, size=[hx + 0.005, hy + 0.005, 0.005],
               rgba=[0.6, 0.6, 0.6, 1.0], mass=0)
    walls = [("outer_front", [hx, t, wall_h], [0, -hy, 0.030]),
             ("outer_back", [hx, t, wall_h], [0, hy, 0.030]),
             ("outer_left", [t, hy, wall_h], [-hx, 0, 0.030]),
             ("outer_right", [t, hy, wall_h], [hx, 0, 0.030])]
    walls += [(f"sep_v{i}", [t, hy, wall_h], [-hx + i * pitch, 0, 0.030]) for i in range(1, cols)]
    walls += [(f"sep_h{i}", [hx, t, wall_h], [0, -hy + i * pitch, 0.030]) for i in range(1, rows)]
    for part, size, pos in walls:
        b.add_geom(name=f"{prefix}{name}_{part}", type=BOX, size=size, pos=pos,
                   rgba=p["rgba"], mass=0)
    return b


def tube(parent, name: str, prefix: str, params: dict):
    p = _params("tube", params)
    r, half = p["radius"], p["length"] / 2
    b = parent.add_body(name=prefix + name)
    b.explicitinertial = True
    b.mass = p["mass"]
    b.ipos = [0, 0, 0.01]
    b.inertia = _scaled(_TUBE_INERTIA, p["mass"], 0.03, [r, half], [0.015, 0.0575])
    _solid(b.add_geom(name=f"{prefix}{name}_body", type=CYL, size=[r, half],
                      rgba=p["rgba"], mass=0))
    b.add_geom(name=f"{prefix}{name}_cap", type=CYL, size=[r + 0.0005, 0.006],
               pos=[0, 0, half + 0.006], rgba=p["cap_rgba"], mass=0)
    return b


def platform(parent, name: str, prefix: str, params: dict):
    p = _params("platform", params)
    sx, sy, h = p["size"]
    b = parent.add_body(name=prefix + name)
    _solid(b.add_geom(name=f"{prefix}{name}_geom", type=BOX, size=[sx / 2, sy / 2, h / 2],
                      pos=[0, 0, h / 2], mass=p["mass"], rgba=p["rgba"]))
    return b


GENERATORS: Dict[str, Callable] = {"cube": cube, "bin": bin, "tube_rack": tube_rack,
                                   "tube": tube, "platform": platform}
