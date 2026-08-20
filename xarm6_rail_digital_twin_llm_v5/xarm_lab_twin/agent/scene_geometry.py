"""Query layer over the MuJoCo scene XML — the single source of truth for geometry.

The scene file is authoritative for *where things are* and *what can move*. Anything
that needs those facts (the LLM system prompt, `agent/objects.json`, the invariant
checks in `scripts/sim_checks.py`) should ask this module rather than hardcoding a
copy. Commit `fde3d22` moved most of the bench and the hand-maintained copies did
not follow; this module exists so that cannot happen twice.

Loads the compiled model rather than parsing XML text, so `<include>`, defaults, and
compiler-applied transforms are all resolved the same way MuJoCo resolves them.

Positions are returned in **millimetres** to match the xArm SDK and the LLM prompt.
MuJoCo works in metres internally; the conversion happens here, once.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import mujoco as mj
import numpy as np

DEFAULT_SCENE = "envs/lab_scene.xml"
M_TO_MM = 1000.0


@dataclass(frozen=True)
class BodyInfo:
    """One body's geometry facts, as the scene defines them."""

    name: str
    pos_mm: tuple[float, float, float]
    has_free_joint: bool
    n_joints: int

    @property
    def xy_mm(self) -> tuple[float, float]:
        return (self.pos_mm[0], self.pos_mm[1])

    @property
    def pushable(self) -> bool:
        """`push_object` requires a free joint; static bodies cannot be moved."""
        return self.has_free_joint


class SceneGeometry:
    """Read-only view of a scene's bodies.

    Built from a *forwarded* MjData so `xpos` reflects the resolved world pose of
    every body, including children of rotated parents — the local `pos` attribute
    in the XML would not.
    """

    def __init__(self, scene_xml: str = DEFAULT_SCENE):
        if not os.path.exists(scene_xml):
            raise FileNotFoundError(f"scene not found: {scene_xml}")
        self.scene_xml = scene_xml
        self._model = mj.MjModel.from_xml_path(scene_xml)
        data = mj.MjData(self._model)
        mj.mj_forward(self._model, data)

        self._bodies: dict[str, BodyInfo] = {}
        for bid in range(self._model.nbody):
            name = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_BODY, bid)
            if not name:
                continue
            adr = int(self._model.body_jntadr[bid])
            num = int(self._model.body_jntnum[bid])
            jtypes = [int(self._model.jnt_type[adr + i]) for i in range(num)]
            self._bodies[name] = BodyInfo(
                name=name,
                pos_mm=tuple(np.asarray(data.xpos[bid], dtype=float) * M_TO_MM),
                has_free_joint=int(mj.mjtJoint.mjJNT_FREE) in jtypes,
                n_joints=num,
            )

        # Geoms carry facts bodies don't — notably the OT-2 deck slot outlines, which
        # are the authoritative slot positions (they move with the OT-2 chassis).
        self._geoms: dict[str, tuple[float, float, float]] = {}
        for gid in range(self._model.ngeom):
            gname = mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_GEOM, gid)
            if gname:
                self._geoms[gname] = tuple(
                    np.asarray(data.geom_xpos[gid], dtype=float) * M_TO_MM)

    # -- lookups ------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._bodies

    def names(self) -> list[str]:
        return sorted(self._bodies)

    def get(self, name: str) -> BodyInfo | None:
        return self._bodies.get(name)

    def xy_mm(self, name: str) -> tuple[float, float] | None:
        b = self._bodies.get(name)
        return b.xy_mm if b else None

    def pushable_bodies(self) -> list[str]:
        """Every body `push_object` could actually move."""
        return sorted(n for n, b in self._bodies.items() if b.pushable)

    def missing(self, names) -> list[str]:
        return [n for n in names if n not in self._bodies]

    def xy_error_mm(self, name: str, expected_xy) -> float | None:
        """Planar distance between a claimed xy and the scene's xy. None if absent."""
        actual = self.xy_mm(name)
        if actual is None:
            return None
        return float(np.hypot(actual[0] - expected_xy[0], actual[1] - expected_xy[1]))


    # -- OT-2 deck ----------------------------------------------------------

    def deck_slots(self) -> dict[str, tuple[float, float, float]]:
        """OT-2 deck slot centres, read from the `ot2_slot_*_outline` geoms.

        Read rather than derived from a pitch constant, so the slots follow the
        chassis automatically if the OT-2 is repositioned in the scene.
        """
        out = {}
        for gname, pos in self._geoms.items():
            if gname.startswith("ot2_slot_") and gname.endswith("_outline"):
                label = gname[len("ot2_slot_"):-len("_outline")]
                out[label] = pos
        return dict(sorted(out.items(),
                           key=lambda kv: (kv[1][0], -kv[1][1])))


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

# Bodies the planner is expected to reason about. Anything here that is missing
# from the scene is reported rather than silently omitted, so a scene edit that
# deletes an object surfaces instead of quietly shrinking the prompt.
PROMPT_BODIES = [
    "red_cube_front", "red_cube_back",
    "green_cube", "blue_cube",
    "green_bin", "blue_bin", "clear_cup",
    "left_tube_rack", "right_tube_rack",
    "tube_L1", "tube_L2", "tube_L3", "tube_R1", "tube_R2", "tube_R3",
    "well_plate_A", "well_plate_B", "tip_box",
    "heater_shaker", "vortex_genie", "pcr_module", "opentrons_ot2",
]


def _as_scene(scene) -> SceneGeometry:
    """Accept a SceneGeometry, a path to a scene, or None (meaning the default)."""
    if isinstance(scene, SceneGeometry):
        return scene
    return load(scene or DEFAULT_SCENE)


def render_geometry_section(scene=None) -> str:
    """Render the authoritative geometry block injected into the system prompt.

    This replaces coordinates that used to be typed into the prompt by hand. Commit
    `fde3d22` moved most of the bench and those hand-typed copies did not follow,
    leaving the planner reasoning about a layout that no longer existed.
    """
    scene = _as_scene(scene)

    movable, static, missing = [], [], []
    for name in PROMPT_BODIES:
        info = scene.get(name)
        if info is None:
            missing.append(name)
            continue
        x, y, z = info.pos_mm
        row = f"  {name:<17} ({x:>6.0f}, {y:>6.0f}, {z:>5.0f})"
        (movable if info.pushable else static).append(row)

    lines = [
        f"GENERATED from {scene.scene_xml} — do not hand-edit these numbers.",
        "All coordinates are world-frame millimetres; xy is the body centre.",
        "",
        "Movable objects (valid `push_object` targets, and graspable):",
        *movable,
    ]
    if static:
        lines += [
            "",
            "Static fixtures — `push_object` on these FAILS (no free joint).",
            "Plan around them; they are obstacles, not cargo:",
            *static,
        ]

    slots = scene.deck_slots()
    if slots:
        deck_z = next(iter(slots.values()))[2]
        lines += ["", f"OT-2 deck slots (deck top z={deck_z:.0f}, SBS 127x85mm):"]
        for label, (x, y, _z) in slots.items():
            lines.append(f"  slot {label:<5} ({x:>6.0f}, {y:>6.0f})")

    if missing:
        lines += ["", f"NOTE: expected but absent from the scene: {', '.join(missing)}"]

    return "\n".join(lines)


def render_object_names(scene=None) -> str:
    """Render the movable-object vocabulary for the dynamic grader's prompt.

    Grouped by a name-pattern guess at type, which is crude but self-updating —
    the previous hardcoded list still advertised `red_bin`, deleted from the
    scene long before, and a grader told to expect an object that cannot exist
    will happily produce success criteria that can never be met.
    """
    scene = _as_scene(scene)

    # Containers come from BIN_BODIES, not from a name pattern. `clear_cup` is a
    # bin with no "bin" in its name, and mis-filing it as generic movable would
    # tell the grader that "<cube> in clear_cup" is not a success shape — which
    # it is, and which physical_outcome() does emit.
    try:
        from sim.mujoco_env import BIN_BODIES  # local import: avoids a cycle
        bins = set(BIN_BODIES)
    except Exception:  # noqa: BLE001 - fall back to the name heuristic
        bins = set()

    groups: dict[str, list[str]] = {
        "Cubes": [], "Bins": [], "Tube racks": [], "Falcon tubes": [],
        "Plates / tip racks": [], "Other movable": [],
    }
    for name in scene.pushable_bodies():
        low = name.lower()
        if name in bins or "bin" in low:
            groups["Bins"].append(name)
        elif "cube" in low:
            groups["Cubes"].append(name)
        elif "rack" in low and "tube" in low:
            groups["Tube racks"].append(name)
        elif low.startswith("tube_"):
            groups["Falcon tubes"].append(name)
        elif "plate" in low or "tip" in low:
            groups["Plates / tip racks"].append(name)
        else:
            groups["Other movable"].append(name)

    lines = []
    for label, names in groups.items():
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return "\n".join(lines)


def worked_example_coords(scene=None) -> dict[str, float]:
    """Coordinates for the prompt's worked example, taken from the live scene.

    The example is the strongest pattern the planner imitates, so a stale
    coordinate here is more damaging than a stale one in a reference table.
    """
    scene = _as_scene(scene)
    cube = scene.get("green_cube")
    binb = scene.get("green_bin")
    if cube is None or binb is None:
        raise ValueError("worked example needs green_cube and green_bin in the scene")
    return {
        "ex_cube_x": round(cube.pos_mm[0]),
        "ex_cube_y": round(cube.pos_mm[1]),
        "ex_bin_x": round(binb.pos_mm[0]),
        "ex_bin_y": round(binb.pos_mm[1]),
    }


@lru_cache(maxsize=4)
def load(scene_xml: str = DEFAULT_SCENE) -> SceneGeometry:
    """Cached loader — the checks query the same scene many times per run."""
    return SceneGeometry(scene_xml)
