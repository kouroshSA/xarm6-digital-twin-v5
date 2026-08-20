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


@lru_cache(maxsize=4)
def load(scene_xml: str = DEFAULT_SCENE) -> SceneGeometry:
    """Cached loader — the checks query the same scene many times per run."""
    return SceneGeometry(scene_xml)
