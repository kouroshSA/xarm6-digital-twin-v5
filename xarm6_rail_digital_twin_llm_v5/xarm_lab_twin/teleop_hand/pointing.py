# teleop_hand/pointing.py
"""From a pointing ray in the world frame to something the arm can be sent to.

    ray  --resolve_pointing-->  PointTarget(kind, name, position_mm, fallbacks)

kind
  grasp   a graspable object lies inside `POINT_CONE_DEG` of the ray. The
          position is a PRIOR (a few mm in the twin, whatever the detector
          gives on hardware); the wrist camera and GG-CNN make the real grasp.
  hover   the ray hits something that is not graspable (bench, instrument,
          bin). The target stops `POINT_STANDOFF_MM` short of the surface,
          along the ray, and never below the floor clearance.
  free    the ray hits nothing in the scene. The target is the farthest point
          along the ray that is inside the workspace box, above the floor
          clearance and within reach of the arm base.

A finger ray is good to a few degrees, so "pointed at" is a CONE, not a line:
at 0.6 m, 8 degrees is 84 mm. That is why the cone test runs before the ray
cast -- a ray that grazes a cube's shoulder would otherwise report the bench
behind it.

Two scene implementations share one interface (`candidates()`, `cast()`,
`bench_top_m`): `TwinScene` (MuJoCo ray caster, the twin's real geometry) and
`SphereScene` (spheres + a plane; the unit tests, and a stand-in for hardware
until the DetectionService feeds object poses).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np

from teleop_hand import config
from vr import transforms as vr_transforms


@dataclasses.dataclass
class RayHit:
    point_m: np.ndarray
    distance_m: float
    name: Optional[str]            # body name, or "bench"


@dataclasses.dataclass
class PointTarget:
    kind: str                      # grasp | hover | free
    name: Optional[str]
    position_mm: np.ndarray        # the primary target, world mm
    angle_deg: Optional[float] = None
    fallbacks: List[np.ndarray] = dataclasses.field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        p = np.round(self.position_mm, 0).astype(int).tolist()
        return f"{self.kind}{'' if not self.name else ' ' + self.name} at {p} mm  {self.note}"


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
class SphereScene:
    """Named spheres (graspable) + other spheres (obstacles) + the bench plane."""

    def __init__(self, graspable: Dict[str, Tuple], bench_top_m: float = 0.75,
                 obstacles: Optional[Dict[str, Tuple]] = None):
        self.graspable = {k: (np.asarray(c, float), float(r)) for k, (c, r) in graspable.items()}
        self.obstacles = {k: (np.asarray(c, float), float(r))
                          for k, (c, r) in (obstacles or {}).items()}
        self.bench_top_m = float(bench_top_m)

    def candidates(self) -> Dict[str, np.ndarray]:
        return {k: c.copy() for k, (c, _) in self.graspable.items()}

    def cast(self, origin_m, direction) -> Optional[RayHit]:
        o, d = np.asarray(origin_m, float), np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        best: Optional[RayHit] = None

        def consider(t, name):
            nonlocal best
            if t > 1e-6 and (best is None or t < best.distance_m):
                best = RayHit(o + t * d, float(t), name)

        for name, (c, r) in {**self.graspable, **self.obstacles}.items():
            oc = o - c
            b = float(np.dot(oc, d))
            disc = b * b - (float(np.dot(oc, oc)) - r * r)
            if disc >= 0:
                consider(-b - np.sqrt(disc), name)
        if d[2] < -1e-9:
            consider((self.bench_top_m - o[2]) / d[2], "bench")
        return best


class TwinScene:
    """The twin's own geometry, via MuJoCo's ray caster.

    The arm is skipped (a ray cast from the operator's side would otherwise
    often stop on a link), and so are the hidden geom groups, through
    `sim.render_options.ray_geomgroup` -- the same mask the camera depth uses.
    """

    MAX_SKIPS = 12

    def __init__(self, arm):
        import mujoco
        from agent.task_validator import _bench_top_mm
        from sim.mujoco_env import GRIPPABLE_BODIES
        from sim.render_options import ray_geomgroup
        self._mj = mujoco
        self.arm = arm
        self.graspable = list(GRIPPABLE_BODIES)
        self.bench_top_m = _bench_top_mm(arm) / 1000.0
        self._group = ray_geomgroup()
        m = arm.model
        root = m.body("xarm_base").id
        self._arm_bodies = set()
        for b in range(m.nbody):
            c = b
            while c > 0 and c != root:
                c = int(m.body_parentid[c])
            if c == root:
                self._arm_bodies.add(b)

    def candidates(self) -> Dict[str, np.ndarray]:
        with self.arm.lock:
            return {n: np.array(self.arm.data.body(n).xpos, float) for n in self.graspable}

    def cast(self, origin_m, direction) -> Optional[RayHit]:
        m, d = self.arm.model, self.arm.data
        o = np.asarray(origin_m, float).copy()
        v = np.asarray(direction, float)
        v = v / np.linalg.norm(v)
        travelled = 0.0
        for _ in range(self.MAX_SKIPS):
            gid = np.zeros(1, np.int32)
            with self.arm.lock:
                dist = self._mj.mj_ray(m, d, o, v, self._group, 1, -1, gid)
            if dist < 0:
                return None
            travelled += dist
            body = int(m.geom_bodyid[gid[0]])
            if body in self._arm_bodies:
                o = o + v * (dist + 1e-3)
                travelled += 1e-3
                continue
            name = m.body(body).name or "bench"
            if m.geom(int(gid[0])).name == "bench_top":
                name = "bench"
            return RayHit(np.asarray(origin_m, float) + v * travelled, travelled, name)
        return None


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def _cone_pick(origin, d, candidates: Dict[str, np.ndarray], cone_deg: float):
    best = None
    for name, c in candidates.items():
        v = c - origin
        t = float(np.dot(v, d))
        if t < 0.05:
            continue
        perp = float(np.linalg.norm(v - t * d))
        ang = float(np.degrees(np.arctan2(perp, t)))
        if ang < cone_deg and (best is None or ang < best[1]):
            best = (name, ang)
    return best


def _reach_ok(p_mm: np.ndarray, base_mm: np.ndarray, floor_mm: float) -> bool:
    inside = np.allclose(vr_transforms.clamp_workspace_mm(p_mm), p_mm, atol=1e-6)
    horiz = float(np.linalg.norm((p_mm - base_mm)[:2]))
    return bool(inside and p_mm[2] >= floor_mm - 1e-6 and horiz <= config.POINT_REACH_MM)


def resolve_pointing(origin_m, direction, scene, base_mm,
                     cone_deg: float = config.POINT_CONE_DEG) -> Optional[PointTarget]:
    """`origin_m` metres and `direction` unit vector, world frame; `base_mm`
    the arm base (it rides the rail). None when nothing safe lies along the ray."""
    o = np.asarray(origin_m, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    base_mm = np.asarray(base_mm, float)
    floor_mm = scene.bench_top_m * 1000.0 + config.POINT_FLOOR_CLEARANCE_MM

    pick = _cone_pick(o, d, scene.candidates(), cone_deg)
    if pick is not None:
        name, ang = pick
        return PointTarget("grasp", name, scene.candidates()[name] * 1000.0, ang,
                           note=f"({ang:.1f} deg off the ray)")

    hit = scene.cast(o, d)
    t_max = config.POINT_MAX_RAY_M
    hit_note = "ray hits nothing"
    if hit is not None:
        t_max = max(hit.distance_m - config.POINT_STANDOFF_MM / 1000.0, 0.0)
        hit_note = f"ray hits {hit.name} at {hit.distance_m:.2f} m"

    # Walk the ray from far to near; the first point that is safe and reachable
    # is the target, the rest are fallbacks for a move the arm refuses.
    ok: List[np.ndarray] = []
    for t in np.linspace(t_max, 0.0, 40):
        p = (o + t * d) * 1000.0
        if _reach_ok(p, base_mm, floor_mm):
            ok.append(p)
        if len(ok) >= 6 and t < t_max * 0.5:
            break
    if not ok:
        return None
    # Spread the fallbacks out instead of taking six neighbours.
    primary, rest = ok[0], ok[1::2][:4]
    return PointTarget("hover" if hit is not None else "free",
                       None if hit is None else hit.name, primary,
                       fallbacks=rest, note=hit_note)


# ---------------------------------------------------------------------------
# dwell: a ray commits only once it has held still
# ---------------------------------------------------------------------------
class PointDwell:
    """Commit a pointing ray after `dwell_s` of holding within `steady_deg` of
    its running mean. Fires once per pointing gesture; the gesture has to end
    before it can fire again."""

    def __init__(self, dwell_s: float = config.POINT_DWELL_S,
                 steady_deg: float = config.POINT_STEADY_DEG):
        self.dwell_s, self.steady_deg = dwell_s, steady_deg
        self.reset()

    def reset(self) -> None:
        self._t0: Optional[float] = None
        self._o: Optional[np.ndarray] = None
        self._d: Optional[np.ndarray] = None
        self._n = 0
        self.fired = False
        self.progress = 0.0

    def update(self, origin_m, direction, now: float):
        """Returns (origin, direction) when it commits, else None."""
        if self.fired:
            return None
        o, d = np.asarray(origin_m, float), np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        if self._d is not None:
            ang = np.degrees(np.arccos(np.clip(float(np.dot(d, self._d)), -1.0, 1.0)))
            if ang > self.steady_deg:
                self._t0 = None
        if self._t0 is None:
            self._t0, self._o, self._d, self._n = now, o.copy(), d.copy(), 1
        else:
            self._n += 1
            a = 1.0 / self._n
            self._o = (1 - a) * self._o + a * o
            m = (1 - a) * self._d + a * d
            self._d = m / np.linalg.norm(m)
        self.progress = min(1.0, (now - self._t0) / self.dwell_s)
        if now - self._t0 >= self.dwell_s:
            self.fired = True
            return self._o.copy(), self._d.copy()
        return None
