# teleop_hand/follow.py
"""Fist-follow: the end effector tracks a fisted hand at a standoff.

Pure: no MuJoCo, no sockets, no clock. The receiver gives it the fist in the
world frame and where the EE is; it returns where the EE should be.

    target = fist + d * unit(EE - fist)        d = clip(|EE - fist|, lo, hi)

So the EE approaches along the line it is already on (it never swings round
the hand), stops at `hi` from the fist, backs off if the hand comes closer than
`lo`, and does not move radially while it is inside the band -- the 100-150 mm
band is what keeps a hand that is merely sitting still from being chased by a
tool that is trying to hit exactly 125 mm.

The floor and workspace clamps come AFTER the standoff, so a fist held low
could push the target inside the minimum distance. The last step moves the
target away from the fist horizontally until the distance is restored: keeping
off the hand outranks keeping off the floor, and the floor is already 120 mm
above the bench.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from teleop_hand import config
from teleop_hand.filters import OneEuro
from vr import transforms as vr_transforms


class FistFollower:
    def __init__(self, standoff_mm=config.FOLLOW_STANDOFF_MM,
                 z_min_mm: float = config.FOLLOW_Z_MIN_MM):
        self.lo, self.hi = (float(v) / 1000.0 for v in standoff_mm)
        self.z_min_m = float(z_min_mm) / 1000.0
        self.filter = OneEuro(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                              config.ONE_EURO_D_CUTOFF)
        self.reset()

    def reset(self) -> None:
        self.filter.reset()
        self._pending: Optional[np.ndarray] = None
        self._pending_n = 0
        self.fist_m: Optional[np.ndarray] = None
        self.n_jumps = 0
        self.status: dict = {}

    # ---- input gating -----------------------------------------------------
    def _accept(self, p_m: np.ndarray) -> bool:
        """One-frame jumps are depth outliers (a fist's edge picks up the
        background); a jump that FOLLOW_JUMP_ACCEPT_FRAMES frames agree on is
        the hand moving."""
        if self.fist_m is None:
            return True
        if np.linalg.norm(p_m - self.fist_m) * 1000.0 < config.FOLLOW_JUMP_MM:
            self._pending, self._pending_n = None, 0
            return True
        if self._pending is not None and \
                np.linalg.norm(p_m - self._pending) * 1000.0 < config.FOLLOW_JUMP_MM:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = p_m.copy(), 1
        if self._pending_n >= config.FOLLOW_JUMP_ACCEPT_FRAMES:
            self._pending, self._pending_n = None, 0
            return True
        self.n_jumps += 1
        return False

    def update(self, fist_world_m, ee_m, dt: float) -> Optional[np.ndarray]:
        """New fist sample -> target EE position (m), or None if the sample was
        rejected as an outlier (the caller keeps the previous target)."""
        p = np.asarray(fist_world_m, float)
        if not self._accept(p):
            return None
        self.fist_m = self.filter(p * 1000.0, dt) / 1000.0
        return self.target(ee_m)

    def target(self, ee_m) -> np.ndarray:
        fist = self.fist_m
        ee = np.asarray(ee_m, float)
        v = ee - fist
        dist = float(np.linalg.norm(v))
        u = v / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0])
        d = float(np.clip(dist, self.lo, self.hi))
        tgt = fist + d * u

        clamped = False
        t_mm = vr_transforms.clamp_workspace_mm(tgt * 1000.0)
        t_mm[2] = max(t_mm[2], self.z_min_m * 1000.0)
        tgt2 = t_mm / 1000.0
        clamped = bool(np.linalg.norm(tgt2 - tgt) > 1e-6)
        tgt = tgt2

        # Restore the minimum distance to the hand if the clamps ate it.
        dv = tgt - fist
        if np.linalg.norm(dv) < self.lo - 1e-6:
            h = dv[:2]
            hn = np.linalg.norm(h)
            uxy = h / hn if hn > 1e-6 else (u[:2] / max(np.linalg.norm(u[:2]), 1e-6))
            need = np.sqrt(max(self.lo ** 2 - dv[2] ** 2, 0.0))
            tgt = np.array([fist[0] + uxy[0] * need, fist[1] + uxy[1] * need, tgt[2]])
            clamped = True
        self.status = {"dist_mm": round(dist * 1000, 1), "standoff_mm": round(d * 1000, 1),
                       "clamped": clamped}
        return tgt
