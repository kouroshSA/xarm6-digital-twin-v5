"""Where the MaxArm actually stops, relative to where it was sent.

MEASURED on the lab unit (2026-11-07/09, `hardware/maxarm/README.md`, "Open,
needs the device" and the arrival-tolerance entry), kept here as the ONE place
the digital twin reads them from. The real backend only needs the tolerance
(`real_maxarm.ARRIVAL_TOL_MM`); the twin needs the error itself, so that a
sequence rehearsed in the twin meets the same misses the bench produces.

The numbers (servo read-back against the commanded wrist point, mm):

* unloaded, stepping down 2 mm at a time near the table: 0.4-1.5
* unloaded, after larger moves: 3.3-3.5
* unloaded, back at a high view pose (z 200, reach ~220): 6.0, all of it z SHORT
* holding a 40 mm wooden cube at z 150: 6.4 (z 5.8 short), still after a
  re-send -- steady gravity sag, not a miss
* the hand-eye calibration's residual, RMS 2.1 mm / max 3.7 mm, is "the size of
  the servo read-back error" (41 views)
* bus-servo pulses are integers: 0.24 deg per count, ~1 mm at full reach

The model below is a FIT to those points, not a measurement of its own:
a gravity sag in -z that grows with height and reach (0 near the table, the
measured 6 mm at the high view pose, more with a load), plus an isotropic
Gaussian scatter whose 2-sigma radius (~3.3 mm) matches the spread seen after
larger moves. Re-fit it when the bench gives more points; do not tune it to
make the twin succeed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Per-axis standard deviation of the scatter, mm. 1.2 per axis gives a
#: median miss of ~1.8 mm and a 95th percentile of ~3.3 mm: the 0.4-3.5 mm
#: band above.
SCATTER_SIGMA_MM = 1.2

#: Gravity sag in -z at the measured high view pose (z 200, reach ~223), mm.
SAG_Z_MM_AT_VIEW = 6.0
#: The pose the 6.0 mm was measured at: wrist z and horizontal reach, mm.
SAG_REF_Z_MM = 200.0
SAG_REF_REACH_MM = 223.0
#: Below this wrist height the sag is taken as zero (0.4-1.5 mm total error
#: was seen stepping down near the table).
SAG_ZERO_Z_MM = 60.0
#: Extra sag while something is held, mm: 5.8 short at z 150 against the
#: unloaded fit's 3.9 at that height.
SAG_LOAD_EXTRA_MM = 1.5

#: Bus-servo resolution: 1000 counts over 240 degrees.
PULSE_DEG = 240.0 / 1000.0


def sag_z_mm(z_mm: float, reach_mm: float, loaded: bool = False) -> float:
    """How far short in z the wrist settles at this height and reach, mm (>= 0)."""
    h = min(max((z_mm - SAG_ZERO_Z_MM) / (SAG_REF_Z_MM - SAG_ZERO_Z_MM), 0.0), 1.0)
    s = SAG_Z_MM_AT_VIEW * h * (reach_mm / SAG_REF_REACH_MM)
    if loaded:
        s += SAG_LOAD_EXTRA_MM * h
    return s


@dataclass
class Repeatability:
    """Perturbs a commanded wrist point the way the bench does. Seeded, so a
    rehearsal can be replayed."""
    seed: int = 0
    scatter_sigma_mm: float = SCATTER_SIGMA_MM
    sag: bool = True

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def settle(self, target_xyz_mm, loaded: bool = False):
        """Where the arm will actually stop for this command (arm frame, mm)."""
        x, y, z = (float(v) for v in target_xyz_mm)
        e = self.rng.normal(0.0, self.scatter_sigma_mm, size=3) if self.scatter_sigma_mm > 0 else np.zeros(3)
        dz = -sag_z_mm(z, math.hypot(x, y), loaded) if self.sag else 0.0
        return (x + e[0], y + e[1], z + e[2] + dz)


def exact() -> Repeatability:
    """A perfect arm: no scatter, no sag (for grading the twin's kinematics)."""
    return Repeatability(seed=0, scatter_sigma_mm=0.0, sag=False)
