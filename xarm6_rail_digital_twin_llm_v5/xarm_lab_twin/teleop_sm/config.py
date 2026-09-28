# teleop_sm/config.py
"""Tunables for SpaceMouse teleoperation.

Same role `vr/config.py` plays for the WebXR path: everything an operator
might want to adjust lives here as a module-level constant rather than as a
literal scattered through the package. `scripts/run_spacemouse.py` mutates a
few of these from CLI flags before the control loop starts.

THESE WANT TUNING ON HARDWARE. The speed caps in particular are first guesses
-- how fast "full deflection" should drive the arm is a matter of taste and of
how stiff the operator holds the cap. Expect to change them; that is why they
are here and exposed as `--pos-speed` / `--rot-speed` overrides.
"""
from __future__ import annotations

import numpy as np

# ---- device ---------------------------------------------------------------
#: spnav raw full-scale. The protocol carries signed counts; a firm push
#: reaches roughly this, and the daemon clamps around +/-350. Axis values are
#: divided by this to land in about [-1, 1] -- "about", because a hard shove
#: can exceed it, so consumers must clip rather than assume the range.
MAX_VALUE: float = 300.0

#: Per-axis deadzone as a fraction of full scale, applied AFTER normalisation.
#: The puck rarely returns to exactly zero, and an integrating jog turns any
#: standing offset into continuous unwanted drift -- so this is not cosmetic,
#: it is what stops the arm walking away while the operator's hand rests.
DEADZONE: float = 0.05

#: Control-loop rate (Hz). VR runs at 60; 50 is plenty for a velocity jog and
#: leaves headroom on the viewer thread.
FREQUENCY: float = 50.0

# ---- speed caps -----------------------------------------------------------
#: Translation speed at full deflection, mm/s.
MAX_POS_SPEED: float = 150.0

#: Rotation speed at full deflection, deg/s.
MAX_ROT_SPEED: float = 45.0

#: Rail speed at full deflection in RAIL mode, mm/s.
MAX_RAIL_SPEED: float = 200.0

#: Joint speed at full deflection in JOINT mode, deg/s. Deliberately slow:
#: joint jogging bypasses IK AND the workspace clamp, so the only guard left
#: is the operator's eyes and the model's own joint limits.
MAX_JOINT_SPEED: float = 20.0

#: Multiplier on every speed cap when FINE is toggled on. One fixed speed is
#: either too fast to place a plate or too slow to cross the bench; the
#: operator feels that within a minute of driving.
FINE_SPEED_FACTOR: float = 0.25

#: A dual-role key (Compact button 1) counts as a TAP if released within this
#: many seconds, otherwise it is a HOLD. Needed because the press tick sees
#: the edge and the held state together.
TAP_MAX_S: float = 0.35

# ---- filtering ------------------------------------------------------------
#: Exponential smoothing on the integrated target before IK. Matches
#: `vr.config.SMOOTH_ALPHA` deliberately: same filter, same feel, one number
#: to reason about across both teleop paths.
SMOOTH_ALPHA: float = 0.3

# ---- servo ----------------------------------------------------------------
#: "direct"    -> solve IK once per tick and write joint ctrl (smooth, skips
#:                the collision validator -- fine for sim servoing).
#: "validated" -> arm.set_position() per tick, through IK + FKValidator.
SERVO_MODE: str = "direct"

#: Speed passed to arm.set_position when SERVO_MODE == "validated".
VALIDATED_SERVO_SPEED_MM_S: float = 400.0

# ---- workspace ------------------------------------------------------------
#: The integrated target is clamped into this box before IK. Shared with the
#: VR path on purpose -- `vr.transforms.clamp_workspace_mm` reads
#: `vr.config.WORKSPACE_AABB_MM`, and this module does NOT define its own
#: copy. Two boxes that start equal and drift apart is defect class #1.
#: Imported here only so the banner can print it.
from vr.config import WORKSPACE_AABB_MM  # noqa: E402,F401

#: Rail travel limits, mm. Same reasoning: re-exported from the VR config
#: rather than duplicated.
from vr.config import RAIL_MAX_MM, RAIL_MIN_MM  # noqa: E402,F401

# ---- axis mapping ---------------------------------------------------------
#: Permutation from the device's axis frame into the twin world frame.
#:
#: MEASURED 2026-09-28 on the SpaceMouse Pro, not inherited. The probe logged
#: six deliberate pushes; the dominant raw axis of each:
#:
#:     cap forward / back  ->  raw tz   (+ forward)
#:     cap right  / left   ->  raw tx   (+ right)
#:     cap up     / down   ->  raw ty   (+ up)
#:
#: so the device frame is X=right, Y=up, Z=forward. Rotations share those
#: axes (a forward push also tilts, which is why rx rides along with tz).
#:
#: The twin frame was anchored the same way rather than assumed: driving the
#: rail +300 mm moved the EE +291 mm in world X, and the workspace box puts Z
#: up. So twin X = along the rail, Y = across the bench, Z = up.
#:
#: Mapping (the work order requires cap-forward -> twin +X):
#:     twin x = raw tz     twin y = raw tx     twin z = raw ty
#:
#: THE VENDOR MATRIX WAS `[[0,0,-1],[1,0,0],[0,1,0]]` -- right except for the
#: SIGN of the tz->x term, which would have driven the arm BACKWARDS along
#: the rail. One sign, invisible without measuring, exactly as predicted.
AXIS_PERMUTATION: np.ndarray = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
], dtype=float)

AXIS_SIGNS: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

# ---- recording ------------------------------------------------------------
#: Whether a Recorder is constructed at all (--no-record sets this False).
RECORD: bool = True

#: Task label written into the session metadata.
TASK_LABEL: str = "spacemouse_teleop"
