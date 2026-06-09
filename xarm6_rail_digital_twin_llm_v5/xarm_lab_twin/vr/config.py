# vr/config.py
"""Tunables for VR teleoperation.

Everything the operator might want to adjust (IPD, world scale, where the
twin workspace lands relative to the standing origin, loop rates, ports,
the safety workspace box, and which servo path to drive the arm with) lives
here as a module-level constant rather than scattered as literals across
the rest of vr/. scripts/run_vr.py mutates a few of these from CLI flags
before the server starts (PORT, MODE, SERVO_MODE, WORLD_SCALE, RECORD).
"""
from __future__ import annotations

import numpy as np

# ---- display / rendering --------------------------------------------------
# "mono"  -> single flat panel (also viewable in a browser tab)
# "stereo"-> per-eye cameras offset by IPD, head-tracked, true depth.
MODE: str = "mono"

# Inter-pupillary distance, metres. Drives the cam_left/cam_right offset in
# the scene; the scene ships with +/-0.0315 m baked in, but the receiver can
# re-derive eye offsets from this if you move the cameras to code-driven
# poses later. 63 mm is the human average.
IPD_M: float = 0.063

# Per-eye offscreen frame size (pixels). 640x480 is a good LAN/JPEG balance;
# bump for clarity at the cost of bandwidth/latency.
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
JPEG_QUALITY: int = 80

# Offscreen render loop rate (Hz). The sim stepper is independent; this only
# caps how often we encode + push frames. Start at 30.
RENDER_HZ: float = 30.0

# ---- control --------------------------------------------------------------
# Teleop control-update rate (Hz). Each tick reads the latest controller
# state and (if the clutch is held) servos the arm toward it.
CONTROL_HZ: float = 60.0

# Servo path the clutch uses to drive the arm:
#   "direct"    -> solve IK once per tick and write joint ctrl directly
#                  (smooth, low-latency, bypasses the collision validator —
#                   fine in sim for continuous servoing).
#   "validated" -> call arm.set_position(...) per tick, reusing the existing
#                  IK + FKValidator + pacing path (safer, slightly jerkier).
SERVO_MODE: str = "direct"

# Speed (mm/s) passed to arm.set_position when SERVO_MODE == "validated".
VALIDATED_SERVO_SPEED_MM_S: float = 400.0

# Exponential smoothing factor applied to the clutch target before IK, to
# damp controller jitter. alpha in (0, 1]; lower = smoother but laggier.
SMOOTH_ALPHA: float = 0.3

# ---- coordinate mapping (XR world -> MuJoCo twin) -------------------------
# Uniform scale on the XR->twin position mapping. 1.0 = 1:1; shrink (<1) to
# bring the whole bench within easy arm's reach of a standing operator.
WORLD_SCALE: float = 1.0

# After the basis change + scale, add this (twin-frame metres) so the
# operator's comfortable hand origin lands near the bench workspace. The
# bench top is ~z=0.76 and the workspace sits in front of the arm.
TWIN_ORIGIN_OFFSET_M: np.ndarray = np.array([0.0, 0.30, 0.95], dtype=float)

# Twin-frame head viewpoint at session start (metres), matching the scene's
# vr_head default pos. Head tracking is *relative* to the first head sample,
# so the operator's absolute standing height never shoves the camera away
# from this comfortable bench-facing viewpoint.
VR_HEAD_BASE_M: np.ndarray = np.array([0.0, -0.6, 1.4], dtype=float)

# ---- safety workspace clamp (twin frame, mm) ------------------------------
# The smoothed clutch target is clamped into this AABB before IK so a wild
# hand motion can't fling an IK target to infinity. Failure is free in sim,
# but clamping keeps the demo smooth. (x, y, z) min/max in mm.
WORKSPACE_AABB_MM: dict = {
    "x": (-700.0, 700.0),
    "y": (-200.0, 900.0),
    "z": (650.0, 1300.0),
}

# ---- rail -----------------------------------------------------------------
RAIL_MIN_MM: float = 0.0
RAIL_MAX_MM: float = 700.0
# How fast the left thumbstick jogs the rail, mm per second at full deflection.
RAIL_JOG_MM_PER_S: float = 300.0
# Thumbstick deadzone (|axis| below this is treated as zero).
STICK_DEADZONE: float = 0.15

# ---- gripper default pose -------------------------------------------------
# When the clutch is engaged we honour the controller's orientation, but a
# downward-pointing controller maps to roll≈180 (gripper down). No config
# needed; documented here for reference.

# ---- server / transport ---------------------------------------------------
HOST: str = "0.0.0.0"
PORT: int = 8443
# Throttle for inbound controller-state messages on the server side (Hz).
# The client also throttles; this is a backstop.
MAX_INPUT_HZ: float = 72.0
# How often the server pushes a HUD status message to the client (Hz).
STATUS_HZ: float = 5.0

# ---- recording ------------------------------------------------------------
# The A button starts/stops takes; this only controls whether the Recorder
# is constructed at all (run_vr.py --no-record sets it False).
RECORD: bool = True
