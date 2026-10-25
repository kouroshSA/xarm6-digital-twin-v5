# teleop_kin/config.py
"""Tunables for kinesthetic teaching (mouse-dragged teach handle, and the
real-arm manual-mode mirror).

Same role `teleop_sm/config.py` plays for the SpaceMouse: everything an
operator might want to adjust is a module-level constant here, and the runners
override a few from CLI flags before the loop starts.
"""
from __future__ import annotations

# ---- loop -----------------------------------------------------------------
#: Control-loop rate (Hz). Matches the Recorder's 60 Hz state rate so every
#: recorded sample sees a fresh setpoint.
FREQUENCY: float = 60.0

# ---- rate limits ----------------------------------------------------------
#: The commanded pose chases the handle at most this fast. A mouse flick can
#: move the handle half a metre in one frame; without a cap the arm would
#: teleport and the recorded actions would contain a step no policy should
#: learn.
MAX_POS_SPEED: float = 150.0      # mm/s
MAX_ROT_SPEED: float = 90.0       # deg/s

#: Below these, the commanded pose has not moved enough to be worth an IK
#: solve. Keeps a resting handle from re-solving (and re-writing ctrl with
#: solver noise) 60 times a second.
SOLVE_MIN_POS_MM: float = 0.5
SOLVE_MIN_ROT_DEG: float = 0.5

#: Reject an IK solution that moves any joint further than this from its
#: current setpoint in one solve: that is a branch flip, not a servo step.
#: At the speed caps above a real step is ~1-2 deg away from singularities;
#: 5 deg per 1/60 s solve is a 300 deg/s ceiling on any joint.
MAX_JOINT_STEP_DEG: float = 5.0

# ---- handle snapping ------------------------------------------------------
#: After the operator lets go, if the handle sits further than this from the
#: actual EE (and the arm has finished chasing it), it is snapped onto the EE,
#: so the next drag always starts where the arm really is.
SNAP_POS_MM: float = 10.0
SNAP_ROT_DEG: float = 5.0
#: ...but only after the arm has been settled this long, so a release while
#: the arm is still catching up does not cut the motion short.
SNAP_GRACE_S: float = 0.3

#: A clamped target differs from the handle by more than this -> the arm is
#: NOT following the handle, and the handle turns red.
CLAMP_TOLERANCE_MM: float = 1.0

#: Handle colours (the ball geom's rgba).
RGBA_TRACKING = (1.0, 0.55, 0.0, 0.6)    # orange: the arm follows
RGBA_BLOCKED = (1.0, 0.0, 0.0, 0.8)      # red: unreachable / outside workspace

#: A FAILED IK solve is expensive -- measured 215 ms, all of it under
#: arm.lock -- so retrying every tick at the reach boundary starves the
#: physics thread (the arm stalled 10-30 deg short of its own setpoint) and
#: drops the loop to ~4 Hz. After a failure, retry only once the handle has
#: moved, and no more often than this.
IK_RETRY_PERIOD_S: float = 0.25

#: Warn about IK failure at most this often, seconds.
IK_WARN_PERIOD_S: float = 1.0

# ---- gravity-sag compensation ---------------------------------------------
#: The twin's joints are bare PD position actuators with no gravity
#: compensation, so the arm settles BELOW its setpoint -- ~1.2 deg at joint 2,
#: ~15 mm at the tool in a typical bench pose (measured; IK itself is exact to
#: 0.07 mm). Every interface inherits that sag. Teach mode is the one place it
#: is intolerable: the tool must end up where the hand put it. So a slow
#: per-joint integral term rides on top of the IK setpoint, like the integral
#: action of the real controller. ctrl (the exported action) is the
#: compensated setpoint -- what the joints were actually told.
GRAVITY_COMP: bool = True
GRAVITY_COMP_KI: float = 6.0          # 1/s; ~0.17 s time constant
GRAVITY_COMP_MAX_DEG: float = 8.0     # clamp on the bias, per joint
#: Only integrate while the joint is within this of its setpoint AND nearly
#: at rest. Sag is a steady-state error; during a drag the error is mostly
#: tracking lag, and integrating that wound the bias up to its clamp (measured:
#: +3 deg mid-drag against a true sag of -1.2 deg) and then had to unwind.
GRAVITY_COMP_WINDOW_DEG: float = 8.0
GRAVITY_COMP_MAX_VEL_DEG_S: float = 3.0

# ---- rail -----------------------------------------------------------------
#: Left/Right arrow step, mm, and the speed the step is paced at.
RAIL_STEP_MM: float = 25.0
MAX_RAIL_SPEED: float = 200.0     # mm/s

#: Rail travel limits, re-exported from the VR config rather than duplicated
#: (two copies that drift is defect class #1).
from vr.config import RAIL_MAX_MM, RAIL_MIN_MM  # noqa: E402,F401

# ---- servo ----------------------------------------------------------------
#: "direct"    -> IK once per tick, joint setpoints straight into ctrl.
#: "validated" -> arm.set_position() per tick (IK + FKValidator + pacing);
#:                paces at wall-clock speed, so it feels sluggish to drag.
SERVO_MODE: str = "direct"
VALIDATED_SERVO_SPEED_MM_S: float = 400.0

# ---- keys (GLFW keycodes, as MuJoCo's viewer hands them over) --------------
#: Verified against MuJoCo 3.8.1 simulate.cc: in the PASSIVE viewer none of
#: these has a built-in binding (Left/Right step physics only in the
#: standalone simulate app; Up/Down/Insert/Delete/Home/End are unbound).
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_INSERT = 260
KEY_DELETE = 261
KEY_HOME = 268
KEY_END = 269

# ---- real arm (run_kinesthetic_real.py) -----------------------------------
#: Seconds to wait for the controller to report mode 2 after set_mode(2)
#: before retrying with set_mode(2, 1) (the Lite 6 Studio variant).
TEACH_MODE_TIMEOUT_S: float = 0.5
#: Release debounce for the hold-to-teach key. X11 key autorepeat delivers a
#: held key as release/press pairs a few ms apart; without this the arm would
#: be switched in and out of manual mode ~30 times a second while held.
TEACH_RELEASE_DEBOUNCE_S: float = 0.12
#: How often to re-read the rail during a real take, seconds. The carriage is
#: not back-drivable by hand, so it cannot move during teaching; this only
#: catches a move made from the pendant.
RAIL_POLL_PERIOD_S: float = 1.0

# ---- recording ------------------------------------------------------------
RECORD: bool = True
TASK_LABEL_SIM: str = "kinesthetic_sim"
TASK_LABEL_REAL: str = "kinesthetic_real"
