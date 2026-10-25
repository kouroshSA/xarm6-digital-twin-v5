# teleop_hand/mapping.py
"""`ArmTeleopMapper`: HandFrames + clutch -> commanded EE pose and rail (§4).

Pure: no MuJoCo, no sockets, no clock of its own. The receiver hands it the
newest frame, the clutch/precision keys, the pose the arm is COMMANDED to,
and where the arm base and rail are; it returns a rate-limited target. The
same mapper drives the twin and the real arm.

Coordinates: palm poses arrive in the CAMERA frame; everything that leaves is
in the robot WORLD frame (metres), the frame IK and `RealXArmAPI.set_position`
take. `R_c2r` (a preset) rotates one into the other.

THE ARM NEVER MOVES ON ITS OWN. Motion needs the clutch held AND a fresh
valid frame. A dropout (no valid frame for DROPOUT_S) or tracker silence
(SILENCE_S) releases the clutch, and re-engaging needs the key let go and
pressed again -- the hand reappearing is not enough.

THE TARGET NEVER JUMPS. Every discontinuity in the input re-anchors instead:
clutch engage, precision on/off, the end of a pinch freeze, rotation input
returning after a hold. Re-anchoring sets the hand reference to the current
palm and the arm reference to the current target, so the mapped target is
continuous by construction; the rate limit is a second line, not the first.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from teleop_hand import config
from teleop_hand.filters import OneEuro, RotLowPass
from teleop_hand.messages import HandFrame
from teleop_hand.retarget import PinchFreeze
from teleop_kin.receiver import _rot_angle_deg, rate_limit_pose
from vr import transforms


@dataclasses.dataclass
class AbsoluteCalib:
    """Two opposite corners of the operator's comfortable hand box, camera
    frame, metres (`--calibrate-absolute` records them)."""
    corner_a: np.ndarray
    corner_b: np.ndarray


@dataclasses.dataclass
class MapperOutput:
    engaged: bool
    target_pos: Optional[np.ndarray]     # rate-limited command, world m; None = hold
    target_rot: Optional[np.ndarray]
    rail_mm: Optional[float]             # new rail setpoint, None = unchanged
    opening: Optional[float]
    status: dict


def _yaw_about_z(r: np.ndarray) -> float:
    """Twist of rotation `r` about world z (swing-twist), radians."""
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return float(2.0 * np.arctan2(z, w))


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ArmTeleopMapper:
    def __init__(self, preset: str = config.PRESET, map_mode: str = config.MAP_MODE,
                 orient_mode: str = config.ORIENT_MODE, rail_mode: str = config.RAIL_MODE,
                 scale: float = config.SCALE, rot_scale: float = config.ROT_SCALE,
                 rail_axis=(1.0, 0.0, 0.0), abs_calib: Optional[AbsoluteCalib] = None,
                 pinch_freeze: bool = config.PINCH_FREEZE):
        if map_mode == "absolute" and abs_calib is None:
            raise ValueError("absolute mapping needs a calibration (--calibrate-absolute)")
        self.r_c2r = np.asarray(config.PRESETS[preset], float)
        self.map_mode, self.orient_mode, self.rail_mode = map_mode, orient_mode, rail_mode
        self.scale, self.rot_scale = float(scale), float(rot_scale)
        a = np.asarray(rail_axis, float)
        self.rail_axis = a / np.linalg.norm(a)
        self.abs_calib = abs_calib
        self.max_pos_speed = config.MAX_POS_SPEED
        self.max_rot_speed = config.MAX_ROT_SPEED

        self.pos_filter = OneEuro(config.ONE_EURO_MIN_CUTOFF, config.ONE_EURO_BETA,
                                  config.ONE_EURO_D_CUTOFF)
        self.rot_filter = RotLowPass(config.ROT_LPF_CUTOFF_HZ)
        self.pinch = PinchFreeze(enabled=pinch_freeze)

        #: Set by the receiver each tick: hold orientation (wrist singularity).
        self.rot_hold_external = False

        self.engaged = False
        self._need_release = False
        self._precision = False
        self._frozen = False
        self._was_frozen = False
        self._rot_latched = False
        self._rot_was_held = False
        self._last_seq: Optional[int] = None
        self._last_frame: Optional[HandFrame] = None
        self._last_frame_t: Optional[float] = None    # tracker clock
        self._last_msg_now: Optional[float] = None
        self._last_valid_now: Optional[float] = None
        self._ref_pos: Optional[np.ndarray] = None    # jump-rejection reference
        self._ref_rot: Optional[np.ndarray] = None
        self._raw_pos: Optional[np.ndarray] = None    # newest accepted, camera
        self._raw_rot: Optional[np.ndarray] = None
        self._palm_pos: Optional[np.ndarray] = None   # filtered, camera
        self._palm_rot: Optional[np.ndarray] = None
        self._opening: Optional[float] = None
        self.n_jumps = 0
        self.n_dropouts = 0

        self._cmd_pos = self._cmd_rot = None          # rate-limited output
        self._good_pos = self._good_rot = None        # last command the arm took
        self._tgt_pos = self._tgt_rot = None          # mapped, before limits
        self._h0_pos = self._h0_rot = None
        self._e0_pos = self._e0_rot = None
        self._abs_base_mm: Optional[np.ndarray] = None

    # ---- frames -----------------------------------------------------------
    def _ingest(self, frame: Optional[HandFrame], now: float) -> bool:
        """Take a new frame if there is one. Returns True if a frame was
        ACCEPTED this tick (valid and not a jump)."""
        if frame is None or frame.seq == self._last_seq:
            return False
        self._last_seq = frame.seq
        self._last_msg_now = now
        self._last_frame = frame
        dt = (frame.t_mono - self._last_frame_t) if self._last_frame_t is not None else 0.0
        self._last_frame_t = frame.t_mono
        if frame.opening is not None:
            self._opening = float(frame.opening)
        if not frame.valid:
            return False

        pos = frame.pos()
        rot = frame.rot()
        if self.engaged and self._ref_pos is not None:
            jump_mm = float(np.linalg.norm(pos - self._ref_pos)) * 1000.0
            jump_deg = (_rot_angle_deg(self._ref_rot, rot)
                        if rot is not None and self._ref_rot is not None else 0.0)
            if jump_mm > config.JUMP_POS_MM or jump_deg > config.JUMP_ROT_DEG:
                # An outlier counts as no frame at all: if the "jump" was
                # real and persists, the dropout timer releases the clutch.
                self.n_jumps += 1
                return False
        self._ref_pos = self._raw_pos = pos
        if rot is not None:
            self._ref_rot = self._raw_rot = rot
        self._palm_pos = self.pos_filter(pos * 1000.0, dt) / 1000.0
        if rot is not None:
            self._palm_rot = self.rot_filter(rot, dt)
        self._last_valid_now = now
        self._frozen = self.pinch.update(frame.opening, frame.t_mono) or False
        return True

    # ---- anchoring --------------------------------------------------------
    def _anchor(self, e_pos, e_rot) -> None:
        # Restart the filters ON the newest raw palm and anchor there. Anchoring
        # on the filtered value would leave its lag (~0.15 s of motion) still
        # draining into the target after the anchor -- measured as several mm
        # of drift after a pinch freeze.
        if self._raw_pos is not None:
            self.pos_filter.reset()
            self._palm_pos = self.pos_filter(self._raw_pos * 1000.0, 0.0) / 1000.0
        if self._raw_rot is not None:
            self.rot_filter.reset()
            self._palm_rot = self.rot_filter(self._raw_rot, 0.0)
        self._h0_pos = None if self._palm_pos is None else self._palm_pos.copy()
        self._h0_rot = None if self._palm_rot is None else self._palm_rot.copy()
        self._e0_pos = np.asarray(e_pos, float).copy()
        self._e0_rot = np.asarray(e_rot, float).copy()
        self._tgt_pos, self._tgt_rot = self._e0_pos.copy(), self._e0_rot.copy()
        self._rot_latched = False

    def _engage(self, ee_pos, ee_rot, base_mm) -> None:
        self.engaged = True
        self._cmd_pos, self._cmd_rot = np.asarray(ee_pos, float).copy(), np.asarray(ee_rot, float).copy()
        self._good_pos, self._good_rot = self._cmd_pos.copy(), self._cmd_rot.copy()
        self._anchor(ee_pos, ee_rot)
        self._abs_base_mm = np.asarray(base_mm, float).copy()
        self._ref_pos = self._palm_pos.copy() if self._palm_pos is not None else None
        self._ref_rot = self._palm_rot.copy() if self._palm_rot is not None else None
        self.pinch.reset()
        self._frozen = self._was_frozen = False
        self._rot_was_held = False

    def _release(self, need_release: bool) -> None:
        self.engaged = False
        self._need_release = self._need_release or need_release

    def force_release(self) -> None:
        """Drop the clutch and require a fresh press (after home/reset/rail
        step moved the arm under the operator)."""
        self._release(need_release=True)

    def reject(self) -> None:
        """The arm could not take the last command (IK failed, a guard
        refused it): roll the command back so it cannot march off."""
        if self._good_pos is not None:
            self._cmd_pos, self._cmd_rot = self._good_pos.copy(), self._good_rot.copy()

    def accept(self) -> None:
        if self._cmd_pos is not None:
            self._good_pos, self._good_rot = self._cmd_pos.copy(), self._cmd_rot.copy()

    # ---- mapping ----------------------------------------------------------
    def _scales(self):
        k_pos = config.PRECISION_POS_FACTOR if self._precision else 1.0
        k_rot = config.PRECISION_ROT_FACTOR if self._precision else 1.0
        return self.scale * k_pos, self.rot_scale * k_rot

    def _map_position(self, s: float) -> np.ndarray:
        if self.map_mode == "absolute":
            c = self.abs_calib
            lo_r = self.r_c2r @ np.asarray(c.corner_a, float)
            hi_r = self.r_c2r @ np.asarray(c.corner_b, float)
            lo, hi = np.minimum(lo_r, hi_r), np.maximum(lo_r, hi_r)
            u = self.r_c2r @ self._palm_pos
            frac = np.clip((u - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)
            b = config.ABSOLUTE_BOX_MM
            box_lo = np.array([b["x"][0] + self._abs_base_mm[0], b["y"][0], b["z"][0]])
            box_hi = np.array([b["x"][1] + self._abs_base_mm[0], b["y"][1], b["z"][1]])
            return (box_lo + frac * (box_hi - box_lo)) / 1000.0
        return self._e0_pos + s * (self.r_c2r @ (self._palm_pos - self._h0_pos))

    def _rotation_ok(self) -> tuple:
        """(ok, reason). Whether this frame's rotation may be used."""
        f = self._last_frame
        if self.orient_mode == "locked":
            return False, "locked"
        if self.rot_hold_external:
            return False, "wrist_singular"
        if f is None or f.palm_rot_cam is None or self._palm_rot is None or self._h0_rot is None:
            return False, "no_rotation"
        if self.orient_mode == "yaw":
            return (f.rot_quality >= config.YAW_MIN_QUALITY), "low_quality"
        if f.approx:
            return False, "approx_frame"
        return (f.rot_quality >= config.FULL_MIN_QUALITY), "low_quality"

    def _map_rotation(self, rs: float) -> tuple:
        ok, reason = self._rotation_ok()
        if not ok:
            self._rot_was_held = self.orient_mode != "locked"
            return self._tgt_rot, (None if self.orient_mode == "locked" else reason)
        if self._rot_was_held:
            # Coming back from a hold: re-anchor rotation only, so the
            # orientation does not leap to wherever the palm went meanwhile.
            self.rot_filter.reset()
            self._palm_rot = self.rot_filter(self._raw_rot, 0.0)
            self._h0_rot = self._palm_rot.copy()
            self._e0_rot = self._tgt_rot.copy()
            self._rot_latched = False
            self._rot_was_held = False
        d_cam = self._palm_rot @ self._h0_rot.T
        d_w = self.r_c2r @ d_cam @ self.r_c2r.T
        if self.orient_mode == "yaw":
            yaw = _yaw_about_z(d_w)
            ang = abs(np.rad2deg(yaw))
            delta = _rz(yaw * rs)
        else:
            rv = Rotation.from_matrix(d_w).as_rotvec()
            ang = float(np.rad2deg(np.linalg.norm(rv)))
            delta = Rotation.from_rotvec(rv * rs).as_matrix()
        # Latching deadband: still until the palm has turned ROT_DEADBAND_DEG
        # from the anchor, then fully tracking until it comes back inside
        # ROT_DEADBAND_EXIT_DEG. Tremor at rest never moves the tool, and a
        # deliberate turn is reproduced 1:1 (no 5 deg loss).
        if not self._rot_latched and ang > config.ROT_DEADBAND_DEG:
            self._rot_latched = True
        elif self._rot_latched and ang < config.ROT_DEADBAND_EXIT_DEG:
            self._rot_latched = False
        if not self._rot_latched:
            return self._e0_rot, None
        return delta @ self._e0_rot, None

    def _rail_push(self, p_mm: np.ndarray, base_mm: np.ndarray, rail_mm: float,
                   dt: float) -> tuple:
        """§4.5 edge-push. Returns (new rail or None, p clamped to the reach
        band along the rail axis)."""
        along = float(np.dot(p_mm - base_mm, self.rail_axis))
        band = config.REACH_HALF_SPAN_MM + config.EDGE_MARGIN_MM
        new_rail = None
        if self.rail_mode == "edge" and abs(along) > band:
            speed = min(config.EDGE_GAIN * (abs(along) - band), config.RAIL_MAX_SPEED)
            cand = float(np.clip(rail_mm + np.sign(along) * speed * dt,
                                 config.RAIL_MIN_MM, config.RAIL_MAX_MM))
            if abs(cand - rail_mm) > 1e-9:
                new_rail = cand
        if abs(along) > band:
            # The arm goes as far as the band; the rail carries it the rest.
            p_mm = p_mm - (along - np.sign(along) * band) * self.rail_axis
        return new_rail, p_mm

    # ---- the tick ---------------------------------------------------------
    def update(self, frame: Optional[HandFrame], now: float, dt: float,
               clutch: bool, precision: bool, ee_pos, ee_rot,
               base_mm, rail_mm: float) -> MapperOutput:
        """`ee_pos/ee_rot`: where the arm is COMMANDED to (world m), used to
        anchor on engage. `base_mm`: arm base, world mm. `rail_mm`: setpoint."""
        accepted = self._ingest(frame, now)
        silent = (self._last_msg_now is None
                  or now - self._last_msg_now > config.SILENCE_S)
        dropout = (self._last_valid_now is None
                   or now - self._last_valid_now > config.DROPOUT_S)
        status = {"silent": silent, "dropout": dropout, "frozen": False,
                  "rot_held": None, "clamped": False, "precision": precision,
                  "need_release": self._need_release, "jumps": self.n_jumps,
                  "accepted": accepted}

        if not clutch:
            self._need_release = False
            if self.engaged:
                self._release(need_release=False)
        elif self.engaged and (dropout or silent):
            self.n_dropouts += 1
            self._release(need_release=True)
        elif (not self.engaged and not self._need_release and not dropout
              and not silent and self._palm_pos is not None):
            self._engage(ee_pos, ee_rot, base_mm)
            self._precision = precision

        status["need_release"] = self._need_release
        if not self.engaged:
            self._precision = precision
            return MapperOutput(False, None, None, None, self._opening, status)

        if precision != self._precision:
            self._precision = precision
            self._anchor(self._tgt_pos, self._tgt_rot)

        if self._frozen:
            status["frozen"] = True
            self._was_frozen = True
        elif self._was_frozen:
            # Freeze over: whatever the palm did during the pinch is dropped.
            self._was_frozen = False
            self._anchor(self._tgt_pos, self._tgt_rot)

        s, rs = self._scales()
        if not self._frozen:
            self._tgt_pos = self._map_position(s)
            self._tgt_rot, status["rot_held"] = self._map_rotation(rs)

        base_mm = np.asarray(base_mm, float)
        new_rail, p_mm = self._rail_push(self._tgt_pos * 1000.0, base_mm, rail_mm, dt)
        clamped_mm = transforms.clamp_workspace_mm(p_mm)
        status["clamped"] = bool(np.linalg.norm(clamped_mm - p_mm) > config.CLAMP_TOLERANCE_MM)

        self._cmd_pos, self._cmd_rot = rate_limit_pose(
            self._cmd_pos, self._cmd_rot, clamped_mm / 1000.0, self._tgt_rot,
            self.max_pos_speed * dt, self.max_rot_speed * dt)
        return MapperOutput(True, self._cmd_pos.copy(), self._cmd_rot.copy(),
                            new_rail, self._opening, status)
