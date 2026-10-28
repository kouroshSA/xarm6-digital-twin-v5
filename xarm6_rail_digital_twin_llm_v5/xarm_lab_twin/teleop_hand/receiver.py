# teleop_hand/receiver.py
"""`HandReceiver.tick(dt)`: newest HandFrame -> mapper -> guarded IK -> twin.

Per tick (§5.1):
  1. drain the frame source (UDP), keep only the newest frame;
  2. `ArmTeleopMapper.update` -> rate-limited target pose, rail setpoint;
  3. arm: guarded IK (`guards.solve_guarded`, branch flip / joint limit /
     wrist band), seeded from qpos, solved with the rail at its SETPOINT,
     written through `TwinServo` (ctrl = q_cmd + gravity-sag bias);
  4. rail: ctrl[RAIL_ACT] = rail_mm / 1000, as SpaceMouse does;
  5. gripper: finger visual from the opening every tick, weld on hysteresis
     transitions (`close_lite6_gripper` / `open_lite6_gripper`);
  6. marker: the teach-handle mocap body (geom group 4, never in recorded
     frames) shows the target. Orange tracking, grey released, red refused,
     blue precision.

Nothing moves without the clutch: no target, no gripper. A failed solve costs
~215 ms under arm.lock, so a refused target is not re-solved every tick (same
back-off as kinesthetic teaching).

All `arm.data` access is under `arm.lock`.
"""
from __future__ import annotations

import enum
import queue
import time
from typing import Callable, Optional

import numpy as np

from sim.mujoco_env import GRIPPER_CLOSED_CTRL_M, GRIPPER_OPEN_CTRL_M, RAIL_ACT
from teleop_hand import config, guards
from teleop_hand.mapping import ArmTeleopMapper
from teleop_hand.retarget import BinaryGrip
from teleop_kin.handle import TeachHandle
from teleop_kin.receiver import _rot_angle_deg, rate_limit_pose
from teleop_kin.servo import TwinServo
from teleop_kin.takes import TakeControl


class HandAction(enum.Enum):
    RECORD_TOGGLE = "record_toggle"
    DISCARD_TAKE = "discard_take"
    HOME = "home"
    RESET_SCENE = "reset_scene"
    RAIL_MINUS = "rail_minus"
    RAIL_PLUS = "rail_plus"


class TwinGeometry:
    """Where the arm base is for a given rail setpoint, and the rail axis --
    read from the model, not hard-coded (§4.5)."""

    def __init__(self, arm):
        m, d = arm.model, arm.data
        with arm.lock:
            self.rail_axis = np.array(m.jnt_axis[arm.rail_jid], float)
            rail_q = float(d.qpos[m.jnt_qposadr[arm.rail_jid]])
            base = np.array(d.body("xarm_base").xpos, float) * 1000.0
        self.base_at_rail0_mm = base - rail_q * 1000.0 * self.rail_axis

    def base_mm(self, rail_mm: float) -> np.ndarray:
        return self.base_at_rail0_mm + rail_mm * self.rail_axis


class HandReceiver:
    def __init__(self, arm, mapper: ArmTeleopMapper, source, *,
                 clutch_fn: Callable[[], bool], precision_fn: Callable[[], bool] = lambda: False,
                 gesture_clutch: bool = False, marker: Optional[TeachHandle] = None,
                 recorder=None, recorder_factory=None,
                 task_label: str = config.TASK_LABEL_SIM, clock=time.monotonic,
                 gestures=None, point_executor=None):
        self.arm = arm
        self.mapper = mapper
        self.source = source
        self.clutch_fn, self.precision_fn = clutch_fn, precision_fn
        self.gesture_clutch = gesture_clutch
        self.marker = marker
        self.clock = clock
        self.takes = TakeControl(recorder, recorder_factory, task_label,
                                 log=lambda m: print(f"[HAND] {m}"))
        self.servo = TwinServo(arm)
        self.servo.adopt_ctrl()
        self.geom = TwinGeometry(arm)
        mapper.rail_axis = self.geom.rail_axis
        self.limits = guards.joint_limits_rad(arm)
        self.grip = BinaryGrip()
        self.rail_mm = self.servo.rail_setpoint_mm()
        self.rail_step_mm = config.RAIL_STEP_MM
        self.max_rail_speed = config.RAIL_MAX_SPEED

        # Gestures (fist-follow, point-to-grasp): None = off.
        self.gestures = gestures
        self.point_executor = point_executor
        self.gesture_out = None
        self._follow_rot = None
        self._g_fail_target = None
        self._g_fail_time = -1e9
        self._g_solved = None
        self.last_point_result = None

        self._actions: "queue.SimpleQueue[HandAction]" = queue.SimpleQueue()
        self._frame = None
        self._solved_pos = self._solved_rot = None
        self._fail_target = None
        self._fail_time = -1e9
        self._last_warn = -1e9
        self.blocked: Optional[str] = None        # guard reason or None
        self.out = None                           # last MapperOutput
        self.n_solves = 0
        self.latencies_s: list = []               # capture -> command
        self._shown_rgba = None

    # ---- input --------------------------------------------------------------
    def post(self, action: HandAction) -> None:
        self._actions.put(action)

    def _clutch(self) -> bool:
        if self.gesture_clutch:
            f = self._frame
            return bool(f is not None and f.aux_hand_fist)
        return bool(self.clutch_fn())

    # ---- tick ---------------------------------------------------------------
    def tick(self, dt: float) -> None:
        while True:
            try:
                self.dispatch(self._actions.get_nowait())
            except queue.Empty:
                break

        frame = self.source.poll()
        if frame is not None:
            self._frame = frame
        now = self.clock()

        self.servo.adopt_ctrl_if_foreign()
        q_cmd = self.servo.q_cmd
        self.mapper.rot_hold_external = (self.mapper.orient_mode in ("yaw", "full")
                                         and guards.in_wrist_band(q_cmd))
        ee_pos, ee_rot = self.servo.commanded_pose()
        out = self.mapper.update(frame, now, dt, self._clutch(), bool(self.precision_fn()),
                                 ee_pos, ee_rot, self.geom.base_mm(self.rail_mm), self.rail_mm)
        self.out = out

        if out.rail_mm is not None:
            self.rail_mm = out.rail_mm
            with self.arm.lock:
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]] = self.rail_mm / 1000.0

        if out.engaged:
            self._reset_gesture_motion()
            if self.gestures is not None:
                self.gestures.pause()
            self._servo(out, frame, now)
            self._gripper(out.opening)
        else:
            self.blocked = None
            self._solved_pos = None
            self._fail_target = None
            if self.gestures is not None:
                self._gestures_tick(now, dt, ee_pos)

        self.servo.apply(dt)
        self._show_marker(out)

    def _servo(self, out, frame, now: float) -> None:
        pos, rot = out.target_pos, out.target_rot
        if self._solved_pos is not None:
            moved_mm = float(np.linalg.norm(pos - self._solved_pos)) * 1000.0
            moved_deg = _rot_angle_deg(self._solved_rot, rot)
            if moved_mm < config.SOLVE_MIN_POS_MM and moved_deg < config.SOLVE_MIN_ROT_DEG:
                if self.blocked is None:
                    self.mapper.accept()
                return
        if self._fail_target is not None and not self._retry_due(pos, rot, now):
            self.mapper.reject()
            return
        check_wrist = self.mapper.orient_mode in ("yaw", "full")
        with self.servo.rail_at_setpoint():
            q, why = guards.solve_guarded(self.arm, pos, rot, self.servo.q_cmd,
                                          self.limits, check_wrist=check_wrist)
        self.n_solves += 1
        if q is not None:
            self.servo.q_cmd = q
            self.mapper.accept()
            self.blocked = None
            self._fail_target = None
            self._solved_pos, self._solved_rot = pos.copy(), rot.copy()
            if frame is not None and out.status.get("accepted"):
                self.latencies_s.append(now - frame.t_mono)
            return
        self.mapper.reject()
        self.blocked = why
        self._fail_target = (pos.copy(), rot.copy())
        self._fail_time = now
        if now - self._last_warn > config.IK_WARN_PERIOD_S:
            self._last_warn = now
            p = pos * 1000.0
            print(f"[HAND] target refused ({why}) at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm"
                  f" -- arm holding")

    # ---- gestures -------------------------------------------------------------
    def _reset_gesture_motion(self) -> None:
        self._follow_rot = None
        self._g_fail_target = None
        self._g_solved = None
        self.gesture_out = None

    def _gestures_tick(self, now: float, dt: float, ee_pos) -> None:
        base = self.geom.base_mm(self.rail_mm)
        gout = self.gestures.update(self._frame, now, dt, ee_pos, base)
        self.gesture_out = gout
        if gout.kind == "follow":
            self._follow(gout.follow_target_m, now, dt, base)
            return
        self._reset_gesture_motion()
        self.gesture_out = gout
        if gout.kind == "point" and self.point_executor is not None:
            self._run_point(gout.point)

    def _follow(self, target_m, now: float, dt: float, base_mm) -> None:
        cur_pos, cur_rot = self.servo.commanded_pose()
        if self._follow_rot is None:
            self._follow_rot = cur_rot.copy()         # orientation stays as it was
        step_pos, step_rot = rate_limit_pose(
            cur_pos, cur_rot, target_m, self._follow_rot,
            config.FOLLOW_MAX_SPEED_MM_S * dt, config.MAX_ROT_SPEED * dt)
        new_rail, p_mm = self.mapper._rail_push(step_pos * 1000.0, base_mm, self.rail_mm, dt)
        if new_rail is not None:
            self.rail_mm = new_rail
            with self.arm.lock:
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]] = self.rail_mm / 1000.0
        pos = p_mm / 1000.0
        if self._g_solved is not None:
            if (float(np.linalg.norm(pos - self._g_solved[0])) * 1000.0 < config.SOLVE_MIN_POS_MM
                    and _rot_angle_deg(self._g_solved[1], step_rot) < config.SOLVE_MIN_ROT_DEG):
                return
        if self._g_fail_target is not None:
            if now - self._g_fail_time < config.IK_RETRY_PERIOD_S or \
                    float(np.linalg.norm(pos - self._g_fail_target)) * 1000.0 < config.SOLVE_MIN_POS_MM:
                return
        with self.servo.rail_at_setpoint():
            q, why = guards.solve_guarded(self.arm, pos, step_rot, self.servo.q_cmd,
                                          self.limits, check_wrist=False)
        self.n_solves += 1
        if q is not None:
            self.servo.q_cmd = q
            self.blocked = None
            self._g_fail_target = None
            self._g_solved = (pos.copy(), step_rot.copy())
            return
        self.blocked = why
        self._g_fail_target, self._g_fail_time = pos.copy(), now
        if now - self._last_warn > config.IK_WARN_PERIOD_S:
            self._last_warn = now
            print(f"[HAND] follow target refused ({why}) -- arm holding")

    def _run_point(self, target) -> None:
        """Blocking, like HOME: nothing else commands the arm meanwhile. The
        clutch key aborts between steps (the executor's `abort_fn`)."""
        if self.marker is not None:
            self.marker.write_pose(np.asarray(target.position_mm) / 1000.0,
                                   self.servo.commanded_pose()[1])
            self.marker.set_color(config.RGBA_GESTURE)
            self._shown_rgba = config.RGBA_GESTURE
        base = self.geom.base_mm(self.rail_mm)
        scene = getattr(self.gestures, "scene", None)
        others = ({n: np.asarray(c) * 1000.0 for n, c in scene.candidates().items()}
                  if scene is not None else {})
        res = self.point_executor.run(target, base, self.geom.rail_axis, self.rail_mm, others)
        self.last_point_result = res
        self.takes.log_command("point", {"kind": target.kind, "name": target.name,
                                         "position_mm": [round(float(v), 1) for v in target.position_mm],
                                         "ok": res.ok, "stage": res.stage, "held": res.held,
                                         "message": res.message})
        self._after_primitive()
        self.gestures.mode = "idle"
        print(f"[HAND] point -> {'OK' if res.ok else 'FAILED'} ({res.stage}): {res.message}")

    def _retry_due(self, pos, rot, now: float) -> bool:
        if now - self._fail_time < config.IK_RETRY_PERIOD_S:
            return False
        f_pos, f_rot = self._fail_target
        return (float(np.linalg.norm(pos - f_pos)) * 1000.0 >= config.SOLVE_MIN_POS_MM
                or _rot_angle_deg(f_rot, rot) >= config.SOLVE_MIN_ROT_DEG)

    def _gripper(self, opening: Optional[float]) -> None:
        if opening is None:
            return
        with self.arm.lock:
            self.arm._set_finger_ctrl(GRIPPER_OPEN_CTRL_M + (1.0 - opening)
                                      * (GRIPPER_CLOSED_CTRL_M - GRIPPER_OPEN_CTRL_M))
        edge = self.grip.update(opening)
        if edge == "close":
            rc = self.arm.close_lite6_gripper()
            self.takes.log_command("gripper", {"closed": True, "rc": rc, "opening": opening})
        elif edge == "open":
            rc = self.arm.open_lite6_gripper()
            self.takes.log_command("gripper", {"closed": False, "rc": rc, "opening": opening})

    # ---- marker -------------------------------------------------------------
    def marker_rgba(self) -> tuple:
        out = self.out
        if self.gesture_out is not None and self.gesture_out.kind == "follow":
            return config.RGBA_BLOCKED if self.blocked else config.RGBA_GESTURE
        if out is None or not out.engaged:
            return config.RGBA_RELEASED
        if self.blocked or out.status.get("clamped"):
            return config.RGBA_BLOCKED
        if out.status.get("precision"):
            return config.RGBA_PRECISION
        return config.RGBA_TRACKING

    def _show_marker(self, out) -> None:
        if self.marker is None:
            return
        g = self.gesture_out
        if g is not None and g.kind == "follow" and g.follow_target_m is not None:
            self.marker.write_pose(g.follow_target_m, self._follow_rot
                                   if self._follow_rot is not None
                                   else self.servo.commanded_pose()[1])
        elif out.engaged and out.target_pos is not None:
            self.marker.write_pose(out.target_pos, out.target_rot)
        else:
            self.marker.write_pose(*self.servo.commanded_pose())
        rgba = self.marker_rgba()
        if rgba != self._shown_rgba:          # geom_rgba: rewrite on change only
            self.marker.set_color(rgba)
            self._shown_rgba = rgba

    # ---- discrete actions -----------------------------------------------------
    def dispatch(self, action: HandAction) -> None:
        if action is HandAction.RECORD_TOGGLE:
            self.takes.toggle()
        elif action is HandAction.DISCARD_TAKE:
            self.takes.discard()
        elif action is HandAction.HOME:
            rc = self.arm.go_home(wait=True)
            self.takes.log_command("gohome", {"rc": rc})
            self._after_primitive()
        elif action is HandAction.RESET_SCENE:
            rc = self.arm.reset_scene()
            self.grip.closed = False
            self.takes.log_command("reset_scene", {"rc": rc})
            self._after_primitive()
        elif action in (HandAction.RAIL_MINUS, HandAction.RAIL_PLUS):
            self._step_rail(-1 if action is HandAction.RAIL_MINUS else 1)

    def _after_primitive(self) -> None:
        """A primitive moved the arm: adopt its ctrl, drop the clutch (a fresh
        press re-anchors on the new pose)."""
        self.servo.adopt_ctrl()
        self.rail_mm = self.servo.rail_setpoint_mm()
        self.mapper.force_release()

    def _step_rail(self, sign: int) -> None:
        if self.mapper.rail_mode != "keys":
            print("[HAND] rail keys are for --rail keys (edge mode drives the rail)")
            return
        if self.mapper.engaged:
            print("[HAND] rail keys work with the clutch RELEASED")
            return
        target = float(np.clip(self.rail_mm + sign * self.rail_step_mm,
                               config.RAIL_MIN_MM, config.RAIL_MAX_MM))
        if abs(target - self.rail_mm) < 1e-6:
            return
        rc = self.arm.set_rail_position(target, speed_mm_s=self.max_rail_speed, wait=True)
        self.takes.log_command("rail", {"target_mm": target, "rc": rc})
        self._after_primitive()
        print(f"[HAND] rail -> {self.rail_mm:.0f} mm")

    # ---- HUD ------------------------------------------------------------------
    def status(self) -> dict:
        out = self.out
        st = dict(out.status) if out is not None else {}
        st.update({
            "engaged": bool(out.engaged) if out else False,
            "blocked": self.blocked,
            "recording": self.takes.recording,
            "rail_mm": round(self.rail_mm, 1),
            "gripper_closed": self.grip.closed,
            "opening": None if out is None else out.opening,
            "rot_quality": None if self._frame is None else self._frame.rot_quality,
            "orient": self.mapper.orient_mode, "map": self.mapper.map_mode,
            "rail_mode": self.mapper.rail_mode,
        })
        if self.gesture_out is not None:
            st["gesture"] = dict(self.gesture_out.status)
        if self.last_point_result is not None:
            r = self.last_point_result
            st["last_point"] = {"ok": r.ok, "stage": r.stage, "held": r.held}
        return st
