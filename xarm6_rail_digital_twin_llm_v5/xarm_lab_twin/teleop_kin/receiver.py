# teleop_kin/receiver.py
"""Teach handle -> rate-limited target -> IK -> ctrl.

The twin's counterpart of a real arm in manual (teach) mode: the operator
drags the handle and the arm follows it.

THE HANDLE IS ABSOLUTE, THE COMMAND IS NOT. Unlike the SpaceMouse (a velocity
jog that integrates into a target it owns), the handle is a pose. But the arm
must not simply be sent to it: a mouse flick moves the handle half a metre in
one frame. So the receiver owns a *commanded* pose that chases the handle at no
more than MAX_POS_SPEED / MAX_ROT_SPEED, and only that is solved and written to
ctrl. The recorded action stream (ctrl) is therefore smooth by construction.

IK FAILURE ROLLS BACK THE COMMAND, NOT THE HANDLE. On failure ctrl keeps its
last value, the commanded pose returns to the last pose IK accepted, and the
handle turns red. The handle stays where the operator put it while they are
still holding it -- yanking it out from under the mouse would fight the drag --
and is snapped back onto the EE the moment they let go.

WHY SNAP ON RELEASE AT ALL. If the handle could stay parked somewhere other
than the EE, the next drag would start there and the arm would lunge to catch
up. After a release, once the arm has finished chasing the handle and settled
for SNAP_GRACE_S, a handle more than SNAP_POS_MM / SNAP_ROT_DEG from the EE is
moved onto it.

Discrete actions (keys) arrive on the VIEWER thread and are only queued there;
`tick()` runs them on the control thread. Home, reset and the rail step block
for the duration of a paced move, and blocking inside the viewer's key callback
would freeze the window.

All `arm.data` / `arm.model` access is under `arm.lock`.
"""
from __future__ import annotations

import enum
import queue
import time
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from sim.mujoco_env import RAIL_ACT
from teleop_kin import config
from teleop_kin.handle import TeachHandle
from teleop_kin.servo import TwinServo
from teleop_kin.takes import TakeControl
from teleop_sm.receiver import servo_direct
from vr import transforms


class KinAction(enum.Enum):
    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    RAIL_MINUS = "rail_minus"
    RAIL_PLUS = "rail_plus"
    RECORD_TOGGLE = "record_toggle"
    DISCARD_TAKE = "discard_take"
    HOME = "home"
    RESET_SCENE = "reset_scene"


def _rot_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Angle of the rotation taking r_a to r_b, degrees."""
    cos = (np.trace(np.asarray(r_a).T @ np.asarray(r_b)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def rate_limit_pose(cur_pos_m, cur_rot, tgt_pos_m, tgt_rot,
                    max_pos_mm: float, max_rot_deg: float) -> tuple:
    """Step (cur_pos, cur_rot) toward the target by at most the given amounts.

    Translation moves along the straight line; rotation along the geodesic
    (constant axis), so a capped step never changes the direction of travel.
    """
    cur_pos_m = np.asarray(cur_pos_m, float)
    delta = np.asarray(tgt_pos_m, float) - cur_pos_m
    dist_mm = float(np.linalg.norm(delta)) * 1000.0
    if dist_mm > max_pos_mm > 0:
        delta = delta * (max_pos_mm / dist_mm)
    new_pos = cur_pos_m + delta

    rel = Rotation.from_matrix(np.asarray(cur_rot).T @ np.asarray(tgt_rot))
    rotvec = rel.as_rotvec()
    ang_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
    if ang_deg > max_rot_deg > 0:
        rotvec = rotvec * (max_rot_deg / ang_deg)
    new_rot = np.asarray(cur_rot) @ Rotation.from_rotvec(rotvec).as_matrix()
    return new_pos, new_rot


class KinestheticReceiver:
    """Drives the twin from the teach handle. Constructor mirrors
    `SpaceMouseReceiver` so the runner and HUD are interchangeable."""

    def __init__(self, arm, handle: Optional[TeachHandle] = None,
                 recorder=None, recorder_factory: Optional[Callable] = None,
                 task_label: str = config.TASK_LABEL_SIM):
        self.arm = arm
        self.handle = handle or TeachHandle(arm)
        self.takes = TakeControl(recorder, recorder_factory, task_label,
                                 log=lambda m: print(f"[KIN] {m}"))

        self.max_pos_speed = config.MAX_POS_SPEED
        self.max_rot_speed = config.MAX_ROT_SPEED
        self.max_rail_speed = config.MAX_RAIL_SPEED
        self.rail_step_mm = config.RAIL_STEP_MM
        self.servo_mode = config.SERVO_MODE

        self.gripper_closed = False
        self.ik_fail = False
        self.clamped = False
        self.n_solves = 0

        self._actions: "queue.SimpleQueue[KinAction]" = queue.SimpleQueue()
        self._was_dragging = False
        self._settled_s = 0.0
        self._last_ik_warn = 0.0
        self._fail_target: Optional[tuple] = None   # (pos, rot) that last failed
        self._fail_time = 0.0
        self._blocked_shown: Optional[bool] = None
        # Last IK solution and the gravity-sag bias riding on it:
        # ctrl = q_cmd + bias each tick (direct mode). Shared with hand teleop.
        self.servo = TwinServo(arm)

        with self.arm.lock:
            self.rail_mm = float(
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]]) * 1000.0
        self._anchor_on_ee()

    # ---- anchoring --------------------------------------------------------
    @property
    def gravity_comp(self) -> bool:
        return self.servo.gravity_comp

    @gravity_comp.setter
    def gravity_comp(self, on: bool) -> None:
        self.servo.gravity_comp = bool(on)

    def commanded_pose(self) -> tuple:
        """(pos_m, rot) the EE is COMMANDED to (TwinServo.commanded_pose):
        where the arm settles once the bias has absorbed gravity sag."""
        return self.servo.commanded_pose()

    def _anchor_on_ee(self) -> None:
        """Handle, commanded pose and last-solved pose all onto the arm.

        Used at start-up and after anything that moves the arm without the
        handle (home, reset, rail), and when a red handle is let go. Setting
        last-solved too means the next tick does NOT re-solve for a pose the
        arm is already at.

        Anchors on the COMMANDED pose, not the sagged EE: anchoring on the sag
        would bake ~15 mm of droop into the target, and the bias would then
        lift the tool 15 mm off the handle.
        """
        # Something other than the receiver set ctrl (home, reset, rail, or
        # start-up): adopt that setpoint and start the bias from zero.
        self.servo.adopt_ctrl_if_foreign()
        pos, rot = self.commanded_pose()
        self.handle.write_pose(pos, rot)
        self._cmd_pos, self._cmd_rot = pos.copy(), rot.copy()
        self._good_pos, self._good_rot = pos.copy(), rot.copy()
        self._solved_pos, self._solved_rot = pos.copy(), rot.copy()
        self.ik_fail = False
        self.clamped = False
        self._fail_target = None
        self._settled_s = 0.0
        self._show_blocked(False)

    # The current Recorder and the last saved take live on `takes`; exposed
    # here because tests and HUDs read them off the receiver.
    @property
    def rec(self):
        return self.takes.rec

    @rec.setter
    def rec(self, recorder) -> None:
        self.takes.rec = recorder

    @property
    def last_saved(self) -> Optional[str]:
        return self.takes.last_saved

    def resync(self) -> None:
        self.servo.q_cmd = None
        self._anchor_on_ee()

    def _apply_gravity_comp(self, dt: float) -> None:
        if self.servo_mode == "direct":
            self.servo.apply(dt)

    @property
    def following(self) -> bool:
        """True when the arm can go where the handle is."""
        return not (self.ik_fail or self.clamped)

    def _show_blocked(self, blocked: bool) -> None:
        # geom_rgba is a render attribute; only rewrite it on change.
        if blocked != self._blocked_shown:
            self.handle.set_blocked(blocked)
            self._blocked_shown = blocked

    # ---- input ------------------------------------------------------------
    def post(self, action: KinAction) -> None:
        """Queue a discrete action. Thread-safe; the viewer key handler calls
        this, and `tick()` runs it on the control thread."""
        self._actions.put(action)

    # ---- main tick --------------------------------------------------------
    def tick(self, dt: float) -> None:
        while True:
            try:
                action = self._actions.get_nowait()
            except queue.Empty:
                break
            self.dispatch(action)

        dragging = self.handle.is_dragging()
        raw_pos, tgt_rot = self.handle.drag_target()

        # Same workspace box as SpaceMouse and VR; no copy of it here.
        tgt_mm = transforms.clamp_workspace_mm(raw_pos * 1000.0)
        self.clamped = bool(np.linalg.norm(tgt_mm - raw_pos * 1000.0)
                            > config.CLAMP_TOLERANCE_MM)
        tgt_pos = tgt_mm / 1000.0

        new_pos, new_rot = rate_limit_pose(
            self._cmd_pos, self._cmd_rot, tgt_pos, tgt_rot,
            self.max_pos_speed * dt, self.max_rot_speed * dt)

        moved_mm = float(np.linalg.norm(new_pos - self._solved_pos)) * 1000.0
        moved_deg = _rot_angle_deg(self._solved_rot, new_rot)
        want_solve = (moved_mm >= config.SOLVE_MIN_POS_MM
                      or moved_deg >= config.SOLVE_MIN_ROT_DEG)
        if want_solve and self.ik_fail and not self._retry_due(tgt_pos, tgt_rot):
            # Still blocked on the same target: hold, and do not re-pay the
            # cost of a failing solve. ctrl stands; the command stays rolled back.
            self._cmd_pos, self._cmd_rot = self._good_pos.copy(), self._good_rot.copy()
        elif want_solve:
            ok = self._servo(new_pos, new_rot)
            self.n_solves += 1
            if ok:
                self.ik_fail = False
                self._fail_target = None
                if self.servo_mode == "direct":
                    # servo_direct wrote the bare IK solution; remember it so
                    # the bias is re-applied on top below.
                    self.servo.q_cmd = self.servo.read_q_ctrl()
                self._cmd_pos, self._cmd_rot = new_pos, new_rot
                self._good_pos, self._good_rot = new_pos.copy(), new_rot.copy()
                self._solved_pos, self._solved_rot = new_pos.copy(), new_rot.copy()
            else:
                # ctrl was left untouched by the failed solve (q_cmd and the
                # bias still stand). Roll the command back so it cannot march
                # off into unreachable space.
                self.ik_fail = True
                self._cmd_pos, self._cmd_rot = self._good_pos.copy(), self._good_rot.copy()
                self._fail_target = (tgt_pos.copy(), np.array(tgt_rot, float))
                self._fail_time = time.time()
                now = time.time()
                if now - self._last_ik_warn >= config.IK_WARN_PERIOD_S:
                    self._last_ik_warn = now
                    p = tgt_mm
                    print(f"[KIN] handle unreachable at ({p[0]:.0f}, {p[1]:.0f}, "
                          f"{p[2]:.0f}) mm -- arm holding; let go to snap it back")
        else:
            # Sub-threshold: accumulate without solving.
            self._cmd_pos, self._cmd_rot = new_pos, new_rot

        self._apply_gravity_comp(dt)
        self._show_blocked(not self.following)
        self._handle_release(dragging, tgt_pos, tgt_rot, dt)
        self._was_dragging = dragging

    def _retry_due(self, tgt_pos, tgt_rot) -> bool:
        if self._fail_target is None:
            return True
        if time.time() - self._fail_time < config.IK_RETRY_PERIOD_S:
            return False
        f_pos, f_rot = self._fail_target
        return (float(np.linalg.norm(tgt_pos - f_pos)) * 1000.0 >= config.SOLVE_MIN_POS_MM
                or _rot_angle_deg(f_rot, tgt_rot) >= config.SOLVE_MIN_ROT_DEG)

    def _servo(self, pos_m: np.ndarray, rot: np.ndarray) -> bool:
        if self.servo_mode == "validated":
            roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(rot)
            p = pos_m * 1000.0
            rc = self.arm.set_position(
                float(p[0]), float(p[1]), float(p[2]), roll, pitch, yaw,
                speed=config.VALIDATED_SERVO_SPEED_MM_S, wait=False)
            return rc == 0
        with self.servo.rail_at_setpoint():
            return servo_direct(self.arm, pos_m, rot,
                                max_joint_step_rad=np.deg2rad(config.MAX_JOINT_STEP_DEG))

    def _handle_release(self, dragging: bool, tgt_pos, tgt_rot, dt: float) -> None:
        if dragging:
            self._settled_s = 0.0
            return
        if self._was_dragging and not self.following:
            # Let go of a red handle: it must not stay stranded out of reach.
            self._anchor_on_ee()
            print("[KIN] handle released out of reach -- snapped back to the arm")
            return
        caught_up = (float(np.linalg.norm(self._cmd_pos - tgt_pos)) * 1000.0
                     < config.SOLVE_MIN_POS_MM
                     and _rot_angle_deg(self._cmd_rot, tgt_rot) < config.SOLVE_MIN_ROT_DEG)
        if not (caught_up or not self.following):
            self._settled_s = 0.0
            return
        self._settled_s += dt
        if self._settled_s < config.SNAP_GRACE_S:
            return
        h_pos, h_rot = self.handle.read_pose()
        e_pos, e_rot = self.handle.ee_pose()
        if (float(np.linalg.norm(h_pos - e_pos)) * 1000.0 > config.SNAP_POS_MM
                or _rot_angle_deg(h_rot, e_rot) > config.SNAP_ROT_DEG):
            self._anchor_on_ee()

    # ---- discrete actions -------------------------------------------------
    def dispatch(self, action: KinAction) -> None:
        if action is KinAction.GRIPPER_OPEN:
            self._set_gripper(closed=False)
        elif action is KinAction.GRIPPER_CLOSE:
            self._set_gripper(closed=True)
        elif action is KinAction.RAIL_MINUS:
            self._step_rail(-self.rail_step_mm)
        elif action is KinAction.RAIL_PLUS:
            self._step_rail(+self.rail_step_mm)
        elif action is KinAction.RECORD_TOGGLE:
            self.takes.toggle()
        elif action is KinAction.DISCARD_TAKE:
            self.takes.discard()
        elif action is KinAction.HOME:
            self._go_home()
        elif action is KinAction.RESET_SCENE:
            self._reset_scene()

    def _log(self, name: str, payload: dict) -> None:
        self.takes.log_command(name, payload)

    def _set_gripper(self, closed: bool) -> None:
        # Through the arm's own gripper calls, so the weld logic runs -- the
        # recorded gripper channel reads weld activation, not this flag.
        rc = (self.arm.close_lite6_gripper() if closed
              else self.arm.open_lite6_gripper())
        self.gripper_closed = closed
        self._log("gripper", {"closed": closed, "rc": rc})
        what = "closed" if closed else "open"
        held = " (nothing in reach)" if closed and rc != 0 else ""
        print(f"[KIN] gripper {what}{held}")

    def _step_rail(self, delta_mm: float) -> None:
        target = float(np.clip(self.rail_mm + delta_mm,
                               config.RAIL_MIN_MM, config.RAIL_MAX_MM))
        if abs(target - self.rail_mm) < 1e-6:
            print(f"[KIN] rail at its limit ({self.rail_mm:.0f} mm)")
            return
        # Paced, and blocking on the control thread: the arm rides the
        # carriage with its joints held, and the handle is re-anchored after.
        rc = self.arm.set_rail_position(target, speed_mm_s=self.max_rail_speed,
                                        wait=True)
        self._sync_rail()
        self._log("rail", {"target_mm": target, "rc": rc})
        self._anchor_on_ee()
        print(f"[KIN] rail -> {self.rail_mm:.0f} mm (rc={rc})")

    def _sync_rail(self) -> None:
        with self.arm.lock:
            self.rail_mm = float(
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]]) * 1000.0

    def _go_home(self) -> None:
        rc = self.arm.go_home(wait=True)
        self._log("gohome", {"rc": rc})
        self._sync_rail()
        self._anchor_on_ee()
        print(f"[KIN] home (rc={rc})")

    def _reset_scene(self) -> None:
        rc = self.arm.reset_scene()
        self.gripper_closed = False
        self._log("reset_scene", {"rc": rc})
        self._sync_rail()
        self._anchor_on_ee()
        print("[KIN] scene reset")

    # ---- HUD --------------------------------------------------------------
    def status(self) -> dict:
        """Same keys as SpaceMouseReceiver.status(), so HUDs are shared."""
        rec_on = self.takes.recording
        return {
            "type": "status",
            "recording": rec_on,
            "gripper_closed": bool(self.gripper_closed),
            "rail_mm": round(float(self.rail_mm), 1),
            "ik_fail": bool(self.ik_fail),
            "clutch": bool(self.handle.is_dragging()),
            "servo_mode": self.servo_mode,
            "clamped": bool(self.clamped),
            "following": bool(self.following),
        }
