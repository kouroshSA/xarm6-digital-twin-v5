# vr/teleop_receiver.py
"""Turns Touch-controller / head state into twin actions.

Framework-agnostic: the server feeds raw input via :meth:`handle_input` and
calls :meth:`tick` at ``config.CONTROL_HZ``. No asyncio in here.

Input contract (one JSON object per message, from xr-client.js)::

    {
      "head":  {"pos": [x,y,z], "quat": [x,y,z,w]},
      "right": {"pos": [x,y,z], "quat": [x,y,z,w],
                "buttons": [bool, ...], "axes": [float, ...]},
      "left":  {"pos": [...], "quat": [...], "buttons": [...], "axes": [...]}
    }

Positions are metres in the WebXR ``local-floor`` frame; quaternions are
[x,y,z,w]. ``buttons`` are *pressed* booleans indexed by the standard
``xr-standard`` gamepad layout; ``axes`` are floats (axes[2],axes[3] are the
thumbstick on that layout).

Touch mapping (right controller unless noted):
    grip / squeeze (buttons[1]) held  -> clutch: arm follows controller
    trigger        (buttons[0]) press -> toggle gripper
    A              (buttons[4]) press -> toggle recording
    B              (buttons[5]) press -> reset_scene
    left stick X   (left axes[2])     -> jog rail along 0..700 mm
    head pose                          -> drive vr_head mocap (stereo viewpoint)

Button presses are edge-detected (act on the rising edge); the grip clutch is
the one while-held input.

All ``arm.data`` / ``arm.model`` access is under ``arm.lock`` (no exceptions).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import mujoco
import numpy as np

from sim.mujoco_env import RAIL_ACT
from vr import config, transforms

# xr-standard gamepad button indices.
BTN_TRIGGER = 0
BTN_GRIP = 1
BTN_A = 4
BTN_B = 5
# xr-standard thumbstick is axes[2] (X) / axes[3] (Y).
AXIS_STICK_X = 2


class TeleopReceiver:
    def __init__(self, arm, recorder=None,
                 recorder_factory: Optional[Callable[[], object]] = None,
                 task_label: str = "vr_teleop"):
        self.arm = arm
        self.rec = recorder
        self._recorder_factory = recorder_factory
        self.task_label = task_label

        # vr_head mocap index (None if the scene lacks the body).
        self._mocap_id = self._resolve_mocap("vr_head")

        # Latest input snapshot + its own lock (server thread writes, control
        # thread reads).
        self._state_lock = threading.Lock()
        self._state: Optional[dict] = None

        # Edge-detect state for buttons, keyed (hand, index) -> bool pressed.
        self._prev_btn: dict = {}

        self.clutch = transforms.Clutch()
        self.smoother = transforms.Smoother(alpha=config.SMOOTH_ALPHA)

        # Continuous state surfaced to the HUD.
        self.gripper_closed = False
        self.ik_fail = False
        _, rail_mm = self.arm.get_rail_position()
        self.rail_mm = float(rail_mm)

        # Relative head tracking: remember the first head sample so the
        # operator's absolute standing height doesn't shove the camera to the
        # ceiling — only head *motion* moves the twin viewpoint.
        self._head_xr0: Optional[np.ndarray] = None

        self._tick_count = 0

    # ---- setup helpers ----------------------------------------------------
    def _resolve_mocap(self, body_name: str) -> Optional[int]:
        try:
            bid = self.arm.model.body(body_name).id
        except (KeyError, ValueError):
            print(f"[VR] WARNING: body '{body_name}' not in scene; "
                  f"head tracking disabled.")
            return None
        mid = int(self.arm.model.body_mocapid[bid])
        if mid < 0:
            print(f"[VR] WARNING: body '{body_name}' is not a mocap body; "
                  f"head tracking disabled.")
            return None
        return mid

    # ---- input intake -----------------------------------------------------
    def handle_input(self, state: dict) -> None:
        """Store the latest controller/head snapshot (called by the server)."""
        with self._state_lock:
            self._state = state

    def _latest(self) -> Optional[dict]:
        with self._state_lock:
            return self._state

    def _edge(self, hand: str, idx: int, buttons) -> bool:
        """True on the rising edge of buttons[idx] for the given hand."""
        pressed = bool(buttons[idx]) if buttons and len(buttons) > idx else False
        key = (hand, idx)
        prev = self._prev_btn.get(key, False)
        self._prev_btn[key] = pressed
        return pressed and not prev

    def _held(self, hand: str, idx: int, buttons) -> bool:
        held = bool(buttons[idx]) if buttons and len(buttons) > idx else False
        self._prev_btn[(hand, idx)] = held
        return held

    # ---- control tick -----------------------------------------------------
    def tick(self, dt: float) -> None:
        """One control update. ``dt`` is the wall-clock seconds since the
        previous tick (used to integrate the rail jog)."""
        state = self._latest()
        if state is None:
            return
        self._tick_count += 1

        self._update_head(state.get("head"))

        right = state.get("right") or {}
        left = state.get("left") or {}
        r_btn = right.get("buttons") or []
        l_axes = left.get("axes") or []

        # --- gripper (trigger rising edge) ---
        if self._edge("right", BTN_TRIGGER, r_btn):
            self._toggle_gripper()

        # --- recording (A rising edge) ---
        if self._edge("right", BTN_A, r_btn):
            self._toggle_recording()

        # --- reset scene (B rising edge) ---
        if self._edge("right", BTN_B, r_btn):
            self._reset_scene()

        # --- clutch (grip held) ---
        grip_held = self._held("right", BTN_GRIP, r_btn)
        self._update_clutch(grip_held, right)

        # --- rail jog (left thumbstick X) ---
        self._update_rail(l_axes, dt)

    # ---- head / mocap -----------------------------------------------------
    def _update_head(self, head: Optional[dict]) -> None:
        if head is None or self._mocap_id is None:
            return
        p_xr = np.asarray(head.get("pos", [0, 0, 0]), float).reshape(3)
        q_xr = head.get("quat", [0, 0, 0, 1])
        if self._head_xr0 is None:
            self._head_xr0 = p_xr.copy()
        pos_twin = config.VR_HEAD_BASE_M + transforms.xr_delta_to_twin(
            p_xr - self._head_xr0)
        quat_twin = transforms.xr_to_twin_head_quat(q_xr)
        with self.arm.lock:
            self.arm.data.mocap_pos[self._mocap_id] = pos_twin
            self.arm.data.mocap_quat[self._mocap_id] = quat_twin

    # ---- clutch / servo ---------------------------------------------------
    def _ee_pose_twin(self) -> tuple:
        """Current EE (pos_m, rot_3x3) in the twin frame."""
        with self.arm.lock:
            mujoco.mj_forward(self.arm.model, self.arm.data)
            pos = self.arm.data.site_xpos[self.arm.ee_site].copy()
            rot = self.arm.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy()
        return pos, rot

    def _update_clutch(self, grip_held: bool, right: dict) -> None:
        if not grip_held:
            if self.clutch.engaged:
                self.clutch.release()
            return

        ctrl_pos_twin = transforms.xr_to_twin_pos(right.get("pos", [0, 0, 0]))
        ctrl_rot_twin = transforms.xr_to_twin_quat(right.get("quat", [0, 0, 0, 1]))

        if not self.clutch.engaged:
            # Engage: freeze the controller<->EE offset so the arm doesn't jump.
            ee_pos, ee_rot = self._ee_pose_twin()
            self.clutch.engage(ctrl_pos_twin, ctrl_rot_twin, ee_pos, ee_rot)
            self.smoother.reset(ee_pos)
            return

        target_pos, target_rot = self.clutch.target(ctrl_pos_twin, ctrl_rot_twin)
        target_pos = self.smoother.update(target_pos)

        # Clamp into the safe workspace (in mm) before IK.
        target_pos_mm = transforms.clamp_workspace_mm(target_pos * 1000.0)

        if config.SERVO_MODE == "validated":
            self._servo_validated(target_pos_mm, target_rot)
        else:
            self._servo_direct(target_pos_mm / 1000.0, target_rot)

        if self.rec is not None and self.rec.is_recording:
            roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(target_rot)
            self.rec.log_command("ee_target", {
                "pos_mm": [float(v) for v in target_pos_mm],
                "rpy_deg": [roll, pitch, yaw],
                "mode": config.SERVO_MODE,
            })

    def _servo_direct(self, target_pos_m: np.ndarray, target_rot: np.ndarray) -> None:
        """Solve IK once and write the six joint targets straight into ctrl.

        Bypasses the FKValidator — fine for continuous servoing in sim.
        Mirrors SimXArmAPI._execute_joint_angles' indexing (act_ids[1:]).
        Holds the lock across the solve (which save/restores qpos) and the
        ctrl write so the stepper can't race it.
        """
        with self.arm.lock:
            seed = np.array([float(self.arm.data.qpos[jid])
                             for jid in self.arm.ik_solver.joint_ids])
            q = self.arm.ik_solver.solve(target_pos_m, target_rot=target_rot,
                                         seed_q=seed)
            if q is not None:
                for i, ang in enumerate(q):
                    self.arm.data.ctrl[self.arm.act_ids[1 + i]] = float(ang)
        self.ik_fail = q is None

    def _servo_validated(self, target_pos_mm: np.ndarray, target_rot: np.ndarray) -> None:
        """Reuse the existing IK + FKValidator + pacing path. set_position
        takes the lock itself, so we must NOT hold it here."""
        roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(target_rot)
        rc = self.arm.set_position(
            float(target_pos_mm[0]), float(target_pos_mm[1]), float(target_pos_mm[2]),
            roll, pitch, yaw,
            speed=config.VALIDATED_SERVO_SPEED_MM_S, wait=False,
        )
        self.ik_fail = (rc != 0)

    # ---- gripper ----------------------------------------------------------
    def _toggle_gripper(self) -> None:
        if self.gripper_closed:
            self.arm.open_lite6_gripper()
            self.gripper_closed = False
        else:
            self.arm.close_lite6_gripper()
            self.gripper_closed = True
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("gripper", {"closed": self.gripper_closed})
        print(f"[VR] gripper {'closed' if self.gripper_closed else 'open'}")

    # ---- recording --------------------------------------------------------
    def _toggle_recording(self) -> None:
        if self.rec is None:
            print("[VR] recording disabled (--no-record); A button ignored.")
            return
        if not self.rec.is_recording:
            self.rec.start()
            print("[VR] recording STARTED")
        else:
            path = self.rec.stop(kept=True, task_label=self.task_label)
            print(f"[VR] recording STOPPED -> {path}")
            # Fresh recorder for the next take.
            if self._recorder_factory is not None:
                self.rec = self._recorder_factory()

    # ---- reset ------------------------------------------------------------
    def _reset_scene(self) -> None:
        self.arm.reset_scene()
        # Releasing any grasp keeps the HUD honest after a reset.
        self.gripper_closed = False
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("reset_scene", {})
        print("[VR] scene reset")

    # ---- rail -------------------------------------------------------------
    def _update_rail(self, axes, dt: float) -> None:
        if not axes or len(axes) <= AXIS_STICK_X:
            return
        x = float(axes[AXIS_STICK_X])
        if abs(x) < config.STICK_DEADZONE:
            return
        # Integrate stick deflection into a rail target and write the actuator
        # ctrl directly (continuous servo, like the direct arm path) so the
        # jog is smooth at tick rate rather than re-paced per tick.
        self.rail_mm = float(np.clip(
            self.rail_mm + x * config.RAIL_JOG_MM_PER_S * dt,
            config.RAIL_MIN_MM, config.RAIL_MAX_MM))
        with self.arm.lock:
            self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]] = self.rail_mm / 1000.0

    # ---- HUD status -------------------------------------------------------
    def status(self) -> dict:
        rec_on = bool(self.rec is not None and self.rec.is_recording)
        return {
            "type": "status",
            "recording": rec_on,
            "gripper_closed": bool(self.gripper_closed),
            "rail_mm": round(float(self.rail_mm), 1),
            "ik_fail": bool(self.ik_fail),
            "clutch": bool(self.clutch.engaged),
            "servo_mode": config.SERVO_MODE,
        }
