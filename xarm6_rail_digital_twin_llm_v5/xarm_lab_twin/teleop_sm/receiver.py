# teleop_sm/receiver.py
"""Turns six-axis SpaceMouse state into twin actions.

Same role `vr/teleop_receiver.py::TeleopReceiver` plays for WebXR, and the
same constructor shape, so a HUD or runner written against one works against
the other. `status()` returns the same keys.

THE ESSENTIAL DIFFERENCE FROM THE VR PATH, and the thing that shapes this
whole module: **a SpaceMouse is a velocity jog, not a pose tracker.** There is
no absolute hand pose, so there is nothing to clutch against -- the VR path's
"freeze the controller-to-EE offset on engage" has no counterpart here.
Instead this integrates deflection into a *target* pose that it owns.

WHY AN OWNED TARGET RATHER THAN READING THE EE EACH TICK. Re-reading the
actual EE and adding a delta to it feeds IK tracking error straight back into
the command: the arm lags, the lag is read as position, the next delta is
added to the lagged value, and the target either creeps or runs away. Holding
an internal target and integrating onto *that* keeps the jog stable and makes
"let go and it stops" true. The cost is that the target and the arm can
disagree; `status()` exposes both so the operator can see it.

IK FAILURE FREEZES THE TARGET. The characteristic failure of an integrating
jog is that the operator keeps pushing after the arm has stopped being able to
follow, the target marches off into unreachable space, and when it finally
does become reachable again the arm leaps. So on IK failure the target is
rolled back to the last feasible pose and `ik_fail` is raised for the HUD.

All `arm.data` / `arm.model` access is under `arm.lock`, no exceptions.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import mujoco
import numpy as np

from sim.mujoco_env import RAIL_ACT
from teleop_sm import config
from teleop_sm.buttons import HELD_ACTIONS, Action, hold_map_for, map_for
from vr import transforms


class Mode:
    """Which DOFs the six axes drive. Cycled with MODE_CYCLE."""

    ARM = "arm"        # 6-axis Cartesian jog of the EE
    RAIL = "rail"      # Y translation drives the rail; everything else ignored

    ORDER = (ARM, RAIL)

    @staticmethod
    def next(current: str) -> str:
        i = Mode.ORDER.index(current)
        return Mode.ORDER[(i + 1) % len(Mode.ORDER)]


class SpaceMouseReceiver:
    """Consumes `SpaceMouseDevice` state, drives the twin.

    Mirrors `TeleopReceiver`'s constructor so runners and HUDs are
    interchangeable between the two teleop paths.
    """

    def __init__(self, arm, recorder=None,
                 recorder_factory: Optional[Callable] = None,
                 task_label: str = config.TASK_LABEL,
                 model: str = "compact"):
        self.arm = arm
        self.rec = recorder
        self._recorder_factory = recorder_factory
        self.task_label = task_label
        self.model = model
        self.button_map = map_for(model)
        self.hold_map = hold_map_for(model)

        self.mode: str = Mode.ARM
        self.gripper_closed = False
        self.ik_fail = False
        self._lock = threading.Lock()

        # The owned target, in twin frame. Seeded from the arm on first tick
        # so the first deflection does not jump.
        self._target_pos_m: Optional[np.ndarray] = None
        self._target_rot: Optional[np.ndarray] = None
        self._last_good_pos_m: Optional[np.ndarray] = None
        self._last_good_rot: Optional[np.ndarray] = None

        self.smoother = transforms.Smoother(alpha=config.SMOOTH_ALPHA)

        with self.arm.lock:
            rail_m = float(self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]])
        self.rail_mm = rail_m * 1000.0

        # Speed caps are instance state so the CLI can override per run.
        self.max_pos_speed = config.MAX_POS_SPEED
        self.max_rot_speed = config.MAX_ROT_SPEED
        self.max_rail_speed = config.MAX_RAIL_SPEED

    # ---- pose helpers -----------------------------------------------------
    def _ee_pose_twin(self) -> tuple:
        """Current EE (pos_m, rot_3x3) in the twin frame."""
        with self.arm.lock:
            mujoco.mj_forward(self.arm.model, self.arm.data)
            pos = self.arm.data.site_xpos[self.arm.ee_site].copy()
            rot = self.arm.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy()
        return pos, rot

    def _seed_target(self) -> None:
        """Anchor the integrator on the arm's actual pose."""
        pos, rot = self._ee_pose_twin()
        self._target_pos_m = pos.copy()
        self._target_rot = rot.copy()
        self._last_good_pos_m = pos.copy()
        self._last_good_rot = rot.copy()
        self.smoother.reset(pos)

    def resync(self) -> None:
        """Re-anchor the target on the arm. Public because the operator needs
        it after a scene reset or a long IK stall."""
        self._seed_target()
        self.ik_fail = False

    # ---- main tick --------------------------------------------------------
    def tick(self, state_twin: np.ndarray, edges: set[int],
             held: set[int], dt: float) -> None:
        """Advance one control step.

        `state_twin` is the device's six axes already permuted into the twin
        frame; `edges` are button indices that went down since the last tick;
        `held` are those currently down. The runner supplies all three, which
        keeps this class free of any spnav dependency and therefore testable
        with plain arrays.
        """
        if self._target_pos_m is None:
            self._seed_target()

        rail_held = self._action_held(Action.RAIL_MODE, held)
        self._handle_edges(edges, suppress_mode_cycle=rail_held)

        effective_mode = Mode.RAIL if rail_held else self.mode

        if effective_mode == Mode.RAIL:
            self._update_rail(state_twin, dt)
        else:
            self._update_arm(state_twin, dt)

    def _action_held(self, action: Action, held: set[int]) -> bool:
        """Is a key bound to `action` currently down?

        Reads the HOLD map, not the edge map: on the Compact one physical key
        carries both roles, so consulting the edge map here would miss it.
        """
        return any(self.hold_map.get(idx) is action for idx in held)

    def _handle_edges(self, edges: set[int], suppress_mode_cycle: bool) -> None:
        for idx in sorted(edges):
            action = self.button_map.get(idx)
            if action is None:
                continue
            # A held RAIL_MODE key must not also fire its edge action; on the
            # Compact those are the same physical button by necessity.
            if action in HELD_ACTIONS:
                continue
            # A key doing double duty (Compact button 1) must not fire its
            # edge action on the press that starts a hold.
            if suppress_mode_cycle and self.hold_map.get(idx) is Action.RAIL_MODE:
                continue
            self.dispatch(action)

    def dispatch(self, action: Action) -> None:
        """Run a semantic action. Public so the keyboard path shares it."""
        if action is Action.GRIPPER_TOGGLE:
            self._toggle_gripper()
        elif action is Action.MODE_CYCLE:
            self._cycle_mode()
        elif action is Action.RECORD_TOGGLE:
            self._toggle_recording()
        elif action is Action.RESET_SCENE:
            self._reset_scene()
        elif action is Action.HOME:
            self._go_home()

    # ---- arm jog ----------------------------------------------------------
    def _update_arm(self, state_twin: np.ndarray, dt: float) -> None:
        lin = np.asarray(state_twin[:3], dtype=float)
        ang = np.asarray(state_twin[3:], dtype=float)

        if not np.any(lin) and not np.any(ang):
            # Nothing commanded: hold position. Explicitly NOT re-seeding from
            # the arm here -- that would let tracking error walk the target.
            return

        delta_pos_mm = lin * self.max_pos_speed * dt
        delta_rpy_deg = ang * self.max_rot_speed * dt

        target_pos_mm = self._target_pos_m * 1000.0 + delta_pos_mm
        target_pos_mm = transforms.clamp_workspace_mm(target_pos_mm)

        # Rotation integrates as a body-frame increment on the target.
        if np.any(delta_rpy_deg):
            inc = _rpy_to_matrix(np.deg2rad(delta_rpy_deg))
            target_rot = self._target_rot @ inc
        else:
            target_rot = self._target_rot

        # THE SMOOTHER IS AN OUTPUT FILTER, NOT PART OF THE INTEGRATOR STATE.
        # Writing its output back into the target makes the smoother's state
        # *be* the target, and then each tick advances by only alpha*delta --
        # the jog runs at alpha times the speed the operator asked for,
        # silently (0.3 * 150 mm/s = 45 mm/s, which just feels "sluggish"
        # rather than "wrong"). So the raw integral is the state, and
        # smoothing is applied only on the way to IK.
        self._target_pos_m = target_pos_mm / 1000.0
        self._target_rot = target_rot

        smoothed_mm = self.smoother.update(target_pos_mm)

        if config.SERVO_MODE == "validated":
            self._servo_validated(smoothed_mm, target_rot)
        else:
            self._servo_direct(smoothed_mm / 1000.0, target_rot)

        if self.ik_fail:
            # Roll back so the operator cannot integrate into unreachable
            # space. Without this the arm leaps when the target re-enters
            # reach -- the main failure mode of an integrating jog.
            self._target_pos_m = self._last_good_pos_m.copy()
            self._target_rot = self._last_good_rot.copy()
            self.smoother.reset(self._target_pos_m * 1000.0)
        else:
            self._last_good_pos_m = self._target_pos_m.copy()
            self._last_good_rot = self._target_rot.copy()

    def _servo_direct(self, target_pos_m: np.ndarray, target_rot: np.ndarray) -> None:
        """Solve IK once and write the six joint targets straight into ctrl.

        Mirrors TeleopReceiver._servo_direct, including holding the lock
        across the solve (which save/restores qpos) and the ctrl write.
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

    def _servo_validated(self, target_pos_mm: np.ndarray,
                         target_rot: np.ndarray) -> None:
        """Reuse the existing IK + FKValidator + pacing path. set_position
        takes the lock itself, so we must NOT hold it here."""
        roll, pitch, yaw = transforms.twin_rot_to_rpy_deg(target_rot)
        rc = self.arm.set_position(
            float(target_pos_mm[0]), float(target_pos_mm[1]), float(target_pos_mm[2]),
            roll, pitch, yaw,
            speed=config.VALIDATED_SERVO_SPEED_MM_S, wait=False,
        )
        self.ik_fail = (rc != 0)

    # ---- rail -------------------------------------------------------------
    def _update_rail(self, state_twin: np.ndarray, dt: float) -> None:
        """Y translation drives the rail; every other axis is ignored.

        Explicit mode switching rather than auto-allocating the 7th DOF: an
        operator can predict a mode, and cannot predict a heuristic.
        """
        y = float(state_twin[1])
        if abs(y) < 1e-9:
            return
        self.rail_mm = float(np.clip(
            self.rail_mm + y * self.max_rail_speed * dt,
            config.RAIL_MIN_MM, config.RAIL_MAX_MM))
        with self.arm.lock:
            self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]] = self.rail_mm / 1000.0
        # The arm moves with the carriage, so the held target is now stale.
        self._seed_target()

    # ---- discrete actions -------------------------------------------------
    def _cycle_mode(self) -> None:
        self.mode = Mode.next(self.mode)
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("spacemouse_mode", {"mode": self.mode})
        print(f"[SM] mode -> {self.mode}")

    def _toggle_gripper(self) -> None:
        if self.gripper_closed:
            self.arm.open_lite6_gripper()
            self.gripper_closed = False
        else:
            self.arm.close_lite6_gripper()
            self.gripper_closed = True
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("gripper", {"closed": self.gripper_closed})
        print(f"[SM] gripper {'closed' if self.gripper_closed else 'open'}")

    def _toggle_recording(self) -> None:
        if self.rec is None:
            print("[SM] recording disabled (--no-record); ignored.")
            return
        if not self.rec.is_recording:
            self.rec.start()
            # Mode is part of how to read the trace; log it at take start so a
            # replayed session is interpretable without guessing.
            self.rec.log_command("spacemouse_mode", {"mode": self.mode})
            print("[SM] recording STARTED")
        else:
            path = self.rec.stop(kept=True, task_label=self.task_label)
            print(f"[SM] recording STOPPED -> {path}")
            if self._recorder_factory is not None:
                self.rec = self._recorder_factory()

    def _reset_scene(self) -> None:
        self.arm.reset_scene()
        self.gripper_closed = False
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("reset_scene", {})
        # The arm has teleported; the integrator must follow it.
        self.resync()
        with self.arm.lock:
            self.rail_mm = float(
                self.arm.data.ctrl[self.arm.act_ids[RAIL_ACT]]) * 1000.0
        print("[SM] scene reset")

    def _go_home(self) -> None:
        rc = self.arm.go_home(wait=False)
        if self.rec is not None and self.rec.is_recording:
            self.rec.log_command("gohome", {"rc": rc})
        self.resync()
        print(f"[SM] home (rc={rc})")

    # ---- HUD --------------------------------------------------------------
    def status(self) -> dict:
        """Same shape as TeleopReceiver.status() so HUD consumers are shared.

        `clutch` is reported True whenever the jog is live. The SpaceMouse has
        no clutch, but the key must exist and mean "is the arm following" for
        an existing consumer to keep working.
        """
        rec_on = bool(self.rec is not None and self.rec.is_recording)
        return {
            "type": "status",
            "recording": rec_on,
            "gripper_closed": bool(self.gripper_closed),
            "rail_mm": round(float(self.rail_mm), 1),
            "ik_fail": bool(self.ik_fail),
            "clutch": True,
            "servo_mode": config.SERVO_MODE,
            # SpaceMouse-specific extras; additive, so VR consumers ignore them.
            "mode": self.mode,
            "device_model": self.model,
        }


def _rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Small-angle-safe XYZ rotation increment from a roll/pitch/yaw triple.

    Used for the per-tick rotation delta, which at 50 Hz is a fraction of a
    degree, so composition order barely matters -- but it is written out
    explicitly rather than approximated so that a large dt (a stalled loop,
    a test feeding a whole second) stays a valid rotation.
    """
    r, p, y = (float(v) for v in rpy_rad)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx
