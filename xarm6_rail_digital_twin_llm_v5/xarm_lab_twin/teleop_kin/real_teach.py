# teleop_kin/real_teach.py
"""Kinesthetic teaching on the REAL arm: manual mode, hold-to-teach, and a
live mirror of the arm into the twin so the existing Recorder records it.

DESIGN: MIRROR, DON'T FORK THE RECORDER. The controller's joint angles are
written into the twin's `ctrl` AND `qpos` every tick. `ctrl` is what
export_lerobot.py reads as the action, so the session needs no new channel;
`qpos` keeps the twin's state from lagging its actuators. A real take is then
an ordinary Recorder session with `interface == "kinesthetic_real"`.

SAFETY INVARIANT: every exit path returns the controller to mode 0 / state 0.
`ManualMode.safe_exit()` is idempotent and is called from the runner's
`finally:`, its SIGTERM handler, and on any controller error. Nothing in this
module commands motion; the only outbound commands are the mode switch and
the gripper.

Everything here takes the controller as a parameter, so `FakeController` /
`FakeBackend` at the bottom exercise it without hardware (tests, and the
runner's `--fake` rehearsal mode).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

from sim.mujoco_env import RAIL_ACT
from teleop_kin import config

MODE_POSITION = 0
MODE_MANUAL = 2


# ---------------------------------------------------------------------------
# hold-to-teach key
# ---------------------------------------------------------------------------
class TeachHold:
    """Debounced press/release of the hold-to-teach key.

    `press()` / `release()` are called from the keyboard thread; `poll()` from
    the control loop returns "enter", "exit" or None. A release only counts
    once the key has stayed up for `debounce_s`: X11 autorepeat turns a held
    key into release/press pairs a few ms apart, and without the debounce the
    arm would be switched in and out of manual mode ~30 times a second.
    """

    def __init__(self, debounce_s: float = config.TEACH_RELEASE_DEBOUNCE_S,
                 clock: Callable[[], float] = time.monotonic):
        self.debounce_s = float(debounce_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._down = False
        self._released_at: Optional[float] = None
        self.engaged = False          # what the control loop last acted on

    def press(self) -> None:
        with self._lock:
            self._down = True
            self._released_at = None

    def release(self) -> None:
        with self._lock:
            if self._down:
                self._down = False
                self._released_at = self._clock()

    def toggle(self) -> None:
        """For toggle-to-teach (no key-release events available). No debounce:
        a toggle is a deliberate single press."""
        with self._lock:
            self._down = not self._down
            self._released_at = None

    def poll(self) -> Optional[str]:
        with self._lock:
            want = self._down or (
                self._released_at is not None
                and self._clock() - self._released_at < self.debounce_s)
        if want and not self.engaged:
            self.engaged = True
            return "enter"
        if not want and self.engaged:
            self.engaged = False
            return "exit"
        return None


# ---------------------------------------------------------------------------
# manual mode
# ---------------------------------------------------------------------------
def controller_mode(ctl) -> Optional[int]:
    """The controller's current mode. XArmAPI exposes it as the `mode`
    property (refreshed from the report socket); a `get_mode()` method is
    preferred where an SDK build has one."""
    fn = getattr(ctl, "get_mode", None)
    if callable(fn):
        ret = fn()
        if isinstance(ret, (tuple, list)):
            return int(ret[-1]) if ret and ret[0] == 0 else None
        return int(ret)
    m = getattr(ctl, "mode", None)
    return None if m is None else int(m)


class ManualMode:
    """Enter/leave the controller's manual (teach) mode.

    enter:  set_mode(2), set_state(0); if the controller does not report mode 2
            within `timeout_s`, retry once with set_mode(2, 1) -- the variant
            some Lite 6 Studio code uses.
    exit:   set_mode(0), set_state(0). Idempotent, never raises.
    """

    def __init__(self, ctl, timeout_s: float = config.TEACH_MODE_TIMEOUT_S,
                 log: Callable[[str], None] = print):
        self.ctl = ctl
        self.timeout_s = float(timeout_s)
        self.log = log
        self.active = False
        self.used_fallback = False

    def _wait_mode(self, want: int) -> bool:
        t_end = time.monotonic() + self.timeout_s
        while time.monotonic() < t_end:
            if controller_mode(self.ctl) == want:
                return True
            time.sleep(0.02)
        return controller_mode(self.ctl) == want

    def enter(self) -> bool:
        # Marked active BEFORE the call: if set_mode succeeds and the next line
        # raises, safe_exit must still know to leave manual mode.
        self.active = True
        self.ctl.set_mode(MODE_MANUAL)
        self.ctl.set_state(0)
        if self._wait_mode(MODE_MANUAL):
            return True
        self.log("[TEACH] mode 2 not reported; retrying with set_mode(2, 1)")
        self.ctl.set_mode(MODE_MANUAL, 1)
        self.ctl.set_state(0)
        if self._wait_mode(MODE_MANUAL):
            self.used_fallback = True
            return True
        self.log(f"[TEACH] controller still reports mode "
                 f"{controller_mode(self.ctl)} -- NOT in manual mode")
        self.safe_exit()
        return False

    def exit(self) -> None:
        self.safe_exit()

    def safe_exit(self) -> None:
        """mode 0 + state 0. Safe to call any number of times, from anywhere,
        including with a dead connection (errors are logged, not raised)."""
        return_to_position_mode(self.ctl, self.log, "[TEACH]")
        self.active = False


def return_to_position_mode(ctl, log=print, tag: str = "") -> bool:
    """set_mode(0) + set_state(0), never raising. Every exit path of every
    real-arm teleop (kinesthetic, hand tracking) ends here. Returns False if
    the controller could not be reached."""
    try:
        ctl.set_mode(MODE_POSITION)
        ctl.set_state(0)
        return True
    except Exception as e:          # the finally: path must not raise
        log(f"{tag} WARNING: could not return the controller to mode 0: {e}")
        return False


# ---------------------------------------------------------------------------
# mirror
# ---------------------------------------------------------------------------
class RealMirror:
    """Copies the real arm's joints (and rail) into the twin each tick."""

    def __init__(self, arm, backend, use_rail: bool = True):
        self.arm = arm
        self.backend = backend
        self.use_rail = bool(use_rail)
        m = arm.model
        self._qadr = [int(m.jnt_qposadr[j]) for j in arm.joint_ids]
        self._dadr = [int(m.jnt_dofadr[j]) for j in arm.joint_ids]
        self._rail_qadr = int(m.jnt_qposadr[arm.rail_jid])
        self._rail_dadr = int(m.jnt_dofadr[arm.rail_jid])
        self._rail_m: Optional[float] = None
        self._rail_read_t = 0.0
        self.last_joints_deg: Optional[np.ndarray] = None
        self.n_reads = 0

    def read_joints_deg(self) -> np.ndarray:
        code, angles = self.backend.get_servo_angle()
        if code != 0 or angles is None:
            raise RuntimeError(f"get_servo_angle failed (code {code})")
        return np.asarray(angles[:6], dtype=float)

    def _maybe_read_rail(self, now: float) -> None:
        if not self.use_rail:
            return
        if self._rail_m is not None and now - self._rail_read_t < config.RAIL_POLL_PERIOD_S:
            return
        code, rail_mm = self.backend.get_rail_position()
        if code == 0 and rail_mm is not None:
            self._rail_m = float(np.clip(rail_mm / 1000.0, 0.0, 0.7))
        self._rail_read_t = now

    def sync(self) -> np.ndarray:
        """One mirror step. Returns the joint angles written (degrees)."""
        q_deg = self.read_joints_deg()
        self._maybe_read_rail(time.monotonic())
        q = np.deg2rad(q_deg)
        a = self.arm
        with a.lock:
            for i, (qa, da) in enumerate(zip(self._qadr, self._dadr)):
                a.data.ctrl[a.act_ids[1 + i]] = q[i]   # the exported action
                a.data.qpos[qa] = q[i]                 # the state
                a.data.qvel[da] = 0.0
            if self._rail_m is not None:
                a.data.ctrl[a.act_ids[RAIL_ACT]] = self._rail_m
                a.data.qpos[self._rail_qadr] = self._rail_m
                a.data.qvel[self._rail_dadr] = 0.0
            mujoco.mj_forward(a.model, a.data)
        self.last_joints_deg = q_deg
        self.n_reads += 1
        return q_deg


# ---------------------------------------------------------------------------
# gripper
# ---------------------------------------------------------------------------
class RealGripper:
    """Opens/closes the real gripper on a worker thread, and the twin's.

    A real gripper call with wait=True blocks for the jaw travel (~1 s); run
    on the control thread it would stall the 60 Hz mirror. So commands are
    queued, and `closed` (the recorded gripper channel, via
    `Recorder(gripper_fn=...)`) flips only when the controller has accepted
    the command AND the jaw has finished moving -- the nearest thing to jaw
    state this rig exposes without polling a position register.
    """

    def __init__(self, arm, open_fn: Callable[[], int], close_fn: Callable[[], int],
                 log: Callable[[str], None] = print,
                 on_done: Optional[Callable[[bool, int], None]] = None):
        self.arm = arm
        self._open, self._close = open_fn, close_fn
        self.log = log
        self.on_done = on_done
        self.closed = False
        self._q: "queue.Queue[Optional[bool]]" = queue.Queue()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def command(self, close: bool) -> None:
        # The twin follows immediately; its weld logic runs as in sim takes.
        if close:
            self.arm.close_lite6_gripper()
        else:
            self.arm.open_lite6_gripper()
        self._q.put(bool(close))

    def _worker(self) -> None:
        while True:
            close = self._q.get()
            if close is None:
                return
            try:
                rc = self._close() if close else self._open()
            except Exception as e:
                self.log(f"[TEACH] gripper command failed: {e}")
                rc = -1
            if rc == 0:
                self.closed = close
            self.log(f"[TEACH] gripper {'closed' if close else 'open'} (rc={rc})")
            if self.on_done is not None:
                self.on_done(close, rc)

    def state(self) -> float:
        return 1.0 if self.closed else 0.0

    def wait_idle(self, timeout: float = 5.0) -> None:
        t_end = time.monotonic() + timeout
        while not self._q.empty() and time.monotonic() < t_end:
            time.sleep(0.01)
        time.sleep(0.05)

    def stop(self) -> None:
        self._q.put(None)


# ---------------------------------------------------------------------------
# fakes (tests + `run_kinesthetic_real.py --fake` rehearsal)
# ---------------------------------------------------------------------------
class FakeController:
    """Stands in for xarm.wrapper.XArmAPI's mode/state surface.

    `needs_detection_param` mimics a Lite 6 that ignores plain set_mode(2).
    If `state_file` is given, every mode change is written there, so a test
    can read the final mode after killing the process that owned it.
    """

    def __init__(self, needs_detection_param: bool = False,
                 state_file: Optional[str] = None):
        self.mode = MODE_POSITION
        self.state = 0
        self.calls: list = []
        self.needs_detection_param = needs_detection_param
        self.state_file = Path(state_file) if state_file else None
        self._dump()

    def _dump(self) -> None:
        if self.state_file is not None:
            self.state_file.write_text(json.dumps({"mode": self.mode, "state": self.state}))

    def set_mode(self, mode, *args):
        self.calls.append(("set_mode", mode, *args))
        if mode == MODE_MANUAL and self.needs_detection_param and args != (1,):
            return 0      # accepted, but the joints stay locked (the Lite 6 case)
        self.mode = int(mode)
        self._dump()
        return 0

    def set_state(self, state):
        self.calls.append(("set_state", state))
        self.state = int(state)
        self._dump()
        return 0


class FakeBackend:
    """Stands in for RealXArmAPI: joint readback, rail, gripper."""

    def __init__(self, ctl: Optional[FakeController] = None, joints_deg=None,
                 rail_mm: Optional[float] = 350.0):
        self.arm = ctl or FakeController()
        self.joints_deg = np.array(
            joints_deg if joints_deg is not None else [0, -30, -25, 0, 55, 0], float)
        self.rail_mm = rail_mm
        self.gripper_calls: list = []
        self._rail_trustworthy = rail_mm is not None

    def get_servo_angle(self):
        return 0, list(self.joints_deg) + [0.0]     # SDK returns 7 values

    def get_rail_position(self):
        return (0, self.rail_mm) if self.rail_mm is not None else (-1, None)

    def open_gripper(self, wait: bool = True) -> int:
        self.gripper_calls.append("open")
        return 0

    def close_gripper(self, wait: bool = True) -> int:
        self.gripper_calls.append("close")
        return 0

    def disconnect(self) -> None:
        pass
