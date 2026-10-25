# teleop_hand/keys.py
"""Keys for hand teleop.

HELD keys (clutch, precision) come from pynput, because holding needs key
RELEASE events and MuJoCo's viewer key callback delivers presses only. pynput
listens globally (whatever window has focus). A USB foot pedal enumerates as
a keyboard, so `--clutch pedal` is the same code with a different key.

DISCRETE keys (record, discard, home, reset, rail step) go through the
viewer's `register_key_handler`, like kinesthetic teaching, and only QUEUE
actions -- home and reset block for a paced move.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from teleop_hand import config
from teleop_hand.receiver import HandAction

#: Autorepeat on X11 arrives as release/press pairs ~30 ms apart while a key is
#: held; a release only counts once it has lasted this long.
RELEASE_DEBOUNCE_S = 0.12

VIEWER_KEYMAP: Dict[int, HandAction] = {
    config.GLFW_INSERT: HandAction.RECORD_TOGGLE,
    config.GLFW_DELETE: HandAction.DISCARD_TAKE,
    config.GLFW_HOME: HandAction.HOME,
    config.GLFW_END: HandAction.RESET_SCENE,
    config.GLFW_LEFT: HandAction.RAIL_MINUS,
    config.GLFW_RIGHT: HandAction.RAIL_PLUS,
}


def make_viewer_handler(post: Callable[[HandAction], None]) -> Callable[[int], bool]:
    def handler(keycode: int) -> bool:
        action = VIEWER_KEYMAP.get(keycode)
        if action is None:
            return False
        post(action)
        return True
    return handler


class HeldKeys:
    """Debounced held-key state from pynput. `pressed(name)` is thread-safe.
    `on_tap(name, fn)` fires fn once per press (for the recovery key)."""

    def __init__(self, names, clock=time.monotonic):
        self.names = set(names)
        self.clock = clock
        self._down: Dict[str, bool] = {n: False for n in self.names}
        self._released_at: Dict[str, Optional[float]] = {n: None for n in self.names}
        self._taps: Dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self.listener = None

    # ---- events (also callable directly from tests) ---------------------------
    def press(self, name: str) -> None:
        fire = None
        with self._lock:
            if name not in self.names:
                return
            was = self._down[name] and self._released_at[name] is None
            self._down[name] = True
            self._released_at[name] = None
            if not was:
                fire = self._taps.get(name)
        if fire:
            fire()

    def release(self, name: str) -> None:
        with self._lock:
            if name in self.names and self._down[name]:
                self._released_at[name] = self.clock()

    def pressed(self, name: str) -> bool:
        with self._lock:
            if not self._down.get(name, False):
                return False
            t = self._released_at[name]
            if t is not None and self.clock() - t >= RELEASE_DEBOUNCE_S:
                self._down[name] = False
                self._released_at[name] = None
                return False
            return True

    def on_tap(self, name: str, fn: Callable[[], None]) -> None:
        self.names.add(name)
        self._down.setdefault(name, False)
        self._released_at.setdefault(name, None)
        self._taps[name] = fn

    # ---- pynput -------------------------------------------------------------
    def start(self) -> bool:
        try:
            from pynput import keyboard
        except Exception as e:                       # noqa: BLE001
            print(f"[HAND] pynput unavailable ({e})")
            return False

        def name_of(key) -> Optional[str]:
            if isinstance(key, keyboard.Key):
                return key.name
            ch = getattr(key, "char", None)
            return ch.lower() if ch else None

        try:
            self.listener = keyboard.Listener(
                on_press=lambda k: self.press(name_of(k)),
                on_release=lambda k: self.release(name_of(k)))
            self.listener.start()
            self.listener.wait()
        except Exception as e:                       # noqa: BLE001
            print(f"[HAND] keyboard listener failed ({e})")
            self.listener = None
            return False
        return True

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()


def describe(clutch: str, clutch_key: str, precision_key: str, rail_mode: str,
             real: bool = False) -> str:
    c = {"key": f"HOLD {clutch_key}", "pedal": f"HOLD pedal ({clutch_key})",
         "gesture": "LEFT FIST (non-controlling hand)",
         "always": "ALWAYS ENGAGED (test mode)"}[clutch]
    lines = [
        f"  clutch       {c}  -- nothing moves without it",
        f"  precision    HOLD {precision_key}  (x{config.PRECISION_POS_FACTOR} position, "
        f"x{config.PRECISION_ROT_FACTOR} rotation)",
        "  gripper      pinch thumb and index (while clutched)",
        "  Insert       start / stop a take     Delete  discard take",
    ]
    if not real:
        lines.append("  Home / End   home (re-anchor) / reset scene")
        if rail_mode == "keys":
            lines.append(f"  Left/Right   rail -/+ {config.RAIL_STEP_MM:.0f} mm (clutch released)")
    else:
        lines.append(f"  {config.RECOVER_KEY.upper():12s} recover from a controller fault "
                     f"(then a fresh clutch press)")
    return "\n".join(lines)
