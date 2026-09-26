# teleop_sm/device.py
"""spnav event loop -> normalised six-axis state + button edges.

`SpaceMouseDevice` owns the connection to **spacenavd** (the daemon), reached
through **libspnav** (the client library) via the `spnav` Python binding. It
runs its own thread: the thread writes the latest state, the control loop
reads it, and the two are separated by a lock -- the same pattern
`vr/teleop_receiver.py` uses for WebXR input.

WHY A DAEMON AND NOT HID. `spnav` is not a HID library; it is a client of
spacenavd over a Unix socket. If the daemon is not running there is nothing to
talk to, and `spnav_open()` fails. That is worth knowing because the failure
looks like a Python problem and is not -- see `_open()` below, which says so.

SIX AXES, NOT THREE. The vendor's LeRobot teleop is a 2-DOF planar jog: it
comments out rotation, forces `dpos[2] = 0`, and hardcodes the gripper open.
That is enough to push a T-block around a table and nothing else. This exposes
all six, because the twin has six to drive.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from teleop_sm import config

#: Order of the six axes everywhere in this package.
AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


class SpaceMouseError(RuntimeError):
    """Device or daemon could not be reached. Carries the remedy, not a trace."""


class SpaceMouseDevice(threading.Thread):
    """Background reader for one SpaceMouse.

    ``state()`` and ``state_twin()`` return a copy of the latest six-axis
    reading; ``button_edge()`` consumes a rising edge; ``button_held()`` is a
    level read. All are safe to call from the control thread.

    ``fake_events`` exists for tests: pass an iterable of event objects and no
    daemon is contacted, so the whole package is testable with no hardware
    attached (work order A.5).
    """

    def __init__(self, fake_events=None, poll_sleep_s: float = 0.002):
        super().__init__(daemon=True, name="spacemouse")
        self._lock = threading.Lock()
        self._axes = np.zeros(6, dtype=float)       # raw counts
        self._buttons_down: set[int] = set()
        self._edges: set[int] = set()
        self._stop = threading.Event()
        self._fake = list(fake_events) if fake_events is not None else None
        self._poll_sleep_s = float(poll_sleep_s)
        self._spnav = None
        self._last_event_t: float = 0.0
        self.event_count: int = 0

    # -- lifecycle ----------------------------------------------------------
    def _open(self):
        """Connect to spacenavd, or raise with the actual fix.

        Three distinct failures with three distinct remedies, and a bare
        traceback points at none of them.
        """
        try:
            import spnav
        except OSError as exc:
            raise SpaceMouseError(
                f"cannot load the spnav client library ({exc}).\n"
                f"  sudo apt install libspnav0 libspnav-dev\n"
                f"The `spacenavd` package is the DAEMON; libspnav is the client\n"
                f"library the Python binding dlopen()s. Installing the daemon\n"
                f"alone is NOT enough -- this bites everyone once.") from exc
        except ImportError as exc:
            raise SpaceMouseError(
                f"the spnav Python binding is missing ({exc}).\n"
                f'  pip install "spnav @ git+https://github.com/kazoo-osaro/spnav"'
            ) from exc

        try:
            spnav.spnav_open()
        except Exception as exc:  # noqa: BLE001 - binding raises its own type
            raise SpaceMouseError(
                f"spnav_open() failed ({type(exc).__name__}: {exc}).\n"
                f"  systemctl status spacenavd      # must be active\n"
                f"  sudo systemctl enable --now spacenavd") from exc
        return spnav

    def run(self) -> None:
        if self._fake is not None:
            self._run_fake()
            return

        self._spnav = self._open()
        from spnav import SpnavButtonEvent, SpnavMotionEvent

        try:
            while not self._stop.is_set():
                ev = self._spnav.spnav_poll_event()
                if ev is None:
                    time.sleep(self._poll_sleep_s)
                    continue
                self._ingest(ev, SpnavMotionEvent, SpnavButtonEvent)
        finally:
            try:
                self._spnav.spnav_close()
            except Exception:  # noqa: BLE001 - shutdown path, nothing to do
                pass

    def _run_fake(self) -> None:
        """Replay recorded events (from `spacemouse_probe.py --dump`)."""
        for ev in self._fake:
            if self._stop.is_set():
                break
            self._ingest(ev, _FakeMotionEvent, _FakeButtonEvent)
            time.sleep(self._poll_sleep_s)

    def _ingest(self, ev, motion_cls, button_cls) -> None:
        """Fold one event into the shared state. Runs on the reader thread."""
        if isinstance(ev, motion_cls):
            axes = np.array(list(ev.translation) + list(ev.rotation), dtype=float)
            with self._lock:
                self._axes = axes
                self._last_event_t = time.time()
                self.event_count += 1
        elif isinstance(ev, button_cls):
            with self._lock:
                if ev.press:
                    # Only a transition counts as an edge; spacenavd can repeat
                    # a press, and a repeat must not fire the action twice.
                    if ev.bnum not in self._buttons_down:
                        self._edges.add(ev.bnum)
                    self._buttons_down.add(ev.bnum)
                else:
                    self._buttons_down.discard(ev.bnum)
                self._last_event_t = time.time()
                self.event_count += 1

    def stop(self) -> None:
        self._stop.set()

    # -- state --------------------------------------------------------------
    def state(self) -> np.ndarray:
        """Six axes normalised to about [-1, 1], deadzoned, device frame.

        Clipped because a hard shove can exceed `MAX_VALUE`; the caller
        multiplies this by a speed cap, so an un-clipped 1.4 would quietly
        exceed the cap the operator set.
        """
        with self._lock:
            raw = self._axes.copy()
        v = np.clip(raw / config.MAX_VALUE, -1.0, 1.0)
        v[np.abs(v) < config.DEADZONE] = 0.0
        return v

    def state_twin(self) -> np.ndarray:
        """`state()` permuted into the twin world frame.

        Translation and rotation triples are permuted separately by the same
        matrix, then per-axis signs are applied.

        THE PERMUTATION IS PROVISIONAL -- see `config.AXIS_PERMUTATION`. Push
        the cap forward and confirm the TCP moves +X before trusting it.
        """
        v = self.state()
        m = config.AXIS_PERMUTATION
        out = np.empty(6, dtype=float)
        out[:3] = m @ v[:3]
        out[3:] = m @ v[3:]
        return out * config.AXIS_SIGNS

    # -- buttons ------------------------------------------------------------
    def button_edge(self, idx: int) -> bool:
        """True once per physical press. Consumes the edge."""
        with self._lock:
            if idx in self._edges:
                self._edges.discard(idx)
                return True
            return False

    def button_held(self, idx: int) -> bool:
        with self._lock:
            return idx in self._buttons_down

    def pending_edges(self) -> set[int]:
        """Drain every pending edge at once (the receiver's per-tick read)."""
        with self._lock:
            out = set(self._edges)
            self._edges.clear()
            return out

    def seconds_since_event(self) -> float:
        with self._lock:
            if self._last_event_t == 0.0:
                return float("inf")
            return time.time() - self._last_event_t


# ---------------------------------------------------------------------------
# test doubles -- importable so tests never touch spnav
# ---------------------------------------------------------------------------
class _FakeMotionEvent:
    """Mirrors `spnav.SpnavMotionEvent`'s attribute surface."""

    def __init__(self, translation, rotation):
        self.translation = tuple(translation)
        self.rotation = tuple(rotation)


class _FakeButtonEvent:
    """Mirrors `spnav.SpnavButtonEvent`'s attribute surface."""

    def __init__(self, bnum: int, press: bool):
        self.bnum = int(bnum)
        self.press = bool(press)


def events_from_dump(path: str) -> list:
    """Rebuild fake events from `spacemouse_probe.py --dump` JSONL.

    This is how a real device session becomes a test fixture: record once with
    the hardware, replay for ever without it.
    """
    import json

    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "motion":
                out.append(_FakeMotionEvent(rec["translation"], rec["rotation"]))
            elif rec.get("type") == "button":
                out.append(_FakeButtonEvent(rec["bnum"], rec["press"]))
    return out


def warn_if_multiple_devices() -> Optional[str]:
    """Return a warning if more than one SpaceMouse is on the USB bus.

    spacenavd binds one device set; with two plugged in, which one drives the
    arm is not something the operator chose. Detect and say so rather than
    silently using the wrong one (work order A.2).

    Best-effort: reads lsusb, and returns None if that is unavailable.
    """
    import subprocess

    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:  # noqa: BLE001 - diagnostics only, never fatal
        return None

    hits = [ln for ln in out.splitlines()
            if "3dconnexion" in ln.lower() or "spacemouse" in ln.lower()]
    if len(hits) > 1:
        return ("more than one 3Dconnexion device is plugged in:\n  "
                + "\n  ".join(hits)
                + "\nspacenavd binds one of them and there is no way here to "
                  "choose; unplug the one you are not driving.")
    return None
