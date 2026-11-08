"""A stand-in for the MaxArm's ESP32, speaking the MicroPython raw REPL.

It runs the code the backend sends, so a test exercises the real protocol and
the real command text, not a mock of either. Board modules (`BusServo`,
`SuctionNozzle`, `ujson`) are replaced with models of what the vendor's do:

* `BusServo.run` moves a servo straight to the pulse (no time passes in
  tests). `stuck` servos ignore it, and `dead` servos fail to answer reads.
* `SuctionNozzle` records pump and nozzle commands.
* `machine.Timer` models the ESP32's hardware timers by id. `stock_loop=True`
  leaves the stock main.py's `tim` (Timer 2) running, as when it finished
  before the backend's interrupt; `timer_stuck` timers ignore deinit.

`board.calls` logs every hardware-level action, so a test can assert that a
refused move sent NOTHING to the servos.
"""
from __future__ import annotations

import builtins
import io
import json
import traceback

from .transport import CTRL_A, CTRL_B, CTRL_C, CTRL_D, RAW_BANNER


class FakeBoard:
    def __init__(self, pulses=(500, 500, 500), *, stuck=(), dead=(),
                 sag_pulses=0, missing_modules=(), micropython=True,
                 stock_loop=False, timer_stuck=(), lag_once=(), deaf_writes=0):
        self.pos = {1: pulses[0], 2: pulses[1], 3: pulses[2]}
        self.stuck = set(stuck)
        self.lag_once = set(lag_once)
        self.deaf_writes = deaf_writes    # the first N writes are ignored (a silent board)     # stop halfway on their next move, then behave
        self.dead = set(dead)
        self.sag = sag_pulses
        self.missing = set(missing_modules)
        self.micropython = micropython
        self.timers = {0: False, 1: True, 2: stock_loop, 3: True}  # running?
        self.timer_stuck = set(timer_stuck)
        self.calls: list[tuple] = []
        self.out = b""
        self.raw = False
        self.closed = False
        self._code = b""
        self.ns: dict = {}
        self._reset_ns()

    # -- link interface (what SerialLink offers) -------------------------------
    def write(self, data: bytes) -> None:
        if self.deaf_writes > 0:
            self.deaf_writes -= 1
            return
        for b in data:
            self._byte(bytes([b]))

    def read_available(self) -> bytes:
        out, self.out = self.out, b""
        return out

    def close(self) -> None:
        self.closed = True

    # -- raw REPL --------------------------------------------------------------
    def _byte(self, b: bytes) -> None:
        if not self.micropython:
            self.out += b"\x00?"          # an Arduino sketch: no REPL at all
            return
        if b == CTRL_C:
            self.calls.append(("interrupt",))
            self._code = b""
        elif b == CTRL_A:
            self.raw = True
            self._code = b""
            self.out += RAW_BANNER
        elif b == CTRL_B:
            self.raw = False
            self.calls.append(("exit_raw",))
        elif self.raw and b == CTRL_D:
            self._run(self._code.decode())
            self._code = b""
        elif self.raw:
            self._code += b

    def _run(self, code: str) -> None:
        stdout, err = io.StringIO(), ""
        self.ns["print"] = lambda *a, **k: print(*a, file=stdout, **k)
        try:
            exec(code, self.ns)
        except Exception:                 # noqa: BLE001 - that is the point
            err = traceback.format_exc()
        self.out += b"OK" + stdout.getvalue().encode() + CTRL_D + err.encode() + CTRL_D + b">"

    # -- board modules -----------------------------------------------------------
    def _reset_ns(self) -> None:
        board = self

        class BusServo:
            def run(self, i, p, t=1000):
                board.calls.append(("run", i, p, t))
                if i in board.lag_once:
                    board.lag_once.discard(i)
                    board.pos[i] = (board.pos[i] + p) // 2
                elif i not in board.stuck:
                    board.pos[i] = p + (board.sag if i == 2 else 0)

            def get_position(self, i):
                if i in board.dead:
                    return False
                return int(board.pos[i])

            def get_vin(self, i):
                return False if i in board.dead else 7400

            def unload(self, i):
                board.calls.append(("unload", i))

        class SuctionNozzle:
            def on(self):
                board.calls.append(("pump_on",))

            def off(self):
                board.calls.append(("pump_off",))

            def set_angle(self, angle=0, duration=1000):
                board.calls.append(("nozzle", angle, duration))

        class Timer:
            def __init__(self, i):
                self.i = i

            def deinit(self):
                board.calls.append(("timer_deinit", self.i))
                if self.i not in board.timer_stuck:
                    board.timers[self.i] = False

            def __repr__(self):
                on = int(board.timers[self.i])
                return f"Timer(3ffe0000; alarm_en={on}, auto_reload=1, counter_en={on})"

        mods = {"ujson": type("m", (), {"dumps": staticmethod(json.dumps)}),
                "machine": type("m", (), {"Timer": Timer}),
                "BusServo": type("m", (), {"BusServo": BusServo}),
                "SuctionNozzle": type("m", (), {"SuctionNozzle": SuctionNozzle})}

        def imp(name, *a, **k):
            if name in board.missing:
                raise ImportError(f"no module named '{name}'")
            if name in mods:
                return mods[name]
            return builtins.__import__(name, *a, **k)

        b = dict(vars(builtins))
        b["__import__"] = imp
        self.ns = {"__builtins__": b}
        if self.timers[2]:
            self.ns["tim"] = Timer(2)

    def motion_calls(self) -> list[tuple]:
        return [c for c in self.calls if c[0] in ("run", "nozzle", "unload")]
