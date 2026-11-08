"""Talk to the MaxArm's ESP32 through its MicroPython raw REPL.

Why the REPL and not the vendor's binary protocol (docs Chapter 10)
-------------------------------------------------------------------
The binary protocol needs a vendor "slave" program flashed to the board, and
it has no acknowledgement for a move. `ESPMax.set_position` returns False on
an IK failure, and the slave's `deal_command` discards it. So a refused move
looks exactly like an executed one. The REPL returns the real return value and
the real exception text, and needs nothing installed on the board.

The raw REPL protocol (MicroPython's own, as used by mpremote/pyboard.py):

    host: CTRL-C CTRL-C        interrupt whatever is running (the stock program)
    host: CTRL-A               enter raw REPL
    board: "raw REPL; CTRL-B to exit\\r\\n>"
    host: <code> CTRL-D        execute
    board: "OK" <stdout> 0x04 <stderr> 0x04 ">"
    host: CTRL-B               leave raw REPL

Opening the port reboots the board (measured on Linux, 2026-11-07)
------------------------------------------------------------------
ESP32 boards wire DTR/RTS to EN/IO0 for auto-reset. pyserial asserts both on
open by default, so `SerialLink` sets dtr/rts False BEFORE opening
(`test_maxarm.check_serial_opened_without_reset` pins that ordering). On the
lab unit (CH340) the board STILL reboots on every open: the Linux tty driver
raises the lines itself before pyserial can lower them. The stock program then
runs (and may move the arm home), so `RawRepl.enter` waits for its `>>> `
prompt before interrupting.

Each command prints one line `@@<json>` built by the on-board helper `_r`
(see `BOARD_PRELUDE` in real_maxarm.py). Anything else printed is passed
through as log text, so vendor `print()` noise cannot be mistaken for a result.
"""
from __future__ import annotations

import json
import time

CTRL_A, CTRL_B, CTRL_C, CTRL_D = b"\x01", b"\x02", b"\x03", b"\x04"
RAW_BANNER = b"raw REPL; CTRL-B to exit\r\n>"
RESULT_TAG = "@@"


class TransportError(RuntimeError):
    """The link failed: the board is gone, wedged, or not running MicroPython."""


class BoardError(RuntimeError):
    """Code ran on the board and raised. `.traceback` is the board's text."""

    def __init__(self, traceback: str):
        self.traceback = traceback
        lines = [ln for ln in traceback.strip().splitlines() if ln.strip()]
        super().__init__(lines[-1] if lines else "board raised with no message")


class SerialLink:
    """A pyserial port, opened with the reset lines deasserted.

    That is not enough on Linux: the lab unit still reboots on open (see
    RawRepl._wait_for_boot). Deasserting keeps the lines from holding it in reset.
    """

    def __init__(self, port: str, baudrate: int = 115200, serial_module=None):
        if serial_module is None:
            import serial as serial_module
        s = serial_module.Serial()
        s.port = port
        s.baudrate = baudrate
        s.timeout = 0.05
        # Deassert BEFORE open: on ESP32 auto-reset wiring, the default
        # (asserted) state on open can pulse EN and reboot the board.
        s.dtr = False
        s.rts = False
        s.open()
        self._s = s

    def write(self, data: bytes) -> None:
        self._s.write(data)
        self._s.flush()

    def read_available(self) -> bytes:
        n = self._s.in_waiting
        return self._s.read(n or 1)

    def close(self) -> None:
        self._s.close()


class RawRepl:
    """Execute code on the board; returns stdout, raises BoardError on error."""

    def __init__(self, link, timeout_s: float = 5.0):
        self.link = link
        self.timeout_s = timeout_s
        self._buf = b""
        self.in_raw = False
        self.log: list[str] = []   # non-result lines the board printed

    # -- low level ----------------------------------------------------------
    def _read_until(self, marker: bytes, timeout_s: float) -> bytes:
        deadline = time.monotonic() + timeout_s
        while marker not in self._buf:
            if time.monotonic() > deadline:
                got, self._buf = self._buf, b""
                raise TransportError(
                    f"timed out after {timeout_s:.1f}s waiting for {marker!r}; "
                    f"board sent {got[-200:]!r}")
            self._buf += self.link.read_available()
        head, _, self._buf = self._buf.partition(marker)
        return head

    def _wait_for_boot(self, listen_s: float = 0.3, boot_s: float = 8.0) -> None:
        """If the board is booting, wait for its friendly prompt.

        Opening the port reboots the lab unit (CH340 + ESP32) whatever SerialLink
        asks: Linux raises DTR/RTS on open before pyserial can lower them. The ROM
        prints at once, then goes silent ~2.8 s before the stock program's
        "Please wait... Start" and the `>>> ` prompt (~3.2 s). Keys sent in that
        window are lost, so a quiet line does not mean ready; wait for the prompt.
        """
        deadline = time.monotonic() + listen_s
        seen = b""
        while time.monotonic() < deadline:
            seen += self.link.read_available()
        if b"rst:" not in seen and b"ets " not in seen:
            return                               # not booting: interrupt as before
        deadline = time.monotonic() + boot_s
        while b">>> " not in seen and time.monotonic() < deadline:
            seen += self.link.read_available()

    def enter(self, attempts: int = 2) -> None:
        """Interrupt the running program and enter the raw REPL.

        Tried twice: twice on 2026-11-09 the lab unit sent nothing at all for
        3 s after the first interrupt (no boot text, so not a reboot), and the
        same keys sent again were answered at once.
        """
        self._wait_for_boot()
        for attempt in range(1, attempts + 1):
            self.link.write(b"\r" + CTRL_C + CTRL_C)
            time.sleep(0.2)
            self._buf = b""
            self.link.read_available()           # drop whatever was in flight
            self.link.write(b"\r" + CTRL_A)
            try:
                self._read_until(RAW_BANNER, 3.0)
                break
            except TransportError as exc:
                if attempt == attempts:
                    raise TransportError(
                        "no MicroPython raw-REPL banner. Either the board is not running "
                        "MicroPython (Arduino firmware?), or this is the wrong port. "
                        f"Detail: {exc}") from None
                print("[MaxArm] no raw-REPL banner; interrupting again")
        self.in_raw = True

    def exit(self) -> None:
        if self.in_raw:
            self.link.write(CTRL_B)
            self.in_raw = False

    def exec(self, code: str, timeout_s: float | None = None) -> str:
        if not self.in_raw:
            raise TransportError("exec before enter()")
        t = self.timeout_s if timeout_s is None else timeout_s
        self.link.write(code.encode() + CTRL_D)
        ack = self._read_until(b"OK", 2.0)
        if ack.strip(b">\r\n "):
            raise TransportError(f"board did not accept the code: {ack!r}")
        out = self._read_until(CTRL_D, t)
        err = self._read_until(CTRL_D, t)
        self._read_until(b">", 1.0)
        if err.strip():
            raise BoardError(err.decode(errors="replace"))
        return out.decode(errors="replace")

    # -- result protocol ----------------------------------------------------
    def call(self, code: str, timeout_s: float | None = None):
        """Run code that ends with exactly one `_r(...)` and return its value.

        Raises BoardError if the call raised on the board. A missing or
        duplicated result line is a TransportError: it means the protocol is
        out of step, and guessing which line was meant would be worse.
        """
        out = self.exec(code, timeout_s)
        results = []
        for line in out.splitlines():
            if line.startswith(RESULT_TAG):
                results.append(json.loads(line[len(RESULT_TAG):]))
            elif line.strip():
                self.log.append(line)
        if len(results) != 1:
            raise TransportError(f"expected one result line, got {len(results)}: "
                                 f"{out[-300:]!r}")
        res = results[0]
        if not res.get("ok"):
            raise BoardError(res.get("e", "unknown board error"))
        return res.get("v")
