"""Hiwonder MaxArm (suction-cup kit, no rail) as an arm backend.

Implements the contract in `arm_backend.ARM_BACKEND_METHODS`. Everything the
MaxArm cannot do is declared with `@unsupported(reason)`. Nothing returns
success for a command that did not execute.

Coordinates are the ARM FRAME (see kinematics.py): mm, origin on the base axis
at the base plate, home at (0, -163, 212). There is no rail and no world
transform yet. When a cell frame is measured, it belongs here, as it does in
`RealXArmAPI`.

How a move works, and why
-------------------------
1. `kinematics.solve` checks the pose on the PC: reach, z ceiling, floor, and
   every servo limit, including the ones the vendor would silently clamp. A
   refusal raises before anything is sent.
2. The three bus-servo pulses are sent directly (`BusServo.run`). The vendor's
   `ESPMax.set_position` is deliberately NOT used. It re-solves IK on the board
   (a second copy of the kinematics, defect class #1) and clamps without saying
   so (class #2). Sending pulses means the PC port is the one source of truth
   for where the arm goes. `maxarm_probe.py --kinematics` checks the port
   against the board's solver.
3. With `wait=True`, the servos are read back after the move, and the reached
   position is computed from the READ pulses and compared with the target. A
   servo that stalls, a dead bus or a brown-out raises `MaxArmError`. It does
   not come back as rc=0 with the arm somewhere else.

The servos interpolate in JOINT space, so the tool path between two points is
not a straight line. Keep moves near the table short, or break them up.

Board side: nothing is installed. The vendor's `espmax.py`, `BusServo.py` and
`SuctionNozzle.py` must already be on the board's filesystem (the probe lists
them). Connecting interrupts whatever program is running, so the phone app
stops working until the board is power-cycled.
"""
from __future__ import annotations

import math
import threading
import time

from arm_backend import unsupported

from . import kinematics as kin
from .transport import BoardError, RawRepl, SerialLink, TransportError

EFFECTORS = ("suction",)

#: Default Cartesian speed, mm/s. The vendor's examples move ~100-150 mm/s.
DEFAULT_SPEED_MM_S = 50.0
#: Shortest move duration sent to the servos, ms.
MIN_DURATION_MS = 300
#: Longest the bus servos accept, ms (`BusServo.run` is uint16; PWMServo caps 30000).
MAX_DURATION_MS = 30000
#: Allowed gap between target and read-back position, mm. **PROVISIONAL**:
#: pulse quantisation alone is 0.24 deg, about 1 mm at full reach. Gravity sag
#: under a payload is unmeasured. Tighten it once measured.
# Measured 2026-11-07/09: unloaded read-back within 0.4-3.5 mm, then 6.0 mm (z short)
# returning to a high pose; holding a 40 mm cube, 6.4 mm (z 5.8 short) at z 150
# even after the re-send: steady gravity sag at reach, not a miss. A stalled
# servo or a blocked path is far larger.
ARRIVAL_TOL_MM = 10.0
# A miss is re-sent once before it is an error: back at a view pose after a
# place, an unloaded move read back 6.0 mm short in z (2026-11-09), the first
# over the tolerance. A servo that is really stalled or blocked misses twice.
RETRY_DURATION_MS = 500
#: Extra time after the commanded duration before reading back, s.
SETTLE_S = 0.25

WAVE_AMPLITUDE_DEG = 15.0

#: Executed once per connection. `_r` is the ONLY way results leave the board,
#: and it reports exceptions instead of letting a bare `except` hide them
#: (which is what the vendor's `ESPMax.set_position` does).
BOARD_PRELUDE = """\
import ujson as _j
def _r(f, *a):
    try:
        v = f(*a)
    except Exception as e:
        print('@@' + _j.dumps({'ok': False, 'e': repr(e)}))
    else:
        print('@@' + _j.dumps({'ok': True, 'v': v}))
from BusServo import BusServo as _B
from SuctionNozzle import SuctionNozzle as _N
try:
    _bs
except NameError:
    _bs = _B()
    _nz = _N()
def _pulses():
    out = []
    for i in (1, 2, 3):
        p = _bs.get_position(i)
        if p is False:
            p = _bs.get_position(i)
        out.append(p)
    return out
def _move(p1, p2, p3, t):
    _bs.run(1, p1, t)
    _bs.run(2, p2, t)
    _bs.run(3, p3, t)
    return True
def _vin():
    return [_bs.get_vin(i) for i in (1, 2, 3)]
from machine import Timer as _T
def _stop_stock_loop():
    t = globals().get('tim')
    if t is not None:
        t.deinit()
    _T(2).deinit()
    return repr(_T(2))
"""

#: The stock main.py ends with `tim = Timer(2); tim.init(period=15, callback=main)`:
#: its Bluetooth, USB-mouse and key handlers (key 1 runs action group 100) keep
#: moving the arm under our commands, and Ctrl-C does not stop a timer. Whether it
#: is running when we connect depends on timing: an interrupt that lands while
#: main.py is still starting Bluetooth aborts it first (measured 2026-11-07).
#: Timer 3 is PWMServo's (the nozzle's smooth moves) and stays.
STOCK_LOOP_TIMER = 2


class MaxArmError(RuntimeError):
    """The MaxArm could not do what was asked. Raised, never returned as a code."""


class RealMaxArmAPI:
    """The MaxArm over USB serial (MicroPython raw REPL)."""

    def __init__(self, port: str | None = None, *, link=None,
                 effector: str = "suction",
                 floor_z_mm: float = kin.DEFAULT_FLOOR_Z_MM,
                 sleep=time.sleep, baudrate: int = 115200):
        if effector not in EFFECTORS:
            raise ValueError(f"effector must be one of {EFFECTORS}, got {effector!r}")
        if (port is None) == (link is None):
            raise ValueError("pass exactly one of port= or link=")
        self.effector = effector
        self.floor_z_mm = floor_z_mm
        self._sleep = sleep
        self._lock = threading.RLock()
        self._nozzle_cmd_deg: float | None = None   # PWM servo: no readback
        self._suction_cmd = False

        self.link = link if link is not None else SerialLink(port, baudrate)
        self.repl = RawRepl(self.link)
        try:
            self.repl.enter()
            self.repl.exec(BOARD_PRELUDE)
        except BoardError as exc:
            raise MaxArmError(
                f"board prelude failed: {exc}. The vendor's BusServo.py / "
                f"SuctionNozzle.py must be on the board; run "
                f"scripts/maxarm_probe.py to list its files.") from exc
        self.stock_loop = self._stop_stock_loop()

        pulses = self._read_pulses()          # proves the servo bus answers
        self.connect_pulses = pulses
        try:
            self.supply_mv = self.repl.call("_r(_vin)")
        except (BoardError, TransportError):
            self.supply_mv = None

    # -- board I/O ------------------------------------------------------------

    def _call(self, code: str, what: str, timeout_s: float | None = None):
        with self._lock:
            try:
                return self.repl.call(code, timeout_s)
            except BoardError as exc:
                raise MaxArmError(f"{what}: the board raised {exc}") from exc

    def _stop_stock_loop(self) -> str:
        """Stop the stock program's control loop; refuse to connect if it runs on."""
        state = self._call("_r(_stop_stock_loop)", "stop the stock control loop")
        if "counter_en=0" not in str(state):
            raise MaxArmError(
                f"the stock program's control loop (Timer {STOCK_LOOP_TIMER}) is "
                f"still running after deinit ({state}); its Bluetooth/mouse/key "
                f"handlers would move the arm under our commands")
        return state

    def _read_pulses(self) -> list[int]:
        p = self._call("_r(_pulses)", "read servo positions")
        if not isinstance(p, list) or len(p) != 3 or any(v is False or v is None for v in p):
            raise MaxArmError(
                f"servo read-back failed ({p!r}). A servo did not answer on the "
                f"bus. Is the arm powered from its own supply, not just USB?")
        return [int(v) for v in p]

    def _send_pulses(self, pulses, duration_ms: int, what: str) -> None:
        bad = kin.check_pulses(pulses)
        if bad:
            raise MaxArmError(f"{what} refused: " + "; ".join(bad))
        p = [int(round(v)) for v in pulses]
        ok = self._call(f"_r(_move, {p[0]}, {p[1]}, {p[2]}, {int(duration_ms)})", what)
        if ok is not True:
            raise MaxArmError(f"{what}: board returned {ok!r}, not True")

    def _duration_ms(self, target_xyz, speed_mm_s: float) -> int:
        start = kin.pulses_to_xyz(self._read_pulses())
        dist = math.dist(start, target_xyz)
        ms = 1000.0 * dist / max(float(speed_mm_s), 1e-3)
        return int(min(max(ms, MIN_DURATION_MS), MAX_DURATION_MS))

    def _await_arrival(self, target_xyz, duration_ms: int, what: str, pulses=None) -> None:
        """Raise unless the arm read back within ARRIVAL_TOL_MM of target_xyz.

        With `pulses` (the command just sent), a miss is re-sent once first; the
        retry is reported, so a pattern of near-misses stays visible.
        """
        self._sleep(duration_ms / 1000.0 + SETTLE_S)
        reached = kin.pulses_to_xyz(self._read_pulses())
        err = math.dist(reached, target_xyz)
        if err > ARRIVAL_TOL_MM and pulses is not None:
            print(f"[MaxArm] {what}: {err:.1f} mm short of "
                  f"({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f}); re-sending once")
            self._send_pulses(pulses, RETRY_DURATION_MS, what)
            self._sleep(RETRY_DURATION_MS / 1000.0 + SETTLE_S)
            reached = kin.pulses_to_xyz(self._read_pulses())
            err = math.dist(reached, target_xyz)
        if err > ARRIVAL_TOL_MM:
            raise MaxArmError(
                f"{what}: arm did not arrive. Target "
                f"({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f}), "
                f"read back ({reached[0]:.1f}, {reached[1]:.1f}, {reached[2]:.1f}), "
                f"{err:.1f} mm off (tolerance {ARRIVAL_TOL_MM}). A stalled servo, "
                f"a blocked path or a sagging supply all look like this.")

    # -- motion ---------------------------------------------------------------

    def set_position(self, x, y, z, roll=None, pitch=None, yaw=None,
                     speed=DEFAULT_SPEED_MM_S, wait=True, **kwargs) -> int:
        """Move the wrist to (x, y, z) in the arm frame, mm. Returns 0 or raises.

        roll/pitch: the MaxArm's parallel linkage keeps the tool vertical, so the
        only accepted orientations are "pointing down" as the rest of the repo
        writes it (roll 180 or 0, pitch 0). Anything else is refused rather than
        silently ignored.
        yaw: the nozzle angle, -90..90 deg, relative to the arm (it turns with
        the base). None leaves the nozzle where it is.
        """
        if roll is not None and min(abs(float(roll)), abs(abs(float(roll)) - 180.0)) > 1e-6:
            raise MaxArmError(f"set_position: roll {roll} unsupported; the MaxArm "
                              f"tool always points straight down")
        if pitch is not None and abs(float(pitch)) > 1e-6:
            raise MaxArmError(f"set_position: pitch {pitch} unsupported; the MaxArm "
                              f"tool always points straight down")
        try:
            sol = kin.solve((x, y, z), self.floor_z_mm)
        except kin.Unreachable as exc:
            raise MaxArmError(f"set_position refused: {exc}") from None
        with self._lock:
            duration = self._duration_ms(sol.xyz_mm, speed)
            if yaw is not None:
                self.set_nozzle_angle(yaw, duration_ms=duration)
            self._send_pulses(sol.pulses, duration, "set_position")
            if wait:
                self._await_arrival(sol.xyz_mm, duration, "set_position", sol.pulses)
        return 0

    def set_servo_angle(self, angle, speed=30.0, wait=True, **kwargs) -> int:
        """Joint move in the vendor's joint angles (degrees).

        `angle` is [j1, j2, j3] or [j1, j2, j3, nozzle]. j1 0..240 (120 = facing
        -y), j2 210..-30 and j3 -120..120 (see kinematics.deg_to_pulse). The
        resulting wrist position must pass the same floor and limit checks as a
        Cartesian move. A joint move is not a way around the gate.
        `speed` is deg/s of the largest joint change.
        """
        a = [float(v) for v in angle]
        if len(a) not in (3, 4):
            raise MaxArmError(f"set_servo_angle: need 3 or 4 angles, got {len(a)}")
        joints = tuple(a[:3])
        xyz = kin.forward(joints)
        problems = kin.check_pulses(kin.deg_to_pulse(joints))
        if xyz[2] < self.floor_z_mm:
            problems.append(f"wrist z {xyz[2]:.1f} below the floor {self.floor_z_mm:.1f}")
        if xyz[2] > kin.Z_MAX_MM:
            problems.append(f"wrist z {xyz[2]:.1f} > {kin.Z_MAX_MM:.0f}")
        if problems:
            raise MaxArmError("set_servo_angle refused: " + "; ".join(problems))
        with self._lock:
            cur = kin.pulse_to_deg(self._read_pulses())
            biggest = max(abs(c - t) for c, t in zip(cur, joints))
            duration = int(min(max(1000.0 * biggest / max(float(speed), 1e-3),
                                   MIN_DURATION_MS), MAX_DURATION_MS))
            if len(a) == 4:
                self.set_nozzle_angle(a[3], duration_ms=duration)
            self._send_pulses(kin.deg_to_pulse(joints), duration, "set_servo_angle")
            if wait:
                self._await_arrival(xyz, duration, "set_servo_angle",
                                    kin.deg_to_pulse(joints))
        return 0

    def set_nozzle_angle(self, angle_deg: float, duration_ms: int = 500) -> int:
        lo, hi = kin.NOZZLE_ANGLE_RANGE_DEG
        if not (lo <= float(angle_deg) <= hi):
            raise MaxArmError(f"nozzle angle {angle_deg} outside {lo}..{hi}")
        self._call(f"_r(_nz.set_angle, {float(angle_deg)!r}, {int(duration_ms)})",
                   "set_nozzle_angle")
        self._nozzle_cmd_deg = float(angle_deg)
        return 0

    def get_position(self):
        """(0, [x, y, z, 180, 0, yaw]) from the servos' READ pulses.

        yaw is the nozzle angle as last COMMANDED. The PWM servo has no
        read-back. It is None until the nozzle has been commanded once, so a
        consumer doing arithmetic on it fails instead of using a guess.
        """
        x, y, z = kin.pulses_to_xyz(self._read_pulses())
        return 0, [x, y, z, 180.0, 0.0, self._nozzle_cmd_deg]

    def get_servo_angle(self):
        """(0, [j1, j2, j3, nozzle]). Joints are read back; nozzle as get_position."""
        j = kin.pulse_to_deg(self._read_pulses())
        return 0, [*j, self._nozzle_cmd_deg]

    def go_home(self, wait: bool = True, **kwargs) -> int:
        """The vendor home point (0, -163, 212), at a gentle speed."""
        return self.set_position(*kin.HOME_XYZ_MM, speed=kwargs.get("speed", 40.0),
                                 wait=wait)

    def wave_goodbye(self, n_waves: int = 3, **kwargs) -> int:
        """Rock the base joint about the current pose and come back."""
        code, angles = self.get_servo_angle()
        base = list(angles[:3])
        for _ in range(max(1, int(n_waves))):
            for d in (WAVE_AMPLITUDE_DEG, -WAVE_AMPLITUDE_DEG):
                self.set_servo_angle([base[0] + d, base[1], base[2]], speed=60.0)
        return self.set_servo_angle(base, speed=60.0)

    # -- end effector (suction cup) --------------------------------------------

    def close_gripper(self, wait: bool = True) -> int:
        """Pump on, valve closed: start sucking."""
        self._call("_r(_nz.on)", "suction on")
        self._suction_cmd = True
        return 0

    def open_gripper(self, wait: bool = True) -> int:
        """Pump off, valve opened for 1 s (the vendor's release), then closed."""
        self._call("_r(_nz.off)", "suction off")
        self._suction_cmd = False
        if wait:
            self._sleep(1.1)
        return 0

    def set_gripper(self, mode: str) -> int:
        if mode != self.effector:
            raise MaxArmError(f"plan expects the {mode!r} effector; the MaxArm has "
                              f"a {self.effector!r} cup and no tool changer")
        return 0

    def verify_grasp(self) -> tuple[str, str]:
        """No vacuum sensor, so the honest answer is 'unknown'.

        Returning 'held' because the pump is on would be the gripper-command
        error the Recorder documents: a COMMAND reported as a STATE. Use the
        wrist camera to confirm a lift.
        """
        return ("unknown", "the MaxArm has no vacuum sensor; pump is "
                + ("ON" if self._suction_cmd else "off")
                + ". Confirm a pick with the wrist camera.")

    # -- what the MaxArm does not have ------------------------------------------

    @unsupported("the MaxArm suction kit has no rail")
    def set_rail_position(self, *a, **k): ...

    @unsupported("the MaxArm suction kit has no rail")
    def get_rail_position(self, *a, **k): ...

    @unsupported("the MaxArm suction kit has no rail")
    def rail_home(self, *a, **k): ...

    @unsupported("the MaxArm has no force/torque sensor")
    def get_ft_data(self, *a, **k): ...

    @unsupported("reads simulator ground truth; on hardware use perception")
    def get_body_pose(self, *a, **k): ...

    @unsupported("reads simulator state; on hardware use perception")
    def physical_outcome(self, *a, **k): ...

    @unsupported("restores simulator state; resetting the table is a manual step")
    def reset_scene(self, *a, **k): ...

    @unsupported("teleports a body in simulation; no physical equivalent")
    def nudge_body(self, *a, **k): ...

    @unsupported("needs live object poses (perception), not written for the MaxArm yet")
    def push_object(self, *a, **k): ...

    @unsupported("there is no rack in the MaxArm's workspace model")
    def place_tube_in_rack(self, *a, **k): ...

    @unsupported("the thermocycler belongs to the xArm cell, not the MaxArm")
    def set_pcr_lid(self, *a, **k): ...

    # -- lifecycle --------------------------------------------------------------

    def disconnect(self, release_suction: bool = False) -> None:
        """Leave the raw REPL and close the port. The servos keep holding.

        Torque is NOT released: `teaching_mode()` (servo unload) would drop the
        arm onto the table. The pump is left as it is unless release_suction is
        True, because switching it off drops whatever is held.
        """
        try:
            if release_suction and self._suction_cmd:
                self.open_gripper()
            elif self._suction_cmd:
                print("[MaxArm] disconnect: pump is still ON (object held). "
                      "Pass release_suction=True to drop it.")
        finally:
            self.repl.exit()
            self.link.close()
