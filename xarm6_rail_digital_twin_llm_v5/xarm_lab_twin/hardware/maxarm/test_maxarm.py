"""Hardware-free tests for the MaxArm backend.

    python -m hardware.maxarm.test_maxarm

The kinematics are checked against vectors produced by COMPILING the vendor's
own `_espmax.cpp` (with an Arduino.h stub), not against this port's own output:
a port checked against itself passes for any mistake made consistently.
Backend tests run the real raw-REPL protocol against `FakeBoard`, and grade what
reached the servos, not the return code.
"""
from __future__ import annotations

import math
import sys

from arm_backend import missing_methods, unsupported_reason

from . import kinematics as kin
from .fake_board import FakeBoard
from .real_maxarm import MaxArmError, RealMaxArmAPI
from .transport import SerialLink, TransportError

# Output of the vendor's _espmax.cpp compiled with g++ (2026-11-07), in the
# C/ARDUINO frame: (input xyz), (joint deg), (pulses), (FK of those joints).
# On the lab unit's firmware that frame IS the arm frame (the ESPMax frame): the
# board's solver lacks the C source's x negation and espmax.py adds one (see
# kinematics.py). Pulses are frame-free.
C_REFERENCE = [
    ((0, -163, 212), (120.000000, 90.027809, 0.332147), (500.000000, 499.884125, 501.383942)),
    ((120, -180, 85), (86.309921, 133.709946, 41.929420), (359.624664, 317.875214, 674.705933)),
    ((-120, -180, 85), (153.690063, 133.709946, 41.929420), (640.375244, 317.875214, 674.705933)),
    ((0, -250, 100), (120.000000, 142.584229, 26.890299), (500.000000, 280.899048, 612.042908)),
    ((150, -150, 150), (75.000000, 116.923630, 20.745861), (312.500000, 387.818207, 586.441101)),
    ((-60, -220, 120), (135.255157, 129.050613, 27.684023), (563.563171, 337.289124, 615.350098)),
    ((200, 10, 100), (27.137589, 123.875412, 41.258057), (113.073288, 358.852448, 671.908569)),
]
C_FK_HOME_PULSES = (-0.000002, -162.940018, 212.800003)   # forward(pulse_to_deg(500,500,500))

PASS, FAIL = "PASS", "FAIL"


def _close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# -- kinematics ---------------------------------------------------------------

def test_ik_matches_compiled_vendor_source():
    worst = 0.0
    for (cx, cy, cz), joints_c, pulses_c in C_REFERENCE:
        j = kin.inverse((cx, cy, cz))
        p = kin.deg_to_pulse(j)
        if not _close(p, pulses_c, 1e-3):
            return f"{FAIL} pulses for C({cx},{cy},{cz}): {p} vs vendor {pulses_c}"
        worst = max(worst, max(abs(a - b) for a, b in zip(j, joints_c)))
    if worst > 1e-3:
        return f"{FAIL} joint angles differ from the vendor's by {worst:.2e} deg"
    return f"{PASS} IK matches compiled _espmax.cpp on {len(C_REFERENCE)} poses (joints within {worst:.1e} deg)"


def test_fk_matches_compiled_vendor_source():
    x, y, z = kin.pulses_to_xyz(kin.HOME_PULSES)
    if not _close((x, y, z), C_FK_HOME_PULSES, 1e-3):
        return f"{FAIL} FK(500,500,500) = {(x, y, z)}; vendor C gives {C_FK_HOME_PULSES}"
    for (cx, cy, cz), joints_c, _ in C_REFERENCE:
        fx, fy, fz = kin.forward(joints_c)
        if not _close((fx, fy, fz), (cx, cy, cz), 1e-3):
            return f"{FAIL} FK of vendor joints for C({cx},{cy},{cz}) gave {(fx, fy, fz)}"
    return f"{PASS} FK matches compiled _espmax.cpp, incl. home pulses -> (0, -162.94, 212.8)"


# Read off the lab unit (maxarm_probe.py --kinematics, 2026-11-07): what the
# board's own solver returns. `espmax` = joints `ESPMax.set_position` commands for
# arm-frame xyz (it calls __espmax.inverse with -x); `raw` = __espmax.inverse
# called on xyz as given. Rounded to 3 decimals by the probe.
BOARD_ESPMAX = [((120, -180, 85), (86.31, 133.71, 41.929)),
                ((-120, -180, 85), (153.69, 133.71, 41.929)),
                ((150, -150, 150), (75.0, 116.924, 20.746)),
                ((-60, -220, 120), (135.255, 129.051, 27.684)),
                ((200, 10, 100), (27.138, 123.875, 41.258))]
BOARD_RAW = [((120, -180, 85), (153.69, 133.71, 41.929)),
             ((150, -150, 150), (165.0, 116.924, 20.746)),
             ((200, 10, 100), (212.862, 123.875, 41.258))]


def test_frame_matches_the_board():
    for p, j in BOARD_ESPMAX:
        if not _close(kin.inverse(p), j, 0.01):
            return f"{FAIL} arm frame {p}: {kin.inverse(p)} vs the board's ESPMax {j} (x mirrored?)"
    for p, j in BOARD_RAW:
        if not _close(kin._solver_inverse(p), j, 0.01):
            return f"{FAIL} solver {p}: {kin._solver_inverse(p)} vs the board's __espmax {j}"
    return f"{PASS} arm frame = the board's ESPMax frame; solver = its __espmax ({len(BOARD_ESPMAX) + len(BOARD_RAW)} measured poses)"


def test_vendor_home_is_reachable_and_near_fk_home():
    sol = kin.solve(kin.HOME_XYZ_MM)
    gap = math.dist(kin.pulses_to_xyz(kin.HOME_PULSES), kin.HOME_XYZ_MM)
    if gap > 1.0:
        return f"{FAIL} vendor ORIGIN is {gap:.2f} mm from FK(500,500,500)"
    return f"{PASS} vendor home solvable, pulses {tuple(round(p) for p in sol.pulses)}, {gap:.2f} mm from FK home"


def test_fk_ik_roundtrip_over_workspace():
    n, worst = 0, 0.0
    for x in range(-300, 301, 15):
        for y in range(-300, 301, 15):
            for z in range(85, 226, 14):
                try:
                    s = kin.solve((x, y, z))
                except kin.Unreachable:
                    continue
                n += 1
                worst = max(worst, math.dist(kin.forward(s.joints_deg), (x, y, z)))
    if n < 1000 or worst > 1e-6:
        return f"{FAIL} {n} reachable poses, worst round trip {worst:.2e} mm"
    return f"{PASS} FK(IK(p)) == p on {n} reachable poses (worst {worst:.1e} mm)"


def test_silent_vendor_clamps_are_refusals_here():
    cases = {
        "z above 225": (0, -163, 230),
        "reach under 50": (10, -20, 150),
        "below the floor": (0, -200, 60),
        "out of reach": (0, -400, 100),
    }
    missed = [name for name, p in cases.items() if not kin.check_pose(p)]
    # servo-range clamps: find a pose that solves but lands outside 470/700
    clamp_hit = None
    for z in range(85, 226, 5):
        for r in range(60, 290, 5):
            try:
                p = kin.deg_to_pulse(kin.inverse((0, -r, z)))
            except kin.Unreachable:
                continue
            if p[2] < kin.SERVO3_PULSE_MIN or p[1] > kin.SERVO2_PULSE_MAX:
                clamp_hit = (0, -r, z)
                break
        if clamp_hit:
            break
    if clamp_hit is None:
        missed.append("no pose found that triggers a servo clamp")
    elif not any("silently clamp" in r for r in kin.check_pose(clamp_hit, floor_z_mm=None)):
        missed.append(f"servo clamp at {clamp_hit} not refused")
    if missed:
        return f"{FAIL} not refused: {missed}"
    return f"{PASS} z>225, reach<50, floor, out-of-reach and servo clamps ({clamp_hit}) all refused"


# -- transport -------------------------------------------------------------------

def test_serial_opened_without_reset():
    events = []

    class FakeSerial:
        def __setattr__(self, k, v):
            events.append(("set", k, v))
            object.__setattr__(self, k, v)

        def open(self):
            events.append(("open",))

    mod = type("serial", (), {"Serial": FakeSerial})
    SerialLink("/dev/null", serial_module=mod)
    i_open = events.index(("open",))
    before = {(e[1], e[2]) for e in events[:i_open] if e[0] == "set"}
    if not {("dtr", False), ("rts", False)} <= before:
        return f"{FAIL} dtr/rts not deasserted before open: {events}"
    return f"{PASS} dtr and rts are False before open() (no ESP32 auto-reset)"


def test_non_micropython_board_is_a_clear_error():
    try:
        RealMaxArmAPI(link=FakeBoard(micropython=False))
    except TransportError as exc:
        if "not running MicroPython" in str(exc):
            return f"{PASS} Arduino/no-REPL board -> TransportError naming the cause"
        return f"{FAIL} wrong message: {exc}"
    return f"{FAIL} connected to a board with no REPL"


def test_missing_vendor_module_is_a_clear_error():
    try:
        RealMaxArmAPI(link=FakeBoard(missing_modules={"BusServo"}))
    except MaxArmError as exc:
        if "BusServo" in str(exc) and "maxarm_probe" in str(exc):
            return f"{PASS} missing BusServo.py -> MaxArmError pointing at the probe"
        return f"{FAIL} message lacks the cause: {exc}"
    return f"{FAIL} connected without BusServo"


# -- backend ---------------------------------------------------------------------

def _arm(**board_kw):
    board = FakeBoard(**board_kw)
    return RealMaxArmAPI(link=board, sleep=lambda s: None), board


def test_contract_is_complete():
    missing = missing_methods(RealMaxArmAPI)
    if missing:
        return f"{FAIL} contract methods neither implemented nor declared: {missing}"
    arm, _ = _arm()
    try:
        arm.set_rail_position(100)
    except NotImplementedError as exc:
        if "rail" in str(exc):
            return f"{PASS} contract complete; rail calls raise ({unsupported_reason(RealMaxArmAPI, 'set_rail_position')})"
    return f"{FAIL} set_rail_position did not raise"


def test_connect_moves_nothing():
    _, board = _arm()
    if board.motion_calls():
        return f"{FAIL} connecting sent motion: {board.motion_calls()}"
    return f"{PASS} connect interrupts the stock program and reads servos, no motion"


def test_connect_stops_the_stock_control_loop():
    board = FakeBoard(stock_loop=True)
    RealMaxArmAPI(link=board)
    if board.timers[2]:
        return f"{FAIL} the stock main.py's Timer 2 still runs after connect"
    if not board.timers[3]:
        return f"{FAIL} connect stopped Timer 3, the nozzle's (PWMServo) timer"
    try:
        RealMaxArmAPI(link=FakeBoard(stock_loop=True, timer_stuck={2}))
    except MaxArmError as exc:
        if "control loop" not in str(exc):
            return f"{FAIL} wrong refusal for a loop that will not stop: {exc}"
    else:
        return f"{FAIL} connected although the stock loop kept running"
    return f"{PASS} connect stops the stock loop (Timer 2), keeps Timer 3, refuses if it runs on"


def test_dead_servo_fails_connect():
    try:
        _arm(dead={2})
    except MaxArmError as exc:
        return f"{PASS} unanswering servo -> MaxArmError at connect ({str(exc)[:50]}...)"
    return f"{FAIL} connected with servo 2 not answering"


def test_move_sends_the_ports_pulses_and_verifies():
    arm, board = _arm()
    target = (-60, -220, 120)
    rc = arm.set_position(*target, roll=180, pitch=0)
    sent = [c for c in board.calls if c[0] == "run"]
    want = [round(p) for p in kin.solve(target).pulses]
    if rc != 0 or [c[2] for c in sent] != want:
        return f"{FAIL} rc={rc}, sent {sent}, expected pulses {want}"
    _, pose = arm.get_position()
    if math.dist(pose[:3], target) > 2.0:
        return f"{FAIL} read-back {pose[:3]} is not the target"
    return f"{PASS} move sends {want}, read-back {tuple(round(v, 1) for v in pose[:3])}"


def test_refused_move_sends_nothing():
    arm, board = _arm()
    for bad in [(0, -163, 240), (0, -200, 40), (0, -400, 100)]:
        try:
            arm.set_position(*bad)
        except MaxArmError:
            continue
        return f"{FAIL} {bad} was not refused"
    if board.motion_calls():
        return f"{FAIL} a refused move still reached the servos: {board.motion_calls()}"
    return f"{PASS} 3 refused moves, zero servo commands sent"


def test_stalled_servo_is_reported_not_returned_as_success():
    arm, board = _arm(stuck={3})
    try:
        arm.set_position(-60, -220, 120)
    except MaxArmError as exc:
        if "did not arrive" in str(exc):
            return f"{PASS} stuck servo 3 -> 'did not arrive', not rc=0"
        return f"{FAIL} raised, but not about arrival: {exc}"
    return f"{FAIL} stuck servo returned success"


def test_orientation_the_arm_cannot_do_is_refused():
    arm, board = _arm()
    for kw in ({"roll": 90}, {"pitch": 30}):
        try:
            arm.set_position(0, -200, 120, **kw)
        except MaxArmError:
            continue
        return f"{FAIL} {kw} accepted"
    arm.set_position(0, -200, 120, roll=-180, pitch=0)
    return f"{PASS} roll 90 / pitch 30 refused; roll -180 (tool down) accepted"


def test_joint_move_cannot_bypass_the_floor():
    arm, board = _arm()
    # joints that put the wrist low: found by solving a sub-floor pose unfiltered
    low = kin.inverse((0, -200, 40))
    try:
        arm.set_servo_angle(list(low))
    except MaxArmError as exc:
        if "floor" in str(exc) and not board.motion_calls():
            return f"{PASS} set_servo_angle to a sub-floor wrist is refused, nothing sent"
        return f"{FAIL} refused for another reason, or sent: {exc}"
    return f"{FAIL} joint move below the floor was executed"


def test_yaw_is_none_until_commanded_and_grasp_is_unknown():
    arm, board = _arm()
    _, pose = arm.get_position()
    if pose[5] is not None:
        return f"{FAIL} uncommanded nozzle yaw reported as {pose[5]}"
    arm.close_gripper()
    state, _ = arm.verify_grasp()
    if state != "unknown":
        return f"{FAIL} pump on reported as {state!r}: a command passed off as a state"
    arm.set_position(0, -200, 120, yaw=30)
    if arm.get_position()[1][5] != 30.0 or ("nozzle", 30.0) != board.calls[-4][:2]:
        return f"{FAIL} yaw not commanded/reported: {board.calls[-4:]}"
    return f"{PASS} yaw None until commanded; pump ON still grades 'unknown'"


def test_disconnect_keeps_torque_and_warns_about_pump():
    arm, board = _arm()
    arm.close_gripper()
    arm.disconnect()
    if any(c[0] == "unload" for c in board.calls):
        return f"{FAIL} disconnect unloaded servos (the arm would fall)"
    if ("pump_off",) in board.calls:
        return f"{FAIL} disconnect switched the pump off and dropped the payload"
    if ("exit_raw",) not in board.calls or not board.closed:
        return f"{FAIL} raw REPL not exited / port not closed"
    return f"{PASS} disconnect: torque kept, pump untouched, REPL exited, port closed"


TESTS = [
    test_ik_matches_compiled_vendor_source,
    test_fk_matches_compiled_vendor_source,
    test_frame_matches_the_board,
    test_vendor_home_is_reachable_and_near_fk_home,
    test_fk_ik_roundtrip_over_workspace,
    test_silent_vendor_clamps_are_refusals_here,
    test_serial_opened_without_reset,
    test_non_micropython_board_is_a_clear_error,
    test_missing_vendor_module_is_a_clear_error,
    test_contract_is_complete,
    test_connect_moves_nothing,
    test_connect_stops_the_stock_control_loop,
    test_dead_servo_fails_connect,
    test_move_sends_the_ports_pulses_and_verifies,
    test_refused_move_sends_nothing,
    test_stalled_servo_is_reported_not_returned_as_success,
    test_orientation_the_arm_cannot_do_is_refused,
    test_joint_move_cannot_bypass_the_floor,
    test_yaw_is_none_until_commanded_and_grasp_is_unknown,
    test_disconnect_keeps_torque_and_warns_about_pump,
]


def main() -> int:
    failures = 0
    for fn in TESTS:
        try:
            line = fn()
        except Exception as exc:  # noqa: BLE001
            line = f"{FAIL}  {fn.__name__}: {type(exc).__name__}: {exc}"
        failures += line.startswith(FAIL)
        print("  " + line)
    print(f"\n  {len(TESTS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
