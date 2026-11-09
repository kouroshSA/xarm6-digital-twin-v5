"""First contact with the Hiwonder MaxArm over USB, and the measurements it needs.

    python scripts/maxarm_probe.py                    # read-only: identify the board
    python scripts/maxarm_probe.py --kinematics       # read-only: PC port vs the board's IK
    python scripts/maxarm_probe.py --frame            # MOVES: which way +x points
    python scripts/maxarm_probe.py --floor            # MOVES: where the cup meets the table

Run from xarm_lab_twin/ in the xarm6sim env. With no --port the first
/dev/ttyUSB* or /dev/ttyACM* is used.

The default mode moves nothing. It interrupts the stock program (the phone app
stops answering until a power cycle), then lists the board's files, MicroPython
version, servo positions and supply voltage. Back up the listed files before
writing anything to the board: the stock program is not published.

--frame and --floor move the arm. Each asks for Enter first, and Ctrl-C stops.
The servos keep holding when the script exits.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

from hardware.maxarm import kinematics as kin                        # noqa: E402
from hardware.maxarm.transport import RawRepl, SerialLink, BoardError  # noqa: E402

PRELUDE_LIST = """\
import os, sys, ujson
print('@@' + ujson.dumps({'ok': True, 'v': {
    'files': sorted(os.listdir()),
    'impl': [sys.implementation.name, list(sys.implementation.version)],
    'platform': sys.platform,
}}))
"""

#: arm-frame points for --kinematics (the board's espmax.py negates x before
#: calling __espmax.inverse, so the board call below passes -x).
CHECK_POINTS = [kin.HOME_XYZ_MM, (120, -180, 85), (-120, -180, 85),
                (0, -250, 100), (150, -150, 150), (-60, -220, 120), (200, 10, 100)]


def find_port(explicit: str | None) -> str:
    if explicit:
        return explicit
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        sys.exit("no /dev/ttyUSB* or /dev/ttyACM*. Check `dmesg | tail` after "
                 "plugging in; silence there usually means the cable is in the controller "
                 "board's power-only USB-A port, not the ESP32's USB-C port.")
    return ports[0]


def identify(repl: RawRepl) -> dict:
    info = repl.call(PRELUDE_LIST)
    print(f"  firmware : {info['impl'][0]} {'.'.join(map(str, info['impl'][1]))} on {info['platform']}")
    print(f"  files    : {', '.join(info['files'])}")
    need = {"BusServo.py", "SuctionNozzle.py", "PWMServo.py"}
    have = set(info["files"])
    missing = need - have
    print("  vendor modules the backend needs: "
          + ("all on the filesystem" if not missing else
             f"{sorted(missing)} not on the filesystem (they may be frozen into "
             f"the firmware; the import below decides)"))
    return info


def kinematics_check(repl: RawRepl) -> int:
    """Board IK vs the PC port on CHECK_POINTS. Reads only; moves nothing."""
    try:
        repl.exec("import __espmax")
    except BoardError as exc:
        print(f"  __espmax not importable on the board ({exc}); cannot compare")
        return 1
    worst, bad = 0.0, 0
    for p in CHECK_POINTS:
        x, y, z = p
        code = (f"_a = __espmax.inverse({kin.L4!r}, ({-float(x)!r}, {float(y)!r}, {float(z)!r}))\n"
                f"print('@@' + __import__('ujson').dumps({{'ok': True, 'v': list(_a)}}))")
        try:
            board = repl.call(code)
        except BoardError as exc:
            print(f"  {p}: board raised {exc}")
            bad += 1
            continue
        mine = kin.inverse(p)
        d = max(abs(a - b) for a, b in zip(mine, board))
        worst = max(worst, d)
        flag = "" if d < 0.01 else "   <-- DIFFERS"
        bad += bool(flag)
        print(f"  {str(p):24s} board {[round(v, 3) for v in board]}  port {[round(v, 3) for v in mine]}{flag}")
    print(f"  worst joint difference {worst:.4f} deg; {bad} disagreement(s)")
    return 1 if bad else 0


def confirm(msg: str) -> None:
    input(f"\n  {msg}\n  Clear the arm's reach, then press Enter (Ctrl-C aborts) ")


def frame_check(arm) -> None:
    """Move to x=+60 then x=-60 so the operator can see which way +x points."""
    confirm("The arm will go home, then to x=+60 and x=-60 at y=-180, z=150.")
    arm.go_home()
    arm.set_position(60, -180, 150, speed=30)
    input("  Now at x = +60. Note which side of the arm the cup is on, press Enter ")
    arm.set_position(-60, -180, 150, speed=30)
    input("  Now at x = -60. Press Enter to go home ")
    arm.go_home()
    print("  Measured 2026-11-07 on the lab unit: +x is the arm's right, seen from"
          " behind (a left-handed frame). If it differs here, update kinematics.py"
          " and hardware/maxarm/README.md.")


def floor_check(arm) -> None:
    """Step down in 2 mm until the operator says the cup touches the table."""
    x, y = 0.0, -200.0
    z = 120.0
    confirm(f"The arm will go to ({x}, {y}, {z}) and step DOWN 2 mm per Enter. "
            f"Type s + Enter the moment the cup lip touches the table.")
    arm.floor_z_mm = None   # the floor is what is being measured
    arm.set_position(x, y, z, speed=30)
    while True:
        ans = input(f"  z = {z:.0f}  [Enter = 2 mm lower, s = touching] ").strip().lower()
        if ans == "s":
            break
        if z - 2 < 20:
            print("  z = 20 reached without contact; stopping. Is the arm on the table?")
            break
        z -= 2
        arm.set_position(x, y, z, speed=10)
    arm.set_position(x, y, z + 30, speed=20)
    print(f"\n  Contact at wrist z = {z:.0f} mm. Put a margin on it (e.g. +10) and set "
          f"kinematics.DEFAULT_FLOOR_Z_MM. Now it is MEASURED, so say so there.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port")
    ap.add_argument("--kinematics", action="store_true")
    ap.add_argument("--frame", action="store_true")
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()

    port = find_port(a.port)
    print(f"[maxarm] {port}")
    link = SerialLink(port)
    repl = RawRepl(link)
    repl.enter()
    rc = 0
    arm = None
    try:
        identify(repl)
        if a.kinematics:
            rc |= kinematics_check(repl)
        if a.frame or a.floor:
            from hardware.maxarm.real_maxarm import RealMaxArmAPI
            repl.exit()            # the backend enters the raw REPL itself
            arm = RealMaxArmAPI(link=link)
            print(f"  servo pulses {arm.connect_pulses}, supply mV {arm.supply_mv}")
            if a.frame:
                frame_check(arm)
            if a.floor:
                floor_check(arm)
        else:
            from hardware.maxarm.real_maxarm import BOARD_PRELUDE
            try:
                repl.exec(BOARD_PRELUDE)
                p = repl.call("_r(_pulses)")
                v = repl.call("_r(_vin)")
                print(f"  servo pulses {p} -> wrist at "
                      f"{tuple(round(c, 1) for c in kin.pulses_to_xyz(p))} (arm frame, mm)")
                print(f"  servo supply {v} mV")
            except BoardError as exc:
                print(f"  could not load the vendor modules: {exc}")
                rc = 1
    finally:
        if arm is not None:
            arm.disconnect()
        else:
            repl.exit()
            link.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
