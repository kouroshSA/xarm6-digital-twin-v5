# Hiwonder MaxArm backend (real hardware only)

A Hiwonder MaxArm (ESP32 + three bus servos + suction cup on a PWM-servo
nozzle, no rail) driven from the PC. There is no digital twin for it.
`RealMaxArmAPI` implements the shared arm-backend contract
(`arm_backend.ARM_BACKEND_METHODS`) and declares what the arm lacks.

Needs `pyserial` (in `requirements.txt`; `pip install pyserial==3.5` into
`xarm6sim`) and, on Linux, membership of the `dialout` group to open the
port: `sudo usermod -aG dialout $USER`, then log out and back in.

```bash
python -m hardware.maxarm.test_maxarm        # no hardware needed
python scripts/maxarm_probe.py               # first contact: read-only
python scripts/maxarm_probe.py --kinematics  # PC port vs the board's own IK, read-only
python scripts/maxarm_probe.py --frame       # MOVES: settle which way +x points
python scripts/maxarm_probe.py --floor       # MOVES: measure where the cup meets the table
```

```python
from hardware.maxarm.real_maxarm import RealMaxArmAPI
arm = RealMaxArmAPI("/dev/ttyUSB0")
arm.go_home()                                 # (0, -163, 212), arm frame, mm
arm.set_position(-60, -220, 120, yaw=30)     # yaw = nozzle angle, -90..90
arm.close_gripper(); arm.open_gripper()       # pump on / release
```

## Files

| file | what |
|---|---|
| `kinematics.py` | FK/IK ported from the vendor's `_espmax.cpp`, plus the gate (`solve`, `check_pose`) |
| `transport.py` | MicroPython raw-REPL client; opens the port without resetting the ESP32 |
| `real_maxarm.py` | `RealMaxArmAPI` |
| `fake_board.py` | a raw-REPL board model for tests |
| `test_maxarm.py` | tests; kinematics graded against the compiled vendor C source |

## Decisions

- **USB + raw REPL, not the vendor's binary protocol.** The Chapter 10 protocol
  (`AA 55 | func | len | data | ~sum`) needs a vendor "slave" program flashed
  to the board, and it never acknowledges a move: `deal_command` discards
  `set_position`'s False. The REPL returns real values and real exceptions, and
  installs nothing.
- **Pulses are computed on the PC and sent to `BusServo.run`.** The vendor's
  `ESPMax.set_position` would re-solve IK on the board (a second copy of the
  kinematics) and silently clamp z > 225, servo 2 > 700, servo 3 < 470, and
  return None for reach < 50. The gate refuses all of these instead, and the
  board's solver is only used by `--kinematics` to check the port.
- **Every waited move is verified** by reading the servos back and running the
  result through FK. A stalled servo raises `MaxArmError("did not arrive")`.
- **`verify_grasp` always says "unknown".** There is no vacuum sensor, and "the
  pump is on" is a command, not evidence of a hold. Use the wrist camera.
- **`disconnect` never unloads the servos** (the arm would drop), and leaves
  the pump alone unless `release_suction=True`.

## Vendor facts worth knowing

- The board runs MicroPython; the compiled `__espmax.mpy` (mpy v5) holds
  L0=84.4, L1=8.14, L2=128.4, L3=138.0, with L4=16.8 passed in. All 21 copies
  of `_espmax.h` in the vendor downloads agree. `espmax.py`'s own L values
  (84.0/8.2/128.0) only feed `ORIGIN`.
- The vendor's Arduino and MicroPython APIs have **opposite x axes**. This
  module uses the MicroPython one (the "arm frame").
- The vendor's Raspberry Pi example passes xyz as a Python **set**
  (`{120, -180, 85}`), so its axis order is arbitrary. Do not copy it.
- Reading angles over the binary protocol returns pulses, while setting them
  takes degrees.

## Bluetooth (probed 2026-11-07, not used)

The arm advertises as `MaxArm` (`C8:85:41:18:1B:6E` on the lab unit) and pairs
with the laptop. GATT: `FFE0/FFE1` (read/write/notify), `FFF0/FFF1..FFF5`, and a
Nordic UART service with RX/TX swapped (`6e400002` notifies, `6e400003` is
written). The stock firmware answered neither the binary protocol's read-XYZ
frame (`AA 55 13 00 EC`) nor a REPL newline on any channel, so the phone app's
protocol is unknown. Two ways to use Bluetooth later:
1. capture the app's traffic (Android "Bluetooth HCI snoop log") and decode it.
   Likely fire-and-forget, with no readback.
2. once USB works, install a small program that serves an acknowledged protocol
   over the Nordic UART service, and add a BLE link next to `SerialLink`.

`bleak` (not in requirements) is what the probe used. Connect to an
already-paired device with a `BLEDevice` built from the BlueZ object path:
`BleakClient("C8:...")` rescans, and a connected peripheral stops advertising.

## Handoff, 2026-11-07 (state when work moved to another computer)

Nothing in this module has run against the real arm yet. Every test so far
used `FakeBoard`.

- **USB: the robot has never enumerated, and the laptop is not the cause.**
  On `ksa-Creator` the kernel logged nothing for the robot on any replug, not
  even a USB error. That includes the port the D435i had just enumerated on at
  SuperSpeed. So the micro-USB cable (probably power-only) or the arm's socket
  is at fault. First step on any machine: plug it in and check
  `dmesg | tail` / `ls /dev/ttyUSB* /dev/ttyACM*`. If nothing appears, change
  the cable before debugging anything else.
- Separate, machine-specific finding: that laptop's Thunderbolt 4 USB
  controller (PCI 00:0d.0, the USB-C ports) was stuck in runtime suspend. It
  was forced on with `power/control=on`. It was not the cause.
- **Bluetooth pairs and connects, but is not usable** (see the section above).
- **The wrist D435i is serial `950323050368`.** `perception/d435i_calib.py`
  holds the factory intrinsics of `027422071693` (the xArm cell's camera).
  Read this unit's own intrinsics off the device before trusting any
  measurement from it. The xArm cell already had an 11.8 px cx error from
  exactly this kind of swap.
- Branch `feat/maxarm` was pushed to the dev repo and not merged into `main`.

## Open, needs the device

- [ ] USB link: see the handoff above. Change the cable first.
- [ ] Which board files exist (`maxarm_probe.py`); back up the stock program.
- [ ] `--kinematics`: confirm the board's IK equals the port.
- [ ] `--frame`: which physical direction is +x. **PROVISIONAL** until then.
- [ ] `--floor`: cup-to-table contact z. `DEFAULT_FLOOR_Z_MM = 85` is the vendor's
      lowest example height, **not measured**.
- [ ] `ARRIVAL_TOL_MM = 6` is provisional; measure read-back error under load.
- [ ] Wrist D435: hand-eye transform from the cup/flange to the camera. Reuse
      `perception/` (`RGBDFrame.pixel_to_world` needs a `cam_to_world`; the
      MaxArm tool is always vertical, so the chain is base yaw + translation).
