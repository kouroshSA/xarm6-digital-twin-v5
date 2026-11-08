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
| `transport.py` | MicroPython raw-REPL client; waits out the reboot that opening the port causes |
| `real_maxarm.py` | `RealMaxArmAPI` |
| `fake_board.py` | a raw-REPL board model for tests |
| `test_maxarm.py` | tests; kinematics graded against the compiled vendor C source |
| `hand_eye.py` | wrist-camera calibration: wrist frame, left/right-handed conversion, solve via `solve_camera_pose` |
| `wrist_camera.py` | `MaxArmWristCamera`: the D435 with frames posed in the arm frame (`pixel_to_world * 1000` -> `set_position`) |
| `hand_eye_calib.json` | the measured camera mount (cam_to_wrist) with its provenance and error figures |
| `test_hand_eye.py` | recovers a known mount; a camera point lands on the arm-frame target; the naive left-handed fit fails |

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
- **x sign:** the published `_espmax.cpp` negates x; the board's compiled
  `__espmax.mpy` does not, and `espmax.py` negates it once. So on the lab unit
  the MicroPython `ESPMax` frame equals the C-source frame. This module uses the
  `ESPMax` frame (the "arm frame"); the first version assumed the opposite and
  was mirrored, which `--kinematics` caught (2026-11-07).
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
- **The camera on the MaxArm's wrist reports serial `027422071693`** (checked
  2026-11-07, -08 and -09; an earlier note here named `950323050368`). The hand-eye
  calibration records it, and `MaxArmWristCamera` opens only that serial by default.
  `perception/d435i_calib.py` holds `033422072806`'s factory intrinsics, so this
  unit's frames use the intrinsics the device itself reports. Read them off the
  device before trusting any measurement: the xArm cell already had an 11.8 px
  cx error from exactly this kind of swap.
- Branch `feat/maxarm` was pushed to the dev repo and not merged into `main`.

**Next-session notes (temporary):** [NOTES_NEXT.md](NOTES_NEXT.md) -- jog teaching (hand-guiding
does not work: measured), and a MaxArm digital twin for Petri-dish lid handling in its own room.

## Open, needs the device

- [x] USB link (2026-11-07): the old cable was power-only; with a data cable
      it is a CH340 at `/dev/ttyUSB0`. **Opening the port reboots the board**
      (Linux raises DTR/RTS before pyserial can lower them); the stock program
      boots in ~3.2 s and `RawRepl.enter` waits for its prompt.
- [x] Board files (2026-11-07): `1.rob`..`8.rob`, `BusServo.py`, `Buzzer.py`,
      `Key.py`, `Led.py`, `PWMServo.py`, `RobotControl.py`, `SuctionNozzle.py`,
      `USBDevice.py`, `__espmax.mpy`, `boot.py`, `espmax.py`, `main.py`;
      MicroPython 1.12 on ESP32; servo supply ~12.3 V.
- [x] Stock program backed up (2026-11-07), all 20 files byte-for-byte and
      checked against the board's own SHA-256: `~/Models/maxarm_board_backup/2026-11-07/`
      on ksa-14Studio, with `MANIFEST.json`. Kept OUT of git (unpublished vendor code).
- [x] **The stock control loop is stopped on connect** (2026-11-07). `main.py`
      homes the arm at boot, then starts `Timer(2)` calling `main()` every 15 ms:
      Bluetooth, USB-mouse and button handlers (a short press of key 1 runs
      action group 100). Ctrl-C does not stop a timer. Whether it is running
      when we connect is timing: an interrupt that lands while main.py is still
      starting Bluetooth aborts it before the homing and the timer. So
      `RealMaxArmAPI` deinits Timer 2 (and main.py's `tim`), refuses to connect
      unless the board reports `counter_en=0`, and keeps Timer 3 (PWMServo, the
      nozzle). Checked on 3 connects, both cases. Every connect still reboots the
      board, and a reboot that runs main.py through homes the arm.
- [x] `--kinematics` (2026-11-07): the board's IK equals the port on all 7
      points, after the x-sign fix above.
- [x] `--frame` (2026-11-07): **+x is the arm's right**, seen from behind the
      base looking out along the reach (-y); judged from the wrist D435 (picture
      top = away from the base) at x = +60 / -60, read-back within 3.3 mm. So the
      arm frame is **left-handed** (z up, reach along -y): negate x before any
      rotation maths, including the camera hand-eye transform below.
- [x] `--floor` (2026-11-07; checked at two more points 2026-11-09, above): the cup lip touches the table at commanded wrist
      **z = 52** (read-back 51.9) at (0, -200), judged by eye (+-2 mm), stepping
      2 mm at 10 mm/s. `DEFAULT_FLOOR_Z_MM = 62` (contact + 10). One point only:
      re-measure if the arm or table moves, or before working near the floor
      across the whole reach.
- [x] **The camera does not turn with the nozzle** (2026-11-09, `maxarm_hand_eye.py
      nozzle`): nozzle 0 / +45 / -45 / 0, seen turning both ways, and the tag held
      within 0.1 px and 0.1 mm. So the calibration holds at every nozzle angle.
- [x] `ARRIVAL_TOL_MM = 10` (2026-11-09, was 6). Unloaded (2026-11-07, ~40 moves):
      0.4-1.5 mm stepping down, 3.3-3.5 mm after the larger moves; then 6.0 mm (z
      short) back at a high view pose. Holding a 40 mm cube: 6.4 mm (z 5.8 short) at
      z 150, still after a re-send, so it is steady gravity sag at reach. A miss is
      now re-sent once before it is an error (`_await_arrival`, printed). A stalled
      servo or a blocked path is far larger than 10.
- [x] Silent board on connect (twice 2026-11-09): no output at all for 3 s after
      the first interrupt (no boot text); `RawRepl.enter` now interrupts twice.
- [ ] **Pick and place** (`scripts/maxarm_pick_place.py look|run`, 2026-11-09):
      Grounding DINO boxes the phrase; the object is the box's dominant-hue
      pixels; its top face is fitted on the plane table contact + size
      (`wrist_camera.pixel_at_height`), because a 40 mm cube's top is inside the
      D435's minimum range from any view the arm reaches (8% depth). Brightness
      did not work for the top face (light across it split it). The object's
      size is checked and a mismatch refuses (it caught a mask leaking onto pink
      paper and a cube half out of view). Results: run 1 carried the green cube
      onto the tag (turned ~25 deg, a few mm off, judged from a photo); run 2
      lifted it and dropped it on an arrival error (fixed: tolerance, and the
      pump now stays on when a move fails); run 3 never lifted it (cup landed,
      no hold) -- not yet known whether off the face centre or a poor seal on the
      wood. The run now FAILS unless the after-look finds the object within
      15 mm of the tag. Note `d435i_calib.DEPTH_MIN_M = 0.105` is optimistic for
      this unit at 640x480: ~135 mm returned almost nothing, ~172 mm was fine.
- [x] Wrist D435 hand-eye (2026-11-08): `hand_eye.py`, `scripts/maxarm_hand_eye.py`
      (`place` / `collect` / `verify`), result in `hand_eye_calib.json`. 41 views
      of a 40 mm tag laid at a known point: RMS 2.1 mm, max 3.7 mm (the size of
      the servo read-back error), spread 0.55; PnP depth agrees with the depth
      sensor to ~2 mm. **Verified:** with the tag moved ~3 cm, camera + calibration
      put its centre at z 52.9 (table 52) and the cup hovered 1-2 mm from it, by
      eye. The camera sits ~35 mm above the wrist point and ~63 mm off the nozzle
      axis, looking down within ~2.8 deg; it does NOT look at the cup, and it
      cannot focus closer than about wrist z 140. Valid at any nozzle angle: the
      camera does not turn with the nozzle (below).
- [x] Calibration wired into frames (2026-11-09): `wrist_camera.MaxArmWristCamera`
      is `RealSenseWristCamera` with `cam_to_world` = `hand_eye.cam_to_arm` (arm
      frame, metres, from the servo read-back), so `frame.pixel_to_world(u, v) * 1000`
      goes straight to `set_position`. The 3x3 block is a reflection (left-handed
      frame): right for points and clouds, not for orientations.
      `maxarm_hand_eye.py frame` runs it on the device: depth
      sensor and PnP agree on the tag's x/y within 0.7 mm over 4 runs.
- [x] **Calibration corrected 5 mm** (2026-11-09). With the cup lowered to 3 mm
      (`frame --hover-mm 3`, so no parallax) it sat 5 mm forward (-y) of the tag at
      both (30, -216) and (-2, -248): 6 judgements 5-7 mm forward, x under 1 mm on
      average with no steady sign. Taken as the tag having sat 5 mm back of where
      `place` recorded it: `resolve --tag-shift-mm 0 5` re-solved the same 41 samples
      (RMS 2.12 -> 1.91 mm, which supports it) and the file keeps both tag positions
      and the reason. A reach-dependent error would look the same this close to
      straight ahead; a tag 45 deg to the side would tell them apart. Second floor
      point: contact at about wrist z 52 at (31, -216) and 53 at (0, -249).
