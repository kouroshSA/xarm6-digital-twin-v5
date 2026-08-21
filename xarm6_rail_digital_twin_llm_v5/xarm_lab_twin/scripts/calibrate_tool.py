#!/usr/bin/env python3
"""Identify the tool load and set the TCP offset on the real arm (preflight §1.2).

**THIS MOVES THE ARM.** `iden_ft_sensor_load_offset()` is an identification
routine: the controller drives the arm through a sequence of poses to measure
what is hanging off the flange. It chooses its own speeds and path. The envelope
must be clear and an operator must be at the e-stop.

Why it matters more than it looks: gravity compensation, collision detection and
every force reading are derived from the load model. With no payload configured
the arm believes it is carrying nothing, and produces readings that are
*wrong but plausible* — worse than having no sensor at all.

Because the F/T sensor sits between the flange and the gripper, one routine
covers both loads that need setting:

  * **TCP load** — what the ARM carries: sensor + gripper + fingertips + anything
    else past the flange. Used for arm dynamics and collision detection.
  * **F/T load offset** — what the SENSOR carries: everything below it, i.e. the
    gripper and down, *not* the sensor itself. Used to zero the force reading.

`set_ft_sensor_load_offset(..., association_setting_tcp_load=True)` derives the
first from the second, so they cannot drift apart.

    python scripts/calibrate_tool.py --ip 192.168.1.229                  # dry run
    python scripts/calibrate_tool.py --ip 192.168.1.229 --i-am-at-the-estop

Order is deliberate: identify with the TCP offset as-is, apply the load, and only
then set the offset — so the identification happens in the configuration it was
measured in.
"""
from __future__ import annotations

import argparse
import sys
import time

# Flange face to gripper tip, mm. Confirmed against the real arm on 2026-08-21:
# with TCP offset [0,0,0] the flange read z=144.95 at the pose where the tip sat
# 25 mm above a benchtop 97 mm below the base, so 145 - (25 - 97) = 217.
DEFAULT_TCP_Z_MM = 217.0

MIN_FIRMWARE_FOR_FT = (1, 8, 3)

# An xArm F/T sensor plus a gripper cannot weigh less than this. Used to reject
# an identification that reports success but measured nothing.
MIN_PLAUSIBLE_MASS_KG = 0.5


def _fw_tuple(version_string: str):
    """Pull a (major, minor, patch) out of e.g. '6,6,XI1305,MC1305,v2.7.1'."""
    for part in str(version_string).replace(",", " ").split():
        if part.startswith("v") and part[1:].replace(".", "").isdigit():
            bits = part[1:].split(".")
            return tuple(int(b) for b in (bits + ["0", "0"])[:3])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", required=True)
    ap.add_argument("--tcp-z", type=float, default=DEFAULT_TCP_Z_MM,
                    help=f"flange-to-tip distance in mm (default {DEFAULT_TCP_Z_MM})")
    ap.add_argument("--i-am-at-the-estop", action="store_true",
                    help="required to move the arm; without it this only reports")
    args = ap.parse_args()

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip)
    try:
        # -- pre-flight reads (no motion) ------------------------------------
        code, version = arm.get_version()
        err = arm.get_err_warn_code()[1]
        print(f"  firmware      : {version}")
        print(f"  serial        : {arm.sn!r}")
        print(f"  state / mode  : {arm.get_state()[1]} / {arm.mode}")
        print(f"  err / warn    : {err}")
        print(f"  TCP offset    : {arm.tcp_offset}")
        print(f"  TCP load      : {arm.tcp_load}")
        print(f"  position      : {arm.get_position()[1]}")

        fw = _fw_tuple(version)
        if fw and fw < MIN_FIRMWARE_FOR_FT:
            print(f"\n  ABORT: firmware {fw} is below the {MIN_FIRMWARE_FOR_FT} "
                  f"required for the F/T identification calls.")
            return 2
        if err[0]:
            print(f"\n  ABORT: controller error {err[0]} is latched. Clear it in "
                  f"Studio and confirm the cause before moving the arm.")
            return 2

        if not args.i_am_at_the_estop:
            print("\n  DRY RUN — nothing was changed and the arm did not move.")
            print("  This routine drives the arm through its own sequence of poses.")
            print("  Re-run with --i-am-at-the-estop once the envelope is clear and")
            print("  you are at the e-stop.")
            return 0

        # -- enable ----------------------------------------------------------
        print("\n  enabling motion (arm becomes live)")
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)

        # -- enable the F/T sensor and prove it is actually reporting --------
        # Learned the hard way: iden_ft_sensor_load_offset() returns code 0 with
        # an all-zero result when the sensor is disabled. A success code with a
        # meaningless payload is worse than an error, so the sensor is verified
        # to be producing real numbers before the identification is trusted.
        print("  enabling the F/T sensor")
        arm.set_ft_sensor_enable(1)
        time.sleep(1.0)
        code, ft = arm.get_ft_sensor_data()
        print(f"  F/T reading   : {[round(v, 2) for v in (ft or [])]}")
        if code != 0 or not ft or all(abs(v) < 1e-6 for v in ft):
            print("\n  ABORT: the F/T sensor reads exactly zero on every axis with a "
                  "gripper hanging off it. It is not reporting, and identification "
                  "would return zeros while claiming success. Check the sensor is "
                  "connected and powered. Nothing was written.")
            arm.set_ft_sensor_enable(0)
            return 2

        # -- identify --------------------------------------------------------
        print("  running iden_ft_sensor_load_offset() — THE ARM IS MOVING NOW")
        print("  (it travels a substantial distance; this is not a small wiggle)")
        code, load = arm.iden_ft_sensor_load_offset()
        if code != 0 or not load:
            print(f"\n  FAILED: identification returned code {code}. Nothing was "
                  f"written. err/warn now {arm.get_err_warn_code()[1]}")
            return 1
        if float(load[0]) <= MIN_PLAUSIBLE_MASS_KG:
            print(f"\n  FAILED: identified mass {load[0]:.3f} kg is below the "
                  f"{MIN_PLAUSIBLE_MASS_KG} kg plausibility floor — an F/T sensor "
                  f"plus gripper cannot weigh that little. The routine reported "
                  f"success but measured nothing. Nothing was written.")
            return 1
        print(f"  identified    : {load}")
        print(f"    mass        : {load[0]:.3f} kg")
        print(f"    centroid    : ({load[1]:.1f}, {load[2]:.1f}, {load[3]:.1f}) mm")

        # -- apply -----------------------------------------------------------
        print("\n  writing F/T load offset (and the matching TCP load)")
        rc = arm.set_ft_sensor_load_offset(load, association_setting_tcp_load=True)
        if rc != 0:
            print(f"  FAILED: set_ft_sensor_load_offset returned {rc}")
            return 1

        print(f"  setting TCP offset [0, 0, {args.tcp_z:.0f}, 0, 0, 0]")
        rc = arm.set_tcp_offset([0, 0, args.tcp_z, 0, 0, 0])
        if rc != 0:
            print(f"  FAILED: set_tcp_offset returned {rc}")
            return 1

        print("  save_conf() — persist across a control-box reboot")
        rc = arm.save_conf()
        if rc != 0:
            print(f"  WARNING: save_conf returned {rc}; settings may be lost on reboot")

        # -- verify ----------------------------------------------------------
        # Read back rather than trusting the write. save_conf is the step most
        # likely to silently not stick, and a lost payload is invisible until
        # collision detection misbehaves.
        # Read back after a pause: an immediate read returns the pre-write value
        # and produced a false MISMATCH the first time this was run.
        time.sleep(1.5)
        print("\n  read-back:")
        tcp_off, tcp_load = arm.tcp_offset, arm.tcp_load
        print(f"    TCP offset  : {tcp_off}")
        print(f"    TCP load    : {tcp_load}")
        print(f"    position    : {arm.get_position()[1]}   <-- now TIP-referenced")

        ok = abs(float(tcp_off[2]) - args.tcp_z) < 1.0 and float(tcp_load[0]) > 0.01
        print()
        if ok:
            print("  OK. Record these in arm_backend.py as measured cell values.")
            print("  Power-cycle the control box and re-run this without the flag")
            print("  to confirm they survived.")
        else:
            print("  MISMATCH: the read-back does not match what was written.")
            print("  Do not rely on collision detection or F/T readings until resolved.")
        return 0 if ok else 1
    finally:
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(main())
