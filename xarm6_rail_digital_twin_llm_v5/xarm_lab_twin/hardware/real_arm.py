# hardware/real_arm.py
"""Real xArm6 hardware wrapper. Only imported when --mode real is used.

Requires `pip install -r requirements.txt` (which pins xarm-python-sdk) and a
network-accessible xArm controller.

Two rules govern everything in this file:

1. **A command that did not execute must never report success.** The previous
   version caught `AttributeError` from a renamed SDK method, logged a warning,
   and returned `0` — the success code. Every caller, including the grader and
   the episode loop, then believed the rail had moved 700 mm when it had not.
   On a rail carrying an arm across a bench of biological samples that is a
   collision waiting to happen.

2. **Version mismatches fail loudly at connect, not silently mid-motion.**
   UFACTORY renamed the linear-rail API between SDK 1.14.x (`set_linear_track_*`)
   and 1.18.x (`set_linear_motor_*`) with no aliases, so the capability probe
   runs once in `__init__` and raises if neither spelling is present.
"""
from __future__ import annotations

from typing import Optional

from xarm.wrapper import XArmAPI

# Effector types this wrapper can drive. "none" is honest about a bare flange
# rather than pretending a gripper is attached.
EFFECTORS = ("standard", "bio", "vacuum", "none")

# Gripper position (0-850ish) below which the standard gripper counts as closed.
# Used only to distinguish "closed on nothing" from "closed on an object"; tune
# against the fitted fingers with the B5 measurement.
GRIPPER_CLOSED_THRESHOLD = 20


class RealArmError(RuntimeError):
    """Raised when the hardware or SDK cannot do what was asked.

    Deliberately not a return code: these are conditions where continuing would
    mean acting on a false belief about where the arm is.
    """


class RealXArmAPI:
    """Thin wrapper over XArmAPI exposing the subset the agent stack uses.

    Method names and return conventions mirror `SimXArmAPI` so the planner,
    dispatcher and recorder are backend-agnostic. Where the real arm genuinely
    cannot provide something the sim can (there is no `nudge_body` on a physical
    bench) the method raises `NotImplementedError` rather than returning a
    misleading success code.
    """

    def __init__(self, ip: str, effector: str = "standard",
                 ft_sensor: bool = True):
        if effector not in EFFECTORS:
            raise ValueError(f"effector must be one of {EFFECTORS}, got {effector!r}")
        self.effector = effector

        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

        self._rail_api = self._probe_rail_api()
        self._log_identity()

        if effector != "none":
            self._enable_effector()
        if ft_sensor:
            self._enable_ft_sensor()

    # -- probing ------------------------------------------------------------

    def _probe_rail_api(self) -> dict:
        """Resolve the linear-rail method names once, at connect.

        Probing per call and swallowing the failure is what hid the original
        bug; probing once and raising surfaces a version mismatch immediately.
        """
        candidates = [
            # SDK >= ~1.15
            {"set": "set_linear_motor_pos", "get": "get_linear_motor_pos",
             "enable": "set_linear_motor_enable", "home": "set_linear_motor_back_origin"},
            # SDK <= 1.14.x
            {"set": "set_linear_track_pos", "get": "get_linear_track_pos",
             "enable": "set_linear_track_enable", "home": "set_linear_track_back_origin"},
        ]
        for names in candidates:
            if all(hasattr(self.arm, n) for n in names.values()):
                print(f"[RealArm] linear rail API: {names['set']}")
                return names
        raise RealArmError(
            "No linear-rail API found in this xArm SDK build. Expected either "
            "set_linear_motor_pos (SDK >= ~1.15) or set_linear_track_pos "
            "(SDK <= 1.14.x). Check the xarm-python-sdk pin in requirements.txt."
        )

    def _log_identity(self) -> None:
        """Record firmware and gripper versions — several SDK calls carry
        firmware minimums, and the G1/G2 gripper distinction changes which
        gripper API is correct."""
        for label, getter in (("firmware", "get_version"),
                              ("gripper", "get_gripper_version")):
            fn = getattr(self.arm, getter, None)
            if fn is None:
                continue
            try:
                print(f"[RealArm] {label}: {fn()}")
            except Exception as exc:  # noqa: BLE001 - informational only
                print(f"[RealArm] {label} query failed: {type(exc).__name__}: {exc}")

    # -- setup --------------------------------------------------------------

    def _enable_effector(self) -> None:
        if self.effector == "standard":
            self._check(self.arm.set_gripper_enable(True), "set_gripper_enable")
            self._check(self.arm.set_gripper_mode(0), "set_gripper_mode")
        elif self.effector == "bio":
            self._check(self.arm.set_bio_gripper_enable(True), "set_bio_gripper_enable")
        # vacuum needs no enable step

    def _enable_ft_sensor(self) -> None:
        """Enable and zero the F/T sensor. Non-fatal: an arm without the sensor
        fitted should still be drivable, but the failure is reported, not hidden."""
        try:
            self._check(self.arm.set_ft_sensor_enable(1), "set_ft_sensor_enable")
            self._check(self.arm.set_ft_sensor_zero(), "set_ft_sensor_zero")
            print("[RealArm] F/T sensor enabled and zeroed")
        except Exception as exc:  # noqa: BLE001
            print(f"[RealArm] F/T sensor unavailable ({type(exc).__name__}: {exc}); "
                  f"force-based checks disabled")

    @staticmethod
    def _check(ret, what: str) -> int:
        """Normalise an SDK return into an int code, raising on failure.

        The SDK returns either an int code or a (code, value) tuple. A non-zero
        code means the controller rejected the command; that must not be
        mistaken for success.
        """
        code = ret[0] if isinstance(ret, (tuple, list)) else ret
        if not isinstance(code, int):
            return 0
        if code != 0:
            raise RealArmError(f"{what} failed with controller code {code}")
        return 0

    # -- rail ---------------------------------------------------------------

    def set_rail_position(self, position_mm: float, speed_mm_s: float = 50.0,
                          wait: bool = True, **kwargs) -> int:
        fn = getattr(self.arm, self._rail_api["set"])
        return self._check(fn(position_mm, speed=speed_mm_s, wait=wait),
                           self._rail_api["set"])

    def get_rail_position(self) -> tuple:
        """Returns (code, position_mm) — read back from the controller, never
        from a cached value. A cached position is how the old wrapper made an
        unmoved rail look like a moved one."""
        ret = getattr(self.arm, self._rail_api["get"])()
        if isinstance(ret, (tuple, list)):
            return ret[0], ret[1]
        return 0, float(ret)

    def rail_home(self, wait: bool = True) -> int:
        fn = getattr(self.arm, self._rail_api["home"])
        return self._check(fn(wait=wait), self._rail_api["home"])

    # -- arm motion ---------------------------------------------------------

    def set_position(self, x, y, z, roll=0, pitch=0, yaw=0,
                     speed=100, wait=True, **kwargs) -> int:
        return self._check(
            self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                  speed=speed, wait=wait),
            "set_position")

    def set_servo_angle(self, angle, speed=30, wait=True, **kwargs) -> int:
        return self._check(self.arm.set_servo_angle(angle=angle, speed=speed, wait=wait),
                           "set_servo_angle")

    def get_position(self):
        return self.arm.get_position()

    def get_servo_angle(self):
        return self.arm.get_servo_angle()

    # -- gripper ------------------------------------------------------------
    # The old wrapper called open_lite6_gripper / close_lite6_gripper, which are
    # Lite6-specific and wrong for an xArm 6 carrying any of these three tools.

    def open_gripper(self, wait: bool = True) -> int:
        if self.effector == "standard":
            return self._check(self.arm.set_gripper_position(850, wait=wait),
                               "set_gripper_position(open)")
        if self.effector == "bio":
            return self._check(self.arm.open_bio_gripper(wait=wait), "open_bio_gripper")
        if self.effector == "vacuum":
            return self._check(self.arm.set_vacuum_gripper(False, wait=wait),
                               "set_vacuum_gripper(off)")
        raise RealArmError("no effector fitted (effector='none')")

    def close_gripper(self, wait: bool = True) -> int:
        if self.effector == "standard":
            return self._check(self.arm.set_gripper_position(0, wait=wait),
                               "set_gripper_position(close)")
        if self.effector == "bio":
            return self._check(self.arm.close_bio_gripper(wait=wait), "close_bio_gripper")
        if self.effector == "vacuum":
            return self._check(self.arm.set_vacuum_gripper(True, wait=wait),
                               "set_vacuum_gripper(on)")
        raise RealArmError("no effector fitted (effector='none')")

    # Names the sim exposes; kept so the dispatcher needs no special-casing.
    # These are NOT the Lite6 SDK calls -- they route to the fitted effector.
    def open_lite6_gripper(self):
        return self.open_gripper()

    def close_lite6_gripper(self):
        return self.close_gripper()

    def verify_grasp(self) -> tuple[str, str]:
        """Report whether something is actually held: ("held"|"empty"|"unknown", detail).

        The sim's grasp is a magnetic weld that never fails, so the planning stack
        has never had to handle a failed grasp. On real hardware a dropped sample
        is a spill, so callers should gate any transit on this. Returning
        "unknown" is deliberate — it is honest about a channel we cannot read,
        and lets callers be written against this interface from the start.
        """
        if self.effector == "vacuum":
            ret = self.arm.get_vacuum_gripper()
            code, val = (ret if isinstance(ret, (tuple, list)) else (0, ret))
            if code != 0:
                return "unknown", f"get_vacuum_gripper returned code {code}"
            return ("held" if val == 1 else "empty"), f"vacuum state={val}"

        if self.effector == "standard":
            ret = self.arm.get_gripper_position()
            code, pos = (ret if isinstance(ret, (tuple, list)) else (0, ret))
            if code != 0 or pos is None:
                return "unknown", f"get_gripper_position returned code {code}"
            # Fully closed means the fingers met nothing. Anything held holds
            # them apart. Threshold needs calibrating per fingertip set (B5).
            if pos <= GRIPPER_CLOSED_THRESHOLD:
                return "empty", f"gripper fully closed (pos={pos})"
            return "held", f"gripper held open by object (pos={pos})"

        if self.effector == "bio":
            ret = self.arm.get_bio_gripper_status()
            return "unknown", f"bio gripper status={ret}; thresholds not yet calibrated"

        return "unknown", "no effector fitted"

    # -- force/torque -------------------------------------------------------

    def get_ft_data(self):
        """(code, [fx, fy, fz, tx, ty, tz]) — filtered, load-compensated."""
        return self.arm.get_ft_sensor_data()

    def identify_tool_load(self) -> int:
        """Re-identify the tool load. Run after EVERY effector change, or force
        thresholds drift with the tool and collision detection misreads."""
        fn = getattr(self.arm, "iden_ft_sensor_load_offset", None)
        if fn is None:
            raise RealArmError("iden_ft_sensor_load_offset absent in this SDK build")
        return self._check(fn(), "iden_ft_sensor_load_offset")

    # -- sim-only operations ------------------------------------------------

    def nudge_body(self, *args, **kwargs):
        raise NotImplementedError(
            "nudge_body teleports a body in simulation; there is no physical "
            "equivalent. Move the object by hand and re-run perception.")

    def reset_scene(self, *args, **kwargs):
        raise NotImplementedError(
            "reset_scene restores simulator state. On hardware, resetting the "
            "cell is a manual operator step -- see docs/hardware_preflight.md.")

    def physical_outcome(self, *args, **kwargs):
        raise NotImplementedError(
            "physical_outcome reads privileged simulator state. On hardware the "
            "equivalent must come from perception (see PerceptionOutcomeReporter).")

    def get_body_pose(self, name: str):
        raise NotImplementedError(
            f"get_body_pose({name!r}) reads simulator ground truth. On hardware "
            f"object poses come from perception, not the controller.")

    # -- lifecycle ----------------------------------------------------------

    def motion_enable(self, enable=True):
        return self.arm.motion_enable(enable=enable)

    def set_mode(self, mode):
        return self.arm.set_mode(mode)

    def set_state(self, state):
        return self.arm.set_state(state)

    def disconnect(self):
        self.arm.disconnect()
