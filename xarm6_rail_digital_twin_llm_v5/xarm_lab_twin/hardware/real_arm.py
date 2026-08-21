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

from arm_backend import (BASE_AT_RAIL_ZERO_MM, WORKSPACE_AABB_MM,
                         WORKSPACE_FLOOR_Z_MM, base_to_world_mm,
                         check_joint_limits_deg, check_workspace_world,
                         unsupported, world_to_base_mm)

# Effector types this wrapper can drive. "none" is honest about a bare flange
# rather than pretending a gripper is attached.
EFFECTORS = ("standard", "bio", "vacuum", "none")

# Home pose. Mirrors SimXArmAPI.go_home so "go home" means the same thing on both
# backends -- a planner that learned a home-relative approach in sim would
# otherwise be wrong on hardware.
HOME_RAIL_MM = 350.0
HOME_RAIL_SPEED_MM_S = 50.0
HOME_JOINTS_DEG = [90.0, 45.0, -45.0, 0.0, 30.0, 0.0]
HOME_JOINT_SPEED_DEG_S = 20.0

# wave_goodbye: joint-6 rocking amplitude and speed. Small and slow on purpose --
# this runs on a real arm near a bench.
WAVE_AMPLITUDE_DEG = 20.0
WAVE_SPEED_DEG_S = 30.0

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
                 ft_sensor: bool = True,
                 base_at_rail_zero=BASE_AT_RAIL_ZERO_MM,
                 floor_z_mm: float = WORKSPACE_FLOOR_Z_MM,
                 workspace_aabb=None):
        if effector not in EFFECTORS:
            raise ValueError(f"effector must be one of {EFFECTORS}, got {effector!r}")
        self.effector = effector

        # Cartesian coordinates in and out of this class are WORLD coordinates,
        # matching the twin. The controller works in its own base frame, so every
        # pose is converted here. Defaults are nominal values from the scene and
        # must be replaced with measured ones per cell (B4).
        self.base_at_rail_zero = tuple(base_at_rail_zero)
        self.floor_z_mm = float(floor_z_mm)
        self.workspace_aabb = dict(workspace_aabb or WORKSPACE_AABB_MM)
        self.workspace_aabb["z"] = (self.floor_z_mm, self.workspace_aabb["z"][1])

        self.arm = XArmAPI(ip)
        self.arm.clean_error()
        self._rail_api = self._probe_rail_api()
        self._sdk_joint_check_broken = self._probe_sn_defect()
        self._log_identity()

        if effector != "none":
            self._enable_effector()
        if ft_sensor:
            self._enable_ft_sensor()

        # Put the arm in a ready state LAST. Enabling the effector and the F/T
        # sensor can leave the controller in state 5 (stopped), so asserting
        # readiness before that setup -- as this did originally -- meant the first
        # motion came back with code -2 and no obvious cause.
        self.ready()

    def ready(self) -> int:
        """Enable motion, position mode, ready state. Safe to re-call."""
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        return self._check(self.arm.set_state(0), "set_state")

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

    def _probe_sn_defect(self) -> bool:
        """True if this controller reports a blank serial number.

        The SDK's joint-limit path does `int(self.sn[2:6])`, so a blank SN makes
        it raise ValueError rather than check anything. The Docker controller used
        for pre-action validation has no SN. Detected once here so the fallback in
        set_servo_angle is a known, announced condition rather than a surprise.
        """
        sn = (getattr(self.arm, "sn", "") or "").strip()
        if sn:
            return False
        print("[RealArm] controller reports a BLANK serial number: the SDK's own "
              "joint-limit check cannot run on this target (it would raise "
              "ValueError). Joint limits are enforced locally instead. Expected "
              "for the Docker controller; NOT expected on real hardware.")
        return True

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
        """Cartesian move to a **world** pose, matching the twin's convention.

        Two things happen before the controller sees anything:

        1. **Bounds, including the floor.** A commanded z below `floor_z_mm`
           would drive the tool into the benchtop. Refused here, in world
           coordinates, where the number is legible to a human.
        2. **Frame conversion.** The twin speaks world; the controller speaks its
           own base frame, which slides with the rail. Passing world coordinates
           straight through -- as this class did before -- sends the arm roughly
           350 mm out in x and 790 mm out in z. That was caught by dry-running a
           real plan against controller firmware, not by inspection.
        """
        bad = check_workspace_world((x, y, z), self.workspace_aabb, self.floor_z_mm)
        if bad:
            raise RealArmError("set_position refused (world frame): " + "; ".join(bad))

        rc, rail_mm = self.get_rail_position()
        if rc != 0 or rail_mm is None:
            raise RealArmError(
                f"set_position: cannot read the rail position (code {rc}), so the "
                f"world->base conversion would be wrong. Refusing to move.")

        bx, by, bz = world_to_base_mm((x, y, z), float(rail_mm),
                                      self.base_at_rail_zero)
        return self._check(
            self.arm.set_position(x=bx, y=by, z=bz, roll=roll, pitch=pitch,
                                  yaw=yaw, speed=speed, wait=wait),
            "set_position")

    def set_servo_angle(self, angle, speed=30, wait=True, **kwargs) -> int:
        """Joint move, degrees. Limits are checked here before the SDK sees them.

        Checking locally is not redundant. The SDK's own check reads the robot's
        serial number (`int(self.sn[2:6])`), which the Docker controller leaves
        blank -- so on that target the SDK raises ValueError instead of checking,
        and every joint move dies. Validating first means limits are enforced on
        both targets, and the SDK check stays active on real hardware.
        """
        bad = check_joint_limits_deg(angle)
        if bad:
            raise RealArmError("set_servo_angle rejected: " + "; ".join(bad))
        try:
            ret = self.arm.set_servo_angle(angle=angle, speed=speed, wait=wait)
        except ValueError:
            if not self._sdk_joint_check_broken:
                raise
            # Known container artifact, announced at connect. Limits were verified
            # above, so retry with the SDK's broken check disabled.
            self._disable_sdk_joint_check()
            ret = self.arm.set_servo_angle(angle=angle, speed=speed, wait=wait)
        return self._check(ret, "set_servo_angle")

    def _disable_sdk_joint_check(self) -> None:
        inner = getattr(self.arm, "arm", None)
        if inner is not None and hasattr(inner, "_check_joint_limit"):
            inner._check_joint_limit = False

    def get_position(self):
        """(code, [x, y, z, roll, pitch, yaw]) in **world** mm/deg.

        Converted back from the controller's base frame so callers see the same
        coordinates they command. An asymmetric wrapper -- world in, base out --
        would be worse than no conversion at all.
        """
        code, pose = self.arm.get_position()
        if code != 0 or not pose:
            return code, pose
        rc, rail_mm = self.get_rail_position()
        if rc != 0 or rail_mm is None:
            return rc, None
        wx, wy, wz = base_to_world_mm(pose[:3], float(rail_mm),
                                      self.base_at_rail_zero)
        return code, [wx, wy, wz, *pose[3:]]

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

    # -- poses the planner uses ---------------------------------------------

    def go_home(self, wait: bool = True, **kwargs) -> int:
        """Canonical home pose. Mirrors the sim's: rail mid-travel, arm folded.

        Implemented rather than declared unsupported -- "go home" is one of the
        planner's most-used actions and is entirely achievable on hardware.
        """
        self.set_rail_position(HOME_RAIL_MM, speed_mm_s=HOME_RAIL_SPEED_MM_S, wait=wait)
        return self.set_servo_angle(HOME_JOINTS_DEG, speed=HOME_JOINT_SPEED_DEG_S,
                                    wait=wait)

    def wave_goodbye(self, n_waves: int = 3, **kwargs) -> int:
        """Wave by rocking joint 6 about the current pose.

        Cosmetic, but the planner emits it and failing the whole plan on a
        greeting would be a silly reason to abort a hardware run.
        """
        code, angles = self.get_servo_angle()
        if code != 0 or not angles:
            raise RealArmError(f"wave_goodbye: could not read joint angles (code {code})")
        base = list(angles[:6])
        for _ in range(max(1, int(n_waves))):
            for delta in (WAVE_AMPLITUDE_DEG, -WAVE_AMPLITUDE_DEG):
                target = list(base)
                target[5] = base[5] + delta
                self.set_servo_angle(target, speed=WAVE_SPEED_DEG_S, wait=True)
        return self.set_servo_angle(base, speed=WAVE_SPEED_DEG_S, wait=True)

    def set_gripper(self, mode: str) -> int:
        """Confirm the fitted effector matches what the plan assumes.

        In sim this swaps a visual and a grasp tolerance. On hardware there is no
        automatic tool changer, so this cannot *change* anything -- it can only
        check. A mismatch raises: silently continuing with the wrong effector
        fitted means every subsequent grasp height and force threshold is wrong.
        """
        if mode not in EFFECTORS:
            raise ValueError(f"mode must be one of {EFFECTORS}, got {mode!r}")
        if mode != self.effector:
            raise RealArmError(
                f"plan expects the {mode!r} effector but {self.effector!r} is "
                f"fitted. There is no automatic tool changer: swap it by hand, "
                f"re-run identify_tool_load(), and restart with "
                f"effector={mode!r}.")
        return 0

    # -- operations with no hardware equivalent -----------------------------
    # Each raises with a reason. None may return a success code: a scene
    # manipulation that quietly does nothing would let a plan report success
    # while the bench was untouched.

    @unsupported("teleports a body in simulation; no physical equivalent. "
                 "Move the object by hand and re-run perception")
    def nudge_body(self, *args, **kwargs): ...

    @unsupported("restores simulator state. On hardware, resetting the cell is a "
                 "manual operator step -- see docs/hardware_preflight.md")
    def reset_scene(self, *args, **kwargs): ...

    @unsupported("reads privileged simulator state. On hardware the equivalent "
                 "must come from perception (PerceptionOutcomeReporter, B5b)")
    def physical_outcome(self, *args, **kwargs): ...

    @unsupported("reads simulator ground truth. On hardware object poses come "
                 "from perception, not the controller (locate(), B5b)")
    def get_body_pose(self, *args, **kwargs): ...

    @unsupported("needs the object's live pose, which only perception can supply "
                 "on hardware. Blocked on locate() (B5b)")
    def push_object(self, *args, **kwargs): ...

    @unsupported("needs live rack-slot occupancy, which only perception can "
                 "supply on hardware. Blocked on locate() (B5b)")
    def place_tube_in_rack(self, *args, **kwargs): ...

    @unsupported("the thermocycler lid is an Opentrons module under software "
                 "control, not something the arm manipulates. Call the Opentrons "
                 "API directly")
    def set_pcr_lid(self, *args, **kwargs): ...

    # -- lifecycle ----------------------------------------------------------

    def motion_enable(self, enable=True):
        return self.arm.motion_enable(enable=enable)

    def set_mode(self, mode):
        return self.arm.set_mode(mode)

    def set_state(self, state):
        return self.arm.set_state(state)

    def disconnect(self):
        self.arm.disconnect()
