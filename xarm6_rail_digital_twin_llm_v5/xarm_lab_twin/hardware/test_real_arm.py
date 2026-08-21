"""Hardware-free tests for hardware/real_arm.py.

Runs with no arm, no network and no controller: `xarm.wrapper.XArmAPI` is
replaced with a fake before importing the wrapper.

These pin down the two properties that made the original file dangerous:

  1. a missing rail method must fail at CONSTRUCTION, not silently mid-motion;
  2. a motion command the controller rejects must return FAILURE, never 0.

    python -m hardware.test_real_arm
"""
from __future__ import annotations

import sys
import types


# --- fake SDK ---------------------------------------------------------------

class FakeArm:
    """Stand-in for XArmAPI. `rail_api` picks which SDK generation to imitate."""

    def __init__(self, ip, rail_api="motor", rail_code=0, gripper_pos=0):
        self.ip = ip
        self.calls = []
        self._rail_code = rail_code
        self._gripper_pos = gripper_pos
        self._rail_pos = 0.0

        if rail_api == "motor":
            self.set_linear_motor_pos = self._rail_set
            self.get_linear_motor_pos = lambda: (0, self._rail_pos)
            self.set_linear_motor_enable = lambda e: 0
            self.set_linear_motor_back_origin = lambda wait=True: 0
        elif rail_api == "track":
            self.set_linear_track_pos = self._rail_set
            self.get_linear_track_pos = lambda: (0, self._rail_pos)
            self.set_linear_track_enable = lambda e: 0
            self.set_linear_track_back_origin = lambda wait=True: 0
        # rail_api == "none" -> neither generation present

    def _rail_set(self, pos, speed=None, wait=True, timeout=100, **kw):
        self.calls.append(("rail", pos))
        if self._rail_code == 0:
            self._rail_pos = pos
        return self._rail_code

    # a real controller reports a serial number; blank is the Docker case
    sn = "XI1305A2024"
    # TCP configured at the gripper tip, which is the correct setup and the one
    # in which twin coordinates transfer directly. tcp_offset=[0,0,0,...] models
    # the unconfigured case and is covered by its own test.
    tcp_offset = [0.0, 0.0, 217.0, 0.0, 0.0, 0.0]

    # motion
    def clean_error(self): return 0
    def motion_enable(self, enable=True): return 0
    def set_mode(self, mode): return 0
    def set_state(self, state): return 0
    def set_position(self, **kw): self.calls.append(("set_position", kw)); return 0
    def set_servo_angle(self, **kw): return 0
    _position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    def get_position(self): return (0, list(self._position))
    def get_servo_angle(self): return (0, [0] * 6)
    def disconnect(self): self.calls.append(("disconnect", None))

    # identity
    def get_version(self): return (0, "fake-firmware")
    def get_gripper_version(self): return (0, "fake-gripper")

    # gripper
    def set_gripper_enable(self, e, **kw): return 0
    def set_gripper_mode(self, m, **kw): return 0
    def set_gripper_position(self, pos, wait=False, **kw):
        self.calls.append(("gripper", pos)); self._gripper_pos = pos; return 0
    def get_gripper_position(self, **kw): return (0, self._gripper_pos)

    # f/t
    def set_ft_sensor_enable(self, on): return 0
    def set_ft_sensor_zero(self): return 0
    def get_ft_sensor_data(self, is_raw=False): return (0, [0.0] * 6)
    def iden_ft_sensor_load_offset(self): return 0


def install_fake_sdk():
    """Insert a fake `xarm.wrapper` into sys.modules so `real_arm` imports cleanly.

    Note the individual tests then patch `real_arm.XArmAPI` directly: real_arm
    does `from xarm.wrapper import XArmAPI`, which binds the name at import
    time, so mutating this module afterwards would have no effect.
    """
    xarm = types.ModuleType("xarm")
    wrapper = types.ModuleType("xarm.wrapper")
    wrapper.XArmAPI = FakeArm
    xarm.wrapper = wrapper
    sys.modules["xarm"] = xarm
    sys.modules["xarm.wrapper"] = wrapper
    return wrapper


# --- tests ------------------------------------------------------------------

def test_missing_rail_api_raises_at_construction(real_arm, wrapper):
    """The original bug: a renamed SDK method was caught per-call and reported
    as success. It must now be impossible to construct the wrapper at all."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="none")
    try:
        real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    except real_arm.RealArmError as exc:
        assert "linear-rail API" in str(exc), f"unexpected message: {exc}"
        return "PASS  missing rail API raises RealArmError at construction"
    return "FAIL  constructing with no rail API did not raise"


def test_both_sdk_generations_probe(real_arm, wrapper):
    """Either SDK naming must work — the probe is what makes the pin non-fatal."""
    for gen, expected in (("motor", "set_linear_motor_pos"),
                          ("track", "set_linear_track_pos")):
        real_arm.XArmAPI = lambda ip, g=gen: FakeArm(ip, rail_api=g)
        arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
        if arm._rail_api["set"] != expected:
            return f"FAIL  {gen}: probed {arm._rail_api['set']}, expected {expected}"
    return "PASS  probe resolves both SDK generations"


def test_failed_rail_move_returns_failure(real_arm, wrapper):
    """A controller rejection must not be reported as success."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor", rail_code=9)
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    try:
        rc = arm.set_rail_position(350.0)
    except real_arm.RealArmError as exc:
        assert "9" in str(exc), f"code missing from message: {exc}"
        return "PASS  rejected rail move raises instead of returning 0"
    return f"FAIL  rejected rail move returned {rc!r} instead of raising"


def test_rail_position_read_from_controller(real_arm, wrapper):
    """Position must be read back, not served from a local cache — a cache is
    how an unmoved rail looked like a moved one."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor", rail_code=9)
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    try:
        arm.set_rail_position(700.0)
    except real_arm.RealArmError:
        pass
    code, pos = arm.get_rail_position()
    if pos != 0.0:
        return f"FAIL  rail reads {pos} after a rejected move; expected 0.0"
    return "PASS  rail position reflects the controller, not the request"


def test_no_lite6_calls_for_xarm6(real_arm, wrapper):
    """open/close must route to the fitted effector, not the Lite6 API."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="standard", ft_sensor=False)
    arm.close_lite6_gripper()
    arm.open_lite6_gripper()
    grips = [c for c in arm.arm.calls if c[0] == "gripper"]
    if grips != [("gripper", 0), ("gripper", 850)]:
        return f"FAIL  unexpected gripper calls: {grips}"
    return "PASS  gripper aliases route to the standard-gripper API"


def test_verify_grasp_distinguishes_held_and_empty(real_arm, wrapper):
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor", gripper_pos=0)
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="standard", ft_sensor=False)
    state_empty, _ = arm.verify_grasp()
    arm.arm._gripper_pos = 300           # fingers held apart by an object
    state_held, _ = arm.verify_grasp()
    if (state_empty, state_held) != ("empty", "held"):
        return f"FAIL  got {(state_empty, state_held)}, expected ('empty', 'held')"
    return "PASS  verify_grasp distinguishes held from empty"


def test_sim_only_methods_raise(real_arm, wrapper):
    """Sim-only operations must raise, never return a misleading success."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    for name in ("reset_scene", "physical_outcome", "nudge_body"):
        try:
            getattr(arm, name)()
        except NotImplementedError:
            continue
        return f"FAIL  {name}() did not raise NotImplementedError"
    return "PASS  sim-only methods raise NotImplementedError"


def test_world_to_base_conversion(real_arm, wrapper):
    """A world pose must reach the controller in base coordinates."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0                      # base sits at world x=0
    arm.set_position(x=0, y=-250, z=830)
    sent = [c for c in arm.arm.calls if c[0] == "set_position"][-1][1]
    # base = world - (-350 + 350, -50, 847) = (0, -200, -17)
    got = (round(sent["x"]), round(sent["y"]), round(sent["z"]))
    if got != (0, -200, -17):
        return f"FAIL  world (0,-250,830) -> base {got}, expected (0, -200, -17)"
    return "PASS  world pose converted to base coordinates before dispatch"


def test_conversion_tracks_the_rail(real_arm, wrapper):
    """The base slides, so the same world pose maps differently per rail position."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    seen = {}
    for rail in (0.0, 700.0):
        arm.arm._rail_pos = rail
        arm.set_position(x=0, y=-250, z=830)
        seen[rail] = round([c for c in arm.arm.calls
                            if c[0] == "set_position"][-1][1]["x"])
    if seen != {0.0: 350, 700.0: -350}:
        return f"FAIL  rail-dependent x was {seen}, expected {{0: 350, 700: -350}}"
    return "PASS  conversion tracks the live rail position"


def test_round_trip_is_symmetric(real_arm, wrapper):
    """get_position must return world, or the wrapper lies about its own frame."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0
    arm.arm._position = [0.0, -200.0, -17.0, 180.0, 0.0, 0.0]  # base frame
    code, pose = arm.get_position()
    got = tuple(round(v) for v in pose[:3])
    if got != (0, -250, 830):
        return f"FAIL  base (0,-200,40) -> world {got}, expected (0, -250, 830)"
    return "PASS  get_position returns world coordinates"


def test_floor_constraint_blocks_the_benchtop(real_arm, wrapper):
    """A pose below the floor must be refused before the controller sees it."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    before = len([c for c in arm.arm.calls if c[0] == "set_position"])
    try:
        arm.set_position(x=0, y=-250, z=700)       # 50 mm INTO the bench
    except real_arm.RealArmError as exc:
        # Match on the benchtop reference rather than an exact phrase: the
        # wording changed once already and a brittle assertion failed a correct
        # refusal. What matters is that it cites the surface being protected.
        if "benchtop" not in str(exc).lower():
            return f"FAIL  refused, but not for the benchtop: {exc}"
        after = len([c for c in arm.arm.calls if c[0] == "set_position"])
        if after != before:
            return "FAIL  refused but still dispatched to the controller"
        return "PASS  sub-floor pose refused and never dispatched"
    return "FAIL  a pose 50mm inside the benchtop was accepted"


def test_floor_tracks_the_tcp_offset(real_arm, wrapper):
    """With no TCP set the controller positions the flange, so the floor must
    rise by a tool length — otherwise a twin z drives the tip into the bench."""
    class NoTcp(FakeArm):
        tcp_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    real_arm.XArmAPI = lambda ip: NoTcp(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    if abs(arm.floor_z_mm - 992.0) > 1.0:
        return f"FAIL  unset TCP gave floor {arm.floor_z_mm:.0f}, expected 992"
    try:
        arm.set_position(x=0, y=-250, z=775)     # a twin-frame grasp height
    except real_arm.RealArmError:
        return "PASS  unset TCP raises the floor and refuses a twin-frame grasp z"
    return "FAIL  unset TCP accepted a z that puts the tip inside the bench"


TESTS = [
    test_missing_rail_api_raises_at_construction,
    test_both_sdk_generations_probe,
    test_failed_rail_move_returns_failure,
    test_rail_position_read_from_controller,
    test_no_lite6_calls_for_xarm6,
    test_verify_grasp_distinguishes_held_and_empty,
    test_sim_only_methods_raise,
    test_world_to_base_conversion,
    test_conversion_tracks_the_rail,
    test_round_trip_is_symmetric,
    test_floor_constraint_blocks_the_benchtop,
    test_floor_tracks_the_tcp_offset,
]


def main() -> int:
    wrapper = install_fake_sdk()
    import importlib
    real_arm = importlib.import_module("hardware.real_arm")

    failures = 0
    for fn in TESTS:
        try:
            line = fn(real_arm, wrapper)
        except Exception as exc:  # noqa: BLE001
            line = f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}"
        if line.startswith("FAIL"):
            failures += 1
        print("  " + line)
    print(f"\n  {len(TESTS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
