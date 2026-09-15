"""Hardware-free tests for hardware/real_arm.py.

Runs with no arm, no network and no controller: `xarm.wrapper.XArmAPI` is
replaced with a fake before importing the wrapper.

These pin down the two properties that made the original file dangerous:

  1. a missing rail method must fail at CONSTRUCTION, not silently mid-motion;
  2. a motion command the controller rejects must return FAILURE, never 0.

    python -m hardware.test_real_arm
"""
from __future__ import annotations

import math
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
            self.get_linear_motor_is_enabled = lambda: (0, 1)
            self.get_linear_motor_on_zero = lambda: (0, 1)
        elif rail_api == "track":
            self.set_linear_track_pos = self._rail_set
            self.get_linear_track_pos = lambda: (0, self._rail_pos)
            self.set_linear_track_enable = lambda e: 0
            self.set_linear_track_back_origin = lambda wait=True: 0
            self.get_linear_track_is_enabled = lambda: (0, 1)
            self.get_linear_track_on_zero = lambda: (0, 1)
            self.get_linear_motor_is_enabled = lambda: (0, 1)
            self.get_linear_motor_on_zero = lambda: (0, 1)
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
    #: Mirrors the real controller's states: 2 = ready, 5 = STATE_NOT_READY.
    #: Starts ready, matching a freshly connected arm.
    _state = 2
    def clean_error(self): return 0
    def motion_enable(self, enable=True): return 0
    def set_mode(self, mode): return 0
    def get_state(self): return (0, self._state)
    def set_state(self, state):
        if state == 0:
            self._state = 2     # matches the real controller after set_state(0)
        return 0
    def set_position(self, **kw):
        # Real controllers refuse motion with code 9 (STATE_NOT_READY) when not
        # in state 2. Reproducing that here is what makes the F/T state-drop
        # tests below a real check rather than a check against a stub that
        # cannot fail the way the hardware does.
        if self._state != 2:
            return 9
        self.calls.append(("set_position", kw))
        # Track the commanded pose so a simulated descent actually descends.
        self._position = [kw.get(k, d) for k, d in
                          zip(("x", "y", "z", "roll", "pitch", "yaw"),
                              self._position)]
        return 0
    def set_servo_angle(self, **kw): return 0
    #: latched controller error/warn, as get_err_warn_code reports it. A test
    #: sets this to simulate a fault mid-motion (e.g. 31, collision).
    _err_warn = [0, 0]
    def get_err_warn_code(self): return (0, list(self._err_warn))
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
    #: force reading get_ft_sensor_data returns; a test can point this at
    #: something nonzero to check get_contact_force_n's magnitude math.
    _ft_force = [0.0, 0.0, 0.0]
    #: BASE z of a simulated RIGID surface, or None for free space. Below it the
    #: reported force rises at _STIFFNESS_N_PER_MM -- calibrated to the real
    #: measurement of 2026-09-14, where a 10 mm overshoot into a benchtop
    #: produced 90 N. That ratio is what makes the step-size test meaningful.
    _contact_z = None
    _STIFFNESS_N_PER_MM = 9.0
    def set_ft_sensor_enable(self, on):
        self._state = 5          # matches the real controller's undocumented drop
        return 0
    def set_ft_sensor_zero(self):
        self._state = 5          # ditto -- both calls do this, not just the first
        return 0
    def get_ft_sensor_data(self, is_raw=False):
        if self._contact_z is not None:
            penetration = self._contact_z - self._position[2]
            if penetration > 0:
                return (0, [0.0, 0.0, penetration * self._STIFFNESS_N_PER_MM,
                            0.0, 0.0, 0.0])
        return (0, list(self._ft_force) + [0.0, 0.0, 0.0])
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
    """A world pose must reach the controller in base coordinates.

    The expectation is the red cube from the BASE_YAW_DEG measurement table in
    arm_backend: world (0, -250) is base (200, 0) at rail 350. This test
    previously expected (0, -200, -17), which is the same conversion done as a
    pure TRANSLATION -- it was written before the base yaw was measured and was
    left asserting the superseded frame, so it failed for a year of commits
    while the code under it was right.
    """
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0                      # base sits at world x=0
    arm.set_position(x=0, y=-250, z=830)
    sent = [c for c in arm.arm.calls if c[0] == "set_position"][-1][1]
    got = (round(sent["x"]), round(sent["y"]), round(sent["z"]))
    if got != (200, 0, -17):
        return f"FAIL  world (0,-250,830) -> base {got}, expected (200, 0, -17)"
    return "PASS  world pose converted to base coordinates before dispatch"


def test_conversion_tracks_the_rail(real_arm, wrapper):
    """The base slides, so the same world pose maps differently per rail position.

    Under the measured -90 deg base yaw the rail runs along base **y**, not base
    x -- the arm's +y points down the rail. Reading x here (as this test used to)
    watches the one axis the rail cannot move, which is why it reported a
    constant 200 and failed.
    """
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    seen = {}
    for rail in (0.0, 700.0):
        arm.arm._rail_pos = rail
        arm.set_position(x=0, y=-250, z=830)
        seen[rail] = round([c for c in arm.arm.calls
                            if c[0] == "set_position"][-1][1]["y"])
    if seen != {0.0: 350, 700.0: -350}:
        return f"FAIL  rail-dependent base y was {seen}, expected {{0: 350, 700: -350}}"
    return "PASS  conversion tracks the live rail position"


def test_round_trip_is_symmetric(real_arm, wrapper):
    """get_position must return world, or the wrapper lies about its own frame.

    An actual round trip rather than two hand-written constants: command a world
    pose, take whatever reached the controller, hand that back as the arm's
    reported base pose, and require the original world pose out. Hand-written
    constants are how this test came to assert a superseded frame while passing
    review -- and the pair it used, base (0,-200,-17), is not the base image of
    world (0,-250,830) under any yaw.

    **All six numbers are checked.** Orientation was excluded here, and that
    omission is exactly why `get_position` returned a base-frame rpy for weeks:
    the only test of the frame contract could not see the half that was wrong.
    """
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0

    sent_world = (0.0, -250.0, 830.0, 180.0, 0.0, 25.0)
    arm.set_position(*sent_world[:3], roll=sent_world[3], pitch=sent_world[4],
                     yaw=sent_world[5])
    sent = [c for c in arm.arm.calls if c[0] == "set_position"][-1][1]
    arm.arm._position = [sent["x"], sent["y"], sent["z"],
                         sent["roll"], sent["pitch"], sent["yaw"]]

    code, pose = arm.get_position()
    got = tuple(round(v, 6) for v in pose[:6])
    if got != sent_world:
        return (f"FAIL  world {sent_world} -> base "
                f"({sent['x']:.0f},{sent['y']:.0f},{sent['z']:.0f},"
                f"{sent['roll']:.0f},{sent['pitch']:.0f},{sent['yaw']:.0f})"
                f" -> world {got}; the round trip is not the identity")
    # A round trip closes for any self-consistent pair of wrong transforms, so
    # pin the base pose too: the orientation must actually have been rotated.
    if round(sent["yaw"]) != 115:      # 25 world - (-90) base yaw
        return (f"FAIL  round trip closes but the controller was sent yaw="
                f"{sent['yaw']:.1f}, expected 115; the orientation is being "
                f"passed through raw and cancelling itself on the way back")
    return "PASS  set/get round trip is the identity in world, orientation included"


def test_rpy_matches_matrix_composition(real_arm, wrapper):
    """The yaw-addition shortcut must equal the full matrix product.

    `base_to_world_rpy_deg` adds BASE_YAW_DEG to yaw and leaves roll and pitch
    alone. That is exact under R = Rz(yaw)Ry(pitch)Rx(roll), because a base yaw
    composes on the left of the Rz -- but "add 90 to one euler angle" is also
    the shape of a great many wrong frame conversions, so prove it rather than
    reason about it. Pitched and rolled poses are included deliberately: a
    careless shortcut agrees with the matrices at pitch=0 and diverges away
    from it.
    """
    import numpy as np
    from arm_backend import BASE_YAW_DEG, base_to_world_rpy_deg
    from perception.d435i_calib import _euler_to_mat

    c, sn = math.cos(math.radians(BASE_YAW_DEG)), math.sin(math.radians(BASE_YAW_DEG))
    rz_base = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])

    worst = 0.0
    for rpy in [(180, 0, 0), (180, 0, 25), (0, 0, 0), (90, 45, -170),
                (-30, 60, 120), (12, -75, 200), (180, 89, -45)]:
        want = rz_base @ _euler_to_mat(*[math.radians(v) for v in rpy])
        got = _euler_to_mat(*[math.radians(v)
                              for v in base_to_world_rpy_deg(rpy)])
        worst = max(worst, float(np.abs(want - got).max()))
    if worst > 1e-9:
        return (f"FAIL  the yaw-addition shortcut differs from Rz(BASE_YAW) @ R "
                f"by up to {worst:.2e}; it is not the same rotation")
    return f"PASS  rpy conversion equals the matrix composition (max {worst:.1e})"


def test_wrist_camera_sees_a_stationary_object_as_stationary(real_arm, wrapper):
    """The CONSUMER test: a fixed object must not move when only the rail does.

    This is the 2026-08-26 hardware failure reproduced with no hardware. With the
    arm held at one joint pose and the rail swept, a stationary cube deprojected
    to world x = -64.9, 33.3, 129.3 at rail 300/400/500 -- tracking the rail
    almost 1:1 instead of staying put, because `cam_to_world` was handed a world
    translation with a base rotation and composed them as though they shared a
    frame.

    Ground truth here is stated once, as a single rigid transform world = Rz(
    BASE_YAW) * base + origin(rail), and the camera chain is required to agree
    with it. The wrapper reaches the same place along two separate code paths
    (`base_to_world_mm` for the translation, `base_to_world_rpy_deg` for the
    rotation); this is what forces those two to describe the SAME transform,
    which is the property that was actually broken.
    """
    import numpy as np
    try:
        from perception import d435i_calib as calib
        from perception.realsense_camera import RealSenseWristCamera
    except Exception as exc:  # noqa: BLE001
        return f"SKIP  perception not importable ({type(exc).__name__}: {exc})"
    from arm_backend import BASE_AT_RAIL_ZERO_MM, BASE_YAW_DEG

    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)

    # One fixed joint pose. The controller reports the same BASE pose at every
    # rail position -- the rail carries the whole base, so nothing changes in
    # base coordinates. That is what makes this sweep a clean test.
    base_pose = [200.0, 0.0, 150.0, 180.0, 0.0, 30.0]
    arm.arm._position = list(base_pose)

    cam = object.__new__(RealSenseWristCamera)
    cam.arm = arm

    c, sn = math.cos(math.radians(BASE_YAW_DEG)), math.sin(math.radians(BASE_YAW_DEG))
    rz_base = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
    r_fc, t_fc = calib.flange_to_color_optical()
    tcp = np.array([0.0, 0.0, float(arm.tcp_offset_z_mm)])

    def truth(rail_mm):
        """(R, t) camera-optical -> world, from the frame definition alone."""
        r_bf = calib._euler_to_mat(*[math.radians(v) for v in base_pose[3:6]])
        t_bf = (np.array(base_pose[:3]) - r_bf @ tcp) / 1000.0
        bx, by, bz = BASE_AT_RAIL_ZERO_MM
        origin = np.array([bx + rail_mm, by, bz]) / 1000.0
        r_wf, t_wf = rz_base @ r_bf, rz_base @ t_bf + origin
        return r_wf @ r_fc, t_wf + r_wf @ t_fc

    cube_world = np.array([0.100, -0.250, 0.780])      # stationary, metres
    seen = []
    for rail in (300.0, 400.0, 500.0):
        arm.arm._rail_pos = rail
        r_true, t_true = truth(rail)
        # Where the cube falls in the camera's own frame at this rail position.
        p_cam = r_true.T @ (cube_world - t_true)
        # Deproject it with the chain under test.
        m = cam.cam_to_world()
        if m is None:
            return "FAIL  cam_to_world returned None with an arm attached"
        seen.append(m[:3, :3] @ p_cam + m[:3, 3])

    spread_mm = float(np.abs(np.array(seen) - np.array(seen[0])).max()) * 1000.0
    err_mm = float(np.abs(np.array(seen) - cube_world).max()) * 1000.0
    if spread_mm > 1e-6:
        drift = ", ".join(f"x={p[0] * 1000:.1f}" for p in seen)
        return (f"FAIL  a stationary cube moved {spread_mm:.1f} mm as the rail "
                f"swept 300->500 ({drift}); the camera pose is not in one frame")
    if err_mm > 1e-6:
        return (f"FAIL  the cube deprojected {err_mm:.1f} mm from where it is; "
                f"stable across the rail but in the wrong place")
    return ("PASS  a stationary cube stays put across a 200 mm rail sweep "
            f"(spread {spread_mm:.1e} mm)")


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


def test_unhomed_rail_refuses_cartesian(real_arm, wrapper):
    """An un-homed rail reports 0 while the carriage sits elsewhere, so every
    world->base conversion would be silently offset. Refuse rather than guess."""
    class Unhomed(FakeArm):
        def __init__(self, ip, **kw):
            super().__init__(ip, **kw)
            self.get_linear_motor_is_enabled = lambda: (0, 1)
            self.get_linear_motor_on_zero = lambda: (0, 0)     # enabled, not homed
    real_arm.XArmAPI = lambda ip: Unhomed(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    before = len([c for c in arm.arm.calls if c[0] == "set_position"])
    try:
        arm.set_position(x=0, y=-250, z=830)
    except real_arm.RealArmError as exc:
        if "homed" not in str(exc):
            return f"FAIL  refused, but not for the rail: {exc}"
        if len([c for c in arm.arm.calls if c[0] == "set_position"]) != before:
            return "FAIL  refused but still dispatched"
        return "PASS  un-homed rail refuses Cartesian moves"
    return "FAIL  un-homed rail accepted a Cartesian move"


def test_ft_sensor_calls_really_do_drop_state(real_arm, wrapper):
    """Negative control: prove the FakeArm's simulated state-drop has teeth.

    Calls the raw SDK F/T-enable with NO readiness reassert after, and checks
    set_position then fails exactly as it does on real hardware (controller
    code 9, STATE_NOT_READY). Without this, the positive test below could pass
    for a reason that has nothing to do with the real bug.
    """
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm.set_ft_sensor_enable(1)          # raw SDK call, no ready() after
    try:
        arm.set_position(x=0, y=-250, z=830)
    except real_arm.RealArmError as exc:
        if "code 9" not in str(exc):
            return f"FAIL  wrong error after the simulated drop: {exc}"
        return "PASS  an un-reasserted F/T call correctly blocks motion"
    return "FAIL  set_position succeeded despite the simulated state drop"


def test_zero_ft_sensor_leaves_the_arm_ready(real_arm, wrapper):
    """zero_ft_sensor must reassert readiness after BOTH the enable and the
    zero call, not just the first -- see zero_ft_sensor's own docstring for
    why one reassert is not enough. If either is skipped, this fails exactly
    as the hardware did on 2026-09-07."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    rc = arm.zero_ft_sensor()
    if rc != 0:
        return f"FAIL  zero_ft_sensor returned {rc}"
    if arm.arm.get_state()[1] != 2:
        return f"FAIL  arm left in state {arm.arm.get_state()[1]}, not ready (2)"
    try:
        arm.set_position(x=0, y=-250, z=830)
    except real_arm.RealArmError as exc:
        return f"FAIL  set_position after zero_ft_sensor still refused: {exc}"
    return "PASS  zero_ft_sensor reasserts readiness after both F/T calls"


def test_get_contact_force_n_is_the_force_magnitude(real_arm, wrapper):
    """get_contact_force_n must report norm(Fx,Fy,Fz), not raw components or
    torque -- a caller comparing this against a newton ceiling needs a single
    number, and a wrong axis or an included torque term would silently pass
    or fail a force-limited descent for the wrong reason."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._ft_force = [3.0, 4.0, 0.0]      # 3-4-5 triangle -> magnitude 5.0
    got = arm.get_contact_force_n()
    if abs(got - 5.0) > 1e-9:
        return f"FAIL  got {got}, expected 5.0"
    return "PASS  get_contact_force_n is the 3-axis force magnitude"


def test_descend_until_contact_finds_a_surface(real_arm, wrapper):
    """The normal case: a surface is there, the descent stops on force."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0
    arm.arm._contact_z = -60.0          # simulated rigid surface, BASE frame
    # world z=830 -> base z=-17, so start clear of it and descend into it
    z, f, hit = arm.descend_until_contact(x=0, y=-250, z_from=830, z_floor=740,
                                          step_mm=2.0, f_max_n=3.0, settle_s=0.0)
    if not hit:
        return f"FAIL  surface at base -60 was not detected (stopped z={z}, F={f})"
    if f <= 3.0:
        return f"FAIL  reported contact at only {f:.2f} N, below the 3.0 ceiling"
    return f"PASS  descent stops on contact (world z={z:.0f}, |F|={f:.1f} N)"


def test_descend_until_contact_raises_when_nothing_is_there(real_arm, wrapper):
    """The 2026-09-15 bug: the loop ran to its floor touching nothing and the
    caller released a plate into thin air, off the edge of the bench.

    No contact must be impossible to walk past -- it raises rather than
    returning a value a caller can forget to check."""
    real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
    arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
    arm.arm._rail_pos = 350.0
    arm.arm._contact_z = None           # nothing beneath the tool
    try:
        arm.descend_until_contact(x=0, y=-250, z_from=830, z_floor=790,
                                  step_mm=2.0, f_max_n=3.0, settle_s=0.0)
    except real_arm.RealArmError as exc:
        if "never made contact" not in str(exc):
            return f"FAIL  raised, but not about contact: {exc}"
        return "PASS  descent over empty space raises instead of returning"
    return "FAIL  descended into nothing and returned normally -- a caller would release here"


def test_descend_step_size_governs_impact_force(real_arm, wrapper):
    """Why step_mm must match the target's STIFFNESS, not the distance.

    Force is only sampled between steps, so the travel left in a step when
    contact begins becomes force. Against a rigid surface a 10 mm step lands
    roughly 5x the force of a 2 mm one -- which is how a benchtop probe reached
    90 N on 2026-09-14. Pins the ratio so the docstring's advice stays true."""
    peak = {}
    for step in (2.0, 10.0):
        real_arm.XArmAPI = lambda ip: FakeArm(ip, rail_api="motor")
        arm = real_arm.RealXArmAPI("127.0.0.1", effector="none", ft_sensor=False)
        arm.arm._rail_pos = 350.0
        arm.arm._contact_z = -60.0
        _, f, hit = arm.descend_until_contact(x=0, y=-250, z_from=830, z_floor=700,
                                              step_mm=step, f_max_n=3.0,
                                              settle_s=0.0)
        if not hit:
            return f"FAIL  step {step}: no contact detected"
        peak[step] = f
    if peak[10.0] <= peak[2.0]:
        return f"FAIL  10mm step gave {peak[10.0]:.1f} N, not more than 2mm's {peak[2.0]:.1f} N"
    return (f"PASS  step size governs impact force "
            f"(2mm -> {peak[2.0]:.1f} N, 10mm -> {peak[10.0]:.1f} N)")


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
    test_rpy_matches_matrix_composition,
    test_wrist_camera_sees_a_stationary_object_as_stationary,
    test_floor_constraint_blocks_the_benchtop,
    test_floor_tracks_the_tcp_offset,
    test_unhomed_rail_refuses_cartesian,
    test_ft_sensor_calls_really_do_drop_state,
    test_zero_ft_sensor_leaves_the_arm_ready,
    test_get_contact_force_n_is_the_force_magnitude,
    test_descend_until_contact_finds_a_surface,
    test_descend_until_contact_raises_when_nothing_is_there,
    test_descend_step_size_governs_impact_force,
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
