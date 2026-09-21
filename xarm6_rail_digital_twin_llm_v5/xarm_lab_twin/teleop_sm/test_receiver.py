# teleop_sm/test_receiver.py
"""Unit tests for teleop_sm/receiver.py.

    python -m teleop_sm.test_receiver     # plain-python, prints PASS/FAIL
    pytest teleop_sm/test_receiver.py

MUST PASS WITH NO HARDWARE and without building the twin: the arm is a stub
that records what was asked of it. That is deliberate -- these tests are about
the *integrator's* behaviour (does the target accumulate correctly, does it
clamp, does it freeze on IK failure), and a real MuJoCo scene would make those
questions harder to ask, not easier.

The one thing a stub cannot check is whether IK is right; `scripts/ik_sanity.py`
owns that.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from teleop_sm import config
from teleop_sm.buttons import Action
from teleop_sm.receiver import Mode, SpaceMouseReceiver
from vr import config as vr_config


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class _FakeIK:
    """Solves everything, unless told to fail."""

    def __init__(self):
        self.joint_ids = list(range(6))
        self.fail = False
        self.calls = []

    def solve(self, target_pos_m, target_rot=None, seed_q=None):
        self.calls.append(np.asarray(target_pos_m, float).copy())
        if self.fail:
            return None
        return np.zeros(6)


class _FakeData:
    def __init__(self):
        self.ctrl = np.zeros(8)
        self.qpos = np.zeros(16)
        # EE parked mid-workspace so a jog has room in every direction.
        self.site_xpos = np.array([[0.0, 0.35, 0.95]])
        self.site_xmat = np.array([np.eye(3).reshape(9)])


class _FakeArm:
    """Minimal SimXArmAPI surface the receiver actually touches."""

    def __init__(self):
        self.lock = threading.RLock()
        self.model = object()
        self.data = _FakeData()
        self.act_ids = [0, 1, 2, 3, 4, 5, 6]
        self.ee_site = 0
        self.ik_solver = _FakeIK()
        self.gripper_calls = []
        self.reset_calls = 0
        self.home_calls = 0
        self.set_position_rc = 0
        self.set_position_calls = []

    def open_lite6_gripper(self):
        self.gripper_calls.append("open")
        return 0

    def close_lite6_gripper(self):
        self.gripper_calls.append("close")
        return 0

    def reset_scene(self):
        self.reset_calls += 1
        return 0

    def go_home(self, wait=False):
        self.home_calls += 1
        return 0

    def set_position(self, x, y, z, roll, pitch, yaw, speed=None, wait=False):
        self.set_position_calls.append((x, y, z, roll, pitch, yaw))
        return self.set_position_rc


class _FakeRecorder:
    def __init__(self):
        self.is_recording = False
        self.commands = []
        self.started = 0
        self.stopped = 0

    def start(self):
        self.is_recording = True
        self.started += 1

    def stop(self, kept=True, task_label=None):
        self.is_recording = False
        self.stopped += 1
        return "/tmp/fake_session"

    def log_command(self, name, payload):
        self.commands.append((name, payload))


def _mj_forward_noop(model, data):
    pass


# mujoco.mj_forward is called inside _ee_pose_twin; the stub has no real model.
import teleop_sm.receiver as _recv_mod  # noqa: E402
_recv_mod.mujoco.mj_forward = _mj_forward_noop


def _rx(model="compact", recorder=None):
    arm = _FakeArm()
    return arm, SpaceMouseReceiver(arm, recorder=recorder, model=model)


def _axes(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0):
    return np.array([tx, ty, tz, rx, ry, rz], dtype=float)


# ---------------------------------------------------------------------------
# target integration
# ---------------------------------------------------------------------------
def test_target_integrates_over_time():
    """Full deflection for 1 s must move the target by the speed cap.

    This is the contract of a velocity jog: deflection is speed, and holding
    it moves further. Testing one tick would pass for an implementation that
    treats deflection as position.
    """
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.01)      # seed
    start = rx._target_pos_m.copy()

    dt = 0.02
    for _ in range(int(1.0 / dt)):
        rx.tick(_axes(tx=1.0), set(), set(), dt)

    moved_mm = (rx._target_pos_m - start) * 1000.0
    # The smoother lags a first-order step, so allow generous tolerance; what
    # matters is the order of magnitude and the direction, not the last mm.
    assert moved_mm[0] > config.MAX_POS_SPEED * 0.5, (
        f"1 s at full deflection moved only {moved_mm[0]:.1f} mm, "
        f"expected order {config.MAX_POS_SPEED}")


def test_zero_input_does_not_move_the_target():
    """Hands off means stop. With an integrator this is not automatic."""
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    before = rx._target_pos_m.copy()
    for _ in range(50):
        rx.tick(_axes(), set(), set(), 0.02)
    assert np.allclose(rx._target_pos_m, before), "target drifted with no input"


def test_target_is_not_reread_from_the_arm():
    """The integrator must own its target.

    If it re-read the EE each tick, a lagging arm would drag the target back
    and the jog would creep or run away. Here the stub arm never moves, so a
    re-reading implementation would show zero net travel.
    """
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    for _ in range(20):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)
    ee_m = arm.data.site_xpos[0]
    assert not np.allclose(rx._target_pos_m, ee_m), (
        "target collapsed back onto the (stationary) EE -- it is being "
        "re-read from the arm rather than integrated")


# ---------------------------------------------------------------------------
# workspace clamp
# ---------------------------------------------------------------------------
def test_workspace_clamp_engages_at_the_boundary():
    """Push +Z for long enough and the target must stop at the AABB ceiling."""
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    zmax_mm = vr_config.WORKSPACE_AABB_MM["z"][1]

    for _ in range(3000):                      # far more than enough
        rx.tick(_axes(tz=1.0), set(), set(), 0.02)

    z_mm = rx._target_pos_m[2] * 1000.0
    assert z_mm <= zmax_mm + 1e-6, f"target escaped the clamp: {z_mm} > {zmax_mm}"
    assert z_mm > zmax_mm - 5.0, (
        f"target stalled well short of the ceiling ({z_mm} vs {zmax_mm}); "
        f"the clamp may be engaging early")


def test_clamp_reuses_the_vr_workspace():
    """There must be exactly ONE workspace box in the tree.

    Two copies that start equal and drift apart is the repo's standing defect
    class #1, so this asserts the identity rather than the values.
    """
    assert config.WORKSPACE_AABB_MM is vr_config.WORKSPACE_AABB_MM


# ---------------------------------------------------------------------------
# IK failure
# ---------------------------------------------------------------------------
def test_target_freezes_on_ik_failure():
    """THE failure mode of an integrating jog.

    Without a rollback the operator keeps pushing, the target marches into
    unreachable space, and the arm leaps when it becomes reachable again.
    """
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    for _ in range(10):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)

    frozen = rx._target_pos_m.copy()
    arm.ik_solver.fail = True

    for _ in range(50):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)

    assert rx.ik_fail is True, "ik_fail flag not raised for the HUD"
    assert np.allclose(rx._target_pos_m, frozen, atol=1e-9), (
        f"target kept integrating through IK failure: "
        f"{(rx._target_pos_m - frozen) * 1000} mm of runaway")


def test_recovers_after_ik_returns():
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    arm.ik_solver.fail = True
    for _ in range(10):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)
    assert rx.ik_fail is True

    arm.ik_solver.fail = False
    before = rx._target_pos_m.copy()
    for _ in range(10):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)
    assert rx.ik_fail is False, "ik_fail stuck on after IK recovered"
    assert rx._target_pos_m[0] > before[0], "jog did not resume"


# ---------------------------------------------------------------------------
# rail mode
# ---------------------------------------------------------------------------
def test_rail_mode_zeroes_the_non_rail_axes():
    """In RAIL mode only Y drives, and the arm target must not move."""
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    rx.mode = Mode.RAIL
    target_before = rx._target_pos_m.copy()
    rail_before = rx.rail_mm

    for _ in range(25):
        rx.tick(_axes(tx=1.0, ty=1.0, tz=1.0, rx=1.0), set(), set(), 0.02)

    assert rx.rail_mm > rail_before, "rail did not move in rail mode"
    # The arm target is re-seeded from the (stationary) stub arm each rail
    # step, so it must equal where it started, not have integrated tx/tz.
    assert np.allclose(rx._target_pos_m, target_before, atol=1e-9), (
        "non-rail axes leaked into the arm target while in rail mode")


def test_rail_clamps_to_travel_limits():
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    rx.mode = Mode.RAIL
    for _ in range(4000):
        rx.tick(_axes(ty=1.0), set(), set(), 0.02)
    assert rx.rail_mm <= config.RAIL_MAX_MM + 1e-9, rx.rail_mm

    for _ in range(8000):
        rx.tick(_axes(ty=-1.0), set(), set(), 0.02)
    assert rx.rail_mm >= config.RAIL_MIN_MM - 1e-9, rx.rail_mm


def test_rail_writes_the_rail_actuator_in_metres():
    """ctrl is metres, rail_mm is millimetres. A unit slip here is a 1000x
    command, which in sim is a teleport."""
    from sim.mujoco_env import RAIL_ACT
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    rx.mode = Mode.RAIL
    for _ in range(10):
        rx.tick(_axes(ty=1.0), set(), set(), 0.02)
    assert abs(arm.data.ctrl[arm.act_ids[RAIL_ACT]] - rx.rail_mm / 1000.0) < 1e-12


def test_hold_rail_button_switches_mode_without_cycling():
    """On the Compact, button 1 is both MODE_CYCLE (edge) and RAIL_MODE (hold).

    Holding it must drive the rail and must NOT also cycle the mode -- that
    overload is forced by having only two buttons, so the disambiguation has
    to be correct.
    """
    arm, rx = _rx(model="compact")
    rx.tick(_axes(), set(), set(), 0.02)
    mode_before = rx.mode
    rail_before = rx.rail_mm

    # Button 1 held AND its edge present, which is what a real press delivers.
    rx.tick(_axes(ty=1.0), {1}, {1}, 0.02)

    assert rx.mode == mode_before, "holding the rail key also cycled the mode"
    assert rx.rail_mm != rail_before, "holding the rail key did not drive the rail"


# ---------------------------------------------------------------------------
# discrete actions
# ---------------------------------------------------------------------------
def test_gripper_toggles_on_edge_only():
    arm, rx = _rx(model="compact")
    rx.tick(_axes(), set(), set(), 0.02)

    rx.tick(_axes(), {0}, {0}, 0.02)
    assert arm.gripper_calls == ["close"], arm.gripper_calls

    # Still held, no new edge: must not toggle again.
    for _ in range(10):
        rx.tick(_axes(), set(), {0}, 0.02)
    assert arm.gripper_calls == ["close"], (
        f"gripper toggled while held: {arm.gripper_calls}")

    rx.tick(_axes(), {0}, set(), 0.02)
    assert arm.gripper_calls == ["close", "open"], arm.gripper_calls


def test_mode_cycles():
    arm, rx = _rx(model="compact")
    rx.tick(_axes(), set(), set(), 0.02)
    assert rx.mode == Mode.ARM
    rx.dispatch(Action.MODE_CYCLE)
    assert rx.mode == Mode.RAIL
    rx.dispatch(Action.MODE_CYCLE)
    assert rx.mode == Mode.ARM, "mode must cycle back round"


def test_unbound_button_is_ignored():
    arm, rx = _rx(model="compact")
    rx.tick(_axes(), set(), set(), 0.02)
    rx.tick(_axes(), {9}, {9}, 0.02)          # not in the compact map
    assert arm.gripper_calls == []


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------
def test_record_toggle_starts_and_stops():
    rec = _FakeRecorder()
    arm, rx = _rx(model="pro", recorder=rec)
    rx.tick(_axes(), set(), set(), 0.02)

    rx.dispatch(Action.RECORD_TOGGLE)
    assert rec.is_recording and rec.started == 1
    rx.dispatch(Action.RECORD_TOGGLE)
    assert not rec.is_recording and rec.stopped == 1


def test_mode_changes_are_logged_for_replay():
    """A replayed session must be interpretable: which mode was active changes
    how the trace reads, so it has to be in the command log."""
    rec = _FakeRecorder()
    arm, rx = _rx(model="pro", recorder=rec)
    rx.tick(_axes(), set(), set(), 0.02)
    rx.dispatch(Action.RECORD_TOGGLE)
    rx.dispatch(Action.MODE_CYCLE)

    names = [n for n, _ in rec.commands]
    assert "spacemouse_mode" in names, rec.commands
    modes = [p["mode"] for n, p in rec.commands if n == "spacemouse_mode"]
    assert Mode.RAIL in modes, modes


def test_no_recorder_is_not_an_error():
    arm, rx = _rx(model="pro", recorder=None)
    rx.tick(_axes(), set(), set(), 0.02)
    rx.dispatch(Action.RECORD_TOGGLE)         # must not raise


# ---------------------------------------------------------------------------
# reset / status
# ---------------------------------------------------------------------------
def test_reset_resyncs_the_target():
    """After a scene reset the arm has teleported; a stale target would fling
    it straight back."""
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    for _ in range(30):
        rx.tick(_axes(tx=1.0), set(), set(), 0.02)
    assert not np.allclose(rx._target_pos_m, arm.data.site_xpos[0])

    rx.dispatch(Action.RESET_SCENE)
    assert arm.reset_calls == 1
    assert np.allclose(rx._target_pos_m, arm.data.site_xpos[0]), (
        "target not re-anchored after reset")


def test_status_has_the_vr_hud_keys():
    """A HUD written against TeleopReceiver.status() must keep working."""
    arm, rx = _rx()
    rx.tick(_axes(), set(), set(), 0.02)
    st = rx.status()
    for key in ("type", "recording", "gripper_closed", "rail_mm", "ik_fail",
                "clutch", "servo_mode"):
        assert key in st, f"status() is missing VR key {key!r}: {st}"
    assert st["type"] == "status"


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
