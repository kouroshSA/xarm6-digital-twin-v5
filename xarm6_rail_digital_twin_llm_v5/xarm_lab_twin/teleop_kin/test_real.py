# teleop_kin/test_real.py
"""Hardware-free tests for real-arm kinesthetic teaching.

    MUJOCO_GL=egl python -m teleop_kin.test_real

What these CAN prove without a robot: the hold-to-teach debounce, the
mode-switch sequence (including the Lite 6 fallback and the failure path),
that the mirror lands the controller's angles in the twin's ctrl and qpos and
from there in the recording and the export, that a real gripper press reaches
the gripper channel with no twin object anywhere near, and -- in a separate
process, killed with SIGTERM while in manual mode -- that the controller ends
in mode 0.

What they CANNOT prove, and docs/kinesthetic_teaching.md lists as operator
exit tests: that the real arm actually releases and locks, gravity
compensation with the real payload, mirror fidelity against a measured TCP,
and the real gripper's timing.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import numpy as np  # noqa: E402

import recording  # noqa: E402
from teleop_kin.real_teach import (FakeBackend, FakeController, ManualMode,  # noqa: E402
                                   RealGripper, RealMirror, TeachHold,
                                   controller_mode)
from teleop_kin.takes import TakeControl  # noqa: E402

SCENE = "envs/lab_scene.xml"


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
def test_hold_enters_and_exits_once():
    clk = _Clock()
    h = TeachHold(debounce_s=0.12, clock=clk)
    assert h.poll() is None
    h.press()
    assert h.poll() == "enter"
    assert h.poll() is None
    h.release()
    clk.t += 0.05
    assert h.poll() is None, "released for less than the debounce: still engaged"
    clk.t += 0.10
    assert h.poll() == "exit"
    assert h.poll() is None


def test_hold_survives_autorepeat():
    """X11 autorepeat: a held key arrives as release/press pairs ~30 ms apart.
    The controller must see ONE enter, not thirty."""
    clk = _Clock()
    h = TeachHold(debounce_s=0.12, clock=clk)
    h.press()
    events = [h.poll()]
    for _ in range(30):
        clk.t += 0.03
        h.release()
        events.append(h.poll())
        clk.t += 0.002
        h.press()
        events.append(h.poll())
    assert [e for e in events if e] == ["enter"], events


def test_toggle_mode():
    h = TeachHold()
    h.toggle()
    assert h.poll() == "enter"
    h.toggle()
    assert h.poll() == "exit"


def test_manual_mode_sequence_xarm6():
    ctl = FakeController()
    m = ManualMode(ctl, log=lambda *_: None)
    assert m.enter()
    assert ctl.calls[:2] == [("set_mode", 2), ("set_state", 0)]
    assert controller_mode(ctl) == 2 and not m.used_fallback
    m.exit()
    assert ctl.calls[-2:] == [("set_mode", 0), ("set_state", 0)]
    assert controller_mode(ctl) == 0


def test_manual_mode_lite6_fallback():
    ctl = FakeController(needs_detection_param=True)
    m = ManualMode(ctl, timeout_s=0.1, log=lambda *_: None)
    assert m.enter()
    assert ("set_mode", 2, 1) in ctl.calls and m.used_fallback
    assert controller_mode(ctl) == 2


def test_manual_mode_failure_leaves_mode_0():
    class Stubborn(FakeController):
        def set_mode(self, mode, *args):
            self.calls.append(("set_mode", mode, *args))
            if mode == 0:
                self.mode = 0
            return 0
    ctl = Stubborn()
    m = ManualMode(ctl, timeout_s=0.05, log=lambda *_: None)
    assert m.enter() is False
    assert controller_mode(ctl) == 0 and not m.active


def test_safe_exit_never_raises():
    class Dead(FakeController):
        def set_mode(self, mode, *args):
            raise ConnectionError("controller gone")
    msgs = []
    m = ManualMode(Dead(), log=msgs.append)
    m.active = True
    m.safe_exit()                     # must not raise
    assert not m.active and msgs, msgs


def test_mirror_recording_and_export():
    """Consumer-side: controller angles -> twin ctrl/qpos -> trajectory.h5 ->
    export arrays. And a real gripper close reaches the gripper channel with
    the twin's gripper nowhere near an object (the weld source would record 0)."""
    from sim.mujoco_env import SimXArmAPI
    with tempfile.TemporaryDirectory() as tmp:
        saved = recording.RECORDINGS_ROOT
        recording.RECORDINGS_ROOT = Path(tmp)
        arm = SimXArmAPI(scene_xml=SCENE, render=False)
        try:
            be = FakeBackend(joints_deg=[10, -20, -30, 5, 40, -15], rail_mm=420.0)
            mirror = RealMirror(arm, be, use_rail=True)
            mirror.sync()
            with arm.lock:
                q = np.rad2deg([arm.data.qpos[arm.model.jnt_qposadr[j]] for j in arm.joint_ids])
                c = np.rad2deg([arm.data.ctrl[a] for a in arm.act_ids[1:7]])
                rail = arm.data.ctrl[arm.act_ids[0]] * 1000.0
            assert np.allclose(q, be.joints_deg, atol=1e-9), q
            assert np.allclose(c, be.joints_deg, atol=1e-9), c
            assert abs(rail - 420.0) < 1e-6, rail

            gripper = RealGripper(arm, be.open_gripper, be.close_gripper, log=lambda *_: None)
            takes = TakeControl(
                recording.Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                                   interface="kinesthetic_real", scene_xml=SCENE,
                                   enable_frames=False, gripper_fn=gripper.state),
                task_label="kinesthetic_real", log=lambda *_: None)
            takes.toggle()
            t_end = time.time() + 2.4
            sweep = np.linspace(-20, 25, 60)
            k = 0
            t_close = t_open = None
            while time.time() < t_end:
                be.joints_deg[1] = sweep[min(k, len(sweep) - 1)]   # "operator moves joint 2"
                k += 1
                mirror.sync()
                el = 2.4 - (t_end - time.time())
                if t_close is None and el > 0.8:
                    t_close = el
                    gripper.command(close=True)
                if t_open is None and el > 1.6:
                    t_open = el
                    gripper.command(close=False)
                time.sleep(1 / 60)
            gripper.wait_idle()
            session = Path(takes.stop(kept=True))
            meta = json.loads((session / "metadata.json").read_text())
            assert meta["interface"] == "kinesthetic_real"

            import h5py
            with h5py.File(session / "trajectory.h5") as f:
                t = np.asarray(f["t_wall"])
                j2 = np.asarray(f["joints_deg"])[:, 1]
                a2 = np.rad2deg(np.asarray(f["ctrl"])[:, 2])
                grip = np.asarray(f["gripper"])
            assert j2.max() - j2.min() > 30, "joint 2 sweep missing from the state"
            assert np.allclose(a2, j2, atol=0.5), "action and state disagree for a mirror"
            assert grip[t < t_close - 0.05].max() == 0.0
            assert grip[(t > t_close + 0.1) & (t < t_open - 0.05)].min() == 1.0, \
                "a real gripper close did not reach the gripper channel"
            assert grip[t > t_open + 0.1].max() == 0.0
            assert be.gripper_calls == ["close", "open"]
            ok, _ = arm.verify_grasp()
            assert ok == "empty", "control: the twin welded nothing, so this IS gripper_fn"

            sys.path.insert(0, str(HERE / "scripts"))
            from export_lerobot import session_arrays
            arr = session_arrays(str(session))
            assert arr is not None and arr["images"] is None      # state-only take
            assert arr["action"][:, 7].max() == 1.0
            out = subprocess.run([sys.executable, "scripts/export_lerobot.py", "--dry-run",
                                  "--recordings", tmp], capture_output=True, text=True,
                                 timeout=120)
            assert out.returncode == 0 and "state-only     : 1" in out.stdout, out.stdout
            gripper.stop()
        finally:
            recording.RECORDINGS_ROOT = saved
            arm.disconnect()


def _run_runner_and_kill(sig: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "ctl.json"
        p = subprocess.Popen(
            [sys.executable, "scripts/run_kinesthetic_real.py", "--fake", "--fake-hold",
             "--no-viewer", "--no-record", "--teach-toggle",
             "--fake-state-file", str(state)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "MUJOCO_GL": "egl"})
        try:
            t_end = time.time() + 60
            while time.time() < t_end:
                if state.exists():
                    try:
                        if json.loads(state.read_text())["mode"] == 2:
                            break
                    except (ValueError, KeyError):
                        pass
                time.sleep(0.05)
            else:
                raise AssertionError("runner never entered manual mode")
            time.sleep(0.3)
            p.send_signal(sig)
            out, _ = p.communicate(timeout=30)
        finally:
            if p.poll() is None:
                p.kill()
        return {"final": json.loads(state.read_text()), "rc": p.returncode, "out": out}


def test_sigterm_in_manual_mode_ends_in_mode_0():
    """Exit test 3 (crash safety), against a real process. SIGKILL is the
    negative control: it cannot be handled, so the fake controller must still
    say mode 2 -- proving the check is reading process state, and that it is
    the SIGTERM handler that does the work."""
    r = _run_runner_and_kill(signal.SIGTERM)
    assert r["final"]["mode"] == 0 and r["final"]["state"] == 0, (r["final"], r["out"][-1500:])
    assert "returned to mode 0" in r["out"], r["out"][-1500:]
    k = _run_runner_and_kill(signal.SIGKILL)
    assert k["final"]["mode"] == 2, ("negative control did not stay in mode 2", k["final"])


def _main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
