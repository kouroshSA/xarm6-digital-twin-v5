# teleop_kin/test_receiver.py
"""Headless tests for kinesthetic teaching in the twin.

    MUJOCO_GL=egl python -m teleop_kin.test_receiver

Unlike teleop_sm's tests these build the REAL twin: the claims are about where
the tool ends up, what lands in ctrl, and what the recording and the export
contain -- the consumer side -- and a stub arm can answer none of those. A
viewer drag is simulated by writing the handle's mocap pose directly, which is
exactly what the passive viewer does (see teleop_kin/handle.py).

Each test builds its own twin, so a failure in one cannot leak state into the
next. Roughly 40 s in total.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")   # before mujoco is imported

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)                               # scene paths are relative

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from transforms3d.euler import euler2mat  # noqa: E402

import recording  # noqa: E402
from sim.mujoco_env import SimXArmAPI  # noqa: E402
from teleop_kin import config, keys  # noqa: E402
from teleop_kin.receiver import KinAction, KinestheticReceiver, _rot_angle_deg  # noqa: E402

SCENE = "envs/lab_scene.xml"
DT = 1.0 / config.FREQUENCY
TOOL_DOWN = euler2mat(np.pi, 0.0, 0.0, axes="sxyz")   # roll=180: tool Z down


def _twin():
    arm = SimXArmAPI(scene_xml=SCENE, render=False)
    # Start at home, where run_kinesthetic_sim.py starts (the scene's all-zeros
    # pose is a wrist singularity). Placed directly rather than via go_home(),
    # which paces the move and costs ~15 s per test.
    with arm.lock:
        for jid, aid, q in zip(arm.joint_ids, arm.act_ids[1:7], arm.HOME_JOINTS_RAD):
            arm.data.qpos[arm.model.jnt_qposadr[jid]] = q
            arm.data.ctrl[aid] = q
        arm.data.qpos[arm.model.jnt_qposadr[arm.rail_jid]] = arm.HOME_RAIL_M
        arm.data.ctrl[arm.act_ids[0]] = arm.HOME_RAIL_M
        arm.data.qvel[:] = 0.0
        mujoco.mj_forward(arm.model, arm.data)
    time.sleep(0.5)                          # settle into the gravity sag
    rx = KinestheticReceiver(arm)
    rx.handle.dragging_override = True       # "the operator is holding it"
    return arm, rx


def _run(rx, seconds: float) -> None:
    """Tick in real time -- the arm is a physical PD loop on its own thread,
    so simulated ticks without wall time would outrun it."""
    t_end = time.time() + seconds
    last = time.time()
    while time.time() < t_end:
        now = time.time()
        rx.tick(max(now - last, 1e-4))
        last = now
        time.sleep(max(0.0, DT - (time.time() - now)))


def _run_sim(rx, seconds: float) -> None:
    """Tick until `seconds` of SIMULATED time have passed. The sim thread runs
    at roughly half wall-clock speed, so "the arm settles within 2 s" is a
    statement about the twin's clock, not the test's."""
    arm = rx.arm
    with arm.lock:
        t_end = float(arm.data.time) + seconds
    _run_until(rx, lambda: float(arm.data.time) >= t_end, timeout=10.0 * seconds)


def _run_until(rx, pred, timeout: float) -> bool:
    t_end = time.time() + timeout
    last = time.time()
    while time.time() < t_end:
        now = time.time()
        rx.tick(max(now - last, 1e-4))
        last = now
        if pred():
            return True
        time.sleep(max(0.0, DT - (time.time() - now)))
    return False


def _caught_up(rx):
    hp, hr = rx.handle.read_pose()
    return (np.linalg.norm(rx._cmd_pos - hp) * 1000.0 < config.SOLVE_MIN_POS_MM
            and _rot_angle_deg(rx._cmd_rot, hr) < config.SOLVE_MIN_ROT_DEG)


def _q_ctrl(arm):
    with arm.lock:
        return np.array([float(arm.data.ctrl[a]) for a in arm.act_ids[1:7]])


def _is_orange(img: np.ndarray) -> np.ndarray:
    """Pixels the colour of the handle ball (1.0, 0.55, 0.0) and nothing else
    on the bench: strong red, mid green, almost no blue."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return (r > 180) & (g > 70) & (g < 170) & (b < 50) & (r - g > 60)


# ---------------------------------------------------------------------------
def test_1_tracking():
    """Handle 100 mm in +X: the TOOL (not the setpoint) ends within 2 mm / 1 deg.

    "Within 2 s" is 2 s of the twin's simulated time after the rate-limited
    command reaches the handle (100 mm at 150 mm/s = 0.67 s).
    """
    arm, rx = _twin()
    try:
        p, r = rx.handle.read_pose()
        rx.handle.write_pose(p + [0.10, 0.0, 0.0], r)
        assert _run_until(rx, lambda: _caught_up(rx), 3.0), "command never reached the handle"
        _run_sim(rx, 2.0)
        hp, hr = rx.handle.read_pose()
        ep, er = rx.handle.ee_pose()
        err_mm = float(np.linalg.norm(hp - ep)) * 1000.0
        err_deg = _rot_angle_deg(hr, er)
        print(f"    tool-to-handle {err_mm:.2f} mm, {err_deg:.2f} deg")
        assert not rx.ik_fail and rx.following
        assert err_mm < 2.0, f"EE {err_mm:.2f} mm from the handle"
        assert err_deg < 1.0, f"EE {err_deg:.2f} deg from the handle"
    finally:
        arm.disconnect()


def _drag_error_mm(gravity_comp: bool) -> float:
    """test 1's drag on an arm whose model gravity compensation is switched
    off -- an uncompensated load, as a held object is -- with the sag bias on
    or off. Returns the tool-to-handle error after settling."""
    arm, rx = _twin()
    try:
        with arm.lock:
            arm.model.body_gravcomp[:] = 0.0
        rx.gravity_comp = gravity_comp
        _run_sim(rx, 1.0)                     # let the bare arm sag first
        p, r = rx.handle.read_pose()
        rx.handle.write_pose(p + [0.10, 0.0, 0.0], r)
        _run_until(rx, lambda: _caught_up(rx), 3.0)
        _run_sim(rx, 2.0)
        hp, _ = rx.handle.read_pose()
        ep, _ = rx.handle.ee_pose()
        return float(np.linalg.norm(hp - ep)) * 1000.0
    finally:
        arm.disconnect()


def test_1b_sag_bias_cancels_an_uncompensated_load():
    """The scene compensates the arm's and tool's own weight (gravcomp, as the
    real controller does for its payload), so test 1 would pass with or without
    the sag bias. What the bias is still for is load the model does NOT
    compensate -- a held object. Remove the compensation to make the whole arm
    such a load: without the bias the tool must end visibly short (negative
    control), with it on the handle."""
    off = _drag_error_mm(gravity_comp=False)
    on = _drag_error_mm(gravity_comp=True)
    print(f"    uncompensated arm: {off:.1f} mm without the bias, {on:.2f} mm with it")
    assert off > 5.0, f"expected visible sag, got {off:.2f} mm"
    assert on < 2.0, f"the bias left the tool {on:.2f} mm from the handle"


def test_2_rate_limit():
    """Handle 300 mm away in one step, one tick: the commanded pose (what lands
    in ctrl, i.e. the exported action) moves no more than MAX_POS_SPEED/60."""
    arm, rx = _twin()
    try:
        cmd0, _ = rx.commanded_pose()
        ee0, _ = rx.handle.ee_pose()
        p, r = rx.handle.read_pose()
        rx.handle.write_pose(p + [0.30, 0.0, 0.0], r)
        rx.tick(DT)
        cmd1, _ = rx.commanded_pose()
        time.sleep(0.3)                      # let the arm reach that setpoint
        ee1, _ = rx.handle.ee_pose()
        limit = config.MAX_POS_SPEED / 60.0 + 1.0
        d_cmd = float(np.linalg.norm(cmd1 - cmd0)) * 1000.0
        d_ee = float(np.linalg.norm(ee1 - ee0)) * 1000.0
        print(f"    commanded moved {d_cmd:.2f} mm, tool {d_ee:.2f} mm (limit {limit:.1f})")
        assert d_cmd <= limit, d_cmd
        assert d_ee <= limit, d_ee
    finally:
        arm.disconnect()


def test_3_unreachable():
    """Handle 2 m away: no exception, the arm holds, the ball turns red, and
    on release the handle snaps back onto the arm."""
    arm, rx = _twin()
    try:
        p, r = rx.handle.read_pose()
        rx.handle.write_pose(p + [0.0, 2.0, 1.0], r)
        rx.tick(DT)
        assert np.allclose(rx.handle.color(), config.RGBA_BLOCKED), \
            "outside the workspace but the ball is not red"
        assert _run_until(rx, lambda: rx.ik_fail, 12.0), "IK never failed 2 m away"
        q_cmd_frozen = rx.servo.q_cmd.copy()
        ctrl_at_fail = _q_ctrl(arm)
        _run(rx, 0.5)
        creep = float(np.max(np.abs(np.rad2deg(rx.servo.q_cmd - q_cmd_frozen))))
        print(f"    IK target creep along the reach boundary: {creep:.3f} deg")
        drift = float(np.max(np.abs(np.rad2deg(_q_ctrl(arm) - ctrl_at_fail))))
        print(f"    ctrl drift while blocked: {drift:.3f} deg (sag bias only)")
        assert drift < 0.5, drift
        assert np.allclose(rx.handle.color(), config.RGBA_BLOCKED)
        # Let the arm arrive at the held setpoint. The sim thread runs at about
        # half wall-clock speed, so this is seconds, not a fixed sleep.
        def _settled():
            cp, _ = rx.commanded_pose()
            ep, _ = rx.handle.ee_pose()
            return float(np.linalg.norm(cp - ep)) * 1000.0 < 0.5
        assert _run_until(rx, _settled, 8.0), "arm never settled at the held pose"

        rx.handle.dragging_override = False  # let go
        rx.tick(DT)
        hp, _ = rx.handle.read_pose()
        cp, _ = rx.commanded_pose()
        ep, _ = rx.handle.ee_pose()
        to_cmd = float(np.linalg.norm(hp - cp)) * 1000.0
        to_ee = float(np.linalg.norm(hp - ep)) * 1000.0
        print(f"    after release: handle {to_cmd:.3f} mm from the setpoint, "
              f"{to_ee:.2f} mm from the tool")
        assert to_cmd < 0.1
        assert to_ee < 1.0, to_ee
        assert np.allclose(rx.handle.color(), config.RGBA_TRACKING)
    finally:
        arm.disconnect()


class _FakeViewer:
    """Just enough of a viewer for _viewer_key_callback to reach nudge_body."""
    def __init__(self, select):
        self.perturb = type("P", (), {"select": select, "active": 0})()


def test_4_key_routing():
    """Up reaches the gripper and NOT nudge_body with the handler registered;
    unregistered, the same key nudges again (negative control)."""
    arm, rx = _twin()
    try:
        nudges, opens = [], []
        arm.nudge_body = lambda name, **kw: nudges.append((name, kw)) or 0
        real_open = arm.open_lite6_gripper
        arm.open_lite6_gripper = lambda: opens.append(1) or real_open()
        arm._viewer = _FakeViewer(select=arm.cube_bids["green_cube"])

        handler = keys.make_handler(rx)
        arm.register_key_handler(handler)
        arm._viewer_key_callback(config.KEY_UP)
        rx.tick(DT)                          # actions run on the control thread
        assert opens == [1], f"gripper open calls: {opens}"
        assert nudges == [], f"nudge_body called despite the handler: {nudges}"
        arm._viewer_key_callback(82)         # 'R' is not a teach key -> falls through
        assert len(nudges) == 1, "a non-teach key must still reach nudge_body"

        arm.unregister_key_handler(handler)
        nudges.clear()
        arm._viewer_key_callback(config.KEY_UP)
        rx.tick(DT)
        assert opens == [1], "gripper opened with no handler registered"
        assert nudges and nudges[0][1] == {"dy_mm": 10.0}, nudges
    finally:
        arm._viewer = None
        arm.disconnect()


def _drag_to(rx, pos, rot, timeout=8.0):
    rx.handle.write_pose(np.asarray(pos, float), rot)
    assert _run_until(rx, lambda: _caught_up(rx), timeout), f"never reached {pos}"
    _run(rx, 0.8)


def test_5_recording_and_export():
    """Scripted pick with the recorder running. Graded on the files: interface,
    ctrl moving during the drag, the gripper channel switching at the key
    presses, and the exporter reading the session."""
    with tempfile.TemporaryDirectory() as tmp:
        saved_root = recording.RECORDINGS_ROOT
        recording.RECORDINGS_ROOT = Path(tmp)
        arm, rx = _twin()
        try:
            rec = recording.Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                                     interface="kinesthetic_sim", scene_xml=SCENE,
                                     enable_frames=True)
            rx.rec = rec
            handler = keys.make_handler(rx)
            arm.register_key_handler(handler)

            arm._viewer_key_callback(config.KEY_INSERT)     # start the take
            rx.tick(DT)
            assert rec.is_recording
            with arm.lock:
                mujoco.mj_forward(arm.model, arm.data)
                # red_cube_back: inside the shared teleop workspace box (the
                # front-row cubes at y=-250 mm sit outside its y>=-200 mm face)
                # and on a clean path from home. Dragging from home toward the
                # green cube runs this elbow branch into joint 3's +11 deg
                # limit, and the handle -- correctly -- turns red.
                cube = arm.data.xpos[arm.cube_bids["red_cube_back"]].copy()
            _drag_to(rx, cube + [0.0, 0.0, 0.12], TOOL_DOWN)
            _drag_to(rx, cube + [0.0, 0.0, 0.025], TOOL_DOWN)
            arm._viewer_key_callback(config.KEY_DOWN)       # close
            rx.tick(DT)
            ok, _ = arm.verify_grasp()
            assert ok == "held", f"the scripted pick did not grasp: {arm.last_refusal}"
            _drag_to(rx, cube + [0.0, 0.0, 0.12], TOOL_DOWN)
            arm._viewer_key_callback(config.KEY_UP)         # open
            rx.tick(DT)
            _run(rx, 0.5)
            arm._viewer_key_callback(config.KEY_INSERT)     # stop
            rx.tick(DT)
            session = Path(rx.last_saved)

            meta = json.loads((session / "metadata.json").read_text())
            assert meta["interface"] == "kinesthetic_sim", meta["interface"]
            cmds = [json.loads(l) for l in (session / "commands.jsonl").read_text().splitlines()]
            grip_t = [c["t"] for c in cmds if c["type"] == "gripper"]
            assert len(grip_t) == 2, cmds

            import h5py
            with h5py.File(session / "trajectory.h5") as f:
                t = np.asarray(f["t_wall"])
                ctrl = np.asarray(f["ctrl"])
                grip = np.asarray(f["gripper"])
                assert "frames" in f and "cam_wrist_color" in f["frames"], "no frames recorded"
            span = np.rad2deg(ctrl[:, 1:7].max(0) - ctrl[:, 1:7].min(0))
            print(f"    {len(t)} samples; joint ctrl span (deg) {np.round(span, 1)}")
            assert span.max() > 5.0, "ctrl barely moved during the drag"
            # gripper: 0 before the close, 1 between, 0 after the open
            before = grip[t < grip_t[0] - 0.05]
            between = grip[(t > grip_t[0] + 0.05) & (t < grip_t[1] - 0.05)]
            after = grip[t > grip_t[1] + 0.1]
            assert before.size and between.size and after.size
            assert before.max() == 0.0 and between.min() == 1.0 and after.max() == 0.0, \
                (before.max(), between.min(), after.max())

            # The export is the consumer. Dry run as the CLI does it...
            out = subprocess.run(
                [sys.executable, "scripts/export_lerobot.py", "--dry-run",
                 "--recordings", tmp],
                capture_output=True, text=True, timeout=120)
            assert out.returncode == 0, out.stderr[-2000:]
            assert "usable         : 1" in out.stdout and "kinesthetic_sim" in out.stdout, \
                out.stdout[-2000:]
            # ...and the arrays it would write.
            sys.path.insert(0, str(HERE / "scripts"))
            from export_lerobot import session_arrays
            arr = session_arrays(str(session))
            assert arr is not None and arr["images"] is not None
            a_grip = arr["action"][:, 7]
            assert a_grip.max() == 1.0 and a_grip.min() == 0.0, "gripper lost in export"
            a_span = arr["action"][:, :6].max(0) - arr["action"][:, :6].min(0)
            assert a_span.max() > 5.0, a_span
        finally:
            recording.RECORDINGS_ROOT = saved_root
            arm.disconnect()


def test_6_handle_hidden_from_data():
    """With the ball parked in front of the wrist camera, recorded frames hold
    no handle-orange pixels. The negative control renders the same view with
    group 4 ON and must find them -- otherwise the check proves nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        saved_root = recording.RECORDINGS_ROOT
        recording.RECORDINGS_ROOT = Path(tmp)
        arm, rx = _twin()
        try:
            cam = arm.model.camera("cam_wrist_color").id
            with arm.lock:
                mujoco.mj_forward(arm.model, arm.data)
                cpos = arm.data.cam_xpos[cam].copy()
                fwd = -arm.data.cam_xmat[cam].reshape(3, 3)[:, 2]   # cameras look down -Z
            rx.handle.dragging_override = None                      # stop following
            rx.handle.write_pose(cpos + 0.12 * fwd, np.eye(3))

            rec = recording.Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                                     interface="kinesthetic_sim", scene_xml=SCENE,
                                     enable_frames=True)
            rec.start()
            time.sleep(0.6)
            session = rec.stop(kept=True)
            import h5py
            with h5py.File(session / "trajectory.h5") as f:
                imgs = np.asarray(f["frames/cam_wrist_color/images"])
            assert len(imgs), "no frames"
            n_rec = int(_is_orange(imgs).sum())

            r = mujoco.Renderer(arm.model, height=imgs.shape[1], width=imgs.shape[2])
            opt = mujoco.MjvOption()
            opt.geomgroup[4] = 1
            with arm.lock:
                r.update_scene(arm.data, camera="cam_wrist_color", scene_option=opt)
                shown = r.render()
            r.close()
            n_ctrl = int(_is_orange(shown).sum())
            print(f"    orange pixels: recorded {n_rec}, group-4-on control {n_ctrl}")
            assert n_ctrl > 50, "control render did not show the handle -- check is blind"
            assert n_rec == 0, f"{n_rec} handle pixels in recorded frames"

            # Rays must agree with the picture (sim/render_options.ray_geomgroup).
            from sim.render_options import ray_geomgroup
            gid = np.array([-1], dtype=np.int32)
            with arm.lock:
                mujoco.mj_ray(arm.model, arm.data, cpos, fwd, ray_geomgroup(), 1, -1, gid)
            hit = mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0]))
            assert not (hit or "").startswith("teach_handle"), f"ray hit {hit}"
        finally:
            recording.RECORDINGS_ROOT = saved_root
            arm.disconnect()


def test_7_mocap_lookups_by_name():
    """Adding a second mocap body must not shift vr_head: both resolve by
    name, to different slots."""
    model = mujoco.MjModel.from_xml_path(SCENE)
    vr = int(model.body_mocapid[model.body("vr_head").id])
    th = int(model.body_mocapid[model.body("teach_handle").id])
    assert vr >= 0 and th >= 0 and vr != th, (vr, th)
    for g in ("teach_handle_ball", "teach_handle_x", "teach_handle_y", "teach_handle_z"):
        gid = model.geom(g).id
        assert model.geom_group[gid] == 4, g
        assert model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0, g
    # The primitive source carries the same handle (the mesh scene is built from it).
    prim = mujoco.MjModel.from_xml_path("envs/lab_scene_primitive.xml")
    assert int(prim.body_mocapid[prim.body("teach_handle").id]) >= 0


def test_8_rail_step_reanchors():
    """Right arrow steps the rail 25 mm and leaves the handle on the arm."""
    arm, rx = _twin()
    try:
        rail0 = rx.rail_mm
        rx.post(KinAction.RAIL_PLUS)
        rx.tick(DT)
        assert abs(rx.rail_mm - min(rail0 + config.RAIL_STEP_MM, config.RAIL_MAX_MM)) < 0.5
        hp, _ = rx.handle.read_pose()
        cp, _ = rx.commanded_pose()
        assert float(np.linalg.norm(hp - cp)) * 1000.0 < 0.1
    finally:
        arm.disconnect()


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
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
