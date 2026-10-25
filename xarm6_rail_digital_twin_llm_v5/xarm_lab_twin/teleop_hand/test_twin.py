# teleop_hand/test_twin.py
"""Hand teleop against the live twin (spec §7 tests 9, 10, 11). Headless.

    MUJOCO_GL=egl python -m teleop_hand.test_twin

Graded on the consumer side: where the EE and rail actually are, what the
recording holds, and what the exporter reads -- not the receiver's flags.
The sim thread runs at roughly half wall-clock speed, so "after settling" is
measured in SIMULATED seconds (same as teleop_kin.test_receiver).
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

import recording  # noqa: E402
from sim.mujoco_env import RAIL_ACT, SimXArmAPI  # noqa: E402
from teleop_hand import config, guards  # noqa: E402
from teleop_hand.mapping import ArmTeleopMapper  # noqa: E402
from teleop_hand.messages import UdpPublisher, UdpReceiver  # noqa: E402
from teleop_hand.receiver import HandAction, HandReceiver  # noqa: E402
from teleop_hand.testing import PALM0, FrameSource, Hand, rot_about  # noqa: E402
from teleop_kin.handle import TeachHandle  # noqa: E402

SCENE = "envs/lab_scene.xml"
DT = 1.0 / config.FREQUENCY
R_FACING = config.PRESETS["facing"]


def _cam(world_delta_m):
    return R_FACING.T @ np.asarray(world_delta_m, float)


class Twin:
    """A homed twin + HandReceiver fed by a scripted hand."""

    def __init__(self, source=None, rail_m=None, **mapper_kw):
        self.arm = arm = SimXArmAPI(scene_xml=SCENE, render=False)
        rail_m = arm.HOME_RAIL_M if rail_m is None else rail_m
        with arm.lock:
            for jid, aid, q in zip(arm.joint_ids, arm.act_ids[1:7], arm.HOME_JOINTS_RAD):
                arm.data.qpos[arm.model.jnt_qposadr[jid]] = q
                arm.data.ctrl[aid] = q
            arm.data.qpos[arm.model.jnt_qposadr[arm.rail_jid]] = rail_m
            arm.data.ctrl[arm.act_ids[RAIL_ACT]] = rail_m
            arm.data.qvel[:] = 0.0
            mujoco.mj_forward(arm.model, arm.data)
        time.sleep(0.5)
        mapper_kw.setdefault("rail_mode", "keys")
        self.mapper = ArmTeleopMapper(**mapper_kw)
        self.source = source or FrameSource()
        self.hand = Hand()
        self.clutch = True
        self.marker = TeachHandle(arm)
        self.rx = HandReceiver(arm, self.mapper, self.source,
                               clutch_fn=lambda: self.clutch, marker=self.marker)
        self.push = isinstance(self.source, FrameSource)

    def ee(self):
        with self.arm.lock:
            mujoco.mj_forward(self.arm.model, self.arm.data)
            return (self.arm.data.site_xpos[self.arm.ee_site].copy(),
                    self.arm.data.site_xmat[self.arm.ee_site].reshape(3, 3).copy())

    def servo_pose(self):
        """Where the arm is COMMANDED (sag-free): the mapper's anchor."""
        return self.rx.servo.commanded_pose()

    def q5_deg(self):
        return float(np.rad2deg(self.rx.servo.q_cmd[guards.WRIST_JOINT]))

    def tick(self):
        if self.push:
            self.source.push(self.hand.frame(time.monotonic()))
        self.rx.tick(DT)

    def run(self, seconds: float, each=None):
        t_end = time.time() + seconds
        while time.time() < t_end:
            t0 = time.time()
            self.tick()
            if each:
                each()
            time.sleep(max(0.0, DT - (time.time() - t0)))

    def run_sim(self, seconds: float, each=None):
        with self.arm.lock:
            t_end = float(self.arm.data.time) + seconds
        deadline = time.time() + 10 * seconds
        while time.time() < deadline:
            self.run(0.1, each)
            with self.arm.lock:
                if float(self.arm.data.time) >= t_end:
                    return

    def move(self, pos=None, rot=None, seconds: float = 1.0, each=None):
        """Human-speed hand motion (a one-frame step is an outlier)."""
        p0, r0 = self.hand.pos.copy(), self.hand.rot.copy()
        p1 = p0 if pos is None else np.asarray(pos, float)
        r1 = r0 if rot is None else np.asarray(rot, float)
        sl = Slerp([0, 1], Rotation.from_matrix(np.stack([r0, r1])))
        t_start = time.time()
        while True:
            a = min(1.0, (time.time() - t_start) / seconds)
            self.hand.pos = p0 + a * (p1 - p0)
            self.hand.rot = sl([a])[0].as_matrix()
            t0 = time.time()
            self.tick()
            if each:
                each()
            if a >= 1.0:
                return
            time.sleep(max(0.0, DT - (time.time() - t0)))

    def close(self):
        self.arm.disconnect()


# ---------------------------------------------------------------------------
def test_9_rail_edge_push_twin():
    """Hand far along the rail: the rail moves toward it at the capped speed
    and the EE ends at the HAND's world x -- not carried past it by the rail.

    Starts at rail 50 mm and reaches toward world x = 250 mm: a corridor
    clear of the bench. (The first version reached to x = 620 from rail 350
    and drove link 5 into the OT-2 -- the twin's direct servo path does no
    collision checking, and the rail actuator stalled against the contact.)"""
    tw = Twin(rail_mode="edge", rail_m=0.05)
    try:
        tw.run(0.5)
        e0, _ = tw.servo_pose()
        rail0 = tw.rx.rail_mm
        target_x = 0.25                                       # world m
        dx = target_x - e0[0]
        rails = []
        tw.move(PALM0 + _cam([dx, 0, 0]), seconds=1.5, each=lambda: rails.append(tw.rx.rail_mm))
        tw.run_sim(6.0, each=lambda: rails.append(tw.rx.rail_mm))
        e1, _ = tw.ee()
        base_x = tw.rx.geom.base_mm(tw.rx.rail_mm)[0]
        band = config.REACH_HALF_SPAN_MM + config.EDGE_MARGIN_MM
        print(f"    rail {rail0:.0f} -> {tw.rx.rail_mm:.0f} mm, EE x {e0[0]*1000:.0f} -> "
              f"{e1[0]*1000:.1f} mm (hand {target_x*1000:.0f})")
        assert tw.rx.rail_mm > rail0 + 50, "rail did not follow"
        assert abs(e1[0] - target_x) * 1000.0 < 3.0, "EE did not end under the hand"
        assert abs((target_x * 1000.0 - base_x) - band) < 2.0, "rail did not stop at the band"
        v = np.diff(rails) * config.FREQUENCY
        assert v.max() <= config.RAIL_MAX_SPEED * 1.6, v.max()   # wall-clock tick jitter
    finally:
        tw.close()


def test_10_guards():
    # (a) branch flip: a 300 mm target step in one tick (rate limit lifted)
    tw = Twin()
    try:
        tw.run(0.5)
        q_before = tw.rx.servo.q_cmd.copy()
        tw.mapper.max_pos_speed = 1e6
        tw.mapper.pos_filter.min_cutoff = 1e4      # unfiltered: the step lands in one tick
        tw.mapper.scale = 10.0
        # 30 mm in ONE frame (under the 50 mm outlier threshold) -> a 300 mm
        # target step in one tick: no IK solution is within 20 deg per joint.
        tw.hand.pos = PALM0 + _cam([0.0, 0.0, -0.03])
        tw.run(0.3)
        assert tw.rx.blocked == guards.BRANCH_FLIP, tw.rx.blocked
        assert np.allclose(tw.rx.servo.q_cmd, q_before), "arm moved on a refused target"
        assert tw.rx.marker_rgba() == config.RGBA_BLOCKED
        assert np.allclose(tw.marker.color(), config.RGBA_BLOCKED)
    finally:
        tw.close()

    # (b) wrist band in full mode: tilting the tool ~45 deg about world -y
    # drives joint 5 through zero (measured: 57 deg at home, 1 deg at 45).
    tw = Twin(orient_mode="full")
    try:
        tw.run(0.5)
        q5s = []
        tw.move(rot=rot_about(_cam([0, -1, 0]), 60), seconds=3.0,
                each=lambda: q5s.append(tw.q5_deg()))
        tw.run(1.0, each=lambda: q5s.append(tw.q5_deg()))
        print(f"    wrist: joint 5 {q5s[0]:.1f} -> min {min(q5s, key=abs):.1f} deg")
        assert min(abs(q) for q in q5s) >= config.WRIST_SINGULAR_DEG, min(q5s, key=abs)
        assert q5s[0] - min(q5s) > 30, "the tilt never brought joint 5 near the band"
        assert tw.rx.blocked == guards.WRIST_SINGULAR, tw.rx.blocked
        assert tw.rx.marker_rgba() == config.RGBA_BLOCKED
    finally:
        tw.close()

    # (c) joint limit: from home toward the green cube joint 3 climbs to its
    # +11 deg limit (measured: +10 deg halfway, then a branch flip).
    tw = Twin()
    try:
        tw.run(0.5)
        e0, _ = tw.ee()
        with tw.arm.lock:
            goal = tw.arm.data.body("green_cube").xpos.copy() + [0, 0, 0.12]
        j3 = []
        tw.move(PALM0 + _cam(goal - e0), seconds=4.0,
                each=lambda: j3.append(np.rad2deg(tw.rx.servo.q_cmd[2])))
        tw.run(1.0, each=lambda: j3.append(np.rad2deg(tw.rx.servo.q_cmd[2])))
        hi = np.rad2deg(guards.joint_limits_rad(tw.arm)[2, 1])
        print(f"    joint 3: max {max(j3):.1f} deg (limit {hi:.0f}, margin "
              f"{config.JOINT_LIMIT_MARGIN_DEG:.0f})")
        # The hard stop is the guarantee (the margin tolerates IK jitter).
        assert max(j3) <= hi - (config.JOINT_LIMIT_MARGIN_DEG - guards.HARD_STOP_DEG) + 1e-6
        assert max(j3) > hi - config.JOINT_LIMIT_MARGIN_DEG - 3.0, "never got near the limit"
        assert tw.rx.blocked == guards.JOINT_LIMIT, tw.rx.blocked
        assert tw.rx.marker_rgba() == config.RGBA_BLOCKED
        # ...and the way BACK must work (found live: an arm parked at the
        # margin edge refused even the return move).
        j3_stuck = float(np.rad2deg(tw.rx.servo.q_cmd[2]))
        tw.move(PALM0, seconds=3.0)
        tw.run(1.5)
        j3_back = float(np.rad2deg(tw.rx.servo.q_cmd[2]))
        print(f"    joint 3 after reversing: {j3_stuck:.1f} -> {j3_back:.1f} deg")
        assert j3_back < j3_stuck - 10.0, "the arm did not come back out of the limit band"
        assert tw.rx.blocked is None, tw.rx.blocked
    finally:
        tw.close()


def test_11_udp_end_to_end_recording_export():
    """HandFrames over real UDP into a headless twin: a 100 mm move is
    followed to < 3 mm, and a scripted pick with a pinch records a session
    the exporter reads."""
    rxsock = UdpReceiver(port=0)
    pub = UdpPublisher(port=rxsock.port)
    hand = Hand()
    stop = threading.Event()

    def tracker():                                   # a stand-in tracker @ 30 Hz
        while not stop.is_set():
            pub.send(hand.frame(time.monotonic()))
            time.sleep(1 / 30)

    th = threading.Thread(target=tracker, daemon=True)
    with tempfile.TemporaryDirectory() as tmp:
        saved_root = recording.RECORDINGS_ROOT
        recording.RECORDINGS_ROOT = Path(tmp)
        tw = Twin(source=rxsock)
        tw.hand = hand
        th.start()
        try:
            tw.run(0.5)
            assert tw.rx.out.engaged, "never engaged over UDP"
            # Measured from the COMMANDED start pose: the arm at rest still
            # carries part of its 15-20 mm gravity sag until the bias settles.
            c0, _ = tw.rx.servo.commanded_pose()
            tw.move(PALM0 + _cam([0.1, 0, 0]), seconds=1.0)
            tw.run_sim(3.0)
            e1, _ = tw.ee()
            err = np.linalg.norm(e1 - (c0 + [0.1, 0, 0])) * 1000.0
            print(f"    100 mm move: EE error {err:.2f} mm after settling")
            assert err < 3.0, err
            # Back to home before the pick: from 100 mm further out, the
            # straight line to the cube crosses joint 3's limit band (the
            # guard refuses it, correctly). From home it does not.
            tw.move(PALM0, seconds=1.0)
            tw.run_sim(1.0)

            rec = recording.Recorder(model=tw.arm.model, data=tw.arm.data, lock=tw.arm.lock,
                                     interface="handtrack_sim", scene_xml=SCENE,
                                     enable_frames=True)
            tw.rx.takes.rec = rec
            tw.rx.post(HandAction.RECORD_TOGGLE)
            tw.run(0.2)
            assert rec.is_recording
            with tw.arm.lock:
                cube = tw.arm.data.xpos[tw.arm.cube_bids["red_cube_back"]].copy()
            # Via a waypoint over +y first: the straight line from home to
            # above the cube passes within 3 deg of joint 3's +11 deg limit,
            # which the guard refuses (kinesthetic teaching has no margin).
            c_now, _ = tw.servo_pose()
            way = np.array([c_now[0], cube[1], c_now[2]])
            tw.move(hand.pos + _cam(way - c_now), seconds=2.0)
            tw.run_sim(0.5)
            tw.move(hand.pos + _cam(cube + [0, 0, 0.12] - way), seconds=3.0)
            tw.run_sim(1.5)
            tw.move(hand.pos + _cam([0, 0, -0.095]), seconds=1.5)
            tw.run_sim(2.0)
            # pinch: the opening falls through GRIP_CLOSE_BELOW
            for o in np.linspace(0.9, 0.1, 15):
                hand.opening = float(o)
                tw.run(1 / 30)
            tw.run(0.5)
            ok, detail = tw.arm.verify_grasp()
            assert ok == "held", f"pinch did not grasp: {detail} / {tw.arm.last_refusal}"
            tw.move(hand.pos + _cam([0, 0, 0.10]), seconds=1.5)
            tw.run_sim(1.0)
            for o in np.linspace(0.1, 0.9, 15):
                hand.opening = float(o)
                tw.run(1 / 30)
            tw.run(0.5)
            tw.rx.post(HandAction.RECORD_TOGGLE)
            tw.run(0.2)
            session = Path(tw.rx.takes.last_saved)

            meta = json.loads((session / "metadata.json").read_text())
            assert meta["interface"] == "handtrack_sim", meta["interface"]
            cmds = [json.loads(l) for l in (session / "commands.jsonl").read_text().splitlines()]
            grip_t = [c["t"] for c in cmds if c["type"] == "gripper"]
            assert len(grip_t) == 2, [c["type"] for c in cmds]
            import h5py
            with h5py.File(session / "trajectory.h5") as f:
                t = np.asarray(f["t_wall"])
                ctrl = np.asarray(f["ctrl"])
                grip = np.asarray(f["gripper"])
                assert "frames" in f and "cam_wrist_color" in f["frames"]
            span = np.rad2deg(ctrl[:, 1:7].max(0) - ctrl[:, 1:7].min(0))
            assert span.max() > 5.0, span
            assert grip[t < grip_t[0] - 0.05].max() == 0.0
            between = grip[(t > grip_t[0] + 0.1) & (t < grip_t[1] - 0.05)]
            assert between.size and between.min() == 1.0, "gripper channel missed the pinch"
            assert grip[t > grip_t[1] + 0.1].max() == 0.0

            sys.path.insert(0, str(HERE / "scripts"))
            from export_lerobot import session_arrays
            arr = session_arrays(str(session))
            assert arr is not None and arr["images"] is not None
            out = subprocess.run([sys.executable, "scripts/export_lerobot.py", "--dry-run",
                                  "--recordings", tmp], capture_output=True, text=True,
                                 timeout=180)
            assert out.returncode == 0, out.stdout + out.stderr
            usable = [l for l in out.stdout.splitlines() if l.strip().startswith("usable")]
            assert usable and usable[0].split(":")[1].strip() == "1", out.stdout
            assert "handtrack_sim" in out.stdout, out.stdout
            lat = np.array(tw.rx.latencies_s) * 1000.0
            print(f"    capture->command latency p50 {np.percentile(lat, 50):.0f} ms, "
                  f"p95 {np.percentile(lat, 95):.0f} ms (UDP + loop; no camera)")
        finally:
            stop.set()
            th.join(timeout=1)
            recording.RECORDINGS_ROOT = saved_root
            tw.close()
            rxsock.close()
            pub.close()


def test_no_clutch_no_motion():
    """Safety: a hand moving with the clutch up moves nothing -- arm, rail,
    gripper -- and the marker is grey."""
    tw = Twin(rail_mode="edge")
    try:
        tw.clutch = False
        tw.run(0.3)
        q0 = tw.rx.servo.q_cmd.copy()
        rail0 = tw.rx.rail_mm
        tw.move(PALM0 + _cam([0.5, 0.1, -0.1]), seconds=1.5)
        for o in np.linspace(0.9, 0.05, 10):
            tw.hand.opening = float(o)
            tw.run(1 / 30)
        tw.run(0.3)
        assert np.allclose(tw.rx.servo.q_cmd, q0)
        assert tw.rx.rail_mm == rail0
        assert not tw.rx.grip.closed and tw.arm.verify_grasp()[0] == "empty"
        assert tw.rx.marker_rgba() == config.RGBA_RELEASED
    finally:
        tw.close()


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
