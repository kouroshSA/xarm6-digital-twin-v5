# teleop_hand/test_mapping.py
"""ArmTeleopMapper tests (spec §7 tests 4-9, mapper half). No MuJoCo.

    python -m teleop_hand.test_mapping

Everything runs on a fake clock at 60 Hz with a frame every tick, so the
numbers (rate limits, dropout timing) are exact rather than wall-clock luck.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from teleop_hand import config  # noqa: E402
from teleop_hand.mapping import AbsoluteCalib, ArmTeleopMapper  # noqa: E402
from teleop_hand.testing import PALM0, Clock, Hand, rot_about  # noqa: E402
from teleop_kin.receiver import _rot_angle_deg  # noqa: E402

DT = 1.0 / 60.0
TOOL_DOWN = rot_about([1, 0, 0], 180)
EE0 = np.array([0.30, -0.05, 1.07])       # ~home, world m
BASE0 = np.array([0.0, -50.0, 1114.0])    # arm base at rail 350, world mm
R_FACING = config.PRESETS["facing"]


class Rig:
    """Mapper + clock + scripted hand + a stand-in arm that is always where
    it was commanded (the mapper anchors on the COMMANDED pose)."""

    def __init__(self, **kw):
        self.m = ArmTeleopMapper(**kw)
        self.clk = Clock()
        self.hand = Hand()
        self.ee_pos, self.ee_rot = EE0.copy(), TOOL_DOWN.copy()
        self.rail = 350.0
        self.clutch = True
        self.precision = False
        self.send = True
        self.outs = []

    def base(self):
        return BASE0 + np.array([self.rail - 350.0, 0, 0])

    def tick(self, n: int = 1):
        out = None
        for _ in range(n):
            self.clk.t += DT
            f = self.hand.frame(self.clk.t) if self.send else None
            out = self.m.update(f, self.clk.t, DT, self.clutch, self.precision,
                                self.ee_pos, self.ee_rot, self.base(), self.rail)
            if out.engaged:
                self.m.accept()
                self.ee_pos, self.ee_rot = out.target_pos, out.target_rot
            if out.rail_mm is not None:
                self.rail = out.rail_mm
            self.outs.append(out)
        return out

    def seconds(self, s: float):
        return self.tick(int(round(s / DT)))

    def move(self, pos=None, rot=None, seconds: float = 0.5):
        """Move the hand there smoothly, at human speed. A step in one frame
        is an OUTLIER to the mapper (> JUMP_POS_MM / JUMP_ROT_DEG) and is
        dropped, by design -- so scripts must not teleport the hand."""
        from scipy.spatial.transform import Rotation, Slerp
        p0, r0 = self.hand.pos.copy(), self.hand.rot.copy()
        p1 = p0 if pos is None else np.asarray(pos, float)
        r1 = r0 if rot is None else np.asarray(rot, float)
        sl = Slerp([0, 1], Rotation.from_matrix(np.stack([r0, r1])))
        n = max(1, int(round(seconds / DT)))
        out = None
        for k in range(1, n + 1):
            a = k / n
            self.hand.pos = p0 + a * (p1 - p0)
            self.hand.rot = sl([a])[0].as_matrix()
            out = self.tick()
        return out


def _cam(world_delta_m):
    """Camera-frame palm displacement that maps to this world displacement."""
    return R_FACING.T @ np.asarray(world_delta_m, float)


# ---------------------------------------------------------------------------
def test_4_relative_mapping_and_clutch():
    for s in (1.0, 0.5):
        rig = Rig(scale=s)
        rig.seconds(0.2)
        p_start = rig.ee_pos.copy()
        rig.move(PALM0 + [0.100, 0.0, 0.0])             # +100 mm along CAMERA x
        out = rig.seconds(2.0)
        d = (out.target_pos - p_start) * 1000.0
        # facing: image-right is the operator's left -> robot -x
        assert abs(d[0] + 100.0 * s) < 1.0 and abs(d[1]) < 1.0 and abs(d[2]) < 1.0, (s, d)

    # release: target freezes; re-engage: no jump
    held = rig.ee_pos.copy()
    rig.clutch = False
    out = rig.tick()
    assert not out.engaged and out.target_pos is None
    rig.move(PALM0 + [-0.05, 0.08, 0.03])               # hand wanders while released
    rig.seconds(0.5)
    assert np.allclose(rig.ee_pos, held)
    rig.clutch = True
    out = rig.tick()
    assert out.engaged
    jump = np.linalg.norm(out.target_pos - held) * 1000.0
    assert jump < 0.01, f"re-engage jumped {jump:.2f} mm"
    # and the NEXT motion is relative to the new anchor
    rig.move(rig.hand.pos + [0.0, 0.0, -0.05])           # toward the camera = +y robot
    out = rig.seconds(1.5)
    d = (out.target_pos - held) * 1000.0
    assert abs(d[1] - 50.0 * 0.5) < 1.0 and abs(d[0]) < 1.0, d


def test_4b_precision_mode():
    rig = Rig()
    rig.seconds(0.2)
    rig.precision = True
    out = rig.tick()
    p0 = out.target_pos.copy()
    assert out.status["precision"]
    rig.move(PALM0 + [0.100, 0.0, 0.0])
    out = rig.seconds(2.0)
    d = (out.target_pos - p0) * 1000.0
    assert abs(d[0] + 100.0 * config.PRECISION_POS_FACTOR) < 1.0, d
    # switching back re-anchors: no jump
    p1 = out.target_pos.copy()
    rig.precision = False
    out = rig.tick()
    assert np.linalg.norm(out.target_pos - p1) * 1000.0 < 0.01


def test_5_absolute_mapping():
    a = PALM0 + [-0.15, -0.10, -0.10]
    b = PALM0 + [0.15, 0.10, 0.10]
    rig = Rig(map_mode="absolute", abs_calib=AbsoluteCalib(a, b))
    box = config.ABSOLUTE_BOX_MM
    for corner in (a, b):
        rig.clutch = False
        rig.tick(3)
        rig.move(np.array(corner, float), seconds=1.0)
        rig.seconds(0.5)                     # let the position filter settle
        rig.outs.clear()
        rig.clutch = True
        rig.seconds(6.0)
        steps = [np.linalg.norm(o2.target_pos - o1.target_pos) * 1000.0
                 for o1, o2 in zip(rig.outs[:-1], rig.outs[1:])]
        cap = config.MAX_POS_SPEED * DT
        assert max(steps) <= cap + 1e-6, f"ramp step {max(steps):.3f} mm > cap {cap:.3f}"
        u = R_FACING @ (np.asarray(corner) - PALM0)
        want = np.array([box[k][0 if u[i] < 0 else 1] for i, k in enumerate("xyz")])
        want[0] += rig.base()[0]
        got = rig.ee_pos * 1000.0
        assert np.linalg.norm(got - want) < 2.0, (got, want)


def test_5b_absolute_box_inside_reach_band():
    band = config.REACH_HALF_SPAN_MM + config.EDGE_MARGIN_MM
    assert max(abs(v) for v in config.ABSOLUTE_BOX_MM["x"]) < band


def test_6_orientation_modes():
    # yaw: palm turns 30 deg about the camera axis that maps to world +z
    # (facing: world z = -camera y, so about camera -y).
    yaw_axis_cam = R_FACING.T @ [0, 0, 1]
    rig = Rig(orient_mode="yaw")
    rig.seconds(0.3)
    rig.move(rot=rot_about(yaw_axis_cam, 30) @ rig.hand.rot)
    out = rig.seconds(1.5)
    want = rot_about([0, 0, 1], 30) @ TOOL_DOWN
    assert _rot_angle_deg(out.target_rot, want) < 1.0, _rot_angle_deg(out.target_rot, want)

    # yaw ignores tilt: a 30 deg turn about world x changes nothing
    rig = Rig(orient_mode="yaw")
    rig.seconds(0.3)
    rig.move(rot=rot_about(R_FACING.T @ [1, 0, 0], 30))
    out = rig.seconds(1.5)
    assert _rot_angle_deg(out.target_rot, TOOL_DOWN) < 0.5

    # full: the same tilt is reproduced, about world x
    rig = Rig(orient_mode="full")
    rig.seconds(0.3)
    rig.move(rot=rot_about(R_FACING.T @ [1, 0, 0], 30))
    out = rig.seconds(1.5)
    want = rot_about([1, 0, 0], 30) @ TOOL_DOWN
    assert _rot_angle_deg(out.target_rot, want) < 1.0, _rot_angle_deg(out.target_rot, want)

    # deadband: a 3 deg wobble moves nothing, in either mode
    for mode in ("yaw", "full"):
        rig = Rig(orient_mode=mode)
        rig.seconds(0.3)
        worst = 0.0
        for k in range(120):
            ang = 3.0 * np.sin(2 * np.pi * k / 30)
            rig.hand.rot = rot_about(yaw_axis_cam, ang) @ rot_about([1, 0, 0], ang)
            out = rig.tick()
            worst = max(worst, _rot_angle_deg(out.target_rot, TOOL_DOWN))
        assert worst < 1e-6, (mode, worst)

    # locked: nothing turns at all
    rig = Rig(orient_mode="locked")
    rig.seconds(0.3)
    rig.move(rot=rot_about([1, 1, 0], 60), seconds=1.0)
    out = rig.seconds(1.0)
    assert _rot_angle_deg(out.target_rot, TOOL_DOWN) < 1e-9


def test_7_dropout_and_jumps():
    rig = Rig()
    rig.seconds(0.3)
    # 200 ms with no hand (tracker alive, detected=false)
    rig.hand.detected = False
    rig.seconds(0.2)
    out = rig.tick()
    assert not out.engaged and out.status["dropout"]
    held = rig.ee_pos.copy()
    rig.hand.detected = True
    rig.move(PALM0 + [0.05, 0.0, 0.0])         # clutch still held
    assert not out.engaged, "re-engaged without a fresh clutch press"
    assert out.status["need_release"]
    assert np.allclose(rig.ee_pos, held)
    rig.clutch = False
    rig.tick()
    rig.clutch = True
    out = rig.tick()
    assert out.engaged and np.linalg.norm(out.target_pos - held) * 1000 < 0.01

    # tracker silence (no messages at all) also releases
    rig.send = False
    out = rig.seconds(0.6)
    assert not out.engaged and out.status["silent"]
    rig.send = True
    rig.clutch = False
    rig.tick()
    rig.clutch = True
    rig.seconds(0.3)

    # single outliers: +300 mm, then +60 deg (full mode)
    for kind in ("pos", "rot"):
        rig = Rig(orient_mode="full")
        rig.seconds(0.5)
        p0, r0 = rig.ee_pos.copy(), rig.ee_rot.copy()
        if kind == "pos":
            rig.hand.pos = PALM0 + [0.3, 0, 0]
        else:
            rig.hand.rot = rot_about([0, 1, 0], 60)
        rig.tick()
        rig.hand.pos, rig.hand.rot = PALM0.copy(), np.eye(3)
        worst_p = worst_r = 0.0
        for _ in range(30):
            out = rig.tick()
            worst_p = max(worst_p, np.linalg.norm(out.target_pos - p0) * 1000.0)
            worst_r = max(worst_r, _rot_angle_deg(out.target_rot, r0))
        assert worst_p < 1.0 and worst_r < 1.0, (kind, worst_p, worst_r)
        assert rig.m.n_jumps == 1


def test_8_pinch_freeze():
    rig = Rig()
    rig.seconds(0.5)
    p0 = rig.ee_pos.copy()
    # close 0.9 -> 0.2 in 0.3 s while the palm drifts 20 mm (the pinch drags it)
    n = int(0.3 / DT)
    worst = 0.0
    for k in range(n + 1):
        rig.hand.opening = 0.9 - 0.7 * k / n
        rig.hand.pos = PALM0 + [0.0, 0.020 * k / n, 0.0]
        out = rig.tick()
        worst = max(worst, np.linalg.norm(out.target_pos - p0) * 1000.0)
    assert worst < 0.5, f"target moved {worst:.2f} mm during the pinch"
    assert any(o.status["frozen"] for o in rig.outs[-n:])
    # after the freeze the drift is NOT applied (re-anchored), no jump
    out = rig.seconds(1.0)
    assert np.linalg.norm(out.target_pos - p0) * 1000.0 < 0.5
    # and ordinary motion works again
    rig.move(rig.hand.pos + [0.05, 0, 0])
    out = rig.seconds(1.5)
    assert abs((out.target_pos - p0)[0] * 1000.0 + 50.0) < 1.0

    # control: with the freeze off, the same pinch drags the target
    rig = Rig(pinch_freeze=False)
    rig.seconds(0.5)
    p0 = rig.ee_pos.copy()
    for k in range(n + 1):
        rig.hand.opening = 0.9 - 0.7 * k / n
        rig.hand.pos = PALM0 + [0.0, 0.020 * k / n, 0.0]
        rig.tick()
    out = rig.seconds(0.5)
    assert np.linalg.norm(out.target_pos - p0) * 1000.0 > 10.0


def test_9_rail_edge_push_mapper():
    rig = Rig(rail_mode="edge")
    rig.seconds(0.3)
    band = config.REACH_HALF_SPAN_MM + config.EDGE_MARGIN_MM
    along0 = rig.ee_pos[0] * 1000.0 - rig.base()[0]
    # hand far right: target world x well past the rail's reach
    rig.move(PALM0 + _cam([1.0, 0, 0]), seconds=1.0)
    rails = []
    for _ in range(int(8.0 / DT)):
        rig.tick()
        rails.append(rig.rail)
    v = np.diff(rails) / DT
    assert v.max() <= config.RAIL_MAX_SPEED + 1e-6, v.max()
    assert v.max() > 0.9 * config.RAIL_MAX_SPEED, "rail never reached its speed cap"
    assert abs(rails[-1] - config.RAIL_MAX_MM) < 1e-9, rails[-1]
    # the commanded arm target never leaves the reach band around the base
    assert along0 < band
    for o in rig.outs:
        if o.target_pos is not None:
            assert o.target_pos[0] * 1000.0 <= 700.0 + 1e-6      # workspace box
    # keys mode: the same hand moves no rail
    rig = Rig(rail_mode="keys")
    rig.seconds(0.3)
    rig.move(PALM0 + _cam([1.0, 0, 0]), seconds=1.0)
    rig.seconds(2.0)
    assert rig.rail == 350.0


def test_gesture_needs_nothing_from_mapper():
    """The gesture clutch is the receiver's business; the mapper only sees a
    boolean. A clutch held with no hand ever seen must not engage."""
    rig = Rig()
    rig.send = False
    out = rig.seconds(0.5)
    assert not out.engaged


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
