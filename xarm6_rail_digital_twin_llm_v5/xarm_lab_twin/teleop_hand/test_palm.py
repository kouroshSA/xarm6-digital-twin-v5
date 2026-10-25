# teleop_hand/test_palm.py
"""Palm frame, depth lookup, fusion, fist gesture (spec §7 test 3). No camera.

    python -m teleop_hand.test_palm
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from teleop_hand import config, palm  # noqa: E402
from teleop_hand.detector import SingleHandDetector  # noqa: E402
from teleop_hand.mapping import ArmTeleopMapper  # noqa: E402
from teleop_hand.testing import Clock, Hand, rot_about  # noqa: E402

#: Palm landmarks in a palm-local frame (x toward the wrist, y normal,
#: z toward the index side), metres. Roughly an adult hand.
LOCAL = {0: [0.045, 0, 0.0], 5: [-0.045, 0, 0.022], 9: [-0.048, 0, 0.0],
         13: [-0.044, 0, -0.020], 17: [-0.036, 0, -0.040]}


def _palm_points(rot, centre, noise_m, rng):
    return {k: centre + rot @ np.array(v) + rng.normal(0, noise_m, 3) for k, v in LOCAL.items()}


def _palm_surface(rot, centre, noise_m, rng, n=21):
    """Interior palm-surface points (what palm_samples_px deprojects)."""
    lo = np.array([-0.045, -0.040]); hi = np.array([0.045, 0.022])
    uv = lo + rng.random((n, 2)) * (hi - lo)
    local = np.stack([uv[:, 0], np.zeros(n), uv[:, 1]], axis=1)
    return centre + local @ rot.T + rng.normal(0, noise_m, (n, 3))


def _frame(rot, centre, noise_m, rng):
    return palm.palm_frame(_palm_points(rot, centre, noise_m, rng),
                           _palm_surface(rot, centre, noise_m, rng))


def _angle_deg(a, b):
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)), -1, 1))))


def test_3_palm_frame_and_quality():
    rng = np.random.default_rng(0)
    rot = rot_about([1, 2, 0.5], 40)
    centre = np.array([0.05, -0.02, 0.6])
    errs, quals = [], []
    for _ in range(200):
        R, noise = _frame(rot, centre, 0.002, rng)
        errs.append(_angle_deg(R[:, 1], rot[:, 1]))
        quals.append(palm.rot_quality(noise))
    assert np.percentile(errs, 95) < 5.0, \
        f"normal error mean {np.mean(errs):.1f}, p95 {np.percentile(errs, 95):.1f} deg"
    assert np.mean(quals) > 0.8, np.mean(quals)
    # The frame itself, not just the normal (axes and handedness).
    R, _ = palm.palm_frame(_palm_points(rot, centre, 0.0, rng))
    assert np.allclose(R, rot, atol=1e-6), R

    noisy = [palm.rot_quality(_frame(rot, centre, 0.020, rng)[1]) for _ in range(200)]
    assert np.mean(noisy) < 0.2 and np.percentile(noisy, 90) < 0.2, \
        (np.mean(noisy), np.percentile(noisy, 90))
    # Control: the same 20 mm noise on the five landmarks ALONE cannot be
    # thresholded -- this is why the interior samples exist.
    five = [palm.rot_quality(palm.palm_frame(_palm_points(rot, centre, 0.020, rng))[1])
            for _ in range(200)]
    assert np.percentile(five, 90) > 0.5, np.percentile(five, 90)
    # Three points always fit a plane: quality must be unknown (0), not 1.
    three = {k: v for k, v in _palm_points(rot, centre, 0.020, rng).items() if k in (0, 5, 9)}
    assert palm.rot_quality(palm.palm_frame(three)[1]) == 0.0


def test_3b_full_mode_refuses_low_quality_rotation():
    """Consumer side of rot_quality: a 30 deg palm turn at quality 0.1 does
    not turn the target in `full` mode; at 0.9 it does."""
    for q, expect_turn in ((0.1, False), (0.9, True)):
        clk = Clock()
        m = ArmTeleopMapper(orient_mode="full")
        h = Hand(quality=q)
        ee = (np.array([0.3, 0.0, 1.0]), rot_about([1, 0, 0], 180))
        for _ in range(5):
            clk.t += 1 / 30
            m.update(h.frame(clk.t), clk.t, 1 / 30, True, False, *ee, np.zeros(3), 350.0)
        h.rot = rot_about([0, 0, 1], 30)
        for _ in range(120):
            clk.t += 1 / 30
            out = m.update(h.frame(clk.t), clk.t, 1 / 30, True, False, *ee, np.zeros(3), 350.0)
        turned = np.rad2deg(np.arccos(np.clip((np.trace(ee[1].T @ out.target_rot) - 1) / 2, -1, 1)))
        assert (turned > 20) == expect_turn, (q, turned)
        if not expect_turn:
            assert out.status["rot_held"] == "low_quality"


def test_frames_share_upstream_convention():
    """The depth frame and MediaPipe's must be the SAME convention or fusing
    them averages different axes. On coplanar points they must coincide."""
    rot = rot_about([0.3, -1, 0.2], 65)
    kp = np.zeros((21, 3))
    pts = {k: rot @ np.array(v) for k, v in LOCAL.items()}
    for k, v in pts.items():
        kp[k] = v
    ours, _ = palm.palm_frame(pts)
    upstream = SingleHandDetector.estimate_frame_from_hand_points(kp)
    assert np.allclose(ours, upstream, atol=1e-9), (ours, upstream)


def test_fusion_rules():
    a, b = np.eye(3), rot_about([0, 0, 1], 40)
    r, src = palm.fuse(a, b, 0.25)
    assert src == "mediapipe" and np.allclose(r, a)
    r, src = palm.fuse(a, b, 0.75)
    ang = np.rad2deg(np.arccos((np.trace(r) - 1) / 2))
    assert src == "fused" and abs(ang - 30.0) < 1e-6, ang
    r, src = palm.fuse(None, b, 0.9)
    assert src == "depth"


def test_depth_lookup_and_deprojection():
    k = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    depth = np.full((480, 640), 0.6, np.float32)
    depth[::7, ::5] = 0.0                       # dropouts the median must skip
    rot = rot_about([1, 0, 0], 90)              # palm facing the camera: plane z = 0.6
    pts_world = {kk: np.array([0.02, 0.01, 0.6]) + rot @ np.array(v) for kk, v in LOCAL.items()}
    kp_px = np.zeros((21, 2))
    for kk, p in pts_world.items():
        kp_px[kk] = (k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2])
        depth[int(round(kp_px[kk, 1])), int(round(kp_px[kk, 0]))] = 0.0   # hole right on it
    pos, R, q, pts = palm.palm_from_depth(kp_px, depth, k)
    assert len(pts) == 5 and q == 1.0, q
    want = np.mean([[p[0], p[1], 0.6] for p in pts_world.values()], axis=0)
    assert np.linalg.norm(pos - want) < 0.002, (pos, want)
    assert _angle_deg(R[:, 1], [0, 0, 1]) < 3.0
    depth[:] = 0.0
    pos, _, _, _ = palm.palm_from_depth(kp_px, depth, k)
    assert pos is None, "no valid depth must give pos_valid=false"


def test_fist_and_debounce():
    open_hand = np.zeros((21, 3))
    for t, m_, x in zip(palm.FINGER_TIPS, palm.FINGER_MCPS, (0.0, 0.01, 0.02, 0.03)):
        open_hand[m_] = [x, 0.08, 0]
        open_hand[t] = [x, 0.16, 0]
    fist = open_hand.copy()
    for t in palm.FINGER_TIPS:
        fist[t] = fist[t] * [1, 0.3, 1]
    assert palm.is_fist(fist) and not palm.is_fist(open_hand)
    d = palm.Debounce(config.FIST_DEBOUNCE_FRAMES)
    seq = [True, True, False, True, True, True, False, False, False]
    states = [d.update(s) for s in seq]
    assert states == [False, False, False, False, False, True, True, True, False], states


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
