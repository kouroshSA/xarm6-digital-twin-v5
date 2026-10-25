# teleop_hand/test_retarget.py
"""Gripper retargeting tests (spec §7, tests 1 and 2), no camera.

    python -m teleop_hand.test_retarget

Needs dex_retargeting (installed in both xarm6sim and xarm6sim_hand).
The fixture's provenance is in fixtures/make_pinch_fixture.py: real
MediaPipe frames plus a synthetic pinch sequence.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from teleop_hand import config  # noqa: E402
from teleop_hand.retarget import (ASSETS, CONFIG_YML, BinaryGrip,  # noqa: E402
                                  ContinuousGrip, GripperRetargeter,
                                  pinch_distance_m, pinch_reference)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pinch_sequence.json"


def _fixture():
    return json.loads(FIXTURE.read_text())


def test_1_retarget_agrees_with_direct_pinch():
    fx = _fixture()
    rt = GripperRetargeter()
    seq = [np.asarray(f["joint_pos"]) for f in fx["sequence"]]
    ret = np.array([rt.opening(jp) for jp in seq])
    ref = np.array([pinch_reference(jp) for jp in seq])
    r = float(np.corrcoef(ret, ref)[0, 1])
    assert r > 0.9, f"correlation {r:.3f}"
    assert ret.min() < 0.15 and ref.min() < 0.15, (ret.min(), ref.min())
    assert ret.max() > 0.85 and ref.max() > 0.85, (ret.max(), ref.max())
    # The two must agree on WHEN it is pinched, not just that it was once.
    assert abs(int(np.argmin(ret)) - int(np.argmin(ref))) <= 8, "pinch timing disagrees"

    # Real MediaPipe frames: independent of the synthetic edit. Same ordering
    # by opening (rank correlation), fresh filter state per frame.
    st = fx["static"]
    rs, fs = [], []
    for f in st:
        rt.reset()
        jp = np.asarray(f["joint_pos"])
        for _ in range(30):                   # let the low-pass settle
            o = rt.opening(jp)
        rs.append(o)
        fs.append(pinch_reference(jp))
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    rho = float(np.corrcoef(rank(rs), rank(fs))[0, 1])
    assert rho > 0.85, (rho, list(zip([f["source"] for f in st], np.round(rs, 2), np.round(fs, 2))))


def test_1_real_d435_pinches():
    """Test 1 on a REAL recorded keypoint log (D435i, 3 pinches): the
    retargeter agrees with the direct pinch distance, reaches both extremes,
    and the binary gripper sees exactly the pinches -- no chatter."""
    fx = json.loads((FIXTURE.parent / "pinch_real_d435.json").read_text())
    rt = GripperRetargeter()
    seq = [np.asarray(f["joint_pos"]) for f in fx["sequence"]]
    t = np.array([f["t"] for f in fx["sequence"]])
    ret = np.array([rt.opening(jp) for jp in seq])
    ref = np.array([pinch_reference(jp) for jp in seq])
    r = float(np.corrcoef(ret, ref)[0, 1])
    assert r > 0.9, r
    assert ret.min() < 0.15 and ret.max() > 0.85, (ret.min(), ret.max())
    g = BinaryGrip()
    edges = [e for o in ret[t > 2.0] if (e := g.update(o))]   # after the hand settled in view
    assert edges == ["close", "open", "close", "open", "close"], edges


def test_1b_fixed_base_is_what_breaks_it():
    """Negative control for the yml's add_dummy_free_joint: the same gripper
    with a FIXED base reads a real, wide-open hand as nearly closed, because
    it only sees the thumb->index vector's projection on its jaw axis."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ASSETS / "xarm_gripper", tmp / "xarm_gripper")
        yml = CONFIG_YML.read_text().replace("add_dummy_free_joint: True",
                                             "add_dummy_free_joint: False")
        (tmp / "fixed.yml").write_text(yml)
        fixed = GripperRetargeter(tmp / "fixed.yml", urdf_dir=tmp)
    free = GripperRetargeter()
    worst = None
    for f in _fixture()["static"]:
        jp = np.asarray(f["joint_pos"])
        if pinch_distance_m(jp) < 0.07:
            continue                            # only clearly open hands
        free.reset()
        fixed.reset()
        for _ in range(30):
            o_free, o_fixed = free.opening(jp), fixed.opening(jp)
        assert o_free > 0.85, (f["source"], o_free)
        worst = o_fixed if worst is None else min(worst, o_fixed)
    assert worst is not None and worst < 0.5, \
        f"fixed-base control did not fail (min opening {worst}); the free joint may be unnecessary"


def test_2_joint_order_by_name():
    """Reorder the joints in a copy of the URDF: finger_joint moves in
    joint_names, the opening must not change.

    Pinocchio orders joints by the kinematic tree (siblings by name), not by
    their order in the file -- swapping the two <joint> blocks changes
    nothing. Renaming the mimic joint so it sorts first does."""
    src = (ASSETS / "xarm_gripper" / "xarm_gripper.urdf").read_text()
    joints = re.findall(r"  <joint .*?</joint>\n", src, flags=re.S)
    assert len(joints) == 2
    shuffled = src.replace(joints[0] + joints[1], joints[1] + joints[0]).replace(
        'name="finger_joint_mimic"', 'name="a_finger_joint_mimic"')
    assert shuffled != src
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "xarm_gripper").mkdir()
        (tmp / "xarm_gripper" / "xarm_gripper.urdf").write_text(shuffled)
        a = GripperRetargeter()
        b = GripperRetargeter(urdf_dir=tmp)
    names_a = list(a.retargeting.joint_names)
    names_b = list(b.retargeting.joint_names)
    assert names_a != names_b, "shuffle did not change joint_names; test proves nothing"
    assert a.ji != b.ji, "finger_joint did not move; reading by position would still pass"
    for f in _fixture()["sequence"][::5]:
        jp = np.asarray(f["joint_pos"])
        oa, ob = a.opening(jp), b.opening(jp)
        assert abs(oa - ob) < 1e-6, (oa, ob)


def test_urdf_gap_matches_config():
    """The URDF's pad gap, the yml's scaling and config.PINCH_CLOSED_M are
    one relation (gap = scaling x closed distance) held in three files."""
    urdf = (ASSETS / "xarm_gripper" / "xarm_gripper.urdf").read_text()
    ys = [abs(float(m)) for m in re.findall(r'<origin xyz="0 (-?[0-9.]+) 0"', urdf)]
    gap = sum(ys)
    rt = GripperRetargeter()
    assert abs(gap - rt.scaling * config.PINCH_CLOSED_M) < 1e-9, (gap, rt.scaling)
    # And the limit is read through the retargeter, not copied.
    assert rt.finger_max == 0.0425 and rt.finger_min == 0.0


def test_effector_logic():
    g = BinaryGrip()
    seq = [0.9, 0.5, 0.31, 0.29, 0.2, 0.5, 0.59, 0.61, 0.7, 0.25]
    edges = [g.update(o) for o in seq]
    assert edges == [None, None, None, "close", None, None, None, "open", None, "close"], edges
    c = ContinuousGrip()
    assert c.update(0.5, 0.0) == 0.5
    assert c.update(0.52, 1.0) is None              # < 3 % change
    assert c.update(0.6, 1.01) == 0.6
    assert c.update(0.9, 1.02) is None              # rate limit (10 Hz)
    assert c.update(0.9, 1.2) == 0.9


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
