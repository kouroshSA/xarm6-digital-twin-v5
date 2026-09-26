# geometry/test_pose_filters.py
"""Unit tests for geometry/pose_filters.py.

Run from the working directory (xarm_lab_twin/):
    python -m geometry.test_pose_filters     # plain-python, prints PASS/FAIL
    pytest geometry/test_pose_filters.py     # if pytest is installed

Needs no hardware and no MuJoCo — pure NumPy/SciPy.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from geometry import pose_filters as pf


def _pose(rot: np.ndarray, trans) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans
    return m


# ---------------------------------------------------------------------------
# slerp
# ---------------------------------------------------------------------------
def test_slerp_matches_scipy():
    """Our slerp must agree with scipy's Slerp, which is the reference."""
    r1 = R.from_euler("xyz", [10, -20, 30], degrees=True)
    r2 = R.from_euler("xyz", [-40, 60, 120], degrees=True)
    ref = Slerp([0.0, 1.0], R.concatenate([r1, r2]))

    for t in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        ours = R.from_quat(pf.slerp(r1.as_quat(), r2.as_quat(), t))
        # Compare as rotations: quaternion sign is not unique, so comparing
        # components directly would spuriously fail on the q/-q ambiguity.
        ang = (ours.inv() * ref(t)).magnitude()
        assert ang < 1e-9, f"t={t}: {np.rad2deg(ang):.3e} deg from scipy"


def test_slerp_takes_the_short_way():
    """A pair in opposite hemispheres must interpolate the short arc."""
    r1 = R.from_euler("z", 0, degrees=True)
    r2 = R.from_euler("z", 20, degrees=True)
    q2_flipped = -r2.as_quat()          # same rotation, opposite hemisphere

    mid = R.from_quat(pf.slerp(r1.as_quat(), q2_flipped, 0.5))
    ang = np.rad2deg(mid.magnitude())
    # Short way is 10 deg; long way would be 170.
    assert abs(ang - 10.0) < 1e-6, f"midpoint {ang:.3f} deg, expected 10"


# ---------------------------------------------------------------------------
# interpolate_poses
# ---------------------------------------------------------------------------
def test_interpolate_poses_endpoints():
    """`weight` is pose1's share: 1.0 -> pose1, 0.0 -> pose2.

    NOTE the direction. The work order specified the opposite
    (`(p1, p2, 0.0) == p1`), but that contradicts both upstream's docstring
    and upstream's behaviour. This test pins what the code actually does, so
    a future "fix" toward the work order's wording breaks here loudly instead
    of silently reversing every interpolation in the tree.
    """
    p1 = _pose(np.eye(3), [1.0, 2.0, 3.0])
    p2 = _pose(R.from_euler("z", 90, degrees=True).as_matrix(), [9.0, 9.0, 9.0])

    assert np.allclose(pf.interpolate_poses(p1, p2, 1.0), p1, atol=1e-12), \
        "weight=1.0 must return pose1"
    assert np.allclose(pf.interpolate_poses(p1, p2, 0.0), p2, atol=1e-12), \
        "weight=0.0 must return pose2"


def test_interpolate_poses_midpoint():
    p1 = _pose(np.eye(3), [0.0, 0.0, 0.0])
    p2 = _pose(R.from_euler("z", 90, degrees=True).as_matrix(), [10.0, 0.0, 0.0])
    mid = pf.interpolate_poses(p1, p2, 0.5)

    assert np.allclose(mid[:3, 3], [5.0, 0.0, 0.0]), mid[:3, 3]
    ang = np.rad2deg(R.from_matrix(mid[:3, :3]).magnitude())
    assert abs(ang - 45.0) < 1e-9, f"midpoint rotation {ang:.6f} deg, expected 45"


# ---------------------------------------------------------------------------
# smooth_poses
# ---------------------------------------------------------------------------
def test_smooth_poses_reduces_noise():
    """On a noisy sine trajectory, smoothing must beat the raw signal."""
    rng = np.random.default_rng(0)
    n = 120
    t = np.linspace(0, 4 * np.pi, n)

    truth = np.stack([_pose(R.from_euler("z", 20 * np.sin(ti), degrees=True).as_matrix(),
                            [ti, np.sin(ti), 0.0]) for ti in t])
    noisy = truth.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.05, (n, 3))
    # Perturb rotations too, so the quaternion path is genuinely exercised.
    for i in range(n):
        jitter = R.from_rotvec(rng.normal(0, 0.02, 3)).as_matrix()
        noisy[i, :3, :3] = jitter @ noisy[i, :3, :3]

    out = pf.smooth_poses(noisy, window_size=11, method="gaussian")

    rms_before = np.sqrt(((noisy[:, :3, 3] - truth[:, :3, 3]) ** 2).mean())
    rms_after = np.sqrt(((out[:, :3, 3] - truth[:, :3, 3]) ** 2).mean())
    assert rms_after < rms_before, f"smoothing made it worse: {rms_after} >= {rms_before}"

    # And the output must still be valid SE(3).
    for i in range(n):
        rot = out[i, :3, :3]
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9), f"frame {i} not orthonormal"
        assert abs(np.linalg.det(rot) - 1.0) < 1e-9, f"frame {i} det != 1"


def test_from_matrix_really_does_flip_hemispheres():
    """Premise check for the regression test below — keep them adjacent.

    The pre-pass is only worth guarding if `R.from_matrix()` can actually emit
    sign flips. It can: as a rotation sweeps past 360 degrees, `w = cos(θ/2)`
    legitimately changes sign and SciPy follows the true value rather than
    forcing `w >= 0`. If a future SciPy canonicalises the sign, this fails and
    tells you the guard below has gone slack — rather than leaving it quietly
    passing for the wrong reason.
    """
    ang = np.linspace(0, 720, 60).reshape(-1, 1)
    quats = R.from_matrix(R.from_euler("z", ang, degrees=True).as_matrix()).as_quat()
    dots = np.sum(quats[1:] * quats[:-1], axis=1)
    assert (dots < 0).sum() >= 1, (
        "R.from_matrix() no longer produces hemisphere flips on a 0..720 deg "
        "sweep, so the pre-pass regression test can no longer fail")


def test_smooth_poses_hemisphere_flip_regression():
    """THE regression test for the q/-q pre-pass. Verified to have teeth.

    A NOTE ON TEST DESIGN, because the obvious approach silently does nothing.
    The tempting test is "build poses from deliberately sign-flipped
    quaternions" — which is what the commissioning work order asked for. It
    cannot work: `smooth_poses` takes (N, 4, 4) *matrices*, and q and -q
    produce the *same* matrix, so the flip is erased by `R.from_matrix()`
    before the pre-pass ever runs. Such a test passes whether or not the
    pre-pass exists — a test with no teeth, which is worse than no test.

    What does work is a sweep through 360 degrees. There `R.from_matrix()`
    emits genuine sign flips (see the premise check above), the pre-pass has
    real work to do, and deleting it fails this assertion.
    """
    n = 90
    angles = np.linspace(0, 720, n)
    poses = np.stack([
        _pose(R.from_euler("z", a, degrees=True).as_matrix(), [0.0, 0.0, 0.0])
        for a in angles])

    out = pf.smooth_poses(poses, window_size=5, method="gaussian")

    # Compare as rotations, not as Euler angles or quaternion components:
    # both of those have their own wrapping artefacts that would muddy the
    # signal we are actually testing.
    ref = R.from_euler("z", angles.reshape(-1, 1), degrees=True)
    err_deg = np.rad2deg((R.from_matrix(out[:, :3, :3]).inv() * ref).magnitude())

    # Interior samples only: 'nearest' padding legitimately biases the ends.
    worst = float(err_deg[5:-5].max())
    assert worst < 2.0, (
        f"hemisphere pre-pass appears broken: smoothing a 0..720 deg sweep "
        f"moved a sample {worst:.1f} deg from truth")


def test_smooth_poses_rejects_unknown_method():
    """Upstream dies on UnboundLocalError naming a private variable, which
    tells the caller nothing. We raise a ValueError that names the options."""
    poses = np.stack([np.eye(4)] * 7)
    try:
        pf.smooth_poses(poses, window_size=5, method="nope")
    except ValueError as exc:
        assert "gaussian" in str(exc), f"message should list valid methods: {exc}"
    else:
        raise AssertionError("unknown method must raise ValueError")


def test_smooth_poses_rejects_even_window():
    poses = np.stack([np.eye(4)] * 7)
    try:
        pf.smooth_poses(poses, window_size=4)
    except ValueError:
        pass
    else:
        raise AssertionError("even window_size must raise")


def test_smooth_poses_all_methods_run():
    rng = np.random.default_rng(1)
    poses = np.stack([_pose(R.from_rotvec(rng.normal(0, 0.05, 3)).as_matrix(),
                            rng.normal(0, 0.1, 3)) for _ in range(30)])
    for method in ("gaussian", "savgol", "ma"):
        out = pf.smooth_poses(poses, window_size=5, method=method)
        assert out.shape == poses.shape, method
        assert np.isfinite(out).all(), f"{method} produced non-finite values"


# ---------------------------------------------------------------------------
# dwell detection
# ---------------------------------------------------------------------------
def test_detect_static_sequence():
    static = np.stack([_pose(np.eye(3), [1.0, 2.0, 3.0])] * 20)
    is_static, td, rd = pf.detect_static_sequence(static)
    assert is_static, f"a constant sequence must read static (td={td}, rd={rd})"
    assert td == 0.0 and rd == 0.0

    t = np.linspace(0, 1, 20)
    moving = np.stack([_pose(R.from_euler("z", 90 * ti, degrees=True).as_matrix(),
                             [ti, 0, 0]) for ti in t])
    is_static, td, rd = pf.detect_static_sequence(moving)
    assert not is_static, f"a moving sequence must not read static (td={td}, rd={rd})"


def test_adaptive_pose_smoothing_survives_even_window():
    """Regression: upstream crashes here.

    `int(base_window * (0.1 / motion))` lands on an EVEN number for a range of
    ordinary motion magnitudes -- 0.05 gives 10, 0.0625 gives 8 -- and
    smooth_poses rejects even windows, so upstream raises
    `AssertionError: window_size must be odd` on perfectly normal input.
    We round up to odd. Not mentioned in the work order; found while vendoring.
    """
    poses = np.stack([_pose(np.eye(3), [0.0, 0.0, 0.0])] * 15)
    for mag in (0.05, 0.0625, 0.1, 0.2, 1.0):
        out = pf.adaptive_pose_smoothing(poses, mag, 0.0, base_window=5)
        assert out.shape == poses.shape, f"motion {mag}"


def test_adaptive_window_widens_for_slow_motion():
    """The whole point of 'adaptive': less motion => more smoothing."""
    rng = np.random.default_rng(2)
    n = 60
    poses = np.stack([_pose(np.eye(3), [0.0, 0.0, 0.0])] * n)
    poses[:, :3, 3] += rng.normal(0, 0.01, (n, 3))

    slow = pf.adaptive_pose_smoothing(poses, 0.001, 0.0, base_window=5)
    fast = pf.adaptive_pose_smoothing(poses, 1.0, 0.0, base_window=5)

    # A wider window leaves less residual jitter.
    var_slow = slow[:, :3, 3].var(axis=0).sum()
    var_fast = fast[:, :3, 3].var(axis=0).sum()
    assert var_slow < var_fast, (
        f"slow-motion smoothing should be stronger: {var_slow:.3e} !< {var_fast:.3e}")


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
