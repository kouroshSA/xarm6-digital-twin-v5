# `geometry/` — SE(3) interpolation, smoothing, dwell detection

Five pose utilities vendored from [Aether](https://github.com/InternRobotics/Aether)
(MIT). Provenance and the exact divergences are in [NOTICE.md](NOTICE.md).

```python
from geometry import smooth_poses, detect_static_sequence

is_static, trans_diff, rot_diff = detect_static_sequence(poses)   # (N,4,4)
clean = smooth_poses(poses, window_size=11, method="gaussian")
```

Run the tests (no hardware, no MuJoCo):

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
python -m geometry.test_pose_filters
```

## The API

| Function | Does |
|---|---|
| `slerp(q1, q2, t)` | shortest-arc quaternion interpolation, `(x,y,z,w)` |
| `interpolate_poses(p1, p2, weight)` | SE(3) interpolation; **`weight` is p1's share** |
| `smooth_poses(poses, window_size, method)` | temporal smoothing, `gaussian`/`savgol`/`ma` |
| `detect_static_sequence(poses, threshold)` | `(is_static, trans_diff, rot_diff)` |
| `adaptive_pose_smoothing(poses, td, rd, base_window)` | window sized inversely to motion |

## Three things that will bite

**`interpolate_poses`' weight runs backwards from `slerp`'s `t`.**
`weight=1.0` returns **pose1**, `weight=0.0` returns **pose2** — opposite to the
`t` convention in `slerp` directly above it. Upstream's docstring agrees with
this; the work order that commissioned the module stated it inverted. The test
suite pins the real direction, so a "fix" toward the work order's wording fails
loudly rather than silently reversing every interpolation.

**Quaternions here are `(x, y, z, w)`, MuJoCo's are `(w, x, y, z)`.**
SciPy's ordering is used throughout because these functions are built on
`scipy.spatial.transform`. Convert at the boundary if you feed MuJoCo poses in.

**Translations are in METRES; the twin works in millimetres.** The hard-coded
constants in `detect_static_sequence` (`threshold=0.01`) and
`adaptive_pose_smoothing` (`0.1 / motion`) only make sense for metres.
`set_position`, the recordings and the LLM plans are all in mm; feed those in
unconverted and a 0.3 mm jitter reads as motion, so the adaptive window always
collapses to `base_window`. Divide by 1000 at the boundary.

**`detect_static_sequence` compares two different units to one threshold.**
`trans_diff` is metres, `rot_diff` is a dimensionless Frobenius norm, and both
are tested against the same `threshold`. That is upstream's choice, kept so
behaviour matches. Do not read the default `0.01` as "one centimetre" — tune it
against whichever term dominates your data.

## Why this is not wired into anything

Deliberately importable and tested, nothing more. The intended consumers, in
the order they are likely to arrive:

1. **Human-demo retargeting** (not built). Hand-pose estimates are jittery and
   need exactly this smoothing, plus `detect_static_sequence` to segment a demo
   into approach / grasp / lift / release.
2. **`scripts/audit_motion_errors.py`**. `detect_static_sequence` is a cheap
   "did the arm actually move" check — worth having given that 40% of the one
   recorded demo episode has zero joint motion.
3. **Trajectory resampling for `replay.py`**. `interpolate_poses` changes
   playback rate without breaking rotations, which naive per-component
   interpolation of a rotation matrix does.

None of those are in scope yet. This file records intent so the next session
knows why a package exists with no callers.
