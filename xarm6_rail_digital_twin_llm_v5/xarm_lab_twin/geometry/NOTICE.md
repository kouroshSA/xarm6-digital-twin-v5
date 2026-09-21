# NOTICE — vendored third-party code

`pose_filters.py` contains code taken from **Aether**.

| | |
|---|---|
| Upstream | https://github.com/InternRobotics/Aether |
| Commit vendored from | `871c6f7ebcd66f3571ed27db5d65264a51d624a3` (2025-10-26) |
| Source file | `aether/utils/postprocess_utils.py` |
| Licence | MIT |
| Vendored on | 2026-09-21 |

## Functions taken

| Function | Upstream line | Changed here? |
|---|---|---|
| `slerp` | 610 | no (type hints, docstring only) |
| `interpolate_poses` | 650 | no (type hints, docstring only) |
| `smooth_poses` | 686 | **yes** — see below |
| `detect_static_sequence` | 354 | no (type hints, docstring only) |
| `adaptive_pose_smoothing` | 368 | **yes** — see below |

Nothing else from Aether is included. In particular this repo takes **no model
weights**, nothing from `aether/pipelines/` (a CogVideoX-5B video-diffusion
pipeline), and none of the torch-based alignment helpers — this repo already
has Procrustes alignment in `perception/extrinsic_calibration.py`.

## Deliberate divergences from upstream

Both are bug fixes found while vendoring; the *maths* is untouched.

1. **`smooth_poses` — unknown `method`.** Upstream leaves `smoothed_trans` /
   `smoothed_quats` unassigned and dies on
   `UnboundLocalError: cannot access local variable 'smoothed_quats'`, which
   names a private variable and tells the caller nothing. Now raises
   `ValueError` listing the valid methods. Input validation on `poses` shape
   and window parity was added for the same reason.

2. **`adaptive_pose_smoothing` — even windows crash.** The adaptive window is
   `int(base_window * (0.1 / motion_magnitude))`, which lands on an even number
   for a wide range of ordinary inputs (motion 0.05 → 10, motion 0.0625 → 8).
   `smooth_poses` requires an odd window, so upstream raises
   `AssertionError: window_size must be odd` on normal data. The window is now
   rounded up to the next odd value. Covered by
   `test_adaptive_pose_smoothing_survives_even_window`.

Note also, for anyone comparing against the commissioning work order: the
hemisphere-flip pre-pass and quaternion renormalisation that the work order
asked to be *added* to `smooth_poses` **already exist upstream** at this
commit. They were kept, and a regression test now guards them.

## MIT License

```
MIT License

Copyright (c) 2025 Aether Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
