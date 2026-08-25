# GG-CNN grasp detection

Wrist-camera depth in, ranked antipodal grasp candidates out, in robot base
coordinates.

```python
from perception import SimWristCamera            # or RealSenseWristCamera
from perception.grasp import GGCNNDetector

det    = GGCNNDetector()
grasps = det.detect(SimWristCamera(arm).capture())
if grasps:
    arm.set_position(*grasps[0].to_arm_pose())
```

## Provenance

| Piece | Source | Licence |
|---|---|---|
| Network definitions (`_ggcnn_net.py`, `_ggcnn2_net.py`) | [GG-CNN](https://github.com/dougsm/ggcnn), Douglas Morrison / ACRV-QUT | BSD-3 (`LICENSE.ggcnn`) |
| Cornell weights, xArm integration | [`ufactory_vision`](https://github.com/xArm-Developer/ufactory_vision) | BSD-3 (`LICENSE.ufactory`) |
| Hand-eye calibration | `ufactory_vision`, via `perception/d435i_calib.py` | BSD-3 |

Local copy of upstream: `~/Models/ufactory_vision`.

## Weights

`weights/*.pt` are `state_dict`s converted from UFACTORY's pickled whole-model
files by `convert_weights.py`. Verify at any time:

```bash
python -m perception.grasp.convert_weights --verify-only
```

Both models reproduce upstream's output **bit-for-bit** (max difference 0.0).

The conversion is not cosmetic. Upstream's files are `torch.save(model)`, which
needs `weights_only=False` — that unpickles arbitrary objects, i.e. executes
code from the file — and hard-codes the class path `models.ggcnn.GGCNN`, so it
only loads if some directory called `models` is on `sys.path`. A `state_dict` is
plain tensors: it loads under `weights_only=True` and binds to the vendored
networks by shape.

| file | net | params |
|---|---|---|
| `ggcnn_epoch_23_cornell.pt` | GGCNN | 62,420 |
| `ggcnn2_epoch_50_cornell.pt` | GGCNN2 | 66,676 |

## What a `Grasp` carries

`quality`, `pixel`, `angle_rad` (image plane), `width_px`, `width_m`, `depth_m`,
`position_cam`, `position_world`, `rotation_world`, `yaw_world_rad`, and
`to_arm_pose()` → `(x, y, z, roll, pitch, yaw)` in mm/degrees for
`set_position`.

`position_world` is `None` — and `to_arm_pose()` raises — when the source frame
carried no `cam_to_world`. That is deliberate: a bare camera on a desk has no
robot pose, and silently treating that as identity would hand the arm camera
coordinates shaped like world ones.

## Four deliberate differences from the upstream demo

Upstream's `RobotGrasp` is a closed-loop visual-servoing controller: it streams
poses in xArm servo mode (`set_mode(7)`), tracks a running maximum across
frames, and owns the pick/place state machine. The twin drives the arm through
its own validated motion primitives, so this module keeps the part that is the
*model* — depth in, candidates out — and stops there.

1. **The camera transform is not re-derived.** Upstream rebuilds the
   base→flange→optical chain from euler triples every frame. Here the grasp
   point goes through `RGBDFrame.pixel_to_world`, the path
   `perception/test_projection.py` validates against MuJoCo's ray caster. One
   transform, already checked, rather than a second that agrees with the first
   only until someone edits either.

2. **The predicted width is delivered.** Upstream computes `width` and
   `depth_center`, returns them, and never reads them again — CLAUDE.md's defect
   class #2, in the reference implementation. `Grasp.width_m` converts the
   network's pixel width into metres at the grasp's own depth, which is what a
   gripper aperture needs.

3. **Inference runs at 300×300.** Upstream feeds the full 480×480 crop. GG-CNN
   is fully convolutional so that runs, but the Cornell weights were trained at
   300×300 and the post-filter sigmas (5.0 / 2.0 px) are tuned for that scale.
   Upstream's own `process_depth_image` defaults to 300; only `get_grasp_img`
   overrides it. `GGCNNDetector(out_size=None)` reproduces upstream.

4. **Top-k candidates, not one tracked maximum.** Frame-to-frame max tracking
   only makes sense inside a servo loop. A planner wants options.

Also: inpainted depth holes are suppressed before peak-finding, so the network
cannot score its own interpolation as a grasp.

## Verifying

```bash
# end-to-end: point the camera at cubes with known positions, check the answer
MUJOCO_GL=egl python -m perception.grasp.test_ggcnn

# weights still match the vendored architecture (also runs in task_sweep.py)
python scripts/task_sweep.py

# look at what it would grasp
python scripts/grasp_check.py --at 0 -150 --target green_cube --out /tmp/g
python scripts/grasp_check.py --real
```

`test_ggcnn.py` is the one that matters. "The network produced a grasp" is not
the claim worth testing — a confident, well-formed, entirely wrong pose passes
that every time. It instead points the wrist camera at cubes whose positions
MuJoCo already knows and requires the returned world coordinate to be that cube,
which exercises render, GG-CNN, deprojection, hand-eye and world composition
together. Current result:

| target | lateral error |
|---|---|
| `green_cube` | 3.6 mm |
| `red_cube_front` | 4.6 mm |
| `blue_cube` | 4.9 mm |

Vertical offset runs 15–30 mm because GG-CNN grasps the visible **top face**,
while the ground truth is the body centre — expected, not error.

## Limits worth knowing

- **Trained on Cornell, not on this bench.** It generalises to simple graspable
  shapes; it has no notion of the tube racks, the OT-2 or the translucent cup.
- **Depth-only.** Colour is used for visualisation, never for inference. A red
  and a blue cube are the same object to it.
- **Real depth is much worse than sim depth** (~50–70% valid vs 100%, with
  dropouts on transparent and specular surfaces). `SimWristCamera(arm,
  noise_std_m=...)` closes part of the gap; it is not a calibration.
- **It ranks graspability, not task relevance.** With several objects in view
  the top candidate is often not the one you meant. Pair it with the object
  registry, or pose the camera so the target dominates the frame.

## Dependencies

`torch` (CPU is fine — 62k parameters, a few ms), `opencv`, `scipy`,
`scikit-image`. All are installed in the `xarm6sim` env. The imports are
deferred, so the sim runs without them and the sweep check SKIPs rather than
fails.
