# Wrist camera — Intel RealSense D435i

An eye-in-hand D435i on the xArm6 flange, present twice: as MuJoCo cameras in
the scene and as the physical device. Both are reached through the same API and
return the same `RGBDFrame`, so a perception routine written against the twin
runs unchanged against the cell.

```python
from perception import SimWristCamera            # or RealSenseWristCamera

cam   = SimWristCamera(arm)                      # arm is a SimXArmAPI
frame = cam.capture()                            # colour + aligned depth
xyz   = frame.pixel_to_world(320, 240)           # metres, robot base frame
```

## Layout

| File | What it holds |
|---|---|
| `d435i_calib.py` | **Every camera number, once.** Intrinsics, extrinsics, hand-eye, stream config. |
| `rgbd.py` | `RGBDFrame` + deprojection, projection, point clouds. |
| `sim_camera.py` | MuJoCo backend. |
| `realsense_camera.py` | Physical-device backend (needs `pyrealsense2`). |
| `sync_scene.py` | Writes the scene XML block from the calibration. |
| `test_projection.py` | Proves the sim camera's optics match what MuJoCo renders. |

Operator-facing entry point: `scripts/wrist_cam_check.py`.

Grasp detection on top of these frames lives in
[`grasp/`](grasp/README.md) — GG-CNN from `ufactory_vision`, depth in and ranked
grasp poses out in robot base coordinates. Entry point:
`scripts/grasp_check.py`.

Language-conditioned targeting sits on top of both:
[`language/`](language/README.md) — "the blue cube" in, the grasp for that cube
out, with depth resolving the ambiguities grounding cannot. Entry point:
`scripts/target_check.py`.

## Observer cameras

Two fixed D435s watch the bench, alongside the wrist camera:
[`observer_calib.py`](observer_calib.py) holds their poses,
[`scene_camera.py`](scene_camera.py) renders them, and they return the same
`RGBDFrame` — so `LanguageTargeter` and `GG-CNN` work through them unchanged.

```python
from perception.scene_camera import SceneCamera
frame = SceneCamera(arm, "cam_overhead").capture()
```

Placement was measured, not chosen (coverage, occlusion and tilt against the
scene). Two results shaped it:

- **Don't mount on the OT-2** — it looks *along* the bench: 1 of 9 objects
  visible, 7 occluded. The most convenient surface, the worst viewpoint.
- **Tilt beats position.** A depth camera's error lies along its view axis, so an
  oblique camera puts ~90% of it into X-Y — the coordinate the arm is commanded
  with. Near-vertical puts 86% into Z, where a top-down grasp barely cares.

Measured through the stack:

| | X-Y error (median) | identity |
|---|---|---|
| `cam_overhead` (15° off vertical) | **7.7 mm** | 2/3 |
| `cam_tripod` (63° off vertical) | 18.9 mm | **3/3** |

They are complements, not spares. Overhead measures better; the oblique view
*identifies* better, because from above a cube shows only its flat top face and
green and blue separate by ~0.03 of grounding score. Raising the observers to
720p did not fix that — the cause is the viewpoint, not the pixel count.

**So: survey and verify with the observers, commit the grasp with the wrist.**
The wrist camera works from ~300 mm with the target centred and sees the sides;
that is where identity and the final pose belong.

## Where the numbers came from

* **Intrinsics and depth→colour extrinsics** — read off the physical device
  (serial `027422071693`) with `rs-enumerate-devices -c` on 2026-08-25. Intel's
  factory calibration, verified against the live device at every connect.
* **Hand-eye (flange → colour optical)** — UFACTORY's published figure for their
  xArm camera stand, taken verbatim from `ufactory_vision`'s D435 GGCNN demo.

> **The one assumption to re-check on real hardware.** The hand-eye transform
> presumes UFACTORY's standard camera stand. If the rig uses a different
> bracket, re-measure it — it is a single constant in `d435i_calib.py`, and
> `python -m perception.sync_scene` pushes the new value into both scenes.
> Everything else is the camera's own factory data and stays valid.

## Changing the calibration

The scene XML is a build product, never hand-edited:

```bash
$EDITOR perception/d435i_calib.py       # change the constant
python -m perception.sync_scene         # rewrite both scenes
python -m perception.test_projection    # re-verify the optics
```

`sync_scene` rewrites the `<body name="wrist_camera">` block inside
`<body name="gripper">` in `lab_scene_primitive.xml` and then reruns
`envs/build_mesh_scene.py`. Putting the camera in the gripper subtree is
deliberate: `build_mesh_scene.py` copies that subtree verbatim, so one edit
reaches both the primitive and the generated mesh scene with no second copy to
keep in step.

`scripts/sim_checks.py::check_wrist_camera_matches_calib` runs in the normal
sweep and fails if either scene drifts from the calibration — including the case
where someone syncs the primitive and forgets to regenerate. A camera 30 mm out
of place still renders a perfectly plausible picture, so this cannot be left to
eyeballing.

## One physical side effect

The housing geom never collides (`contype`/`conaffinity` 0), so it cannot
invalidate a collision-free trajectory. It does carry the device's real **72 g**,
taking the gripper subtree from 190 g to 262 g — a real D435i on the flange
really does weigh that, and a twin that draws the camera but treats it as
weightless is lying about the arm. At 5% of the xArm6's 5 kg rated payload it
changes nothing about reachability.

## Frames, and the two conventions that meet here

* **optical** (RealSense / ROS / OpenCV): `+x` right, `+y` down, `+z` forward.
  All intrinsics and every `RGBDFrame` coordinate use this.
* **MuJoCo camera**: `+x` right, `+y` up, `−z` forward.

They differ by 180° about `x`. `d435i_calib.mujoco_mount_quat()` and
`sim_camera._MUJOCO_TO_OPTICAL` are the only two places that flip; nothing else
should hand-roll it.

Two conversions in this module were **measured, not derived**, because the
obvious reasoning gives the wrong answer (see `principal_pixel`'s docstring):

* MuJoCo's `principalpixel="0 0"` puts the optical axis at `((W−1)/2, (H−1)/2)`,
  not `(W/2, H/2)`.
* Increasing `principalpixel` moves the axis toward **lower** u *and* v — both
  axes are negated, not just y.

Getting either wrong translates the image by a few pixels while leaving focal
length correct, straight lines straight, and every round-trip through the same
intrinsics self-consistent. It is invisible except against an independent
reference, which is why `test_projection` checks the renderer against MuJoCo's
**ray caster** rather than against itself, and carries a negative control that
fails if a deliberately corrupted principal point would still pass.

## Aligned depth

`capture(align=True)` (the default) is what `rs.align` does on the real device:
depth is reprojected into the colour frame, so the frame then carries the
**colour** intrinsics. Deprojecting aligned depth with the depth imager's matrix
is a ~15 mm lateral bias that looks entirely reasonable in a point cloud.
`RGBDFrame.intrinsics` always carries the right matrix, so no caller has to
choose.

## Sim is not real

| | sim | real (this device, pointed at a room) |
|---|---|---|
| valid depth | 100% | ~71% |
| depth noise | none (1 mm quantisation only) | several mm, range-dependent |
| transparent / specular | exact geometry | dropouts — the translucent cup mostly vanishes |

`SimWristCamera(arm, noise_std_m=...)` adds flat Gaussian depth noise, which
closes part of that gap and is not a calibration. A routine that only ever ran
on clean sim depth is not yet validated for the cell. Use the twin for the
geometry, the device for the perception.

## Verifying

```bash
MUJOCO_GL=egl python -m perception.test_projection   # optics vs MuJoCo's ray caster
python scripts/wrist_cam_check.py --both --out /tmp/w  # sim + device, saves PNGs
python scripts/task_sweep.py                         # includes the drift check
```

`--real` works with the camera on a desk — no robot needed. With no arm
attached, frames carry no pose and `pixel_to_world` raises rather than quietly
returning camera coordinates dressed as world ones.
