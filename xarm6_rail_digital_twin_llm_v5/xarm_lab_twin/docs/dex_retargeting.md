# Hand tracking, dex-retargeting, and the two gestures

How a bare hand in front of a RealSense D435 drives the twin (and, later, the
real xArm6), what `dex_retargeting` does inside that, and the two gestures built
on top of it: **fist-follow** and **point-to-grasp**.

Code: [`teleop_hand/`](../teleop_hand/). Status, measured findings and the to-do
list for the clutch-driven teleop are in [`teleop_hand/README.md`](../teleop_hand/README.md);
this file explains the ideas and documents the gestures.

---

## 1. The pipeline

```
 tracker process (env xarm6sim_hand)               teleop process (env xarm6sim)
 ───────────────────────────────────               ─────────────────────────────
 D435 colour+depth                                  HandFrame (JSON over UDP :47100)
   └─ MediaPipe HandLandmarker ── 21 landmarks          │
        ├─ palm pose (depth)  ────────────────┐         ├─ ArmTeleopMapper   clutch, filters, rail
        ├─ dex-retargeting ── gripper opening ├─ UDP ──►│     └─ guarded IK ─► TwinServo ─► MuJoCo
        └─ gesture classifier ── fist / point ┘         └─ GestureController (clutch released)
                                                              ├─ fist-follow
                                                              └─ point-to-grasp
```

Two processes because MediaPipe pulls `opencv-contrib-python`, which collides with
the `opencv-python` the sim env uses. The tracker publishes one JSON datagram per
camera frame, including frames with no hand, so silence on the socket always
means "tracker dead" and never "no hand".

## 2. What dex-retargeting does

**Retargeting** = turning a human hand pose into joint commands for a robot hand
whose geometry differs. Joint angles cannot be copied (a gripper has no
knuckles), so `dex_retargeting` (AnyTeleop, v0.5.0) solves an optimisation each
frame:

1. **Task vectors.** From the 21 MediaPipe keypoints take the vectors that
   matter, here *thumb tip → index tip* (landmarks 4 and 8).
2. **Robot side.** Load the gripper URDF (`xarm_gripper.urdf`). Its two finger
   links define the matching robot vector *left finger → right finger*.
3. **Objective.** Find the robot joint values `q` that minimise the Huber
   (SmoothL1) distance between the **scaled** human vector and the robot vector.
   `scaling_factor` (1.5 here) compensates for the gripper being a different
   size from a hand.
4. **Solver.** NLopt SLSQP over `q`, with forward kinematics and Jacobians from
   **Pinocchio** and the loss evaluated in **PyTorch**. The previous solution
   seeds the next frame (warm start), and an exponential low-pass
   (`low_pass_alpha`) smooths the output.

The result is one number, `finger_joint`, normalised to a 0 (closed) … 1 (open)
**opening**. The receiver turns that into the visual finger position every tick,
and into a weld/release on hysteresis (close below 0.30, open above 0.60).

Files: `retarget.py` (`GripperRetargeter`), `assets/retarget/xarm_gripper.yml`,
`assets/retarget/xarm_gripper/xarm_gripper.urdf`.

### Why the config has a dummy free joint

A 1-DoF gripper with a fixed base can only match the *projection* of the human
vector onto its jaw axis. On real landmarks the thumb→index direction swings so
far between hand poses that this projection changes sign, and an open hand read
as closed. `add_dummy_free_joint: True` gives the optimiser a floating base so it
can align the jaw with the human vector first. Only the vector's **length** then
drives `finger_joint`. Consequence: joints are looked up **by name**
(`finger_joint`), never by index.

### What dex-retargeting does *not* do here

It handles only the gripper opening. The arm and rail are driven separately: the
palm pose is measured from depth (`palm.py`), mapped by `ArmTeleopMapper`
(`mapping.py`) with a clutch, One-Euro filtering, jump rejection and a rail
edge-push, and solved by guarded IK (`guards.py`). A dexterous hand later would
be a new URDF + YAML, not new code.

### Software stack

| Piece | Role |
|---|---|
| RealSense D435 + `pyrealsense2` | colour + aligned depth, 30 fps |
| MediaPipe HandLandmarker (Tasks API) | 21 image + 21 metric world landmarks |
| `dex_retargeting` 0.5.0 | gripper opening (above) |
| Pinocchio (`pin`), NLopt, PyTorch | inside dex_retargeting |
| One-Euro / slerp low-pass (`filters.py`) | palm position / rotation smoothing |
| twin IK (`sim/ik_solver.py`) + `guards.py` | arm joints, with FK verification |
| MuJoCo | the twin |
| UDP, localhost | tracker → teleop |

---

## 3. The gestures

Both gestures are **absolute** ("where my fist is", "what my finger points at"),
while the clutch teleop is **relative**, so they need the camera's full pose in
the robot world frame (§4). Both are active **only while the clutch is
released**: a fisted hand closes the pinch, which under the clutch would mean
"close the gripper".

### Setting up on another machine

```bash
git clone https://github.com/kouroshSA/xarm6-digital-twin-v5_dev.git   # private; needs access
cd xarm6-digital-twin-v5_dev/xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
conda create -n xarm6sim python=3.11 -y && conda activate xarm6sim
pip install -r requirements.txt                    # includes dex_retargeting==0.5.0
conda create -n xarm6sim_hand --clone xarm6sim -y  # tracker env (mediapipe)
conda activate xarm6sim_hand && pip install -r requirements-hand.txt
```

The GG-CNN weights are in the repo (`perception/grasp/weights/`). The MediaPipe
hand model downloads on first tracker run. Both gestures run in the twin with no
robot attached; the only hardware needed is a D435 (`run_hand_tracker.py --demo`
and the tests need none). Check the install with `python -m teleop_hand.test_gestures`
(xarm6sim) and `MUJOCO_GL=egl python -m teleop_hand.test_gestures_twin`. The
gestures use the placeholder camera pose until the tracker camera is calibrated.

Enable on both processes:

```bash
python scripts/run_hand_tracker.py --gestures --show            # xarm6sim_hand
python scripts/run_handteleop_sim.py --gestures --operator-view # xarm6sim
```

The tracker classifies the **main hand** (`--hand`, default Right). The left-hand
fist (`--aux-gesture`, `--clutch gesture`) is the clutch and is unrelated.

### 3.1 Recognising them (`gestures.py`)

Pure NumPy on MediaPipe's metric world landmarks, debounced over 3 frames:

| gesture | rule |
|---|---|
| **fist** | all four fingertips closer to the wrist than their MCP knuckles |
| **point** | index extended (tip farther from the wrist than the PIP joint, MCP→PIP→tip bent < 40°) **and** middle, ring, pinky curled. The thumb is ignored |

Anything else, including a peace sign or a half-open hand, is `none`.

### 3.2 Fist-follow (`follow.py`)

The tool follows the fist at **100–150 mm**, continuously and in real time.

```
target = fist + d · unit(EE − fist)        d = clip(|EE − fist|, 100, 150) mm
```

* The EE approaches **along the line it is already on**, never swinging round
  the hand. It stops at 150 mm, backs off if the fist comes inside 100 mm, and
  does not move radially while inside the band (so a still hand is not chased
  by a tool trying to hit exactly 125 mm).
* A fist must be held **0.25 s** before the arm moves (accidental fists while
  the hand travels), then it tracks every camera frame. The standoff is
  re-evaluated from the *current* EE every tick.
* Fist position is One-Euro filtered. A single-frame jump over 120 mm is treated
  as a depth outlier (a fist's edge can pick up the background) and ignored
  unless 3 frames agree.
* Speed is capped at 150 mm/s by the same `rate_limit_pose` the other teleops
  use. **Orientation is locked** to whatever it was when the fist engaged.
* Floor and workspace clamps come after the standoff. If they would push the
  target inside 100 mm of the hand, the target moves away from the fist
  horizontally instead: **keeping off the hand outranks keeping off the floor**.
  The floor is 870 mm (the bench top is at 750 mm).
* The rail follows through the mapper's existing edge-push when `--rail edge`.
* Opening the fist, losing the hand for 150 ms, or pressing the clutch stops it.
  The arm holds where it is; a new fist has to be held again.

### 3.3 Point-to-grasp (`pointing.py`, `point_grasp.py`)

**The ray.** Origin = the index MCP, deprojected with its *measured depth*.
Direction = unit MCP→tip from MediaPipe's world landmarks. Deliberately **not**
the fingertip's depth: a fingertip is a few pixels wide, so its depth sample is
usually the background behind it, giving a ray through a point metres past the
hand. A position on the hand is measured; a direction is what the landmark model
is good at.

**Commit.** The ray must hold within 4° of its running mean for **1.0 s**. It
then fires once; the gesture has to end before it can fire again. A wobbling
finger never fires.

**Resolve** (`resolve_pointing`), first match wins:

| kind | when | target |
|---|---|---|
| **grasp** | a graspable object lies within **8°** of the ray | the object (a *prior*, see below) |
| **hover** | the ray hits something that is not graspable (bench, instrument, bin) | stops **80 mm short** of the surface along the ray, never below 100 mm above the bench |
| **free** | the ray hits nothing | the farthest point along the ray that is inside the workspace box, ≥ 100 mm above the bench, and within 550 mm of the arm base |

"Pointed at" is a **cone**, not a line: a finger ray is good to a few degrees,
and 8° at 0.6 m is 84 mm. The cone test runs before the ray cast, otherwise a ray
that grazes a cube's shoulder would report the bench behind it. Among objects in
the cone, the smallest *angle* wins, not the nearest object.

`TwinScene` casts rays with MuJoCo's ray caster (skipping the arm and hidden geom
groups, via the same mask the camera uses). `SphereScene` is the unit-test and
hardware stand-in until the DetectionService supplies object poses.

**Grasp sequence** (`PointGraspExecutor`, blocking; pressing the clutch aborts
between steps):

1. bring the object within reach along the rail if needed
2. fly to 300 mm above the pointed position, tool down
3. capture the **wrist camera** and run **GG-CNN**, UFACTORY's grasp network
   (vendored in `perception/grasp/`)
4. keep grasps within 25 mm of the pointed object and take the **nearest to it**,
   then refuse any grasp that is nearer a *different* known object
5. 100 mm above the grasp point, descend, close, lift 120 mm

It **refuses, and says why, rather than descending on the prior.** If GG-CNN
proposes nothing near the pointed object, the object was *seen*, but nothing
verified the jaws can close there.

Two decisions here came from measurement, not design: selecting the
highest-quality grasp near the object first picked the heater-shaker's edge
(it sits 9 mm from the red cube), since GG-CNN ranks *graspability*, not task
relevance. Hence "nearest", plus the wrong-object check.

**Roles of the two cameras.** The tracker's D435 (the *observer*) sees the hand
and gives the ray and the coarse target. The **wrist camera** makes the actual
grasp, because it is close, sees the object's sides, and has it centred. Same
division of labour the rest of the repo uses for the overhead observer.

---

## 4. Camera calibration (not done on hardware yet)

The gestures need the tracker camera's pose in the **world** frame (the base
rides the rail, but the camera is fixed to the lab). `camera_pose.py`:

* `HandCameraPose.nominal()` is a **placeholder** so the gestures can run in the
  twin: camera at (350, 550, 1050) mm with the `facing` preset's orientation.
  The runner prints a warning whenever it is in use.
* `HandCameraPose.from_correspondences(palm_cam_m, ee_world_m)` is the real
  procedure: hold the palm against the gripper at **6+ well-spread arm poses**;
  the palm position in the camera frame and the EE position from forward
  kinematics are the point pairs. It uses
  `perception.extrinsic_calibration.solve_camera_pose` (Kabsch, reflection-guarded)
  and **refuses** coplanar or too-few sets rather than solving them. There is
  deliberately no second solver.
* `save()` / `load()` write the JSON that `--cam-extrinsic FILE` reads.

A `scripts/calibrate_hand_camera.py` that collects those pairs live is **not
written yet**; it is the first thing to do on the real rig. A 1° rotation error
at 1 m is ~17 mm, so expect the fist standoff to be fine and the pointing
selection to be the part that is sensitive to it.

---

## 5. Safety and limits

* Nothing moves without a fresh valid frame; a 150 ms dropout (500 ms silence)
  drops to idle.
* The twin's direct servo path has **no collision checking** (README finding 10),
  so fist-follow in the twin can push a link into an instrument. The real arm's
  controller collision detection applies. A keep-out check for fist-follow is
  still to do before the real arm.
* Point moves use `arm.set_position`, i.e. the planner's IK + collision
  validators, at ≤ 80 mm/s (50 mm/s on the descent).
* A finger ray is a few degrees accurate. Fine for "that cube", not for exact
  positions.
* The point sequence is blocking. Only the clutch key interrupts it, between steps.
* Gestures need depth on the knuckle; with the webcam fallback (no depth) pointing
  is disabled.

## 6. Tests

```bash
python -m teleop_hand.test_gestures                      # no hardware: classifier, ray, follow, resolution, state machine
MUJOCO_GL=egl python -m teleop_hand.test_gestures_twin   # live twin, real wrist camera + GG-CNN
```

The twin test grades **physically**: where the EE is relative to the fist, and
*which body the gripper holds and how far it rose*, not return codes. A
negative control checks that with GG-CNN returning nothing the arm does not
descend.

## 7. Not done / next

* Live on the D435: the classifier thresholds are verified on a synthetic hand
  skeleton (any orientation, 2 mm noise), not yet on real landmarks.
* `calibrate_hand_camera.py`, then re-measure pointing accuracy on the rig.
* Real-arm runner (`run_handteleop_real.py`) does not exist, so neither does the
  real-arm path for these gestures.
* Keep-out check for fist-follow.
* Wrist-camera capture while the passive viewer is open (GLFW vs EGL) is untested;
  the tests are headless.
* Releasing a grasped object by gesture (open hand) is not implemented; use the
  existing gripper/clutch keys, `Home`, or `End` (reset scene).
