# xArm6 Digital Twin — Project Conventions for Claude Code

This repo holds two MuJoCo-based simulations of a UFACTORY xArm6 on a 700mm
rail. See [README.md](README.md) for the overall layout, install, and quick
start. This file is project-level guidance for Claude Code sessions opened
inside this directory.

## Working directory layout

- `xarm6_rail_sim_interactive_basic_v5/` — manual-control sim (keyboard, Tk
  sliders, terminal REPL). No API key required.
- `xarm6_rail_digital_twin_llm_v5/` — Claude-driven sim. Needs an Anthropic
  API key in `xarm_lab_twin/.env` (gitignored).

For both: `cd <project>/xarm_lab_twin && python <script>.py`. Conda env is
`xarm6sim` (Python 3.11). The detailed setup is in
[`xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/README.md).

## Scene files (GENERATED default — read before editing)

The digital-twin arm now uses the real UFACTORY xArm6 visual meshes. This
splits the scene into a source and a generated file:

- `envs/lab_scene_primitive.xml` — **hand-edited SOURCE**. Primitive
  box/cylinder arm + all scene furniture (bench, rail, cubes/bins/tubes,
  OT-2, instruments). Edit scene content **here**. Also the `--scene
  primitive` fallback.
- `envs/lab_scene.xml` — **GENERATED default** (has a "DO NOT EDIT" banner).
  Every entry point loads this; it's the primitive source with the six arm
  links swapped for xArm6 meshes at the true URDF kinematics.

After editing `lab_scene_primitive.xml`, regenerate with
`python envs/build_mesh_scene.py`. Do **not** hand-edit `lab_scene.xml` — the
next regen overwrites it. Meshes + provenance live in `envs/assets/xarm6/`
(UFACTORY BSD-3); `--scene {meshes,primitive}` on `run_task.py` picks the arm
(default `meshes`). Validate a scene change with `scripts/ik_sanity.py`,
`scripts/pickplace_check.py`, and `scripts/task_sweep.py`.

## Episode learning loop

When iterating on a task that fails, prefer `--loop`:

```bash
python scripts/run_task.py "<task>" --model haiku --loop --max-episodes 10
```

The loop:

- Resets the scene between episodes (`arm.reset_scene()`)
- Reads `physical_outcome()` to grade success *physically*, not just by
  command return codes (a command sequence can return all-zeros and still
  leave the cube in the wrong place — that's a failure for the loop's
  purposes)
- Accumulates learned constraints across episodes and injects them into the
  next attempt's task prompt
- Appends one entry to `lessons.md` per episode (label includes `[ep N/M]`)

When extending the loop:

- Add new failure patterns to `agent/episode_loop.py::analyse_command_failure`
- Add new task patterns to `agent/outcome_checker.py::expected_outcome`
- Do **not** bypass `arm.reset_scene()` between episodes — the loop assumes
  a deterministic starting state. If your change requires preserving state
  across episodes, that's a different feature (multi-objective episodes,
  not yet implemented).
- **Test a task with snapshots and parallel sessions.** `run_task.py --snapshot-dir`
  saves state after every command and at stages inside the paw/put_in primitives;
  `scripts/render_snapshots.py` renders them offscreen (four views) afterwards;
  `scripts/run_parallel_sessions.py` runs N live sessions side by side (one folder,
  one episode each -- they share the room's memory files) and writes a summary with
  the failure screenshots. Use `--speed-tier fast`; in Claude Code run one sub-agent
  per session that looks at the screenshots. See the README section.
- **The grader judges the END state, in the scene's words** (`dynamic_grader._drop_reclosed`,
  `outcome_checker._matches_with_swap`): a door the task opens and later closes is not
  expected open (only the task's own words count, not the appended scene facts);
  "X closed" becomes `NOT X open`; a bare "X moved" is met by "X in/on ..."; objects by
  scene name ("pastry" is the cupcake). Each of these failed a run that did everything.
- **A drop is not a placement.** At `--stringency normal|strict` the gripper's
  release is measured: an object let go more than 30 / 15 mm above the surface
  under it (not over a bin or container) is reported `<name> dropped from N mm`,
  and `check_outcome` fails it unless the task is to make something fall. Loose
  (the default) reports nothing new. A live loop "succeeded" three times by
  letting a cucumber fall 70 mm because the failure analyser said "lift z"; a
  refusal to set a held object DOWN now says the spot is occupied instead.

## Learning architecture (Phases 1/2/3 + grader fallback)

Three layers sit on top of the per-episode failure analyser to let the
system accumulate knowledge across episodes and sessions. Each is
non-fatal: API failure or malformed output is logged, and the session
result is returned unchanged.

- **Phase 1 — in-session plan pinning.** `EpisodeContext.successful_plans`
  records every successful plan and renders them into subsequent
  episodes' prompts as exploration-friendly references ("plans that
  succeeded, but the task likely admits cleaner solutions"). The reuse-
  rate metric in the end-of-session summary tells you whether the
  planner converged on the pinned shape or kept finding independent
  solutions. Destructive successes (`off bench`, `fell to floor`) are
  skipped to avoid pinning false positives.
- **Phase 2 — Opus session review.** When a session of 3+ episodes
  finishes, `agent/review_session.py` invokes Opus 4.7 on the full
  session and asks for *abstracted observations* (phrased as
  hypotheses, never as rules). Markdown writeup is appended to
  `reviews.md` at the project root; structured fields (false-positive
  flags, exploration diagnoses, cross-task observations) come back in
  a fenced JSON block.
- **Phase 3 — cross-task world model.** `world_model.md` accumulates
  *invariants* across sessions in four sections (geometric,
  object-class, primitive, grader). Each entry tracks a corroboration
  list; confidence = high (3+ sessions), medium (2), provisional (1).
  Phase 2's Opus call decides per-observation whether to merge into an
  existing entry or create a new one. The rendered world model gets
  injected into every future `LLMBrain` system prompt. A scene-hash
  banner fires when `envs/lab_scene.xml` has changed since entries were
  recorded.
- **Dynamic grader.** `outcome_checker.expected_outcome` only recognises
  three task templates. For anything else, `agent/dynamic_grader.py`
  makes one Haiku call at session start to produce a
  `(mode, expected_substrings)` spec restricted to the
  `physical_outcome()` vocabulary. The result is cached per session
  and consulted by `check_outcome` as a fallback. The same module
  hosts the `--speed-tier` inference (see below).

When extending any of these:
- New cross-task observation categories: edit `SECTIONS` in
  `agent/world_model.py` *and* the schema docs in
  `agent/review_session.py::SYSTEM_PROMPT` together; the index-based
  merge_with field assumes both sides agree on category names.
- New regex grader templates go in
  `agent/outcome_checker.py::expected_outcome` first; if the dynamic
  fallback keeps grading them, the regex stays out of the LLM call
  path. Faster *and* deterministic.

## Wrist camera (Intel RealSense D435i)

An eye-in-hand D435i lives in `xarm_lab_twin/perception/` — MuJoCo cameras in
the scene and the physical device behind one API, both returning the same
`RGBDFrame`. Full detail in
[`perception/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/perception/README.md).

```python
from perception import SimWristCamera        # or RealSenseWristCamera
frame = SimWristCamera(arm).capture()
xyz   = frame.pixel_to_world(320, 240)       # metres, base frame
```

- **`perception/d435i_calib.py` is the single source of truth** for every camera
  number. The scene XML is a build product: edit the calibration, then run
  `python -m perception.sync_scene` (rewrites the block in
  `lab_scene_primitive.xml` and reruns `build_mesh_scene.py`). Never hand-edit
  the camera block — this is defect class #1 waiting to happen, and
  `check_wrist_camera_matches_calib` in the sweep fails if either scene drifts.
- The camera block sits inside `<body name="gripper">` **on purpose**:
  `build_mesh_scene.py` copies that subtree verbatim, so one edit reaches both
  the primitive and the mesh scene with no second copy.
- Intrinsics are the physical device's factory values (serial `027422071693`);
  the hand-eye transform is UFACTORY's camera-stand calibration, lifted from
  `~/Models/ufactory_vision`. **If the real rig uses a non-UFACTORY bracket,
  that one constant needs re-measuring** — everything else stays valid.
- Two MuJoCo conversions in `d435i_calib.principal_pixel` were *measured*, not
  derived: `principalpixel="0 0"` centres on `((W-1)/2, (H-1)/2)`, and both axes
  are negated relative to image u/v. A wrong value here translates the image a
  few pixels while every self-consistent round-trip still closes — so
  `perception/test_projection.py` checks the renderer against MuJoCo's **ray
  caster**, with a negative control proving the check has teeth. Run it after
  touching anything optical: `MUJOCO_GL=egl python -m perception.test_projection`.
- `capture(align=True)` carries the **colour** intrinsics, because aligned depth
  has been reprojected into the colour frame. `RGBDFrame.intrinsics` always
  holds the right matrix so callers never pick.
- **A camera vantage is a FLANGE height.** The camera rides on the flange, but
  `set_position` takes the TCP, 214.5 mm further out. Write a look as a flange
  height and subtract `sim.tool.TCP_FROM_FLANGE_M`: looks written as TCP heights
  moved the camera 125 mm away when the tool became the real one, and GG-CNN
  stopped proposing grasps on 40 mm cubes (point-to-grasp, the GG-CNN and
  vision-dispatch tests, `vision_demo.py`, the planner's look guidance).
- Sim depth is exact and 100% valid; the real device runs ~71% valid with
  dropouts on transparent and specular surfaces. Validate perception on the
  device, not only on the twin.

## Grasp detection (GG-CNN)

`xarm_lab_twin/perception/grasp/` turns wrist-camera depth into ranked grasp
poses in base coordinates. Vendored from `~/Models/ufactory_vision` (BSD-3).
Detail in [`perception/grasp/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/perception/grasp/README.md).

```python
from perception.grasp import GGCNNDetector
grasps = GGCNNDetector().detect(SimWristCamera(arm).capture())
arm.set_position(*grasps[0].to_arm_pose())      # mm + degrees
```

- **Detection only — no servo loop.** Upstream's `RobotGrasp` streams poses in
  xArm servo mode and owns its own pick/place state machine; none of that
  transfers, because the twin drives the arm through its own validated
  primitives. This module stops at candidates.
- **Do not re-derive the camera transform.** Grasp points go through
  `RGBDFrame.pixel_to_world`, the path `perception/test_projection.py` already
  validates against MuJoCo's ray caster. A second euler chain would be defect
  class #1.
- Weights are `state_dict`s converted from upstream's pickles (which need
  `weights_only=False`, i.e. code execution, and an importable `models` package).
  `python -m perception.grasp.convert_weights --verify-only` proves they still
  match upstream bit-for-bit. `check_ggcnn_weights_load` in the sweep catches
  drift between `weights/*.pt` and the vendored `_ggcnn*_net.py`.
- **`MUJOCO_GL=egl python -m perception.grasp.test_ggcnn` is the test that
  matters** — it points the camera at cubes whose positions MuJoCo knows and
  requires the returned world coordinate to be that cube (currently 3.6–4.9 mm).
  "A grasp was produced" would pass for any pose, including a wrong one. Too
  slow for the sweep, so run it after touching anything optical or geometric.
- Trained on Cornell, depth-only, and it ranks *graspability*, not task
  relevance — with several objects in view the top candidate is often not the
  one you meant. Pair it with the object registry, or frame the target.
- Needs torch/opencv/scipy/scikit-image (installed in `xarm6sim`); imports are
  deferred so the sim runs without them.

## Language-conditioned targeting

`xarm_lab_twin/perception/language/` resolves a phrase to a grasp on the object
it names. Grounding DINO proposes regions, depth turns each into a measured
object, GG-CNN supplies the grasp. Detail in
[`perception/language/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/perception/language/README.md).

```python
from perception.language import LanguageTargeter
grasp = LanguageTargeter().grasp_for("the blue cube", frame, max_size_m=0.08)
```

- **The depth stage is load-bearing — do not remove it.** Asked for "a green
  cube", Grounding DINO returns the green **bin** as confidently as the cube,
  often scoring higher: a bin is a green box, and an image carries no scale.
  `max_size_m` resolves it by physical measurement. `check_size_filter_is_what_
  fixes_it` asserts both halves (unfiltered picks the bin, filtered picks the
  cube), so deleting the depth stage fails the suite rather than silently
  degrading targeting.
- The depth band centres on the region's **lower quartile** depth, not its
  median. For a small object the median is the bench, and a band around it would
  mask the bench instead of the object.
- `grasp_for` returns `None` rather than falling back to the region centroid
  when GG-CNN finds no grasp there. A centroid is a position, not a grasp — no
  jaw angle, no evidence the gripper can close. Use `target().position_world` if
  the position is what you want.
- **`MUJOCO_GL=egl python -m perception.language.test_targeting`** is the test
  that matters: phrases are checked against the scene's own body positions
  (currently 2.7–4.9 mm), and "a rubber duck" must return `None`. Too slow for
  the sweep, which only confirms the checkpoint is cached.
- `xarm6sim` runs `torch 2.13.0+cu126`, so `device="auto"` uses the RTX 3080:
  **0.37 s/call** against 4.90 s on CPU. Same detections either way (boxes agree
  to 3e-4 px, scores to 8e-4 — close but not bitwise, so do not assert exact
  equality across devices). CUDA costs ~3 s more to initialise, so build the
  targeter once rather than per call.
- No spatial language ("the cube behind the bin") — Grounding DINO grounds
  spatial relations poorly and nothing here adds any. `position_world` is in base
  coordinates, so relational filtering belongs above this layer.

## Vision in the planner's dispatch

Three actions in `LLMBrain._dispatch` let a plan name things by appearance
instead of by body name. `agent/vision_targeting.py` is the adapter; see its
docstring for the design constraint below.

| action | does |
|---|---|
| `locate_object` | measure where a described object is; prints it, registers the sighting |
| `move_to_object` | locate it, hover `dz_mm` above it |
| `grasp_object` | locate it, approach, descend on the GG-CNN grasp, close |

```json
{"action": "grasp_object", "params": {"description": "the blue cube"}}
```

- **They are self-contained on purpose — do not add late-bound refs.** The
  tempting design is `locate_object` binding a name and `move_to` taking
  `{"ref": "target"}`. It would break the pre-action gate in
  `scripts/validate_plan.py`, which checks *every* pose for reachability before
  *anything* is dispatched, precisely because a per-command check on real
  hardware comes after the prefix has already run. A pose that does not exist
  until dispatch cannot be pre-checked. Resolve-check-move inside one action
  keeps the property the gate protects: nothing moves before the pose is known.
- The gate knows these actions via `validate_plan.DEFERRED` and **says so out
  loud even when the plan passes** — an accepted plan containing unchecked poses
  must not print a clean bill of health. Adding a vision action means adding it
  to `DEFERRED` too; `check_vision_actions_wired` fails the sweep otherwise
  (without it the gate reports an unrecognised action and rejects the whole plan
  in `--mode real` only, which is the worst place to find out).
- **The planner does not see a sighting in the turn that produced it** — same
  single-shot limitation `get_pose` documents. Delivery happens through the
  registry: `VisionTargeting` writes each sighting back as a `seen:*` object, so
  it lands in the *next* turn's prompt. Break that write-back and
  `locate_object` becomes a `print()`, which is defect class #2 exactly.
- **Grounding DINO never returns nothing.** Ask for an object that is not
  present and it grounds to the backdrop — "a rubber duck" over this bench
  matched a 226 mm region of benchtop at score 0.44 and the arm grasped a cube
  anyway. Two guards catch it: `DEFAULT_MAX_AREA_FRAC` in the targeter (a
  graspable object is a small part of a wrist view; the bench is 30% of it) and
  `DEFAULT_MAX_SIZE_M` for phrases with no size word. Do not remove either
  because absent objects "obviously" return None — they do not.
- `grasp_object` refuses when the object was seen but GG-CNN proposed no grasp
  on it, rather than descending on the centroid. Seen is not graspable.
- **`MUJOCO_GL=egl python -m agent.test_vision_dispatch`** grades physically:
  which body the weld actually holds and how far it rose, not the return code.
  A confident grasp of the wrong cube returns 0 and looks fine in the log.

## SpaceMouse teleop

3Dconnexion SpaceMouse teleoperation of the twin lives in
`xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/teleop_sm/` and runs via
`scripts/run_spacemouse.py` — see [`teleop_sm/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/teleop_sm/README.md)
and [`docs/spacemouse_setup.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/docs/spacemouse_setup.md).
Sim-only, and **no LeRobot dependency** — it reuses the existing
`SimXArmAPI → IKSolver → ctrl → MuJoCo → Recorder` path, with `vr/`'s IK,
workspace clamp and smoother, so it runs under Python 3.11.

```bash
python scripts/run_spacemouse.py --device pro        # or --device compact
```

- **Install is three pieces, not two.** `spacenavd` is the daemon; **`libspnav0`
  / `libspnav-dev` is the client library the Python binding `dlopen()`s**, and
  installing the daemon alone fails with `libspnav.so: cannot open shared
  object file`. Then `pip install "spnav @ git+…"`.
- Unlike `run_vr.py` this launches the **interactive viewer** and must NOT set
  `MUJOCO_GL=egl` — the operator watches the monitor, and an EGL context would
  contend for the GL device.
- **A SpaceMouse is a velocity jog, not a pose tracker**, so there is no clutch.
  The receiver integrates deflection into a target pose it *owns*; re-reading
  the EE each tick would feed IK tracking error back into the command and the
  target would creep or run away.
- **Smoothing is an output filter, never part of the integrator state.** Writing
  the smoother's output back into the target makes the smoother's state *be*
  the target, so each tick advances by only `alpha*delta` and the jog silently
  runs at 30% of the requested speed. This was a real bug during development;
  `test_target_integrates_over_time` asserts distance over a full second, which
  is what catches it — a single-tick test would not.
- On IK failure the target is **frozen at the last feasible pose**, otherwise
  the operator integrates into unreachable space and the arm leaps when it
  becomes reachable again.
- 6 joints + rail = 7 DOF against 6 axes, so `MODE_CYCLE` switches between arm
  and rail rather than auto-allocating. Explicit beats clever: an operator can
  predict a mode, not a heuristic.
- **Two things are PROVISIONAL until measured on the device**: the axis
  permutation in `teleop_sm/config.py` (vendor's matrix, and spacenavd already
  applies `swap y-z invert y-z` before we see the axes) and the Pro's button
  indices (the daemon remaps them). `scripts/spacemouse_probe.py` settles both;
  `--dump` writes fixtures that `teleop_sm.device.events_from_dump()` replays.
- Tests need no hardware: `python -m teleop_sm.test_device`,
  `python -m teleop_sm.test_receiver`.

## Kinesthetic teaching (twin mouse-drag + real-arm manual mode)

`xarm_lab_twin/teleop_kin/` — see [`teleop_kin/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/teleop_kin/README.md)
and the operator guide [`docs/kinesthetic_teaching.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/docs/kinesthetic_teaching.md).

```bash
python scripts/run_kinesthetic_sim.py                       # drag the teach handle
python scripts/run_kinesthetic_real.py --ip <ip> [--robot lite6]
python scripts/run_kinesthetic_real.py --fake               # no robot
```

- **The teach handle is a mocap body in geom group 4** (`teach_handle`, in
  `lab_scene_primitive.xml`, propagated by `build_mesh_scene.py`). Group 4 is
  hidden by the default `MjvOption`, so recorded frames and the wrist camera
  never show it; `SimXArmAPI.set_viewer_geomgroup(4, True)` shows it in the
  interactive viewer only. `sim/render_options.ray_geomgroup` derives its mask
  from that same option — it used to cast against every group, so rays would
  have hit a ball the picture does not show.
- **The passive viewer moves mocap bodies itself** on Ctrl-drag
  (`mjv_applyPerturbPose(..., 0)` in every `sync()`), so the mocap pose IS the
  drag target. Verified on 3.8.1; `handle.drag_target()` is the one place to
  change if that ever stops being true.
- **`SimXArmAPI.register_key_handler(fn)`** gives a teleop first refusal on
  viewer keys (`fn(keycode) -> True` consumes). Empty by default, so the nudge
  keymap is unchanged. Handlers run on the viewer thread: queue, don't block.
- **The arm and tool are gravity-compensated** (`gravcomp="1"` on link1-6 and
  the gripper subtree, as the real controller compensates its payload); without
  it the bare PD arm sagged ~20 mm under the real tool and `set_position` landed
  short while returning 0. A **held object is not compensated**: teach mode
  cancels that with a per-joint integral bias gated on the joint being at rest;
  ctrl (the exported action) is the compensated setpoint. Don't "fix" the
  gating by integrating during motion — that winds up on tracking lag.
- **IK is solved with the rail at its SETPOINT** (`TwinServo.rail_at_setpoint`, `teleop_kin/servo.py`). The rail
  deflects a few mm while the arm moves; solving on the deflected value left the
  tool 1.5–3 mm off once it sprang back. SpaceMouse/VR still solve on the live
  rail.
- **`servo_direct(..., max_joint_step_rad=)`** (in `teleop_sm/receiver.py`,
  shared) rejects IK branch flips — a 293° jump was measured at the reach
  boundary. A *failed* solve costs ~215 ms under `arm.lock`; teach mode backs
  off retries so it cannot starve the physics thread. Any per-tick servo loop
  that retries IK every tick at a boundary has the same problem.
- **Real arm: every exit path runs `set_mode(0)` + `set_state(0)`** — key
  release, Ctrl-C, SIGTERM (turned into `SystemExit`), exceptions, controller
  errors. `test_sigterm_in_manual_mode_ends_in_mode_0` kills a real process,
  with SIGKILL as the negative control. Nothing in the runner commands motion.
- **Real takes take the gripper channel from the REAL gripper**
  (`Recorder(gripper_fn=...)`). The twin is only a mirror, so its weld state
  says nothing about a real object; the weld source would record 0 through
  every real grasp.
- Tests: `MUJOCO_GL=egl python -m teleop_kin.test_receiver` (twin) and
  `MUJOCO_GL=egl python -m teleop_kin.test_real` (fake controller).

## Hand-tracking teleop (dex-retargeting, IN PROGRESS)

`xarm_lab_twin/teleop_hand/` — bare-hand teleop of rail + arm + gripper from a
D435 facing the operator. Status, findings, settled parameters and the TO-DO
list are in [`teleop_hand/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/teleop_hand/README.md);
read it before continuing. Twin side works and was tested live (steps 1-5);
`run_handteleop_real.py` and `docs/hand_teleop.md` are NOT written yet.

- **Two processes, two envs**: tracker (`run_hand_tracker.py`, env
  `xarm6sim_hand` = xarm6sim + mediapipe) publishes `HandFrame` JSON on
  udp/47100; the teleop side (`run_handteleop_sim.py`) runs in `xarm6sim`.
- **mediapipe >= 0.10.30 has no `mp.solutions`**: `detector.py` is upstream's
  detector ported to the Tasks HandLandmarker. Handedness is NOT inverted on a
  D435 (upstream assumes a mirrored image).
- **`guards.solve_guarded` checks FK of every IK answer.** `IKSolver.solve`
  treats orientation as best-effort and can return a rotation-missed solution
  as a success -- the same trap applies to any caller passing `target_rot`.
- **Guards must let the arm move OUT of a band** (joint-limit margin, wrist
  singularity), or an arm parked at the edge freezes: IK jitter makes even the
  return move read as "inside". Found live.
- The twin servo core (gravity-sag bias, IK on the rail SETPOINT) is shared
  with kinesthetic teaching via `teleop_kin/servo.py::TwinServo`.
- Every real-arm exit goes through `teleop_kin.real_teach.return_to_position_mode`.
- Tests: `python -m teleop_hand.test_retarget`, `.test_palm`, `.test_mapping`,
  `MUJOCO_GL=egl python -m teleop_hand.test_twin`.
- **Gestures: fist-follow and point-to-grasp** (`--gestures` on tracker AND
  runner; explained in [`docs/dex_retargeting.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/docs/dex_retargeting.md)).
  Active only while the clutch is released. They are ABSOLUTE, so they need the
  tracker camera's pose in the world frame (`teleop_hand/camera_pose.py`);
  `nominal()` is a PLACEHOLDER and the runner warns while it is in use.
  `from_correspondences` reuses `perception.extrinsic_calibration.solve_camera_pose`
  -- do not add a second solver (defect class #1).
- **Pointing ray = index-MCP depth + landmark direction, never fingertip depth**
  (a fingertip's depth sample is the background). "Pointed at" is an 8 deg CONE
  tested before the ray cast. The grasp picks the GG-CNN candidate NEAREST the
  pointed object, not the highest quality: the first live run chose the
  heater-shaker's edge, 9 mm from the cube. It refuses (and does not descend on
  the prior) when GG-CNN proposes nothing near, or the grasp is nearer another object.
- **Calibrate the tracker camera with `scripts/calibrate_hand_camera.py`** (ArUco tag on the
  gripper, PnP camera-side, real joints through the twin's FK world-side; never commands the
  arm). Use a tag >= 100 mm: a 60 mm tag at ~1.2 m is ~28 px wide and biased by centimetres.
  `--fake` rehearses it; `python -m teleop_hand.test_hand_calib` (xarm6sim_hand) tests it.
  Lab layout: world +x is WEST (OT-2 at x=+0.85), so an operator east of the rail is at -x.
- Gesture tests: `python -m teleop_hand.test_gestures` (no hardware),
  `MUJOCO_GL=egl python -m teleop_hand.test_gestures_twin` (graded on which body
  is welded and how far it rose, not return codes).

## Rooms and SimReady assets (branch feat/rooms-simready)

`xarm_lab_twin/rooms/<id>/room.json` describes a scene: environment, robots,
objects. `twin_rooms/` validates and builds them; see
[`rooms/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/rooms/README.md)
and [`robots/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/robots/README.md).
Entry points take `--room <id>`; **without it they load exactly the scene file
they always did** (`lab_a2s2` is a passthrough room wrapping `envs/lab_scene.xml`).

- **Code uses LOGICAL names** (the lab's: `joint1`, `gripper`, `red_cube`). A
  built room prefixes them (`arm0/joint1`, `obj/red_cube`). In `SimXArmAPI`,
  look every name up through `self._mj(name, kind)`, and report names with
  `self._logical(...)`. A bare `self.model.body("gripper")` works in the lab
  and fails in every built room.
- **Use the per-room lists** `self.grippable`, `cube_bodies`, `bin_bodies`,
  `tube_bodies`, `rack_tube_groups`, not the module constants; for the lab they
  equal the constants (asserted in `twin_rooms.test_sim_rooms`).
- **The lab's system prompt is byte-identical to before rooms.** Its lab-only
  text lives in `rooms/lab_a2s2/agent_prompt.md`, spliced in by
  `agent/room_prompt.py`. Edit lab knowledge there, not in `llm_brain.py`.
- **Agent memory is per room** (`twin_rooms/memory.py`): the lab keeps
  `lessons.md`, `reviews.md`, `world_model.md` at the project root; other rooms
  use `memory/<id>/`.
- **CLIs:** `python -m twin_rooms list|new|build|validate|view|export-usd`;
  `python -m simready_import search|fetch|place|remove|list` (see
  [`simready_import/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/simready_import/README.md)).
  Imported assets live in `assets/simready/<slug>/` with provenance in `manifest.json`.
  Their origin is the bbox CENTRE, because the grasp gate measures the jaws
  against the body origin. Tall objects (the 231 mm bottle) cannot be grasped
  from above by the weld gripper.
- **Mesh objects:** `object_width_m` / `object_z_extent_m` bound a mesh geom by
  its vertices (`_geom_world_bounds`). The bounding sphere made a 52 mm cucumber
  read as 204 mm wide.
- **Imported visuals are geom group 1, never 2.** The lab hides group 2 (its
  backdrop) in the viewer, every offscreen renderer and `mj_ray`; in group 2 every
  imported object was invisible to the viewer, recorded frames and the wrist camera.
  `simready_import.test_importer` renders the demo room with the lab's own
  `scene_option()` and requires every object in the picture.
- **Articulated assets** (`fetch --articulated`): links with hinge/slide joints,
  CoACD on every link plus thin support plates under floors and shelves (CoACD
  left holes a dropped box fell through), handles from `--handle LINK=p0:p1[:normal]`
  as capsule sites (x = outward normal). The gripper holds a handle with a CONNECT
  (a point), so `SimXArmAPI.operate_part("incubator.Door", 0.7)` carries the grip
  point along the hinge arc or slide line. It is a planner action
  (`{"action": "operate_part", "params": {"part": "incubator.Door", "opening": 1.0}}`),
  offered only in rooms with handles (`room_prompt._generated_commands`, which
  lists the parts) and DEFERRED in the plan gate like the vision actions.
  Success is judged where the part ENDS, after the arm is home: `go_home` is
  not collision-checked, and from beside the opened incubator door it swept
  the door shut. In rooms with parts, go_home and the back-off route home by
  `_route_home`: the true home straight or folded first, else the same pose
  one turn round on joint 1, every leg sweep-checked (a reset therefore still
  ends at the true home); with no clear route the arm stays parked and says
  so. The approach prefers rail positions where the part's swing clears the
  arm's base (cabinet door_02 hit it at 61 deg), and each step re-commands
  the tool's residual error (the upper shelf stalled at 35 of 80 mm). Spring-return
  parts (the toaster lever) are not offered. Test: `MUJOCO_GL=egl python -m
  agent.test_operate_dispatch`.
- **Doors use the paw_approach** (user's design, `SimXArmAPI._paw`): the handle only
  cracks a door (`PAW_CRACK_DEG`); the closed gripper's BODY then sweeps it open from
  inside the gap, and closing never uses the handle (outer face, or the palm beside the
  handle bar). Held by the handle all the way, the arm followed the door's arc through
  contorted poses. The sweep is planned in full first, in PUSHES (the paw is set again
  between them, like a cat), as joint waypoints on one elbow branch, each pose checked
  on a scratch scene with the door patched to the angle it will have reached
  (`_solve_at_rail(patch=)`, `_scratch_validator`); only the gripper may touch that door
  (`FKValidator.contact_ok_bodies`) -- allowed for the whole arm, the forearm shoved a
  door shut. A door is opened only as far as the paw can push it shut again
  (`_paw_closable`: cabinet door_01 75 deg); the handle is the fallback. Geometry is
  measured, not assumed: panel axes by PCA, outward judged with the door turned SHUT,
  box geoms by their ORIENTED corners (`_geom_points`; a world AABB made a 66 mm door
  read 290 mm thick). Upright unwinding (`UPRIGHT_JOINTS_DEG`) keeps go_home off the
  one-turn-round home.
- **`put_in(object, container)`** is a planner action (offered for articulated objects
  with a vertical-hinge door; slide parts are its shelves): a horizontal side grasp
  (`close_lite6_gripper(side=True)`; tipped 15-30 deg for flat objects -- a 48 mm
  cucumber is lower than the 75 mm hand), turned only about the vertical so the object
  arrives standing; the shelf found from collision-geom columns (5-ray cross: wire
  shelves have 11 mm holes; collision geoms are alpha 0 so `mj_ray` skips them); every
  waypoint planned as a joint chain before moving (`_chain`), rail moves made tucked
  over the base facing the back of the bench, several releases kept for the one grasp.
  Refuses naming a closed door; re-takes an object the planner already picked from
  above; reports anything it knocked. Shelves 165 mm apart against a 214.5 mm tool are
  why it exists. Tests: `agent.test_operate_dispatch` (put_in cases),
  `python -m agent.test_grader_end_state`.
- **True scale, always** -- the robot is small next to real furniture, and the
  recordings train VLAs. Do not rescale assets for looks.
- **Recordings** carry `room_id` / `room_hash` (see Recording format), and
  replay rebuilds the room.
- Tests: `python -m twin_rooms.test_schema`, `.test_robot`;
  `MUJOCO_GL=egl python -m twin_rooms.test_builder`, `.test_sim_rooms`, `.test_prompt`,
  `.test_recordings`, `.test_lab_rebuilt`, `.test_entry_points`;
  `MUJOCO_GL=egl python -m simready_import.test_importer [--network]`.

## Hiwonder MaxArm (real hardware, no twin)

`xarm_lab_twin/hardware/maxarm/`: a second, unrelated arm (ESP32, 3 bus servos,
suction cup, no rail), driven over USB through the board's MicroPython raw REPL.
Detail and open items in [`hardware/maxarm/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/hardware/maxarm/README.md).

- **The PC-side port in `kinematics.py` is the one source of truth for motion.**
  Moves send bus-servo pulses directly. Do not route them through the vendor's
  `ESPMax.set_position`: it re-solves IK on the board and silently clamps z,
  servo 2, servo 3 and short reaches (defect classes #1 and #2 in a single
  call). The board's solver is only a cross-check (`maxarm_probe.py --kinematics`).
- The arm frame is the board's MicroPython `ESPMax` frame. **Trust the board over
  the published source**: `_espmax.cpp` negates x, the board's compiled
  `__espmax.mpy` does not, so on the lab unit `ESPMax` equals the C-source frame.
  The first port assumed otherwise and was mirrored; `--kinematics` against the
  real board caught it. **+x is the arm's right** (seen from behind, reach along
  -y), so the arm frame is LEFT-HANDED: negate x before any rotation maths (the
  camera hand-eye chain). The table floor and the arrival tolerance are still
  PROVISIONAL until measured on the device.
- **Opening the port reboots the board** (CH340 at `/dev/ttyUSB0`): Linux raises
  DTR/RTS on open before pyserial's deasserted values apply. The stock program
  boots in ~3.2 s with a silent gap, so `RawRepl.enter` waits for its `>>> `
  prompt; keys sent earlier are lost. The port needs the `dialout` group.
- **The stock program is not stopped by Ctrl-C.** `main.py` runs its Bluetooth /
  mouse / key handlers from `Timer(2)`, which would move the arm under ours;
  `RealMaxArmAPI` deinits it on connect and refuses unless the board reports it
  stopped. Leave Timer 3 alone (PWMServo, the nozzle).
- Tests: `python -m hardware.maxarm.test_maxarm`. The IK is graded against
  vectors from the vendor's compiled `_espmax.cpp` and against values read off
  the board (`test_frame_matches_the_board`), not against itself.

## Pose geometry helpers

`xarm_lab_twin/geometry/` holds SE(3) interpolation, smoothing and dwell
detection vendored from InternRobotics/Aether (MIT) — see
[`geometry/NOTICE.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/geometry/NOTICE.md)
for provenance and the two upstream bugs fixed on the way in. Pure NumPy/SciPy;
**no torch** (upstream's module imports torch/einops/plyfile at the top, which
is why these are vendored rather than depended on).

Deliberately not wired into anything yet — importable and tested. Intended
consumers are listed in `geometry/README.md`.

- `interpolate_poses(p1, p2, weight)`: **`weight` is p1's share**, so 1.0 → p1
  and 0.0 → p2 — the opposite of the `t` convention in `slerp` beside it.
- Quaternions are SciPy's `(x,y,z,w)`, **not** MuJoCo's `(w,x,y,z)`.

## VR teleop

Meta Quest 3 teleoperation of the digital twin lives in
`xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/` and runs via
`scripts/run_vr.py` — see [`vr/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/README.md)
for setup, the WebXR secure-context requirement, Quest pairing, and the
control map. **To connect a headset, prefer `adb reverse tcp:8443 tcp:8443` +
`http://localhost:8443`** — no cert, no browser flags; a bare self-signed cert
actively *disables* WebXR. It is **sim-only** and depends on nothing from GR00T: it reuses
the existing `SimXArmAPI → IKSolver → ctrl → MuJoCo → Recorder` path, with a
human hand (Touch controllers, streamed over WebXR) as the EE-target source
instead of the LLM.

- Run with `render=False` — the headset is the viewer, so the GLFW passive
  viewer is **not** launched (it would contend with the EGL offscreen
  renderer for GL). Export `MUJOCO_GL=egl` (run_vr.py does this before
  importing mujoco).
- Two display modes: `--mode mono` (single flat panel, also viewable in a
  browser tab) and `--mode stereo` (per-eye cameras `cam_left`/`cam_right` on
  the `vr_head` mocap body added to `envs/lab_scene.xml`, head-tracked).
- Two servo paths: `--servo direct` (IK→ctrl per tick, smooth, bypasses the
  validator — fine in sim) and `--servo validated` (routes through
  `set_position`). All tunables are in `vr/config.py`.
- The **A** button records standard `Recorder` takes, so VR demos land as
  ordinary `recordings/` sessions and replay through `replay.py` unchanged.
- All `arm.data`/`arm.model` access in `vr/` holds `arm.lock`. Tests:
  `python -m vr.test_transforms` and `MUJOCO_GL=egl python -m vr.smoke_test`.

## Speed caps and motion pacing

`--speed-tier {crazy_fast,fast,medium,slow,very_slow,auto}` on every LLM
entry point pins the session ceiling (`auto` or omit = Haiku reads cues
from the task prompt). Per-command tier downgrades via
`{"speed_tier": "<tier>"}` in any motion command are clamped to the
session ceiling. Caps in mm/s: crazy_fast=None (uncapped), fast=120,
medium=80, slow=40, very_slow=15.

The cap is enforced **twice**:
1. `LLMBrain._clamp_speed` clamps the LLM-emitted `speed_mm_s` before
   passing it to the sim.
2. `SimXArmAPI._execute_paced_arm` / `_execute_paced_rail` interpolate
   actuator targets at ~50 Hz over `distance / speed` wall-clock
   seconds. MuJoCo position actuators have no built-in velocity limit,
   so without this pacing layer the cap would only be cosmetic.

`push_object` takes its own `speed_mm_s` kwarg that's threaded through
all its internal `set_position` / `set_rail_position` calls. The
dispatch resolves the effective cap via
`LLMBrain._effective_speed_mm_s(per_cmd_tier)`.

## Recording format

Every script that touches the sim writes one folder per session under
`recordings/<timestamp>_session_<id>/` containing:

- `metadata.json` — task label, model used, outcome, augmentation config
- `commands.jsonl` — sparse action log (one JSON object per line)
- `trajectory.h5` — 60 Hz state: rail/joints/EE/body poses/weld states/gripper,
  plus 10 Hz image frames in a `/frames` group (`--save-frames`, **on by
  default since 2026-09-16**)
- `llm_session.jsonl` — LLM prompt + response + dispatch trail (LLM runs only)

**`/frames` layout** — `/frames/t_wall` is shared, and each camera gets its own
subgroup: `/frames/<camera>/images`, uint8 `(N, H, W, 3)`. Which cameras are
recorded is set by `Recorder(frame_cameras=...)`, defaulting to
`("cam_wrist_color",)` — the view a policy will actually have at inference.
Pass `frame_cameras=(None,)` for the old free-orbit third-person camera, which
is good for a human reviewing a session and useless as a policy observation.

**Rooms (additive, 2026-10-29)** — `metadata.json` carries `room_id`,
`room_hash` (sha256[:12] of the room definition: room.json, the files it
names, its robot types), `room_variant` and `robots` (instance, type,
end_effector, controllable); `trajectory.h5` root attrs add `room_id` and
`room_hash`. Names in `body_names` / `eq_names` are LOGICAL (`red_cube`, not
`obj/red_cube`). A Recorder given no room finds it from `scene_xml`, so lab
sessions record `lab_a2s2`. `replay.py` rebuilds the recorded room
(`twin_rooms.recordings.replay_room`), warns if its hash changed, and treats
a session with no `room_id` as the project's default room. The lab's dataset
names and layout are unchanged.

**`gripper`** — one float32 per state sample, `1.0` when something is welded to
the gripper (or, when the Recorder is given `gripper_fn`, whatever that returns —
`run_kinesthetic_real.py` passes the real gripper's state, since the mirrored
twin's welds cannot see a real object). It is a STATE ("is it holding"), not a COMMAND ("was it told to
close"), because the twin has no actuated fingers (see the magnetic-gripper
hack below). The two differ on any failed grasp. On real hardware the honest
source is jaw position, which the Recorder cannot currently see — it is handed
`model`/`data`, not an arm.

**Frame rendering was broken from its introduction until 2026-09-16** and
nobody noticed, because the failure was caught, printed, and then the session
was written without a `/frames` group while still reporting success — defect
class #2. The `Renderer` owns an EGL context and EGL contexts are
thread-affine; it was built in `Recorder.__init__` (caller's thread) and used
from the sampler thread, so every render raised `EGL_BAD_ACCESS`. All 954
recordings made before that date have no images at all. It is now built lazily
inside `_ensure_renderer()`, on the sampler thread, and a construction failure
disables frames loudly and once.

The format is designed for VLA training data export. When adding new
quantities to the trajectory, update both `recording.py::_sample_one()` and
`recording.py::_write_trajectory()`, and document the new fields in this
file's "Recording format" section so consumers can find them.

## Safety conventions

- **Never commit secrets.** `.gitignore` excludes `.env`, `*.key`,
  `secrets/`, and `How-to-run.txt` (which historically had pasted API keys).
  Before any `git add .`, scan for `sk-ant-` to be safe.
- **Two repos; identify them by URL, not by remote name.** Remote names
  differ between clones, so run `git remote -v` before any push.
  - **Private dev repo** `github.com/kouroshSA/xarm6-digital-twin-v5_dev`:
    the working repo. `main` tracks it, and bare `git push` goes there.
  - **Public repo** `github.com/kouroshSA/xarm6-digital-twin-v5`: a
    *filtered mirror*. `Claude-Session:` links and `Co-Authored-By: …
    <noreply@anthropic.com>` trailers are stripped, so its hashes differ from
    dev's. Never push to it directly; that would upload the unfiltered history.
  - Clone layouts seen so far:
    - `dev` = private, `origin` = public.
    - `origin` = private, `public-v5` = public, with its push URL set to
      `DISABLED_…` so a direct push fails.
  - Publish only with `tools/sync_public.sh` (dry run) then
    `tools/sync_public.sh --push`. The filter is deterministic, so routine
    syncs are fast-forwards; the script refuses a rewrite unless given
    `--force`. **The script hardcodes `SRC_REMOTE=dev` / `DST_REMOTE=origin`**:
    run it only in a clone with the first layout. In the second layout it
    stops at `git fetch dev`, because `dev` does not exist.
- **The tool is the cell's:** F/T sensor + camera adapter (42.5 mm, derived from
  the measured 217 mm flange-to-tip) + UFACTORY xArm Gripper (TCP 172 mm), so the
  TCP is 214.5 mm from the flange and the payload 1.247 kg (`CELL_TCP_LOAD`).
  One source: `sim/tool.py`; the end_effector site in `lab_scene_primitive.xml`
  is a hand-edited copy. The fingers are the gripper's real four-bar, cosmetic,
  driven in radians; a grasp closes them to the object's width.
- **Magnetic-gripper hack.** This sim doesn't have actuated gripper fingers.
  `gripper_close` activates a MuJoCo `<weld>` constraint between the
  gripper body and the nearest cube/tube/bin/rack. If you're adding a new
  graspable body to the LAB scene, also add it to `GRIPPABLE_BODIES` in
  `sim/mujoco_env.py` and the lab's `rooms/lab_a2s2/room.json`, plus an
  `<equality>` entry in the scene XML. (`FKValidator`'s held-geom table is
  derived from the welds now -- the hand-written one had drifted.) In a
  built room the weld is generated for every `grippable` object; see
  `rooms/README.md`.
- **IK fallback.** `pink` IK is preferred but its API doesn't match
  MuJoCo's types in pink≥4.2, so the iterative Jacobian solver in
  `sim/ik_solver.py` is what actually runs. The one-time warning at startup
  is expected; don't silence it.

## The two standing defect classes

Nearly every real bug found in this repo has been one of two shapes. Neither
is a mistake in arithmetic, and neither shows up as a wrong-looking log line
-- which is exactly why they survive.

### 1. One fact, several copies, and the copies drift

The same measurement stored in more than one place. The copies start equal and
diverge silently, because nothing compares them.

Found so far: the joint limits in **four** places with three different values
(`arm_backend`, `build_mesh_scene.py`, both scene XMLs); the cup's dimensions
in three; the home pose in two, with the twin homing into the PCR module while
the cell homed somewhere safe; the SDK rail-API claim contradicted in
`real_arm.py`'s docstring after `docs/` had already corrected it.

**Rule:** when you add anything to the scene, add a sweep check that ties the
code's view of it back to the scene XML. Prefer deriving one copy from the
other -- `build_mesh_scene.py` now imports `XARM6_JOINT_LIMITS_DEG` rather
than repeating it, which removes the copy instead of re-synchronising it.

### 2. A value computed correctly and delivered to nobody

The component does the right maths, produces the right answer, and hands it to
something that is not the consumer -- a `print()`, a discarded variable, a
return value nobody reads. Every unit passes its own test. The system fails.

Found so far, all in one week:

| The value | Where it went instead |
|---|---|
| `grasp at z=807` | `print()`; the planner never saw it and guessed 795 |
| IK error, measured correctly | written into live `qpos` and never restored, so every move became instant |
| "nothing in reach" | printed, then `return 0` (success) |
| `return 2` for a refusal | discarded by a hardcoded `os._exit(0)` |
| `X on Y` outcome vocabulary | never added to the grader's list, so a toppled block scored as success |
| placement height 867 | computed, but only emitted for unambiguously-resolved referents |
| "no reachable grasp pose" | assigned to `_w` and thrown away |
| *why* a move was refused | printed; the caller received a bare `rc=2` |

**The trap is that the logs make it look right.** The console prints the
correct number, so reading the output you conclude the system knows it. It
does know. It never told anyone.

**Rule:** test the CONSUMER's behaviour, not the producer's output. "Does the
placement fact exist" passes for any number, including a wrong one. "Does
releasing at the reported height produce a stack, and does 20 mm lower fail"
exercises producer, delivery and consumer together -- and would have caught
six of the eight above. When you add a value for another component, trace it
to that component and assert it arrives; a `print()` is not delivery.

## Things that often go wrong

- **Bin/tube push** uses a "fly-over + weld" pattern (see
  `sim/mujoco_env.py::push_object`) rather than the cube grasp+drag, because
  the gripper geometry can't cleanly grasp tall objects with position-only
  IK. Don't try to unify the two paths.
- **Off-bench targets** are snapped to z=800 mm (mid-air past the edge) so
  gravity drops the object visibly. On-bench targets snap to the body's
  bottom-on-bench z. Per-object-type snap z in `push_object`.
- **Compound prompts** (e.g. "put all three cubes in all three bins") often
  overrun Haiku. Escalate to Sonnet via `--model sonnet`. The `--loop` flag
  helps: even if the first plan is bad, the loop retries with constraints.
- **Speed is only paced because of explicit interpolation.** MuJoCo
  position actuators have no built-in velocity limit -- writing a
  ctrl target sends the joint there at full PD authority. If you add
  a new motion primitive on `SimXArmAPI`, you must pace it via
  `_execute_paced_arm` / `_execute_paced_rail` (or call an existing
  primitive that already does), otherwise `--speed-tier slow` will be
  cosmetic for your new path. The bug existed for the first two weeks
  after `--speed-tier` shipped: `set_position`, `set_rail_position`,
  and `set_servo_angle` all accepted a `speed` kwarg they then ignored,
  so the dispatch clamp looked correct in logs while motion ran at
  full speed. See commit `2e87fe4` for the fix.
- **Adding a new task to the grader.** Two layers: (1) the regex
  grader in `agent/outcome_checker.py::expected_outcome` is fastest
  and deterministic; (2) the Haiku fallback in
  `agent/dynamic_grader.py::infer_criteria` only fires when the regex
  returns None. If you add a regex pattern, make sure the existing
  Haiku call wouldn't have produced the same answer -- otherwise
  you're paying tokens for nothing.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **xarm6-digital-twin-v5_dev** (2843 symbols, 5764 relationships, 244 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact analysis before editing.** Use `impact({target: "symbolName", direction: "upstream"})` (MCP) or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .` (CLI fallback); report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/xarm6-digital-twin-v5_dev/context` | Codebase overview, check index freshness |
| `gitnexus://repo/xarm6-digital-twin-v5_dev/clusters` | All functional areas |
| `gitnexus://repo/xarm6-digital-twin-v5_dev/processes` | All execution flows |
| `gitnexus://repo/xarm6-digital-twin-v5_dev/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
