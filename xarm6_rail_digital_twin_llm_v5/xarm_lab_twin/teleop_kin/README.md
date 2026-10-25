# `teleop_kin/` — kinesthetic teaching

Collect demonstrations by moving the end effector by hand, in two places, both
producing ordinary `Recorder` sessions that `scripts/export_lerobot.py` reads
unchanged:

| | how the arm is moved | runner | `interface` |
|---|---|---|---|
| **Twin** | drag a teach handle with the mouse; the arm follows through IK | `scripts/run_kinesthetic_sim.py` | `kinesthetic_sim` |
| **Real xArm 6 / Lite 6** | push the arm by hand in the controller's manual mode; the twin mirrors it | `scripts/run_kinesthetic_real.py` | `kinesthetic_real` |

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
python scripts/run_kinesthetic_sim.py                       # twin
python scripts/run_kinesthetic_real.py --ip 192.168.1.xxx   # real arm
python scripts/run_kinesthetic_real.py --fake               # rehearse, no robot
```

Operator directions (safety, payload, keys): [`docs/kinesthetic_teaching.md`](../docs/kinesthetic_teaching.md).

Tests, no hardware:

```bash
MUJOCO_GL=egl python -m teleop_kin.test_receiver   # twin: 9 tests, ~35 s
MUJOCO_GL=egl python -m teleop_kin.test_real       # real path with a fake controller: 9 tests
```

## Phase 0 findings (MuJoCo 3.8.1, settled before Phase 1)

**The drag target is the mocap pose itself (spec "source 1").** The passive
viewer's `Simulate::Sync` ends every sync with
`mjv_applyPerturbPose(m, d, &pert, 0)` — "mocap bodies only" — so while the
operator Ctrl-drags a selected mocap body, the viewer writes
`perturb.refpos/refquat` into `data.mocap_pos/quat` itself. Confirmed on the
live viewer: with the handle selected and a translate perturbation active,
`mocap_pos` moved 97.7 mm of a 100 mm `refpos` offset with no code of ours
writing it. `handle.drag_target()` is still the single entry point, so if a
future MuJoCo stops doing this only that function changes.

**None of the teach keys has a built-in binding in the passive viewer.** From
`simulate.cc`: Left/Right step physics *only when not passive*
(`!is_passive_ && ...`); Up, Down, Insert, Delete, Home and End are not bound
at all. (PageUp *is* bound — "select parent body" — which is why it is not a
teach key; the pre-existing nudge keymap already shares it.)

## How the twin teach mode works

```
handle (mocap, group 4) ──drag_target──▶ clamp (vr.transforms) ──▶ rate limit ──▶ IK ──▶ ctrl = q_cmd + sag bias
          ▲                                                                       │
          └──── snap onto the COMMANDED pose on release / after home, reset, rail ┘
```

- **Rate-limited command.** The handle is absolute, but the arm chases it at
  ≤ 150 mm/s and 90°/s (`config.MAX_POS_SPEED/MAX_ROT_SPEED`), so a mouse
  flick never teleports the arm and the recorded actions stay smooth.
- **Red handle = the arm cannot follow**: IK failed or the workspace clamp
  moved the target. ctrl holds; letting go snaps the handle back onto the arm.
- **Hidden from data.** The handle's geoms are in group 4, which MuJoCo's
  default `MjvOption` hides; only the interactive viewer turns it on
  (`SimXArmAPI.set_viewer_geomgroup`). `sim/render_options.ray_geomgroup` now
  derives its mask from the same option, so ray casts agree with the picture.
- **Keys.** `SimXArmAPI.register_key_handler` gives a teleop first refusal on
  viewer keys; with nothing registered the nudge keymap is unchanged. The
  handler only *queues* actions — home, reset and the rail step block for a
  paced move, and blocking in the viewer's callback would freeze the window.

## Things measured while building it

Each of these was found by a test failing, and each is now guarded.

1. **The twin sags ~15–20 mm under gravity** — bare PD position actuators, no
   gravity compensation (joint 2 settles 1.2° below its setpoint at home). IK
   is exact; the arm just doesn't get there. Every interface inherits this
   (SpaceMouse, VR, `set_position`). Teach mode adds a slow per-joint integral
   bias (`GRAVITY_COMP_*`), integrated only while a joint is near its setpoint
   **and nearly at rest** — integrating during a drag wound it up to its clamp
   on tracking lag. `test_1b` is the negative control: bias off, 21 mm.
2. **The rail deflects a few mm while the arm moves** and springs back. IK
   solving on the deflected rail left the tool 1.5–3 mm off once it recovered.
   The receiver solves and computes FK with the rail at its **setpoint**
   (`TwinServo.rail_at_setpoint`, `teleop_kin/servo.py`). Tool-to-handle went from ~2 mm to 0.07 mm.
3. **IK flips branch at the reach boundary** — measured a 293° jump on one
   joint, which would whip the arm round. `servo_direct(max_joint_step_rad=)`
   now rejects any solution that moves a joint > 5° in one solve
   (opt-in; SpaceMouse behaviour is unchanged).
4. **A failed IK solve costs ~215 ms, all under `arm.lock`.** Retrying every
   tick at the boundary starved the physics thread (the arm stalled 10–30°
   short of its own setpoint). After a failure the receiver retries only when
   the handle has moved, at most every 0.25 s.
5. **The scene's start pose (all zeros) is a wrist singularity** — any drag
   there trips the branch guard. The runner goes home first (`--no-home` to
   skip).
6. **The shared workspace box excludes the front-row cubes.**
   `vr.config.WORKSPACE_AABB_MM` has `y >= -200 mm`; `red_cube_front` and
   `blue_cube` sit at `y = -250`. Not changed here — the box is shared with
   SpaceMouse and VR — but the handle turns red there for that reason.

## Real arm: what the mirror records

Joint angles from `get_servo_angle()` go into the twin's `ctrl` **and**
`qpos` each tick, so action and state are both the real arm's. The rail is
read at 1 Hz (the carriage cannot be back-driven by hand). The gripper channel
comes from the **real** gripper via `Recorder(gripper_fn=...)`: the twin's weld
cannot know about a real object, so the default weld source would record 0
through every real grasp. It flips when the controller has accepted the
command and the jaw has finished moving (the call runs on a worker thread so
the 60 Hz mirror never stalls).

**Lite 6:** the twin is an xArm 6. Joint data is recorded faithfully; the
twin's EE and body poses are not the Lite 6's. Each take logs the robot model
as its first command.
