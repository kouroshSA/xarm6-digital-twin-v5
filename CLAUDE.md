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

## VR teleop

Meta Quest 3 teleoperation of the digital twin lives in
`xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/` and runs via
`scripts/run_vr.py` — see [`vr/README.md`](xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/README.md)
for setup, the HTTPS/secure-context requirement, Quest pairing, and the
control map. It is **sim-only** and depends on nothing from GR00T: it reuses
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
- `trajectory.h5` — 60 Hz state: rail/joints/EE/body poses/weld states, plus
  optional 10 Hz image frames in a `/frames` group (`--save-frames`)
- `llm_session.jsonl` — LLM prompt + response + dispatch trail (LLM runs only)

The format is designed for VLA training data export. When adding new
quantities to the trajectory, update both `recording.py::_sample_one()` and
`recording.py::_write_trajectory()`, and document the new fields in this
file's "Recording format" section so consumers can find them.

## Safety conventions

- **Never commit secrets.** `.gitignore` excludes `.env`, `*.key`,
  `secrets/`, and `How-to-run.txt` (which historically had pasted API keys).
  Before any `git add .`, scan for `sk-ant-` to be safe.
- **Magnetic-gripper hack.** This sim doesn't have actuated gripper fingers.
  `gripper_close` activates a MuJoCo `<weld>` constraint between the
  gripper body and the nearest cube/tube/bin/rack. If you're adding a new
  graspable body, also add it to `GRIPPABLE_BODIES` in `sim/mujoco_env.py`
  and `HELD_CUBE_GEOMS` in `sim/fk_validator.py`, plus an `<equality>` entry
  in the scene XML.
- **IK fallback.** `pink` IK is preferred but its API doesn't match
  MuJoCo's types in pink≥4.2, so the iterative Jacobian solver in
  `sim/ik_solver.py` is what actually runs. The one-time warning at startup
  is expected; don't silence it.

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
