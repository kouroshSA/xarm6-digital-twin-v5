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
