# xArm6 + Rail Digital Twin (LLM v5) — Setup & Quick Start

A Claude-driven MuJoCo digital twin of a UFACTORY xArm6 on a 700mm linear rail,
with a cubes-and-bins scene, two Falcon-tube racks, magnetic-gripper grasping,
multi-episode auto-play, random-play data generation, and cross-session lesson
memory. Spec doc: `../xarm6_rail_digital_twin_llm_v5.md` (companion to this
folder; explains the design in depth).

This file is the install + quick-start. Claude Code or a human can follow it
top-to-bottom on a fresh machine and get a working sim with one task executed
end-to-end.

---

## Prerequisites (all platforms)

- **Python 3.10 or newer** (3.11 recommended).
- **Conda** (Miniconda or Anaconda) for environment isolation.
- An **Anthropic API key** if you want the LLM-driven scripts. Get one at
  https://console.anthropic.com/settings/keys. The sim works without a key,
  but `scripts/run_task.py` / `auto_play.py` will not.
- ~500 MB of disk for the conda env + dependencies.
- A display (X11 / WSLg / native macOS) if you want to *see* the viewer. The
  scripts also run headless with the right env vars.

---

## Platform setup

### Ubuntu / Debian Linux

```bash
# 1) System packages (Tkinter for the optional Tk panel, OSMesa for headless
#    offscreen rendering when you ask for image frames).
sudo apt-get update
sudo apt-get install -y python3-tk libosmesa6 libosmesa6-dev libgl1 libglu1-mesa

# 2) Conda (skip if you already have it)
wget -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash   # then open a new shell, or `source ~/.bashrc`
```

**Display notes:** the live MuJoCo viewer needs X11. On Wayland (default in
Ubuntu 22.04+) most things work, but the viewer is happier on X11. If you see
GUI weirdness, log out and pick "Ubuntu on Xorg" at login. For *headless*
frame rendering (`--save-frames` without a display), prepend
`MUJOCO_GL=osmesa` or `MUJOCO_GL=egl` to the run command.

### Windows (WSL2)

The project runs inside WSL2 Ubuntu, not native Windows.

```powershell
# 1) Enable WSL2 + install Ubuntu (PowerShell as admin, one-time)
wsl --install -d Ubuntu-22.04

# 2) Open the Ubuntu shell from the Start menu, then follow the
#    Linux instructions above (apt-get install ..., miniconda, etc.)
```

**Display notes:**
- **Windows 11**: WSLg is built in — the MuJoCo viewer window opens automatically
  on your Windows desktop, no extra setup. Just run the scripts as you would on
  native Linux.
- **Windows 10**: WSLg is not available. Install an X server on Windows
  (VcXsrv or X410), set `export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0`
  inside WSL, and start the X server with "Disable access control" before
  launching the script.

### macOS

```bash
# 1) Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2) Conda via Miniconda
brew install --cask miniconda
conda init zsh   # then open a new terminal

# 3) Tkinter ships with the Python from conda; no extra packages needed
#    for the basic flow.
```

**Display notes:** the native macOS viewer opens fine — no X server needed.
For headless frame rendering, the default `MUJOCO_GL` (CGL) works; no env-var
override needed.

**One macOS-specific permission**: if you also use the manual `realtime_keyboard.py`
flow (basic v5 — not the LLM v5 default), `pynput` needs Accessibility permission:
*System Settings → Privacy & Security → Accessibility → enable Terminal / iTerm2 /
the app you launch the script from*. The LLM scripts don't need this.

---

## Step 1 — Get the code

```bash
# Put the project anywhere; xarm_lab_twin/ is the working directory.
cd ~/path/to/xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
```

This `xarm_lab_twin/` directory is the *project root* for every command in
this document. `cd` here before running anything.

## Step 2 — Create the conda environment

```bash
conda create -n xarm6sim python=3.11 -y
conda activate xarm6sim
```

## Step 3 — Install Python dependencies

```bash
# Required
pip install mujoco numpy transforms3d h5py anthropic

# Recommended (faster IK; falls back to iterative Jacobian automatically if missing)
pip install pin pin-pink

# Optional — only needed for the basic-v5 manual keyboard mode (not used by
# the LLM v5 scripts in this folder)
pip install pynput

# Optional — only if --mode real (real hardware) or scan_to_mesh
pip install xarm-python-sdk open3d
```

## Step 4 — Set up the API key (only for LLM-driven scripts)

The repo includes a `.env` loader. Put your key in `.env` at the project root:

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=sk-ant-REPLACE-ME
EOF
chmod 600 .env             # restrict to your user

# Sanity check (value is masked; only the prefix and length are printed)
python env_loader.py
```

Expected output:
```
[env_loader] 1 variable(s) declared in .env:
  ANTHROPIC_API_KEY = sk-ant-a...XXXX  (len=108)
```

`.env` is gitignored so it won't end up in version control.

## Step 5 — Verify the install

```bash
# (a) Scene loads
python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('envs/lab_scene.xml')
print(f'scene OK: nbody={m.nbody} ngeom={m.ngeom} neq={m.neq}')
"
# Expected: scene OK: nbody=26 ngeom=60 neq=9

# (b) Live viewer (close the window to continue)
python -c "
import mujoco, mujoco.viewer
mujoco.viewer.launch(mujoco.MjModel.from_xml_path('envs/lab_scene.xml'))
"
```

You should see the arm in a bent "ready" pose at mid-rail, three RGB cubes
and matching bins on the bench, and two 4×2 tube racks (one on each side)
with 6 Falcon tubes (3 orange caps, 3 blue caps) distributed across them.

---

## Quick start — five commands

All commands assume you are inside `xarm_lab_twin/` with the `xarm6sim` conda
env active and a valid API key in `.env`.

```bash
# 1) Single LLM-driven task with the viewer. Wait ~2s for "Response in X.Xs"
#    before assuming the arm is stuck -- Claude needs to plan first.
python scripts/run_task.py \
    "Put the red cube in the red bin" \
    --model haiku

# 2) Tube task: pick from one rack, place in an empty slot of the other rack
python scripts/run_task.py \
    "Pick up tube_L1 from the left rack and place it in an open slot of the right rack" \
    --model haiku

# 3) Auto-play -- Claude generates N diverse tasks and executes them back-to-back
python scripts/auto_play.py --episodes 5 --model haiku --save-all

# 4) Random-play -- no LLM in the loop, random reachable poses for VLA data
python scripts/random_play.py --episodes 10 --moves-per-episode 8 --save-all

# 5) Replay a saved session (lists sessions if no arg)
python replay.py             # list
python replay.py 0           # replay session #0
python replay.py 0 --plan-only   # show Claude's plan + dispatch log without viewer
```

### Adding image frames for VLA training data

State + cube/tube world poses + weld activations are always recorded.
Image frames are off by default (they add ~10-15 MB per minute of recording).
Add `--save-frames` to any script and set the GL backend if you're on Linux
and the live viewer is also running:

```bash
MUJOCO_GL=egl python scripts/auto_play.py --episodes 5 --save-frames
```

The frames land in `recordings/<session>/trajectory.h5` under the `/frames`
group (`images` at uint8 320×240×3, `t_wall` per frame).

### Capping motion speed (safety override)

By default (when `--speed-tier` is omitted, or set to `auto`) the loop
infers a session-level speed cap from the task prompt: Haiku reads cues
like "quickly" / "carefully" / "fragile" and picks one of `crazy_fast`
/ `fast` / `medium` / `slow` / `very_slow`. When no cue is present the
inference falls back to `medium` = 80 mm/s. The LLM may also downgrade
individual commands by attaching `"speed_tier": "<tier>"` to any
`move_to` / `set_rail` / `set_joints` it emits, but per-command tiers
can never exceed the session ceiling.

Pass `--speed-tier <name>` to ANY of the LLM entry points
(`run_task.py`, `auto_play.py`, `run_task_augmented.py`) to override
the Haiku inference and pin the session ceiling deterministically:

```bash
# Force every motion to very_slow (15 mm/s) regardless of prompt phrasing
python scripts/run_task.py \
    "pick up the blue-cap tube" \
    --loop --max-episodes 10 --speed-tier very_slow

# Pin the ceiling to fast (120 mm/s) on a prompt with no speed cue,
# overriding the medium default
python scripts/run_task.py \
    "transfer tube_L2 to the right rack" \
    --speed-tier fast

# Auto-play a batch of mixed tasks, all bounded to slow (40 mm/s)
python scripts/auto_play.py --episodes 20 --speed-tier slow

# Explicit opt-in to Haiku inference (same as omitting the flag, but
# makes the intent visible in saved commands and scripts)
python scripts/run_task.py \
    "carefully transfer the blue-cap tube" \
    --speed-tier auto
```

There's a complementary `--led` flag on the same entry points: pass it
to enable two 700 mm rainbow LED strips beside the rail. When the rail
moves, the rainbow flows in the direction of motion at a rate matching
the active speed tier (off by default; flow rates: crazy_fast=5 Hz,
fast=3, medium=2, slow=1, very_slow=0.5; dim warm-white standby when
the rail is idle but `--led` is on).

Tiers and caps: `crazy_fast` = uncapped, `fast` = 120, `medium` = 80,
`slow` = 40, `very_slow` = 15 mm/s, plus `auto` = use Haiku inference.
A named-tier override skips the Haiku call entirely (small token-cost
saver). Per-command tier downgrades within the LLM plan continue to
work and are clamped to the CLI-set ceiling.

The cap is enforced two ways: (1) the LLMBrain dispatch clamps every
`speed_mm_s` value the planner emits before passing it to the sim;
(2) `SimXArmAPI.set_position` / `set_rail_position` / `set_servo_angle`
actually *pace* the motion by interpolating actuator targets at ~50 Hz
over `distance / speed` wall-clock seconds. So a 200 mm move at
`speed=40` takes about 5 seconds of real time, not the milliseconds it
would take if you slammed the ctrl in one shot. `push_object` honors
the same cap on all of its internal sub-moves. The pacing is what makes
"slow" actually look slow on screen; without it the cap would just be
a number in the log.

### Learning architecture (Phase 1 / 2 / 3)

Beyond the per-episode failure analyser that the `--loop` flag has had
from the start, three additional layers let the system accumulate
knowledge across episodes and sessions.

**Phase 1 — In-session positive-signal reinforcement.** When an
episode succeeds, its full command plan is recorded in
`EpisodeContext.successful_plans` and rendered into the prompt for
subsequent episodes of the same session, framed as "plans that
satisfied the grader, but the task likely admits shorter or cleaner
solutions" so the planner is invited to improve rather than locked
into the first working plan. A simple reuse-rate metric tells you at
the end of the run whether the planner converged on the pinned shape
or kept finding independent solutions. Defensive: physically
destructive successes ("off bench", "fell to floor") are skipped to
avoid pinning false positives.

**Phase 2 — End-of-session Opus review.** After all N episodes finish
(if `N >= 3`), the loop invokes Claude Opus 4.7 once on the full
session and asks for *abstracted observations* phrased as hypotheses,
not rules. Output is appended to `reviews.md` at the project root,
tagged by timestamp + task, with structured fields for false-positive
flags and exploration diagnoses. The review is non-fatal: API failure
or malformed output is logged and the session result is returned
unchanged.

**Phase 3 — Cross-task world model.** `world_model.md` accumulates
*invariants* across sessions in four sections (geometric, object-class,
primitive, grader). Each entry tracks its corroboration count; on read,
confidence = high (3+ sessions), medium (2), provisional (1). The
Opus reviewer (Phase 2) reads existing entries and decides per
new-observation whether to merge into one (incrementing corroboration)
or create a fresh entry. The world model gets injected into every
future LLMBrain system prompt, so each new task starts with the
system's accumulated knowledge. A scene-hash banner fires when
`envs/lab_scene.xml` has changed since entries were recorded.

**Dynamic grader.** For task phrasings the regex grader doesn't
recognise (anything outside push-off / placement / sort templates), the
loop calls Haiku once at session start (`agent/dynamic_grader.py`) to
produce a `(mode, expected_substrings)` spec that plugs into
`check_outcome` as a fallback. Cached for the session; non-fatal on
API failure (falls back to "ungraded" as before). Same module hosts
the speed-tier inference behind `--speed-tier`.

**Body-aware failure analyser.** `analyse_command_failure` in
`agent/episode_loop.py` now cross-references failed `move_to`
coordinates against the registry: when a validation failure happens
inside a known body's collision envelope, the emitted constraint
names the body and recommends the empirically-working approach /
grasp heights from `_WORKING_HEIGHTS`. For `place_tube_in_rack` IK
failures, it also names the rack's `optimal_rail_mm`.

**Extended physical_outcome vocabulary.** `SimXArmAPI.physical_outcome`
now snapshots body positions at scene reset and emits two new fact
shapes on top of the four categorical events (`in <bin>`,
`in <rack>`, `off bench`, `fell to floor`):

- `<X> moved (Δx, Δy)mm` — per-object displacement >= 20 mm
- `<a> closer to <b>` / `<a> farther from <b>` — inter-object xy
  distance changed by >= 20 mm; pair members alphabetical for
  deterministic substring matching.

This vocabulary is what lets tasks like "push the green bin closer to
the blue bin" be physically gradable; otherwise the grader has no
substring to match against.

### Inspecting a recording

```python
import h5py
with h5py.File('recordings/<session>/trajectory.h5', 'r') as f:
    t       = f['t_wall'][:]           # (N,) wall-clock seconds since start
    joints  = f['joints_deg'][:]       # (N, 6) joint angles
    ee_pos  = f['ee_pos_mm'][:]        # (N, 3) end-effector position
    bodies  = f['body_poses'][:]       # (N, 9, 7) cubes+tubes xyz + quat
    welds   = f['weld_active'][:]      # (N, 9) per-cube/tube grip state
    if 'frames' in f:
        images = f['frames/images'][:] # (M, 240, 320, 3) uint8
```

The recording format and downstream usage is documented in detail in the
spec doc: `../xarm6_rail_digital_twin_llm_v5.md`, "File Format Reference".

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[ik_solver] pink init failed ...` (one-time warning at start) | Harmless. The iterative Jacobian fallback runs and works fine. |
| `qpsolvers UserWarning: no QP solver found` | Cosmetic. Optional: `pip install qpsolvers[open_source_solvers]`. |
| Viewer window opens, arm doesn't move for ~3s | That's Claude planning. Wait for `[LLMBrain] Response in X.Xs ...` |
| `Segmentation fault (core dumped)` *after* `[Recorder] OK Saved` | Benign — GLFW/threading teardown after data is persisted. |
| `[Recorder] Frame renderer init failed (...)` with `--save-frames` | Prepend `MUJOCO_GL=egl` (Linux with viewer) or `MUJOCO_GL=osmesa` (pure headless) to the command. |
| WSL viewer doesn't open | On Win10, install VcXsrv + set `$DISPLAY`; on Win11, WSLg should be automatic — try `sudo apt install -y mesa-utils && glxinfo \| head` to verify GL is available. |
| `BadAccess (GLX)` from the renderer | Same as above — switch GL backend. |

For deeper diagnostic patterns (LLM not responding, IK collisions, tube
release dynamics, etc.) see the spec doc's "Honest Caveats" section and the
inline comments in `sim/mujoco_env.py` and `sim/ik_solver.py`.

---

## Project layout

```
xarm_lab_twin/
├── .env                    # API key (gitignored)
├── env_loader.py           # parses .env into os.environ
├── lessons.md              # one-line outcome log, auto-appended per episode
├── reviews.md              # Opus session-end abstracted writeups (Phase 2)
├── world_model.md          # cross-task invariants accumulated over sessions (Phase 3)
├── envs/
│   ├── lab_scene.xml       # MuJoCo scene (arm + rail + cubes/bins + tubes/racks + OT-2 [11 colored slots + walls] + 2x 96-well plates + tip box + heater-shaker + Vortex-Genie 2 + LED strips)
│   └── scene_randomizer.py # object-pose jitter for augmented runs
├── sim/
│   ├── mujoco_env.py       # SimXArmAPI -- physical_outcome + paced motion + push macros
│   ├── fk_validator.py     # FK + collision validation
│   └── ik_solver.py        # pink IK + iterative Jacobian fallback
├── agent/
│   ├── llm_brain.py        # Claude planner loop; speed-cap dispatch
│   ├── object_registry.py  # cube/tube/bin/rack metadata + grip configs
│   ├── prompt_variants.py  # diverse-task and paraphrase generators
│   ├── outcome_checker.py  # regex grader (push-off / placement / sort)
│   ├── dynamic_grader.py   # Haiku fallback grader + speed-tier inference
│   ├── episode_loop.py     # EpisodeRetry, in-session plan pinning, body-aware analyser
│   ├── review_session.py   # Phase 2 Opus session review pass
│   ├── world_model.py      # Phase 3 cross-task invariant store
│   └── lessons.py          # lessons.md + reviews.md read/write
├── hardware/
│   └── real_arm.py         # RealXArmAPI wrapper (--mode real)
├── scripts/
│   ├── run_task.py         # single task (+ --loop for EpisodeRetry)
│   ├── run_task_augmented.py  # variants + scene jitter
│   ├── auto_play.py        # multi-episode LLM-generated tasks
│   ├── random_play.py      # multi-episode random-pose sampler
│   └── scan_to_mesh.py     # optional: point cloud -> OBJ for custom scenes
├── recording.py            # Recorder + LLMSessionLog
├── replay.py               # playback + soft-delete + trash
└── recordings/             # auto-created per session
    └── <timestamp>_session_<id>/
        ├── metadata.json
        ├── commands.jsonl
        ├── trajectory.h5    # state + body poses + welds + (optional) frames
        └── llm_session.jsonl (LLM runs only)
```
