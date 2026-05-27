# Episode Learning Loop — Implementation Guide
## For: `xarm6_rail_digital_twin_llm_v5/xarm_lab_twin`

This guide is tailored to your actual repo as it exists today. It adds an
**episode learning loop** that retries failed tasks, learning from each
attempt, while reusing your existing infrastructure (`lessons.md`,
`physical_outcome()`, `Recorder`, `reset_scene()`).

---

## What You Already Have (Don't Reinvent)

Your repo already has 80% of what's needed for a learning loop:

| Existing piece | What it does | How the loop uses it |
|---|---|---|
| `agent/lessons.py` | Appends one-line lessons after each run, capped at 20 entries, most-recent first. Injected into the system prompt. | **Use as-is.** Loop will write one lesson per episode. |
| `sim/mujoco_env.py::physical_outcome()` | Returns ground-truth state (e.g., `"red_cube in red_bin"`, `"blue_cube fell to floor"`). | **This is the real success signal.** Don't trust `result == 0` alone. |
| `sim/mujoco_env.py::reset_scene()` | Resets cubes/tubes/arm to canonical start. | **Call between episodes.** Lets the loop start fresh each try. |
| `agent/llm_brain.py::SYSTEM_PROMPT_TEMPLATE` | Already has `{lessons_section}` placeholder. | **Inject learned constraints here** by extending `read_lessons_section()` or adding a parallel `read_session_constraints_section()`. |
| `scripts/auto_play.py` | Multi-episode pattern: one viewer, per-episode `Recorder`, `reset_scene` between, `append_lesson` after. | **Copy this structure** for the loop. |
| `recording.py::LLMSessionLog` | Logs the LLM prompt, response, dispatched commands, and results per session. | **Reuse for each episode** — gives you a debuggable trail. |

**What's missing:** the `execute → evaluate → analyze → refine → retry` loop. That's what we add.

---

## What "Success" Really Means in This Repo

Critical insight: **`result == 0` from `_dispatch()` does NOT mean the task succeeded.** It only means the command was accepted/executed. The cube might still have:
- Fallen off the bench (`physical_outcome` → `"red_cube fell to floor"`)
- Landed in the wrong bin (`"red_cube in blue_bin"`)
- Not been displaced at all (`"no objects displaced"`)

The loop must check **two things** to decide if an episode succeeded:

1. **No command-level failures** (no `result != 0` on non-advisory actions like `move_to`, `set_rail`, `gripper_*`)
2. **Physical outcome matches intent** — the cube ended up where the task said it should

For #2, you have two options:
- **Manual:** Have the user grade success/failure (like `auto_play.py --eval`)
- **Automatic:** Parse the task prompt for the target (e.g., "red cube → red bin") and check if `physical_outcome()` contains `"red_cube in red_bin"`. **We'll implement this** for the loop.

---

## File Changes Summary

| Action | File | Why |
|---|---|---|
| **CREATE** | `agent/episode_loop.py` | The new loop logic: retry, analyze, refine. |
| **CREATE** | `agent/outcome_checker.py` | Parses task prompts to extract expected outcomes; compares to `physical_outcome()`. |
| **MODIFY** | `scripts/run_task.py` | Add `--loop` and `--max-episodes` flags; wire in `EpisodeRetry`. |
| **MODIFY** | `agent/llm_brain.py` | Add an **optional** `extra_context` parameter to `execute_task()` so the loop can inject learned constraints. (Minimal change, fully backward-compatible.) |
| **MODIFY (optional)** | `agent/lessons.py` | Add a sibling `append_episode_lesson()` that records loop episodes with their constraint context. Or just reuse the existing `append_lesson()`. |
| **CREATE (optional)** | `CLAUDE.md` at repo root | Tells Claude Code (when invoked in this directory) about the loop convention. |

No existing functionality is broken. The loop is opt-in via `--loop`.

---

## Implementation

### 1. `agent/outcome_checker.py` (NEW)

Parses the natural-language task to figure out what success looks like, then
compares to `physical_outcome()`.

```python
# agent/outcome_checker.py
"""
Outcome checker: parse a task prompt to determine the expected physical
state, then compare to physical_outcome() to decide if the episode succeeded.

This is best-effort, not exhaustive. It handles the most common task patterns:
  - "put/place/move <color> cube in <color> bin"   -> expect "<color>_cube in <color>_bin"
  - "push <object> off the bench/edge/table"        -> expect "<object> fell to floor" OR "off bench"
  - "put all cubes in <color> bin"                  -> expect all three "*_cube in <color>_bin"
  - "sort the cubes [by color]"                     -> expect each cube in matching bin

For tasks the parser doesn't recognise, returns None (success unknown -- the
loop should fall back to command-level success, or ask the user).
"""
import re
from typing import Optional, List, Tuple

COLORS = ("red", "green", "blue")


def expected_outcome(task: str) -> Optional[List[str]]:
    """
    Return a list of substrings that ALL must appear in physical_outcome()
    for the task to count as a success. Returns None if the parser can't
    determine an expectation (caller should fall back to command-level check).
    """
    t = task.lower()

    # "push X off the bench/edge/table/floor" -> X should fall or be off bench
    if re.search(r"\b(push|knock|drop|shove|slide).*\b(off|edge|floor|away)\b", t):
        # Find the target object
        for color in COLORS:
            if f"{color} cube" in t or f"{color}_cube" in t:
                return [f"{color}_cube fell to floor", f"{color}_cube off bench"]  # either is OK
        # Generic "clear the table" / "everything off"
        if re.search(r"\b(all|everything|clear)\b", t):
            return ["fell to floor"]  # at least something fell
        return None

    # "put/place all cubes in <color> bin" -> all three cubes in that bin
    if re.search(r"\b(all|every|each).*\bcubes?\b", t):
        for color in COLORS:
            if f"{color} bin" in t:
                return [f"{c}_cube in {color}_bin" for c in COLORS]
        return None

    # "sort the cubes [by color]" -> each in its matching bin
    if re.search(r"\bsort.*cubes?\b", t):
        return [f"{c}_cube in {c}_bin" for c in COLORS]

    # "put/place/move <color> cube in <color> bin" (canonical pick-and-place)
    m = re.search(r"\b(red|green|blue)\s*(cube|block)\b.*\b(red|green|blue)\b.*\b(bin|container|box)\b", t)
    if m:
        src = m.group(1)
        dst = m.group(3)
        return [f"{src}_cube in {dst}_bin"]

    return None  # Unknown task pattern


def check_outcome(task: str, physical: str) -> Tuple[Optional[bool], str]:
    """
    Compare expected outcome (from task) to actual physical_outcome() string.

    Returns:
        (success, reason)
        success: True/False/None (None = couldn't determine)
        reason:  human-readable explanation
    """
    expected = expected_outcome(task)
    if expected is None:
        return (None, f"Task pattern not recognised; physical: {physical or 'no displacement'}")

    # For "push off" tasks, ANY of the substrings counts
    if any("fell to floor" in e or "off bench" in e for e in expected):
        if any(e in physical for e in expected):
            return (True, f"Met physical condition: {physical}")
        return (False, f"Expected one of {expected}; got: {physical or 'no displacement'}")

    # For placement tasks, ALL expected substrings must appear
    missing = [e for e in expected if e not in physical]
    if not missing:
        return (True, f"All expected placements present: {physical}")
    return (False, f"Missing: {missing}; got: {physical or 'no displacement'}")
```

### 2. `agent/episode_loop.py` (NEW)

The actual loop. Reuses your `Recorder`, `LLMBrain`, `lessons`, and `reset_scene()`.

```python
# agent/episode_loop.py
"""
Episode learning loop.

Runs a task across up to N episodes. Between episodes, analyses failures and
injects learned constraints into the next attempt. Stops on success or on
reaching max_episodes.

Designed as a thin wrapper around LLMBrain -- does NOT modify llm_brain.py's
core logic. Each episode:
  1. arm.reset_scene()
  2. start a fresh Recorder
  3. brain.execute_task(task, extra_context=<learned constraints>)
  4. wait for physics to settle
  5. arm.physical_outcome() + check_outcome(task, ...) -> success?
  6. if failed: analyse the failure, append a constraint, append a lesson, retry
  7. if succeeded: append a "SUCCESS after N episodes" lesson, stop
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from agent.outcome_checker import check_outcome
from agent.lessons import append_lesson
from recording import Recorder


@dataclass
class EpisodeContext:
    """Accumulates learned constraints across episodes within one task run."""
    task: str
    max_episodes: int = 10
    episode_num: int = 1
    learned_constraints: List[str] = field(default_factory=list)
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    final_result: Optional[Dict[str, Any]] = None
    final_physical: str = ""

    def add_constraint(self, constraint: str) -> None:
        if constraint and constraint not in self.learned_constraints:
            self.learned_constraints.append(constraint)
            print(f"[EpisodeLoop] Learned: {constraint}")

    def constraints_block(self) -> str:
        """Render constraints for injection into the next prompt."""
        if not self.learned_constraints:
            return ""
        lines = ["", "## Constraints learned in earlier episodes of this task",
                 "(These come from real failures observed just now. Respect them.)"]
        for c in self.learned_constraints:
            lines.append(f"- {c}")
        return "\n".join(lines) + "\n"


def analyse_command_failure(failed_step: Dict[str, Any],
                            planned_command: Dict[str, Any]) -> str:
    """
    Given a failed step from results[], infer a constraint string.

    Failed step format: {"action": str, "result": int}
    Planned command:    {"action": str, "params": {...}}

    Returns a short, actionable constraint string for the next episode's prompt.
    """
    action = failed_step.get("action", "unknown")
    code = failed_step.get("result", -1)
    params = planned_command.get("params", {}) if planned_command else {}

    if action == "move_to":
        x, y, z = params.get("x"), params.get("y"), params.get("z")
        roll, pitch, yaw = params.get("roll"), params.get("pitch"), params.get("yaw")
        if code == 1:
            return (f"move_to ({x}, {y}, {z}) mm with roll/pitch/yaw "
                    f"({roll}/{pitch}/{yaw})deg is unreachable (IK failed). "
                    f"Try a different approach pose -- raise z, change xy, or "
                    f"re-orient the wrist.")
        if code == 2:
            new_z = z + 50 if isinstance(z, (int, float)) else "higher"
            return (f"move_to ({x}, {y}, {z}) mm fails collision/FK validation. "
                    f"Lift z to >= {new_z} mm, or change rail position so the "
                    f"arm approaches from a less obstructed side.")

    if action == "set_rail":
        pos = params.get("position_mm")
        return (f"set_rail to {pos} mm failed (code {code}). Choose a rail "
                f"position closer to optimal_rail_mm for the target object.")

    if action == "set_joints":
        angles = params.get("angles_deg")
        return (f"set_joints {angles} deg failed (code {code}). Check joint "
                f"limits; the xArm6 joints are restricted (~+/-360 for J1, "
                f"~+/-118 for J2, etc.). Prefer move_to over raw joint angles.")

    if action == "push_object":
        tgt = params.get("target_name")
        return (f"push_object on '{tgt}' to "
                f"({params.get('to_x_mm')}, {params.get('to_y_mm')}) failed "
                f"(code {code}). The arm couldn't follow the drag path -- try "
                f"a closer destination or push from a different side.")

    if action == "place_tube_in_rack":
        return (f"place_tube_in_rack {params.get('rack_name')} failed "
                f"(code {code}). Either no tube is held, no empty slot, or "
                f"the slot is unreachable. Make sure a tube is grasped first.")

    if action == "gripper_close":
        return ("gripper_close did not grasp the intended object. The EE site "
                "must be within ~70 mm of the object centre when closing. "
                "Lower z further or re-centre xy before closing.")

    return f"Action '{action}' failed with code {code}. Avoid this exact step."


def analyse_physical_failure(task: str, physical: str,
                             reason: str) -> Optional[str]:
    """
    Commands all returned 0, but the physical outcome doesn't match the task.
    Infer a constraint from the discrepancy.
    """
    if not physical or physical == "no objects displaced":
        return (f"All commands succeeded but nothing moved. The grasp likely "
                f"failed: gripper_close probably ran with the EE too far from "
                f"the object. Re-check the cube position from the registry, "
                f"set rail closer, and approach with z=795 mm (just above "
                f"cube top at 780 mm).")

    # Wrong bin / fell to floor / off bench
    if "fell to floor" in physical:
        return (f"An object fell to the floor when it shouldn't have. Either "
                f"the release height was too high, the lift cleared the bin "
                f"opening, or the trajectory took it past the bench edge. "
                f"Release at z=830 mm directly above the bin centre.")

    if "off bench" in physical and "push" not in task.lower():
        return (f"An object ended up off the bench but the task wasn't a push. "
                f"Your transit path went past bench edge. Keep xy inside "
                f"[-750..+750, -450..+450] mm during transit.")

    # Wrong-bin placement
    if "_cube in" in physical and reason and "Missing" in reason:
        return (f"Cube ended up in the WRONG bin ({physical}). Re-read the "
                f"registry for the target bin's xy; you placed at the wrong "
                f"destination.")

    return None


class EpisodeRetry:
    """Wraps LLMBrain with episode-by-episode retry + constraint learning."""

    def __init__(self, brain, arm, registry, recorder_factory,
                 max_episodes: int = 10,
                 settle_seconds: float = 1.5):
        """
        Args:
            brain:            LLMBrain instance (already constructed).
            arm:              SimXArmAPI (needs reset_scene() and physical_outcome()).
            registry:         ObjectRegistry (unchanged across episodes).
            recorder_factory: callable() -> Recorder, called fresh per episode,
                              or None to skip per-episode recording.
            max_episodes:     Cap on retries.
            settle_seconds:   How long to wait after the last command before
                              calling physical_outcome().
        """
        self.brain = brain
        self.arm = arm
        self.registry = registry
        self.recorder_factory = recorder_factory
        self.max_episodes = max_episodes
        self.settle_seconds = settle_seconds

    def run(self, task: str) -> Dict[str, Any]:
        ctx = EpisodeContext(task=task, max_episodes=self.max_episodes)

        while ctx.episode_num <= ctx.max_episodes:
            print(f"\n{'=' * 70}")
            print(f"[EpisodeLoop] Episode {ctx.episode_num}/{ctx.max_episodes}: {task}")
            if ctx.learned_constraints:
                print(f"[EpisodeLoop] Carrying {len(ctx.learned_constraints)} "
                      f"learned constraint(s) into this attempt")
            print('=' * 70)

            # 1. Reset the scene and (re-)attach a fresh recorder.
            self.arm.reset_scene()
            time.sleep(0.5)

            recorder = None
            if self.recorder_factory is not None:
                recorder = self.recorder_factory()
                recorder.start()
                # The brain captures the recorder it was constructed with, so
                # if we want per-episode recording we have to swap it in:
                self.brain.recorder = recorder

            # 2. Build the prompt: original task + learned constraints block.
            #    (See "Hooking the constraints into the prompt" below for the
            #    minimal llm_brain.py change that makes this work.)
            episode_task = task + ctx.constraints_block()

            # 3. Execute.
            try:
                result = self.brain.execute_task(episode_task, dry_run=False)
            except Exception as e:
                print(f"[EpisodeLoop] execute_task raised: {e}")
                result = {"commands": [], "results": [], "error": True}
            ctx.final_result = result

            # 4. Wait for physics to settle and inspect.
            time.sleep(self.settle_seconds)
            physical = self.arm.physical_outcome() if hasattr(
                self.arm, "physical_outcome") else ""
            ctx.final_physical = physical
            print(f"[EpisodeLoop] Physical outcome: {physical or '(none)'}")

            # 5. Did any command fail at the dispatch level?
            cmd_failure = _first_command_failure(result.get("results", []))
            if cmd_failure is not None:
                idx, failed = cmd_failure
                planned = result.get("commands", [])[idx] if idx < len(
                    result.get("commands", [])) else {}
                constraint = analyse_command_failure(failed, planned)
                ctx.add_constraint(constraint)
                ctx.failure_history.append({
                    "episode": ctx.episode_num, "kind": "command",
                    "step": idx, "action": failed.get("action"),
                    "code": failed.get("result"), "constraint": constraint,
                })
                _record_lesson(task, self.brain, result, physical, ctx)
                _close_recorder(recorder)
                ctx.episode_num += 1
                continue

            # 6. All commands returned 0. Did the physical state match the task?
            success, reason = check_outcome(task, physical)
            print(f"[EpisodeLoop] Outcome check: success={success} ({reason})")

            if success is True or (success is None and physical):
                # Treat "unknown but something happened" as soft success --
                # surfaces it for the user instead of looping forever on tasks
                # the parser can't grade.
                ctx.success = bool(success) or False
                if success is True:
                    print(f"[EpisodeLoop] SUCCESS on episode {ctx.episode_num}")
                    ctx.success = True
                else:
                    print(f"[EpisodeLoop] STOPPING (outcome unclear -- check manually)")
                _record_lesson(task, self.brain, result, physical, ctx)
                _close_recorder(recorder)
                break

            if success is False:
                # Commands ran but physical state is wrong. Infer a constraint.
                constraint = analyse_physical_failure(task, physical, reason)
                if constraint:
                    ctx.add_constraint(constraint)
                ctx.failure_history.append({
                    "episode": ctx.episode_num, "kind": "physical",
                    "physical": physical, "reason": reason,
                    "constraint": constraint,
                })

            _record_lesson(task, self.brain, result, physical, ctx)
            _close_recorder(recorder)
            ctx.episode_num += 1

        # End of loop. Print summary.
        print(f"\n{'=' * 70}")
        print(f"[EpisodeLoop] Done after {ctx.episode_num - 1} episode(s). "
              f"success={ctx.success}")
        if ctx.learned_constraints:
            print(f"[EpisodeLoop] Learned constraints ({len(ctx.learned_constraints)}):")
            for c in ctx.learned_constraints:
                print(f"  - {c}")
        print('=' * 70)

        return {
            "success": ctx.success,
            "episodes_run": ctx.episode_num - 1,
            "final_result": ctx.final_result,
            "final_physical": ctx.final_physical,
            "learned_constraints": ctx.learned_constraints,
            "failure_history": ctx.failure_history,
        }


def _first_command_failure(results: List[Dict[str, Any]]
                          ) -> Optional[Tuple[int, Dict[str, Any]]]:
    """Return (index, result_dict) of the first hard failure, or None."""
    advisory = {"done", "wait", "get_pose", "search_workspace"}
    for i, r in enumerate(results):
        if r.get("result", 0) != 0 and r.get("action") not in advisory:
            return (i, r)
    return None


def _record_lesson(task: str, brain, result: Dict[str, Any],
                   physical: str, ctx: EpisodeContext) -> None:
    """Append a one-liner to lessons.md for this episode."""
    label = f"{task} [ep {ctx.episode_num}/{ctx.max_episodes}]"
    append_lesson(
        task_prompt=label,
        model_short=brain.model_short,
        planned_commands=result.get("commands", []),
        results=result.get("results", []),
        physical_outcome=physical,
    )


def _close_recorder(recorder) -> None:
    """Stop the recorder without prompting the user (loop mode is non-interactive)."""
    if recorder is None or not recorder.is_recording:
        return
    # Replicate auto_play.stop_recorder_silent: write trajectory + metadata
    # without asking for input.
    from datetime import datetime
    recorder._recording = False
    if recorder._state_thread is not None:
        recorder._state_thread.join(timeout=1.0)
    recorder._session.ended_at_iso = datetime.now().isoformat()
    recorder._session.duration_s = time.time() - recorder._start_wall_time
    recorder._session.n_state_samples = len(recorder._state_buffer)
    if recorder._commands_file is not None:
        recorder._commands_file.close()
        recorder._commands_file = None
    recorder._write_trajectory()
    recorder._session.kept = True
    recorder._write_metadata()
```

### 3. `agent/llm_brain.py` — Minimal Change

To inject learned constraints between episodes, the loop appends them onto
the task prompt string itself (see `EpisodeContext.constraints_block()` and
the call `episode_task = task + ctx.constraints_block()` in `EpisodeRetry.run()`).

**This requires no change to `llm_brain.py`.** The constraints just ride
along in the user message. Claude reads them as part of the task.

If you want the constraints in the **system prompt** instead (cleaner
separation, doesn't drift through the conversation history), add an optional
parameter to `execute_task()`:

```python
# In llm_brain.py, modify execute_task signature:
def execute_task(self, task_prompt: str, dry_run: bool = False,
                 extra_system_context: str = "") -> dict:
    ...
    system = SYSTEM_PROMPT_TEMPLATE.format(
        registry_context=self.registry.to_llm_context(),
        lessons_section=lessons if lessons else "(no prior lessons yet)",
    )
    if extra_system_context:
        system = system + "\n\n" + extra_system_context
    ...
```

Then in `episode_loop.py`, pass it via `execute_task(task, extra_system_context=ctx.constraints_block())`
and stop appending to the task string. **Either approach works.** The
user-message approach (default above) requires zero changes to `llm_brain.py`.

### 4. `scripts/run_task.py` — Add `--loop`

Two small changes:

**A. Add the flags:**

```python
# After the existing parser.add_argument(...) lines:
parser.add_argument("--loop", action="store_true",
                    help="Enable episode learning loop: retry on failure, "
                         "learning constraints between attempts.")
parser.add_argument("--max-episodes", type=int, default=10,
                    help="Max episodes for --loop (default: 10).")
```

**B. Replace the single `brain.execute_task(...)` call with a branch:**

```python
# Replace this existing block:
#     try:
#         result = brain.execute_task(args.task, dry_run=args.dry_run)
#     except Exception as e:
#         print(f"[System] Task failed: {e}")
#         result = {"commands": [], "results": [], "error": True}
#
# With this:

if args.loop and not args.dry_run:
    from agent.episode_loop import EpisodeRetry

    # Build a factory so each episode gets its own Recorder. (The outer
    # `recorder` from setup is stopped here; the loop manages its own.)
    if recorder is not None:
        recorder.stop_and_prompt(prompt=False)
        recorder = None

    def _recorder_factory():
        return Recorder(
            model=arm.model, data=arm.data, lock=arm.lock,
            interface="episode_loop", scene_xml="envs/lab_scene.xml",
            enable_frames=args.save_frames,
        ) if hasattr(arm, "model") else None

    loop = EpisodeRetry(
        brain=brain, arm=arm, registry=registry,
        recorder_factory=_recorder_factory,
        max_episodes=args.max_episodes,
    )
    summary = loop.run(args.task)
    # Surface the final episode's commands/results to the existing reporting code
    result = summary["final_result"] or {"commands": [], "results": []}
    physical = summary["final_physical"]
    # Skip the post-task append_lesson() below: the loop already appended
    # per-episode lessons. We'll signal this with a flag.
    _loop_handled_lessons = True
else:
    try:
        result = brain.execute_task(args.task, dry_run=args.dry_run)
    except Exception as e:
        print(f"[System] Task failed: {e}")
        result = {"commands": [], "results": [], "error": True}
    _loop_handled_lessons = False
```

**C. Skip the duplicate lesson append when the loop already did it:**

```python
# Before the existing append_lesson(...) call, add:
if not args.dry_run and not _loop_handled_lessons:
    append_lesson(...)  # existing call, unchanged
```

That's it. The rest of `run_task.py` (printing the planned sequence, recording teardown, force-exit) works as-is.

---

## Usage

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
conda activate xarm6sim

# Single attempt (existing behaviour):
python scripts/run_task.py "Put the red cube in the red bin" --model haiku

# Episode learning loop (NEW):
python scripts/run_task.py "Put the red cube in the red bin" \
    --model haiku --loop --max-episodes 8

# Harder task; give it more attempts and a stronger model:
python scripts/run_task.py "Sort all cubes by color into matching bins" \
    --model sonnet --loop --max-episodes 15
```

---

## Expected Behaviour, Step by Step

For `"Put the red cube in the red bin" --loop --max-episodes 5`:

```
======================================================================
[EpisodeLoop] Episode 1/5: Put the red cube in the red bin
======================================================================
[LLMBrain] Using model: haiku (claude-haiku-4-5-20251001)
[LLMBrain] Response in 2.4s (1820->410 tokens)
[Agent] set_rail({'position_mm': 200, 'speed_mm_s': 100}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 795, ...}) -> 0
[Agent] gripper_close({}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 350, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 350, 'z': 810, ...}) -> 2
[EpisodeLoop] Physical outcome: red_cube fell to floor
[EpisodeLoop] Learned: move_to (-200, 350, 810) mm fails collision/FK validation.
              Lift z to >= 860 mm, or change rail position so the arm approaches
              from a less obstructed side.

======================================================================
[EpisodeLoop] Episode 2/5: Put the red cube in the red bin
[EpisodeLoop] Carrying 1 learned constraint(s) into this attempt
======================================================================
[LLMBrain] Response in 2.6s (1995->420 tokens)
[Agent] set_rail({'position_mm': 200, 'speed_mm_s': 100}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 795, ...}) -> 0
[Agent] gripper_close({}) -> 0
[Agent] move_to({'x': -200, 'y': 150, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 350, 'z': 870, ...}) -> 0
[Agent] move_to({'x': -200, 'y': 350, 'z': 830, ...}) -> 0
[Agent] gripper_open({}) -> 0
[Agent] done({'message': 'Red cube placed in red bin'}) -> 0
[EpisodeLoop] Physical outcome: red_cube in red_bin
[EpisodeLoop] Outcome check: success=True (All expected placements present: red_cube in red_bin)
[EpisodeLoop] SUCCESS on episode 2

======================================================================
[EpisodeLoop] Done after 2 episode(s). success=True
[EpisodeLoop] Learned constraints (1):
  - move_to (-200, 350, 810) mm fails collision/FK validation. Lift z to >= 860 mm...
======================================================================
```

After this run, `lessons.md` will have two new entries (one per episode) at
the top, and the next session (whether `--loop` or not) will read them and
already know to avoid `z=810` for the bin release.

---

## Why This Design (Important Trade-offs)

1. **Constraints go in the task prompt, not the system prompt, by default.**
   This means each episode is a fresh API call where the constraints are
   part of the user message. Cost: a few extra tokens per episode. Benefit:
   zero changes to `llm_brain.py`. If you'd rather have them in the system
   prompt, see the optional change in section 3.

2. **`physical_outcome()` is checked AFTER every episode, even when all
   commands returned 0.** A successful command sequence that still leaves
   the cube on the floor is a failure for the loop's purposes. This is the
   single most important behaviour and the reason the loop is more useful
   than just "retry on non-zero return".

3. **`reset_scene()` between episodes, not between commands within an
   episode.** Each episode is an independent attempt from the canonical
   starting state. We're not trying to recover mid-trajectory -- that's
   much harder and the LLM isn't equipped for it.

4. **Per-episode `Recorder`, not one for the whole loop.** Mirrors
   `auto_play.py`. Each episode gets its own session under `recordings/`
   so you can replay any single attempt to see what happened. They share a
   common `parent_session_id` via the loop run, which you could add if you
   want to query "all episodes of this loop run" later.

5. **`lessons.md` is reused, not duplicated.** Each episode appends a
   lesson with `[ep N/M]` in the label. The 20-entry cap will trim older
   single-task lessons faster, which is a real cost -- if you run many
   loop tasks, consider bumping `MAX_LESSONS` in `agent/lessons.py` from
   20 to e.g. 50.

6. **The outcome checker is best-effort.** It handles the canonical
   pick-and-place and push-off-the-bench patterns. For tasks it can't
   parse (`success is None`), the loop stops after one episode and asks
   you to grade manually. This is intentional: silent infinite retries on
   tasks we can't grade would burn API credits.

---

## Optional: `CLAUDE.md` at Repo Root

If you want Claude Code itself to be aware of the loop when invoked in
this repo, drop a `CLAUDE.md` at the repo root with the following content.
Claude Code reads this automatically.

```markdown
# xArm6 Digital Twin -- Project Conventions for Claude Code

## Episode learning loop

When iterating on a task that fails, prefer to use `--loop`:

    python scripts/run_task.py "<task>" --model haiku --loop --max-episodes 10

The loop:
  - Resets the scene between episodes
  - Reads `physical_outcome()` to grade success physically (not just by
    command return codes)
  - Accumulates learned constraints across episodes and injects them into
    the next attempt's prompt
  - Appends one entry to lessons.md per episode

When extending the loop:
  - Add new failure patterns to `agent/episode_loop.py::analyse_command_failure`
  - Add new task patterns to `agent/outcome_checker.py::expected_outcome`
  - Do NOT bypass `arm.reset_scene()` between episodes -- the loop assumes
    a deterministic starting state.

## Other conventions in this repo

(... existing project conventions here ...)
```

---

## Quick Validation Plan

Before trusting the loop on a complex task, validate it on three known cases:

1. **Trivially succeeds:** `python scripts/run_task.py "Go home" --loop`
   Expected: success on episode 1, no constraints learned.

2. **Eventually succeeds:** `python scripts/run_task.py "Put the red cube in the red bin" --loop`
   Expected: success within 1-3 episodes. If Haiku already gets it first try,
   it just confirms the loop's no-op success path works.

3. **Should learn something:** Edit `lessons.md` to remove all entries first,
   then run a tube task:
   `python scripts/run_task.py "Pick up tube_L1 and place it in the right rack" --loop --max-episodes 6`
   Tubes are harder than cubes (taller, rack walls in the way). You should
   see at least one constraint learned about z-clearance before success.

If those three behave as expected, the loop is wired correctly.

---

## What's Deliberately Out of Scope

These would be reasonable next steps, but they're separate features:

- **Persistent cross-session constraint store.** Right now constraints
  survive across sessions only by virtue of being summarised in
  `lessons.md`. A dedicated `session_constraints.md` (or JSON) would be
  richer but adds another file to maintain.
- **Constraint deduplication / generalisation.** After many runs, you'll
  accumulate near-duplicate constraints ("z=810 fails", "z=815 fails", ...).
  A periodic pass to merge them into "z<860 fails near bins" would help.
- **Learning from successes, not just failures.** Recording successful
  trajectories as positive exemplars (and showing them to Claude on the
  next run) would tighten the loop further. Out of scope for v1.
- **Multi-objective episodes.** A "sort all 3 cubes" task is really three
  sub-episodes; if cube 2 fails, ideally cube 1's success is preserved
  and only cube 2 is retried. The current loop resets between episodes
  and retries the whole task. Worth doing later.
