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
import re
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


# Task-family classifiers. These mirror the regexes in outcome_checker.py so
# the loop reasons about WHAT the user asked for (push? place? sort?) rather
# than WHICH object they named. Add new families here as new task templates
# are introduced.
_PUSH_OFF_RE  = re.compile(r"\b(push|knock|drop|shove|slide)\b.*\b(off|edge|floor|away)\b")
_PLACEMENT_RE = re.compile(r"\b(put|place|move|drop|stash)\b.*\b(in|into|inside)\b.*\b(bin|container|box|rack|slot)\b")
_SORT_RE      = re.compile(r"\bsort\b.*\bcubes?\b")


def _classify_task(task: str) -> str:
    """Return one of: 'push_off', 'placement', 'sort', 'unknown'."""
    t = task.lower()
    if _PUSH_OFF_RE.search(t):
        return "push_off"
    if _SORT_RE.search(t):
        return "sort"
    if _PLACEMENT_RE.search(t):
        return "placement"
    return "unknown"


def analyse_physical_failure(task: str, physical: str,
                             reason: str) -> Optional[str]:
    """
    Commands all returned 0, but the physical outcome doesn't match the task.
    Infer a task-family-appropriate constraint -- the hint depends on whether
    the user asked to push, place, or sort, NOT on which specific object was
    named. Object-specific advice (cube vs tube vs rack heights) should come
    from the registry the LLM already sees in the system prompt.
    """
    phys = physical or ""
    nothing_moved = (not phys) or phys == "no objects displaced"
    family = _classify_task(task)
    placement_wrong_dest = bool(reason and "Missing" in reason and " in " in phys)

    if family == "push_off":
        if nothing_moved:
            return ("Push did not displace anything. The arm must actually "
                    "make contact and drag the target: use `push_object` with "
                    "target_name matching the intended body, and aim "
                    "to_x_mm/to_y_mm at the nearest bench edge "
                    "(|x|>750 mm or |y|>450 mm). `move_to` alone will not "
                    "push -- it just relocates the EE. Set the rail position "
                    "so the EE can reach the object's xy with a clear "
                    "approach from the opposite side of the chosen edge.")
        # Got "in <bin>" segments but no off-bench segments for any target.
        if " in " in phys and "off bench" not in phys and "fell to floor" not in phys:
            return ("Push deposited the target into a bin instead of off the "
                    "bench. Re-aim to_x_mm/to_y_mm toward the nearest bench "
                    "edge, not toward another container.")
        # We got SOME off-bench/floor events, but check_outcome graded False
        # -- meaning the wrong object was affected.
        if "off bench" in phys or "fell to floor" in phys:
            return (f"Push affected the wrong object(s) -- observed: {phys}. "
                    "Use `push_object` with target_name set to the body the "
                    "task names, and double-check the registry for that "
                    "body's current xy before choosing the push destination.")
        return (f"Push completed but the result is unclear: {phys}. Re-read "
                "the task target and push harder toward the bench edge.")

    if family in ("placement", "sort"):
        if nothing_moved:
            return ("All commands succeeded but nothing moved -- the grasp "
                    "likely failed. The EE site must be within ~70 mm of the "
                    "target body's centre when `gripper_close` fires. Use "
                    "the registry's grip height for the target object "
                    "(cubes/tubes/bins/racks have different heights), "
                    "set the rail closer, and verify xy matches the object's "
                    "current position from the registry. For tube-into-rack "
                    "tasks, use `place_tube_in_rack` after grasping -- it "
                    "snaps the held tube into the rack's first open slot.")
        if "fell to floor" in phys:
            return ("An object fell to the floor during placement. Likely "
                    "causes: release was too high above the destination "
                    "(bounce-out), lift trajectory cleared the destination's "
                    "opening sideways, or transit path crossed the bench "
                    "edge. Release just above the destination's top surface, "
                    "directly over its xy from the registry.")
        if "off bench" in phys:
            return ("An object ended up off the bench during placement. Your "
                    "transit path went past the bench edge. Keep xy inside "
                    "[-750..+750, -450..+450] mm while carrying.")
        if placement_wrong_dest:
            return (f"Object placed in the WRONG destination: {phys}. The "
                    "placement xy didn't match the task target. Re-read the "
                    "registry for the intended destination's xy.")
        return None

    # Unknown task family: give a generic nudge so something is logged.
    return (f"Physical state doesn't match the task: {phys}. Re-check the "
            "task target and use registry coordinates rather than "
            "estimating positions.")


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
                if recorder is not None:
                    recorder.start()
                    # The brain captures the recorder it was constructed with,
                    # so if we want per-episode recording we swap it in:
                    self.brain.recorder = recorder

            # 2. Build the prompt: original task + learned constraints block.
            #    (Constraints ride along inside the user message; no change
            #    to llm_brain.py required.)
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

        # End of loop. Print summary. Count = current iter when we break on
        # success/unknown (didn't increment), or current iter - 1 when we ran
        # all max_episodes (incremented at end of last iter).
        episodes_run = (ctx.episode_num
                        if ctx.episode_num <= ctx.max_episodes
                        else ctx.max_episodes)
        print(f"\n{'=' * 70}")
        print(f"[EpisodeLoop] Done after {episodes_run} episode(s). "
              f"success={ctx.success}")
        if ctx.learned_constraints:
            print(f"[EpisodeLoop] Learned constraints ({len(ctx.learned_constraints)}):")
            for c in ctx.learned_constraints:
                print(f"  - {c}")
        print('=' * 70)

        return {
            "success": ctx.success,
            "episodes_run": episodes_run,
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
