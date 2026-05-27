# agent/episode_loop.py
"""
Episode learning loop (training-session mode).

Runs the task across exactly N episodes (no early stopping on success).
Between episodes, analyses failures and injects learned constraints into
the next attempt. After all episodes finish, reports per-episode outcomes
and a first-half vs. second-half success breakdown so the user can see
whether the model is actually improving across the session.

Designed as a thin wrapper around LLMBrain -- does NOT modify llm_brain.py's
core logic. Each episode:
  1. arm.reset_scene()
  2. start a fresh Recorder
  3. brain.execute_task(task, extra_context=<learned constraints>)
  4. wait for physics to settle
  5. arm.physical_outcome() + check_outcome(task, ...) -> success/failure/ungraded
  6. append per-episode lesson; if failed, append a learned constraint too
  7. always continue to the next episode until max_episodes is reached
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from agent.outcome_checker import check_outcome, classify_task
from agent.lessons import append_lesson
from recording import Recorder


@dataclass
class EpisodeContext:
    """Accumulates learned constraints + per-episode outcomes across one task run."""
    task: str
    max_episodes: int = 10
    episode_num: int = 1
    learned_constraints: List[str] = field(default_factory=list)
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    # `success` = "did any episode succeed". The loop now runs all episodes,
    # so this is sticky-OR rather than the early-stop terminator it used to be.
    success: bool = False
    final_result: Optional[Dict[str, Any]] = None
    final_physical: str = ""
    # Per-episode outcomes, in run order. True/False/None where:
    #   True  = grader said success
    #   False = command failure OR grader said failure
    #   None  = grader couldn't classify the task (no learning possible)
    episode_outcomes: List[Optional[bool]] = field(default_factory=list)

    # Phase 1: soft positive-signal learning. Successful plans are exposed
    # to subsequent episodes as exploration-friendly references, NOT as
    # prescriptive exemplars. See successes_block() for the framing.
    successful_plans: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {"episode": int, "commands": [...], "n_commands": int,
    #              "physical": str, "stringency": str}

    best_plan_idx: Optional[int] = None
    # Index into successful_plans of the current "best" plan by n_commands.
    # None until first success.

    success_diversity: List[int] = field(default_factory=list)
    # Hash of each successful plan's action sequence (just the "action"
    # strings, ignoring params). Used to compute reuse rate in the summary.

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

    def add_successful_plan(self, commands: List[Dict[str, Any]],
                            physical: str, stringency: str) -> None:
        """Record a success. Gates on clean physical outcome to avoid
        pinning destructive plans the grader missed.
        """
        # Defensive gate: reject if physical outcome has destructive markers,
        # even though check_outcome returned True. False-positive mitigation.
        if physical and any(marker in physical for marker in
                            ("off bench", "fell to floor")):
            print(f"[EpisodeLoop] Success grader fired but physical='{physical}' "
                  f"looks destructive -- NOT pinning this plan.")
            return

        n_cmds = len(commands)
        entry = {
            "episode": self.episode_num,
            "commands": commands,
            "n_commands": n_cmds,
            "physical": physical,
            "stringency": stringency,
        }
        self.successful_plans.append(entry)

        # Track action-sequence hash for reuse-rate metric in summary.
        action_seq = tuple(c.get("action", "?") for c in commands)
        self.success_diversity.append(hash(action_seq))

        # Update best_plan_idx by n_commands (fewer = better as a default
        # proxy). Change this metric here if needed later.
        if (self.best_plan_idx is None or
                n_cmds < self.successful_plans[self.best_plan_idx]["n_commands"]):
            self.best_plan_idx = len(self.successful_plans) - 1
            print(f"[EpisodeLoop] New best plan: {n_cmds} commands "
                  f"(episode {self.episode_num}).")

        # Cap at the most recent 3 to bound prompt growth, but always keep
        # the best one even if it would otherwise fall out of the window.
        if len(self.successful_plans) > 3:
            best = self.successful_plans[self.best_plan_idx]
            recent = [p for p in self.successful_plans[-3:] if p is not best]
            self.successful_plans = [best] + recent[-2:]
            self.best_plan_idx = 0

    def successes_block(self) -> str:
        """Render successful plans with EXPLORATION-FRIENDLY framing.

        The framing here is doing real work: we want the LLM to have
        known-good plans as reference, but we explicitly tell it the task
        likely admits better solutions and invite it to find one. Direct
        framing would cause lock-in.
        """
        if not self.successful_plans:
            return ""

        import json
        best = self.successful_plans[self.best_plan_idx]
        best_n = best["n_commands"]

        lines = [
            "",
            "## Plans that have succeeded on this task",
            f"(The current best plan has {best_n} commands. Plans below "
            f"satisfied the grader, but this task likely admits shorter "
            f"or cleaner solutions. You may adapt these approaches, but "
            f"a plan that achieves the goal in fewer commands or with "
            f"a safer trajectory is preferred.)",
            "",
        ]
        for entry in self.successful_plans:
            marker = " (current best)" if entry is best else ""
            lines.append(f"- Episode {entry['episode']} "
                         f"({entry['n_commands']} commands){marker}:")
            lines.append("  ```json")
            plan_json = json.dumps(entry["commands"], indent=2)
            for jline in plan_json.split("\n"):
                lines.append(f"  {jline}")
            lines.append("  ```")
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


# Task-family-aware failure analyser. Family is determined by
# outcome_checker.classify_task() (push_off / placement / sort / unknown).


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
    family = classify_task(task)
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
                 settle_seconds: float = 1.5,
                 stringency: str = "loose"):
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
            stringency:       loose / normal / strict -- forwarded to
                              arm.physical_outcome() to tighten what counts
                              as a placed object. See STRINGENCY_LEVELS in
                              sim/mujoco_env.py.
        """
        self.brain = brain
        self.arm = arm
        self.registry = registry
        self.recorder_factory = recorder_factory
        self.max_episodes = max_episodes
        self.settle_seconds = settle_seconds
        self.stringency = stringency

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
            episode_task = (task
                            + ctx.constraints_block()
                            + ctx.successes_block())

            # 3. Execute.
            try:
                result = self.brain.execute_task(episode_task, dry_run=False)
            except Exception as e:
                print(f"[EpisodeLoop] execute_task raised: {e}")
                result = {"commands": [], "results": [], "error": True}
            ctx.final_result = result

            # 4. Wait for physics to settle and inspect.
            time.sleep(self.settle_seconds)
            physical = self.arm.physical_outcome(self.stringency) if hasattr(
                self.arm, "physical_outcome") else ""
            ctx.final_physical = physical
            print(f"[EpisodeLoop] Physical outcome: {physical or '(none)'} "
                  f"(stringency={self.stringency})")

            # 5. Classify this episode's outcome (success / failure / ungraded).
            #    The loop runs ALL episodes regardless -- we never short-circuit
            #    on success, because the goal is to measure whether the model
            #    *consistently* solves the task, not just whether it can once.
            episode_outcome: Optional[bool] = None  # True / False / None
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
                episode_outcome = False
            else:
                # All commands returned 0. Did the physical state match the task?
                success, reason = check_outcome(task, physical)
                print(f"[EpisodeLoop] Outcome check: success={success} ({reason})")

                if success is True:
                    print(f"[EpisodeLoop] SUCCESS on episode {ctx.episode_num}")
                    ctx.success = True
                    ctx.add_successful_plan(
                        commands=result.get("commands", []),
                        physical=physical,
                        stringency=self.stringency,
                    )
                    episode_outcome = True
                elif success is False:
                    constraint = analyse_physical_failure(task, physical, reason)
                    if constraint:
                        ctx.add_constraint(constraint)
                    ctx.failure_history.append({
                        "episode": ctx.episode_num, "kind": "physical",
                        "physical": physical, "reason": reason,
                        "constraint": constraint,
                    })
                    episode_outcome = False
                else:
                    # success is None -- parser couldn't classify the task.
                    # No constraint to learn from, but we keep iterating so
                    # the user can still measure command-level reliability.
                    print(f"[EpisodeLoop] UNGRADED (grader cannot classify this task)")
                    episode_outcome = None

            ctx.episode_outcomes.append(episode_outcome)
            _record_lesson(task, self.brain, result, physical, ctx,
                           episode_outcome, self.stringency)
            _close_recorder(recorder)
            ctx.episode_num += 1

        # --- End of loop. Build the training-quality summary. ---
        total = len(ctx.episode_outcomes)
        n_succ     = sum(1 for o in ctx.episode_outcomes if o is True)
        n_fail     = sum(1 for o in ctx.episode_outcomes if o is False)
        n_ungraded = sum(1 for o in ctx.episode_outcomes if o is None)

        # First half = ceil(total / 2). For odd N the first half gets the
        # extra episode (e.g., 31 -> 16 + 15) per the design spec.
        half = (total + 1) // 2
        first_half  = ctx.episode_outcomes[:half]
        second_half = ctx.episode_outcomes[half:]
        succ_first  = sum(1 for o in first_half  if o is True)
        fail_first  = sum(1 for o in first_half  if o is False)
        succ_second = sum(1 for o in second_half if o is True)
        fail_second = sum(1 for o in second_half if o is False)

        print(f"\n{'=' * 70}")
        print(f"[EpisodeLoop] Done after {total} episode(s). "
              f"any_success={ctx.success}")
        ungraded_tail = (f"   Ungraded: {n_ungraded}" if n_ungraded else "")
        print(f"[EpisodeLoop] Totals: {n_succ} success / {n_fail} failure"
              f" out of {total}{ungraded_tail}")
        if total >= 2:
            print(f"[EpisodeLoop] First half  (ep 1-{half}):  "
                  f"{succ_first} success, {fail_first} failure")
            print(f"[EpisodeLoop] Second half (ep {half + 1}-{total}): "
                  f"{succ_second} success, {fail_second} failure")
            # Convergence hint: did we improve?
            if succ_second > succ_first:
                print(f"[EpisodeLoop] -> IMPROVING (+{succ_second - succ_first} "
                      f"more successes in second half)")
            elif succ_second < succ_first:
                print(f"[EpisodeLoop] -> REGRESSING ({succ_first - succ_second} "
                      f"fewer successes in second half)")
            else:
                print(f"[EpisodeLoop] -> FLAT (same success count in both halves)")
        if ctx.learned_constraints:
            print(f"[EpisodeLoop] Learned constraints ({len(ctx.learned_constraints)}):")
            for c in ctx.learned_constraints:
                print(f"  - {c}")

        if ctx.successful_plans:
            print(f"[EpisodeLoop] Successful plans pinned: {len(ctx.successful_plans)}")
            best = ctx.successful_plans[ctx.best_plan_idx]
            print(f"[EpisodeLoop] Best plan: {best['n_commands']} commands "
                  f"(episode {best['episode']})")

            # Reuse-rate metric: of the successes in this run, how many used
            # the same action sequence as a previous success? Tells the user
            # whether the model is converging on the pinned shape (high reuse)
            # or independently finding solutions (low reuse).
            if len(ctx.success_diversity) >= 2:
                unique_hashes = len(set(ctx.success_diversity))
                total_successes = len(ctx.success_diversity)
                reuse_rate = 1.0 - (unique_hashes / total_successes)
                print(f"[EpisodeLoop] Success-plan diversity: "
                      f"{unique_hashes}/{total_successes} unique action "
                      f"sequences (reuse rate {reuse_rate:.0%})")
                if reuse_rate >= 0.5:
                    print(f"[EpisodeLoop] -> Model is CONVERGING on pinned plan shape")
                else:
                    print(f"[EpisodeLoop] -> Model is finding INDEPENDENT solutions")
        print('=' * 70)

        return {
            "success": ctx.success,
            "episodes_run": total,
            "final_result": ctx.final_result,
            "final_physical": ctx.final_physical,
            "learned_constraints": ctx.learned_constraints,
            "failure_history": ctx.failure_history,
            "episode_outcomes": ctx.episode_outcomes,
            "totals": {"success": n_succ, "failure": n_fail, "ungraded": n_ungraded},
            "halves": {
                "first":  {"success": succ_first,  "failure": fail_first,  "n": len(first_half)},
                "second": {"success": succ_second, "failure": fail_second, "n": len(second_half)},
            },
            "successful_plans": ctx.successful_plans,
            "best_plan_idx": ctx.best_plan_idx,
            "success_diversity": ctx.success_diversity,
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
                   physical: str, ctx: EpisodeContext,
                   episode_outcome: Optional[bool],
                   stringency: str = "loose") -> None:
    """Append a one-liner to lessons.md for this episode.

    `episode_outcome` is the grader's verdict (True/False/None) -- it gets
    passed through to append_lesson so the lesson string accurately reflects
    whether the TASK succeeded, not just whether the commands ran without
    errors. Without this, "knocked the rack off the bench" while trying to
    place a tube would be recorded as SUCCESS and poison future runs.

    `stringency` is the grading mode the episode was scored under; passed
    through so the entry can be tagged in lessons.md.
    """
    label = f"{task} [ep {ctx.episode_num}/{ctx.max_episodes}]"
    append_lesson(
        task_prompt=label,
        model_short=brain.model_short,
        planned_commands=result.get("commands", []),
        results=result.get("results", []),
        physical_outcome=physical,
        task_success=episode_outcome,
        stringency=stringency,
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
