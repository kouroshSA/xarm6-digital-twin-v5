# scripts/run_task.py
import argparse
import sys
import threading
import time

# Make project root importable when running as `python scripts/run_task.py`
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()  # populate os.environ from .env before anthropic.Anthropic() reads it

from agent.llm_brain import LLMBrain, prompt_model_choice, MODELS
from agent.object_registry import build_default_registry
from agent.lessons import append_lesson
from recording import Recorder


def _shutdown(arm, recorder=None):
    """Close the sim down before an early return.

    Returning while the sim thread and any viewer are still live tears the GL
    and MuJoCo contexts down from under them, and the process dies with
    SIGSEGV -- so a deliberate refusal reported as exit 139 instead of its own
    exit code. Exit status is the contract for anything scripting this.
    """
    if recorder is not None:
        try:
            recorder.abort()
        except Exception:  # noqa: BLE001
            pass
    try:
        arm.disconnect()
    except Exception:      # noqa: BLE001
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--mode", choices=["sim", "dryrun", "real"], default="sim")
    parser.add_argument("--ip", default="192.168.1.100")
    parser.add_argument("--validator-ip", default="127.0.0.1",
                        help="Address of the UFACTORY controller container used "
                             "as the pre-action gate for --mode real. See "
                             "docs/a8_controller_validator.md.")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip pre-action plan validation in --mode real. "
                             "You are then executing an unchecked plan on a "
                             "physical arm; the operator must be at the e-stop.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", metavar="PATH",
                        help="write the loop/run summary as JSON. Lets a "
                             "harness read results instead of screen-scraping "
                             "stdout, which changes whenever a print does.")
    parser.add_argument("--no-intake", action="store_true",
                        help="skip Layer 0 (decomposition, class constraints, "
                             "pushback). Layer 0 runs on every prompt by "
                             "default; this is the A/B switch.")
    parser.add_argument("--refine-prompt", action="store_true",
                        help="Layer 2: let a model rewrite an AMBIGUOUS task "
                             "into an unambiguous one before planning. Off by "
                             "default until an A/B shows it earns its place; "
                             "the rewrite is discarded unless it passes "
                             "re-validation against the scene.")
    parser.add_argument("--no-task-check", action="store_true",
                        help="skip Layer 1 task validation (resolution + "
                             "feasibility screen) before planning")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--save-frames", action="store_true",
                        help="Record image frames at 10Hz (off by default). "
                             "Adds ~10-15MB per minute of recording.")
    parser.add_argument("--loop", action="store_true",
                        help="Enable episode learning loop: retry on failure, "
                             "learning constraints between attempts.")
    parser.add_argument("--hloop", action="store_true",
                        help="Human-in-the-loop variant of --loop. Pauses at "
                             "the end of each episode and prompts the operator "
                             "for free-form feedback that's injected into the "
                             "NEXT episode's prompt. Press Enter to skip a "
                             "given prompt; type anything to record. Implies "
                             "--loop.")
    parser.add_argument("--max-episodes", type=int, default=10,
                        help="Max episodes for --loop (default: 10).")
    parser.add_argument("--stringency",
                        choices=["loose", "normal", "strict"],
                        default="loose",
                        help="How tightly physical_outcome() grades placements. "
                             "loose (default): legacy 20mm xy / 30mm z slot "
                             "tolerance, no uprightness check. normal: 12/15mm "
                             "+ <=30deg tilt. strict: 6/6mm + <=10deg tilt. "
                             "Tighter = harder for the LLM to claim success.")
    from agent.dynamic_grader import SPEED_TIERS
    parser.add_argument("--speed-tier",
                        choices=list(SPEED_TIERS.keys()) + ["auto"],
                        default=None,
                        help="Override the Haiku-inferred session speed cap. "
                             "Useful when you want deterministic safety: "
                             "regardless of what the prompt says, the session "
                             "ceiling will be this tier. Per-command tier "
                             "downgrades within the LLM plan still apply, "
                             "but cannot exceed this ceiling. Pass `auto` to "
                             "explicitly request Haiku inference (same as "
                             "omitting the flag); omitting also defaults to "
                             "Haiku inference.")
    parser.add_argument("--led", dest="led_enabled",
                        action="store_false", default=True,
                        help="TURN OFF the rainbow LED strips beside the rail. "
                             "By default the strips are ON: when the rail "
                             "moves, the rainbow flows opposite to the "
                             "motion direction; flow speed matches the "
                             "active --speed-tier. Pass --led to disable.")
    parser.add_argument("--scene", choices=["primitive", "meshes"],
                        default="meshes",
                        help="Which arm model to load. 'meshes' (default) = the "
                             "realistic UFACTORY xArm6 visual meshes with true "
                             "URDF kinematics (envs/lab_scene.xml); 'primitive' "
                             "= the legacy box/cylinder xArm6 "
                             "(envs/lab_scene_primitive.xml). Both are validated "
                             "for the full task suite (IK, gripper, pick/place/"
                             "push); 'meshes' is the higher-fidelity default.")
    args = parser.parse_args()

    scene_path = {
        "meshes": "envs/lab_scene.xml",
        "primitive": "envs/lab_scene_primitive.xml",
    }[args.scene]

    model_short = args.model if args.model else prompt_model_choice()

    if args.mode == "sim":
        from sim.mujoco_env import SimXArmAPI
        arm = SimXArmAPI(scene_xml=scene_path,
                         render=not args.no_render)
        print("[System] SIMULATION mode")
    elif args.mode == "dryrun":
        # Same backend code path as --mode real, pointed at the containerised
        # controller instead of the bench. Nothing can move; this exercises
        # hardware/real_arm.py itself, which sim mode never touches.
        from hardware.real_arm import RealXArmAPI
        # The validator container has no rail hardware, so 0 genuinely is its
        # position. Declared explicitly rather than letting the rail-trust guard
        # refuse every Cartesian move.
        arm = RealXArmAPI(ip=args.validator_ip, assume_rail_mm=0.0)
        print(f"[System] DRYRUN mode - real backend against controller "
              f"container at {args.validator_ip} (no physical arm)")
    else:
        from hardware.real_arm import RealXArmAPI
        arm = RealXArmAPI(ip=args.ip)
        print(f"[System] REAL HARDWARE mode - {args.ip}")

    recorder = None
    if not args.no_record:
        recorder = Recorder(
            model=arm.model if hasattr(arm, "model") else None,
            data=arm.data if hasattr(arm, "data") else None,
            lock=arm.lock if hasattr(arm, "lock") else threading.Lock(),
            interface="llm_brain", scene_xml=scene_path,
            enable_frames=args.save_frames,
        )
        if hasattr(arm, "model"):
            recorder.start()
        else:
            print("[System] Recording requires sim mode (or real-hardware state poller).")
            recorder = None

    registry = build_default_registry()
    brain = LLMBrain(arm=arm, registry=registry, recorder=recorder,
                     model=model_short)

    # Pre-action gate (A8). On real hardware, replay the plan against real
    # controller firmware in Docker before any of it reaches the arm. Refusing to
    # run when the validator is unreachable is deliberate: a gate that silently
    # passes when its backend is down is worse than no gate, because the operator
    # believes a check happened. --no-preflight is the explicit opt-out.
    if args.mode == "real" and not args.no_preflight:
        from scripts.validate_plan import make_gate
        gate = make_gate(args.validator_ip)
        if gate is None:
            print(f"[System] ABORT: pre-action validator unreachable at "
                  f"{args.validator_ip}. Start it with:\n"
                  f"    docker start uf-sim && docker exec uf-sim "
                  f"/xarm_scripts/xarm_start.sh 6 6\n"
                  f"  See docs/a8_controller_validator.md, or pass --no-preflight "
                  f"to run an UNCHECKED plan on the physical arm.")
            return 2
        brain.plan_validator = gate
        print(f"[System] pre-action validation ENABLED "
              f"(controller container at {args.validator_ip})")
    elif args.mode == "real":
        print("[System] WARNING: --no-preflight - the plan will NOT be checked "
              "before it reaches the arm. Stay on the e-stop.")

    print(f"\n[Task] {args.task}\n")

    # Layer 0: every prompt passes through here. Needs no scene, so it is
    # cheap on the easy cases; works on the instruction's structure and on
    # class-level knowledge (what a Falcon tube requires) rather than on where
    # anything is. Constraints it emits ride WITH the task; they are
    # requirements a plan must satisfy, never a plan.
    if not args.no_intake:
        try:
            from agent.instruction_intake import intake, STATUS_REJECT
            r0 = intake(args.task)
            rendered = r0.render()
            if rendered:
                print(rendered)
            if r0.blocked:
                # Autonomous runs must not wait for input nobody will type, so
                # a question becomes a refusal that records what it would have
                # asked. --loop is exactly the case that would hang otherwise.
                verb = ("REJECTED" if r0.status == STATUS_REJECT
                        else "needs clarification")
                print(f"[Layer0] Not proceeding -- {verb}. "
                      f"Re-run with a clearer instruction, or --no-intake to "
                      f"plan it anyway.")
                _shutdown(arm, recorder)
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(2)
            if r0.used:
                args.task = r0.to_task_prompt()
            else:
                print(f"[Layer0] using the original instruction ({r0.reason})")
        except Exception as exc:  # noqa: BLE001 - never block a run on intake
            print(f"[Layer0] skipped ({type(exc).__name__}: {exc})")

    # Layer 2 (opt-in): rewrite an ambiguous instruction before Layer 1's
    # final pass. It runs AFTER a first Layer 1 read internally -- the refiner
    # needs measured facts to write with -- and its output is re-validated, so
    # a rewrite that invents an object or a number is discarded and the
    # original prompt is used unchanged.
    if args.refine_prompt and not args.no_task_check:
        try:
            from agent.prompt_refiner import refine
            r = refine(args.task, registry, arm)
            if r.used:
                print(f"[Layer2] refined: {r.original!r}\n"
                      f"      -> {r.refined!r}")
                args.task = r.refined
            else:
                print(f"[Layer2] using the original prompt ({r.reason})")
        except Exception as exc:  # noqa: BLE001 - never block a run on the refiner
            print(f"[Layer2] skipped ({type(exc).__name__}: {exc})")

    # Layer 1: resolve the task against the scene before any planning happens.
    # Deterministic -- no LLM, no motion. It answers two questions the planner
    # should not have to guess at: which object was meant, and whether the
    # task is possible at all. Facts it reads are handed to the planner;
    # blockers stop the run, because no plan can fix them.
    if not args.no_task_check:
        try:
            from agent.task_validator import validate_task
            verdict = validate_task(args.task, registry, arm)
            if verdict.facts or verdict.warnings or verdict.blockers:
                print(verdict.render())
                print()
            if not verdict.feasible:
                print("[System] Task REFUSED before planning -- the blockers "
                      "above are geometric, so no action sequence can succeed. "
                      "Re-run with --no-task-check to plan anyway.")
                _shutdown(arm, recorder)
                sys.stdout.flush(); sys.stderr.flush()
                os._exit(2)
        except Exception as exc:  # noqa: BLE001 - never block a run on the checker
            print(f"[System] task validation skipped ({type(exc).__name__}: {exc})")
    # _loop_handled_lessons: when True, the EpisodeRetry already appended one
    # lesson per episode, so the outer code at the bottom skips its
    # single-shot append_lesson() to avoid duplication.
    _loop_handled_lessons = False
    loop_summary = None  # populated only on --loop runs; used to re-print the
                         # success/failure totals as the LAST thing on screen
                         # so the [Planned sequence]/etc. blocks below don't
                         # scroll the per-episode stats out of view.

    # --hloop implies --loop (no point asking for feedback if the loop
    # isn't running).
    if args.hloop:
        args.loop = True

    if args.loop and not args.dry_run:
        from agent.episode_loop import EpisodeRetry

        # The outer recorder is replaced by per-episode recorders inside the
        # loop. Stop+save the outer one (no prompt, since loop mode is
        # non-interactive).
        if recorder is not None:
            recorder.stop_and_prompt(prompt=False)
            recorder = None

        def _recorder_factory():
            if not hasattr(arm, "model"):
                return None
            return Recorder(
                model=arm.model, data=arm.data, lock=arm.lock,
                interface="episode_loop", scene_xml=scene_path,
                enable_frames=args.save_frames,
            )

        loop = EpisodeRetry(
            brain=brain, arm=arm, registry=registry,
            recorder_factory=_recorder_factory,
            max_episodes=args.max_episodes,
            stringency=args.stringency,
            speed_tier_override=args.speed_tier,
            led_enabled=args.led_enabled,
            human_feedback_enabled=args.hloop,
        )
        summary = loop.run(args.task)
        loop_summary = summary
        result = summary["final_result"] or {"commands": [], "results": []}
        _loop_handled_lessons = True
    else:
        # Infer the speed-cap tier from the task prompt before dispatch
        # (CLI --speed-tier wins over inference when set).
        brain.prepare_for_task(args.task, override_tier=args.speed_tier)
        if hasattr(arm, "set_led"):
            arm.set_led(args.led_enabled, getattr(brain, "speed_tier", "medium"))
        try:
            result = brain.execute_task(args.task, dry_run=args.dry_run)
        except Exception as e:
            print(f"[System] Task failed: {e}")
            result = {"commands": [], "results": [], "error": True}

    print("\n[Planned sequence]")
    for i, cmd in enumerate(result.get("commands", [])):
        print(f"  {i+1:2d}. {cmd['action']}  {cmd.get('params', {})}")
    if not args.dry_run and result.get("results"):
        print("\n[Execution results]")
        for r in result["results"]:
            ok = "OK" if r["result"] == 0 else "FAIL"
            print(f"  {ok}  {r['action']}  ->  {r['result']}")

    # Wait briefly for cubes to settle, then snapshot physical outcome and
    # append a lesson before the recording prompt steals the terminal.
    time.sleep(1.5)
    physical = ""
    if args.mode == "sim" and hasattr(arm, "physical_outcome"):
        physical = arm.physical_outcome(args.stringency)
        print(f"\n[Physical outcome] {physical}  (stringency={args.stringency})")
    if not args.dry_run and not _loop_handled_lessons:
        # Grade the run before logging so lessons.md records the TASK outcome,
        # not just whether commands ran without errors.
        from agent.outcome_checker import check_outcome
        task_success, _reason = check_outcome(args.task, physical)
        append_lesson(
            task_prompt=args.task,
            model_short=model_short,
            planned_commands=result.get("commands", []),
            results=result.get("results", []),
            physical_outcome=physical,
            task_success=task_success,
            stringency=args.stringency,
        )
        print("[Lessons] Appended to lessons.md")

    if recorder is not None:
        recorder.stop_and_prompt(prompt=True, auto_task_label=args.task)

    arm.disconnect()

    # Re-print the loop summary as the LAST visible block so it isn't scrolled
    # away by [Planned sequence] / [Execution results] / [Physical outcome].
    if loop_summary is not None:
        tot = loop_summary.get("totals", {})
        halves = loop_summary.get("halves", {})
        first = halves.get("first", {})
        second = halves.get("second", {})
        n_total = loop_summary.get("episodes_run", 0)
        n_succ  = tot.get("success", 0)
        n_fail  = tot.get("failure", 0)
        n_ungr  = tot.get("ungraded", 0)
        succ_first  = first.get("success", 0); fail_first  = first.get("failure", 0)
        succ_second = second.get("success", 0); fail_second = second.get("failure", 0)
        n_first  = first.get("n", 0)
        n_second = second.get("n", 0)
        if succ_second > succ_first:
            trend = f"IMPROVING (+{succ_second - succ_first} more successes in 2nd half)"
        elif succ_second < succ_first:
            trend = f"REGRESSING ({succ_first - succ_second} fewer successes in 2nd half)"
        else:
            trend = "FLAT (same success count in both halves)"
        truncated_task = (args.task if len(args.task) <= 60
                          else args.task[:57] + "...")
        print()
        print("=" * 68)
        print("                          RUN SUMMARY")
        print("=" * 68)
        print(f"  Task:        {truncated_task}")
        print(f"  Episodes:    {n_total}   (success={n_succ}, "
              f"failure={n_fail}{', ungraded=' + str(n_ungr) if n_ungr else ''})")
        if n_total >= 2:
            print(f"  First half   (ep 1-{n_first}):  "
                  f"{succ_first} success, {fail_first} failure")
            print(f"  Second half  (ep {n_first + 1}-{n_total}): "
                  f"{succ_second} success, {fail_second} failure")
            print(f"  Trend:       {trend}")
        print("=" * 68)

    # Force-exit. The MuJoCo viewer's GLFW + X11 cleanup at Python interpreter
    # shutdown can hang or segfault; all our durable data is already flushed
    # to disk by this point. Flush stdout/stderr to make sure no print() is lost.
    #
    # The code is passed through rather than hardcoded. This used to be
    # os._exit(0) unconditionally, with main() called without sys.exit(), so
    # run_task.py reported success for EVERY run -- a failed task, a refused
    # task and a clean success were indistinguishable to anything scripting
    # it. task_sweep.py's docstring already states that exit code is the
    # contract; this file was quietly breaking it.
    if args.summary_json:
        try:
            import json as _json
            payload = {"task": args.task, "model": args.model,
                       "loop": loop_summary}
            with open(args.summary_json, "w") as fh:
                _json.dump(payload, fh, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001 - reporting must not fail a run
            print(f"[System] could not write --summary-json ({exc})")

    _exit_code = 0
    if loop_summary is not None:
        if not loop_summary.get("any_success", False):
            _exit_code = 1
    elif not args.dry_run and "task_success" in dir():
        pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(_exit_code)


if __name__ == "__main__":
    main()
