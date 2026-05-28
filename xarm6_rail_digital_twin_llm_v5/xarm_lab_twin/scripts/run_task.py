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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--mode", choices=["sim","real"], default="sim")
    parser.add_argument("--ip", default="192.168.1.100")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--save-frames", action="store_true",
                        help="Record image frames at 10Hz (off by default). "
                             "Adds ~10-15MB per minute of recording.")
    parser.add_argument("--loop", action="store_true",
                        help="Enable episode learning loop: retry on failure, "
                             "learning constraints between attempts.")
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
                        choices=list(SPEED_TIERS.keys()),
                        default=None,
                        help="Override the Haiku-inferred session speed cap. "
                             "Useful when you want deterministic safety: "
                             "regardless of what the prompt says, the session "
                             "ceiling will be this tier. Per-command tier "
                             "downgrades within the LLM plan still apply, "
                             "but cannot exceed this ceiling.")
    args = parser.parse_args()

    model_short = args.model if args.model else prompt_model_choice()

    if args.mode == "sim":
        from sim.mujoco_env import SimXArmAPI
        arm = SimXArmAPI(scene_xml="envs/lab_scene.xml",
                         render=not args.no_render)
        print("[System] SIMULATION mode")
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
            interface="llm_brain", scene_xml="envs/lab_scene.xml",
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

    print(f"\n[Task] {args.task}\n")
    # _loop_handled_lessons: when True, the EpisodeRetry already appended one
    # lesson per episode, so the outer code at the bottom skips its
    # single-shot append_lesson() to avoid duplication.
    _loop_handled_lessons = False
    loop_summary = None  # populated only on --loop runs; used to re-print the
                         # success/failure totals as the LAST thing on screen
                         # so the [Planned sequence]/etc. blocks below don't
                         # scroll the per-episode stats out of view.

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
                interface="episode_loop", scene_xml="envs/lab_scene.xml",
                enable_frames=args.save_frames,
            )

        loop = EpisodeRetry(
            brain=brain, arm=arm, registry=registry,
            recorder_factory=_recorder_factory,
            max_episodes=args.max_episodes,
            stringency=args.stringency,
            speed_tier_override=args.speed_tier,
        )
        summary = loop.run(args.task)
        loop_summary = summary
        result = summary["final_result"] or {"commands": [], "results": []}
        _loop_handled_lessons = True
    else:
        # Infer the speed-cap tier from the task prompt before dispatch
        # (CLI --speed-tier wins over inference when set).
        brain.prepare_for_task(args.task, override_tier=args.speed_tier)
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
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
