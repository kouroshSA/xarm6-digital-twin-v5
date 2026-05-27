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
        )
        summary = loop.run(args.task)
        result = summary["final_result"] or {"commands": [], "results": []}
        _loop_handled_lessons = True
    else:
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
        physical = arm.physical_outcome()
        print(f"\n[Physical outcome] {physical}")
    if not args.dry_run and not _loop_handled_lessons:
        append_lesson(
            task_prompt=args.task,
            model_short=model_short,
            planned_commands=result.get("commands", []),
            results=result.get("results", []),
            physical_outcome=physical,
        )
        print("[Lessons] Appended to lessons.md")

    if recorder is not None:
        recorder.stop_and_prompt(prompt=True, auto_task_label=args.task)

    arm.disconnect()
    # Force-exit. The MuJoCo viewer's GLFW + X11 cleanup at Python interpreter
    # shutdown can hang or segfault; all our durable data is already flushed
    # to disk by this point. Flush stdout/stderr to make sure no print() is lost.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
