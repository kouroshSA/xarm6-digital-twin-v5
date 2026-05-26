# scripts/auto_play.py
"""
Auto-play: Claude generates N diverse task prompts using the scene primitives,
then the LLMBrain executes each one back-to-back in a single viewer window.

Each episode is recorded into its own session under recordings/ (linked via
parent_session_id so you can find sibling episodes later). Lessons.md is
appended after every episode.

Usage:
  python scripts/auto_play.py --episodes 5 --model haiku --save-all
  python scripts/auto_play.py --episodes 3 --model sonnet --eval
"""
import argparse
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()

from agent.llm_brain import LLMBrain, MODELS
from agent.object_registry import build_default_registry
from agent.prompt_variants import generate_episodes
from agent.lessons import append_lesson
from recording import Recorder


def stop_recorder_silent(recorder, kept: bool = True) -> Path:
    if not recorder.is_recording:
        return None
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
    recorder._session.kept = kept
    recorder._write_metadata()
    return recorder._session_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes (default 5)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="haiku",
                        help="Claude model for prompt generation + execution")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval", action="store_true",
                            help="Prompt s/f/d between episodes")
    eval_group.add_argument("--save-all", action="store_true",
                            help="Save every episode without prompting (default)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the prompt generator (forwards to Claude implicitly)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Record image frames at 10Hz per episode (off by default).")
    args = parser.parse_args()

    # 1) Ask Claude for diverse task prompts
    print(f"\n[AutoPlay] Generating {args.episodes} diverse prompts via {args.model}...")
    try:
        tasks = generate_episodes(args.episodes, model=MODELS[args.model])
    except Exception as e:
        print(f"[AutoPlay] Prompt generation failed: {e}")
        sys.exit(1)

    print(f"[AutoPlay] Got {len(tasks)} prompts:")
    for i, t in enumerate(tasks, 1):
        print(f"  {i}. {t}")

    # 2) Spawn one SimXArmAPI with viewer (reused across episodes)
    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=True)
    registry = build_default_registry()
    run_id = uuid.uuid4().hex[:8]
    print(f"\n[AutoPlay] Run ID: {run_id}  (each episode's metadata links here)")

    # 3) Execute each prompt
    saved_dirs = []
    for i, task in enumerate(tasks, start=1):
        print(f"\n{'=' * 70}\nEpisode {i}/{len(tasks)}: {task}\n{'=' * 70}")
        arm.reset_scene()
        time.sleep(0.5)

        recorder = Recorder(
            arm.model, arm.data, arm.lock,
            interface="auto_play",
            scene_xml="envs/lab_scene.xml",
            enable_frames=args.save_frames,
        )
        recorder.start()
        recorder.session.parent_session_id = run_id
        recorder.session.cycle_index = i
        recorder.session.task_label = task[:64]

        brain = LLMBrain(arm=arm, registry=registry,
                         recorder=recorder, model=args.model)
        try:
            result = brain.execute_task(task)
        except Exception as e:
            print(f"  Task error: {e}")
            result = {"commands": [], "results": [], "error": True}

        time.sleep(1.5)
        physical = arm.physical_outcome()
        print(f"  [Physical outcome] {physical}")

        # Auto-append lesson regardless of eval mode
        append_lesson(
            task_prompt=task, model_short=args.model,
            planned_commands=result.get("commands", []),
            results=result.get("results", []),
            physical_outcome=physical,
        )

        # Eval prompt (only if --eval)
        kept = True
        if args.eval:
            try:
                ans = input("  Outcome [s=success / f=failure / d=delete / Enter=keep]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans.startswith("s"):
                recorder.session.outcome = "success"
            elif ans.startswith("f"):
                recorder.session.outcome = "failure"
            elif ans.startswith("d"):
                kept = False

        saved = stop_recorder_silent(recorder, kept=kept)
        if kept and saved:
            saved_dirs.append(saved)
            print(f"  Saved: {saved.name}")
        elif saved:
            shutil.rmtree(saved)
            print(f"  Deleted: {saved.name}")

    arm.disconnect()
    print(f"\n[AutoPlay] Done. {len(saved_dirs)} episodes saved under recordings/.")
    print(f"           Parent run id: {run_id}")
    print(f"           Inspect any with: python replay.py <index>")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
