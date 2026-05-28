# scripts/run_task_augmented.py
import argparse
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()  # populate os.environ from .env before anthropic.Anthropic() reads it

import numpy as np

from agent.llm_brain import (
    LLMBrain, prompt_model_choice, MODELS, resolve_model,
)
from agent.object_registry import build_default_registry
from agent.prompt_variants import generate_variants, preview_and_confirm
from envs.scene_randomizer import (
    randomize_scene, DEFAULT_POS_JITTER_MM, DEFAULT_ROT_JITTER_DEG,
)
from recording import Recorder


def run_one_cycle(task_prompt, cycle_index, parent_session_id, original_prompt,
                  model_short, pos_jitter_mm, rot_jitter_deg, seed,
                  render, no_record, dry_run,
                  speed_tier_override: Optional[str] = None,
                  led_enabled: bool = False) -> Optional[Path]:
    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=render)

    scene_summary = {}
    if cycle_index > 1 and (pos_jitter_mm > 0 or rot_jitter_deg > 0):
        rng = np.random.default_rng(seed)
        with arm.lock:
            scene_summary = randomize_scene(
                arm.model, arm.data,
                pos_jitter_mm=pos_jitter_mm,
                rot_jitter_deg=rot_jitter_deg,
                initial_joint_jitter_deg=0.0,
                rng=rng,
            )

    recorder = None
    if not no_record:
        recorder = Recorder(
            model=arm.model, data=arm.data, lock=arm.lock,
            interface="llm_brain_augmented", scene_xml="envs/lab_scene.xml",
        )
        recorder.start()
        recorder.session.parent_session_id = parent_session_id
        recorder.session.cycle_index = cycle_index
        recorder.session.augmentation_config = {
            "parent_prompt": original_prompt,
            "variant_prompt": task_prompt,
            "object_pose_jitter_mm": pos_jitter_mm if cycle_index > 1 else 0.0,
            "object_rotation_jitter_deg": rot_jitter_deg if cycle_index > 1 else 0.0,
            "scene_perturbations": scene_summary,
            "seed": seed if cycle_index > 1 else 0,
        }

    registry = build_default_registry()
    brain = LLMBrain(arm=arm, registry=registry, recorder=recorder,
                     model=model_short)
    print(f"\n  [Cycle {cycle_index}] prompt: \"{task_prompt}\"")
    # Per-task speed-cap inference (variant prompts may shift the tier).
    # CLI --speed-tier (when set) wins over Haiku inference.
    brain.prepare_for_task(task_prompt, override_tier=speed_tier_override)
    if hasattr(arm, "set_led"):
        arm.set_led(led_enabled, getattr(brain, "speed_tier", "medium"))
    try:
        result = brain.execute_task(task_prompt, dry_run=dry_run)
    except Exception as e:
        print(f"  [Cycle {cycle_index}] task error: {e}")
        result = {"commands": [], "results": [], "error": True}
    n_cmds = len(result.get("commands", []))
    print(f"  [Cycle {cycle_index}] {n_cmds} commands, "
          f"{result.get('latency_s', 0):.1f}s latency")

    saved_path = None
    if recorder is not None:
        time.sleep(1.5)
        saved_path = _stop_silent(recorder)
    arm.disconnect()
    return saved_path


def _stop_silent(recorder):
    if not recorder.is_recording: return None
    recorder._recording = False
    if recorder._state_thread is not None:
        recorder._state_thread.join(timeout=1.0)
    recorder._session.ended_at_iso = datetime.now().isoformat()
    recorder._session.duration_s = time.time() - recorder._start_wall_time
    recorder._session.n_state_samples = len(recorder._state_buffer)
    if recorder._commands_file is not None:
        recorder._commands_file.close(); recorder._commands_file = None
    recorder._write_trajectory()
    recorder._session.kept = True
    recorder._write_metadata()
    return recorder._session_dir


def batch_annotate(saved_dirs):
    import json
    print("\n" + "=" * 70)
    print("  BATCH ANNOTATION")
    print("=" * 70)
    print(f"  {sum(1 for d in saved_dirs if d)} recordings created.")
    print("-" * 70)
    for i, d in enumerate(saved_dirs):
        if d is None: continue
        meta_path = d / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        variant_prompt = meta.get("augmentation_config", {}).get("variant_prompt", "")
        cycle = meta.get("cycle_index", "")
        print(f"\n[{i+1}/{len(saved_dirs)}] {d.name}")
        print(f"     cycle: {cycle}  prompt: \"{variant_prompt}\"")
        try:
            outcome = input("     Outcome [s/f/d/Enter]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if outcome.startswith("d"):
            print(f"     x Deleting {d.name}"); shutil.rmtree(d); continue
        if outcome.startswith("s"):   meta["outcome"] = "success"
        elif outcome.startswith("f"): meta["outcome"] = "failure"
        try:
            note = input("     Notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
        if note: meta["notes"] = note
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Original task command")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--variant-model", default="haiku")
    parser.add_argument("--pos-jitter-mm", type=float, default=DEFAULT_POS_JITTER_MM)
    parser.add_argument("--rot-jitter-deg", type=float, default=DEFAULT_ROT_JITTER_DEG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--auto-accept-variants", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    from agent.dynamic_grader import SPEED_TIERS
    parser.add_argument("--speed-tier",
                        choices=list(SPEED_TIERS.keys()) + ["auto"],
                        default=None,
                        help="Override the Haiku-inferred speed cap for "
                             "every cycle. Deterministic safety override. "
                             "Pass `auto` (or omit) to use Haiku inference.")
    parser.add_argument("--led", dest="led_enabled",
                        action="store_false", default=True,
                        help="TURN OFF the rainbow LED strips beside the rail. "
                             "LEDs are ON by default; pass --led to disable.")
    args = parser.parse_args()

    model_short = args.model if args.model else prompt_model_choice()

    n_variants = max(args.cycles - 1, 0)
    if n_variants > 0:
        print(f"\n[Augment] Generating {n_variants} variants via "
              f"{args.variant_model}...")
        try:
            variants = generate_variants(
                original_prompt=args.task, n_variants=n_variants,
                model=resolve_model(args.variant_model),
            )
        except Exception as e:
            print(f"[Augment] Variant generation failed: {e}")
            sys.exit(1)
        if args.auto_accept_variants:
            print(f"[Augment] Auto-accepted {len(variants)} variants.")
        else:
            variants = preview_and_confirm(args.task, variants)
            if not variants:
                print("[Augment] Variants rejected - exiting."); return
    else:
        variants = []

    base_seed = args.seed if args.seed is not None else int(time.time())
    print(f"\n[Augment] Base seed: {base_seed}")
    print(f"[Augment] Cycles: {args.cycles}")
    print(f"[Augment] Robot model: {model_short}")
    print(f"[Augment] Pos jitter: +/-{args.pos_jitter_mm}mm  "
          f"Rot jitter: +/-{args.rot_jitter_deg} deg")

    parent_id = uuid.uuid4().hex[:8]
    all_prompts = [args.task] + variants[:args.cycles - 1]
    saved_dirs = []

    for i, prompt in enumerate(all_prompts, start=1):
        cycle_seed = base_seed + i * 1000
        saved = run_one_cycle(
            task_prompt=prompt, cycle_index=i,
            parent_session_id=parent_id, original_prompt=args.task,
            model_short=model_short,
            pos_jitter_mm=args.pos_jitter_mm,
            rot_jitter_deg=args.rot_jitter_deg,
            seed=cycle_seed,
            render=not args.no_render,
            no_record=args.no_record,
            dry_run=args.dry_run,
            speed_tier_override=args.speed_tier,
            led_enabled=args.led_enabled,
        )
        saved_dirs.append(saved)

    if not args.no_record and not args.no_annotate:
        batch_annotate(saved_dirs)

    n = sum(1 for d in saved_dirs if d)
    print(f"\nOK {n} sessions saved.")
    print(f"  Parent ID: {parent_id}")
    print(f"  Inspect: python replay.py")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
