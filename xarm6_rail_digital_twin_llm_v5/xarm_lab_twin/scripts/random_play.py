# scripts/random_play.py
"""
Random-play: sample diverse reachable poses, validate (IK + collision),
and execute. Pure data-generation routine -- no LLM in the motion loop, since
random sampling is more efficient than asking Claude for random coordinates.

Each episode does K random Cartesian targets. The Recorder logs the full
trajectory at 60Hz, the dispatched commands, and the EE pose. Output is the
same format as LLM-driven recordings, so it slots into the same VLA training
pipeline.

Usage:
  python scripts/random_play.py --episodes 10 --moves-per-episode 8 --save-all
  python scripts/random_play.py --episodes 5 --eval --seed 42
"""
import argparse
import os
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()

from recording import Recorder


# Workspace bounds (mm) -- conservative reachable region above the bench
# that avoids the bins (y >= 350 dropped to keep arm clear unless on a rail offset)
WORKSPACE = {
    "rail_mm": (80.0,  620.0),
    "x_mm":    (-280.0, 280.0),
    "y_mm":    (60.0,   340.0),
    "z_mm":    (820.0, 1200.0),
}
ORIENTATION_RPY_DEG = (180.0, 0.0, 0.0)  # consistent grasp-down for valid IK
MAX_SAMPLES_PER_MOVE = 15                 # how hard to try before giving up


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


def sample_pose(rng: np.random.Generator) -> dict:
    return {
        "rail_mm": float(rng.uniform(*WORKSPACE["rail_mm"])),
        "x_mm":    float(rng.uniform(*WORKSPACE["x_mm"])),
        "y_mm":    float(rng.uniform(*WORKSPACE["y_mm"])),
        "z_mm":    float(rng.uniform(*WORKSPACE["z_mm"])),
    }


def execute_random_episode(arm, recorder, n_moves: int,
                           rng: np.random.Generator) -> dict:
    """Pick n_moves random reachable poses and drive the arm to each.
    Skips poses where IK fails or validation reports a collision."""
    n_attempted = n_succeeded = n_skipped = 0

    for move_idx in range(n_moves):
        sampled = None
        for _ in range(MAX_SAMPLES_PER_MOVE):
            p = sample_pose(rng)
            # Set rail first (it changes the arm base x and therefore reachability)
            arm.set_rail_position(position_mm=p["rail_mm"], wait=True)
            # Try IK + validation via the public API (which already wires them
            # up). set_position returns 0 on success, 1 on IK failure,
            # 2 on validation collision.
            ret = arm.set_position(x=p["x_mm"], y=p["y_mm"], z=p["z_mm"],
                                   roll=ORIENTATION_RPY_DEG[0],
                                   pitch=ORIENTATION_RPY_DEG[1],
                                   yaw=ORIENTATION_RPY_DEG[2],
                                   wait=True)
            n_attempted += 1
            if ret == 0:
                sampled = p
                n_succeeded += 1
                recorder.log_command("random_move", {
                    "move_idx": move_idx, **p,
                    "roll": ORIENTATION_RPY_DEG[0],
                    "pitch": ORIENTATION_RPY_DEG[1],
                    "yaw": ORIENTATION_RPY_DEG[2],
                })
                break
        else:
            n_skipped += 1
            print(f"    move {move_idx+1}: no valid pose after "
                  f"{MAX_SAMPLES_PER_MOVE} tries -- skipping")

    return {
        "n_moves_target": n_moves,
        "n_attempted_samples": n_attempted,
        "n_succeeded": n_succeeded,
        "n_skipped": n_skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes (default 5)")
    parser.add_argument("--moves-per-episode", type=int, default=8,
                        help="Random move targets per episode (default 8)")
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument("--eval", action="store_true",
                            help="Prompt s/f/d between episodes")
    eval_group.add_argument("--save-all", action="store_true",
                            help="Save every episode without prompting (default)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random sampler (reproducible runs)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Record image frames at 10Hz per episode (off by default).")
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else int(time.time())
    print(f"\n[RandomPlay] {args.episodes} episodes x {args.moves_per_episode} "
          f"moves each.")
    print(f"[RandomPlay] Base seed: {base_seed}")
    print(f"[RandomPlay] Workspace (mm): rail={WORKSPACE['rail_mm']}, "
          f"x={WORKSPACE['x_mm']}, y={WORKSPACE['y_mm']}, z={WORKSPACE['z_mm']}")

    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=True)
    run_id = uuid.uuid4().hex[:8]
    print(f"[RandomPlay] Run ID: {run_id}")

    saved_dirs = []
    for i in range(1, args.episodes + 1):
        print(f"\n{'=' * 70}\nEpisode {i}/{args.episodes}\n{'=' * 70}")
        print(f"  [RandomPlay] returning to home pose...")
        arm.reset_scene()
        time.sleep(1.5)   # Pause at home so the visual transition is obvious

        recorder = Recorder(
            arm.model, arm.data, arm.lock,
            interface="random_play",
            scene_xml="envs/lab_scene.xml",
            enable_frames=args.save_frames,
        )
        recorder.start()
        recorder.session.parent_session_id = run_id
        recorder.session.cycle_index = i
        recorder.session.task_label = f"random_play_e{i}"

        # Per-episode seed so each episode is reproducible from base_seed+i
        rng = np.random.default_rng(base_seed + i * 1000)
        stats = execute_random_episode(arm, recorder,
                                       args.moves_per_episode, rng)
        recorder.session.augmentation_config = {
            "mode": "random_play",
            "base_seed": base_seed,
            "episode_seed": base_seed + i * 1000,
            "moves_per_episode": args.moves_per_episode,
            **stats,
        }
        print(f"  {stats['n_succeeded']}/{stats['n_moves_target']} moves "
              f"executed ({stats['n_attempted_samples']} samples), "
              f"{stats['n_skipped']} skipped")

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
    print(f"\n[RandomPlay] Done. {len(saved_dirs)} episodes saved.")
    print(f"             Parent run id: {run_id}")
    print(f"             Inspect any with: python replay.py <index>")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
