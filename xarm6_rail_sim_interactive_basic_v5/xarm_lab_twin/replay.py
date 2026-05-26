# replay.py
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from recording import (
    RECORDINGS_ROOT, TRASH_DIR,
    soft_delete_session, restore_session, purge_trash,
)

SCENE_XML = "envs/basic_scene.xml"


def list_sessions(include_trash: bool = False, filter_outcome: str = None):
    paths = []
    if RECORDINGS_ROOT.exists():
        for d in sorted(RECORDINGS_ROOT.iterdir()):
            if d.is_dir() and d.name != "trash":
                paths.append((d, False))
    if include_trash and TRASH_DIR.exists():
        for d in sorted(TRASH_DIR.iterdir()):
            if d.is_dir():
                paths.append((d, True))

    sessions = []
    for path, in_trash in paths:
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}
        if filter_outcome and meta.get("outcome", "") != filter_outcome:
            continue
        sessions.append((path, meta, in_trash))
    return sessions


def print_session_list(sessions):
    if not sessions:
        print("No sessions found."); return
    print(f"\n{'#':>3}  {'status':>7}  {'name':<48}  task                   outcome    dur")
    print("-" * 120)
    for i, (path, meta, in_trash) in enumerate(sessions):
        tag = "TRASH" if in_trash else ("kept" if meta.get("kept") else "draft")
        task     = (meta.get("task_label") or "")[:22]
        outcome  = (meta.get("outcome") or "")[:10]
        duration = meta.get("duration_s", 0)
        cycle    = meta.get("cycle_index", 0)
        cycle_str = f" c{cycle}" if cycle else ""
        print(f"[{i:>2d}] {tag:>7}  {path.name:<48}  {task:<22} {outcome:<10} {duration:>5.1f}s{cycle_str}")
    print()


def resolve_session(arg: str, sessions: list) -> Path:
    if arg is None:
        return None
    if arg.isdigit():
        idx = int(arg)
        if 0 <= idx < len(sessions):
            return sessions[idx][0]
    p = Path(arg)
    if p.is_dir():
        return p
    for candidate_root in (RECORDINGS_ROOT, TRASH_DIR):
        p2 = candidate_root / arg
        if p2.is_dir():
            return p2
    return None


def replay_trajectory(session_dir: Path, speed: float = 1.0, loop: bool = False):
    traj_path = session_dir / "trajectory.h5"
    if not traj_path.exists():
        print(f"No trajectory.h5 in {session_dir}"); return
    with h5py.File(traj_path, "r") as f:
        rail_mm    = f["rail_mm"][:]
        joints_deg = f["joints_deg"][:]
        t_wall     = f["t_wall"][:]
        n_samples  = int(f.attrs["n_samples"])
    if n_samples == 0:
        print("Empty trajectory."); return

    print(f"\nReplaying {session_dir.name}  ({n_samples} samples, "
          f"{t_wall[-1]:.1f}s, {speed}x)")
    print("  Press Esc in the viewer to stop.\n")

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    lock  = threading.Lock()
    act_ids = [model.actuator(n).id for n in
               ["act_rail","act1","act2","act3","act4","act5","act6"]]

    def play():
        while True:
            t0 = time.time()
            for i in range(n_samples):
                target_t = t_wall[i] / speed
                wait = target_t - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
                with lock:
                    data.ctrl[act_ids[0]] = rail_mm[i] / 1000.0
                    for j in range(6):
                        data.ctrl[act_ids[1 + j]] = np.deg2rad(joints_deg[i, j])
            if not loop:
                break
            t0 = time.time()

    threading.Thread(target=play, daemon=True).start()

    def sim_loop():
        while True:
            with lock:
                mujoco.mj_step(model, data)
            time.sleep(0.002)
    threading.Thread(target=sim_loop, daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            with lock:
                v.sync()
            time.sleep(0.016)


def confirm(prompt: str, magic_word: str) -> bool:
    try:
        ans = input(f"{prompt}\nType '{magic_word}' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans == magic_word


def main():
    parser = argparse.ArgumentParser(description="Playback and manage recorded sessions.")
    parser.add_argument("session", nargs="?",
                        help="Index, name, or path. Omit to list.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--include-trash", action="store_true")
    parser.add_argument("--outcome",
                        help="Filter by outcome: success / failure")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--restore",
                        help="Restore a named session from trash")
    parser.add_argument("--purge-trash", action="store_true")
    args = parser.parse_args()

    if args.purge_trash:
        if not TRASH_DIR.exists():
            print("Trash is empty."); return
        n = sum(1 for d in TRASH_DIR.iterdir() if d.is_dir())
        if n == 0:
            print("Trash is empty."); return
        if confirm(f"Permanently delete {n} session(s)?", magic_word="purge"):
            count = purge_trash()
            print(f"Purged {count}.")
        else:
            print("Cancelled.")
        return

    if args.restore is not None:
        restore_session(args.restore); return

    sessions = list_sessions(include_trash=args.include_trash,
                             filter_outcome=args.outcome)

    if args.session is None:
        print_session_list(sessions); return

    target = resolve_session(args.session, sessions)
    if target is None:
        print(f"Could not resolve: {args.session}")
        print_session_list(sessions); sys.exit(1)

    if args.delete:
        if confirm(f"Move '{target.name}' to trash?", magic_word="delete"):
            soft_delete_session(target)
        else:
            print("Cancelled.")
        return

    replay_trajectory(target, speed=args.speed, loop=args.loop)

    print()
    try:
        ans = input("Delete this session? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans == "y":
        if confirm(f"Move '{target.name}' to trash?", magic_word="delete"):
            soft_delete_session(target)


if __name__ == "__main__":
    main()
