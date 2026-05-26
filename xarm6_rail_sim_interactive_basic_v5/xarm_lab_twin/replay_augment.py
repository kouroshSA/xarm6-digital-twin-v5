# replay_augment.py
import argparse
import json
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from recording import Recorder, RECORDINGS_ROOT, JOINT_NAMES, ACT_NAMES
from envs.scene_randomizer import (
    randomize_scene,
    DEFAULT_POS_JITTER_MM,
    DEFAULT_ROT_JITTER_DEG,
)

DEFAULT_TIMING_JITTER_MS = 50.0
DEFAULT_DURATION_JITTER_PCT = 0.10
SCENE_XML = "envs/basic_scene.xml"


def load_session(session_dir: Path) -> dict:
    meta_path = session_dir / "metadata.json"
    cmd_path  = session_dir / "commands.jsonl"
    traj_path = session_dir / "trajectory.h5"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {session_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    commands = []
    if cmd_path.exists():
        with open(cmd_path) as f:
            for line in f:
                if line.strip():
                    commands.append(json.loads(line))
    duration = meta.get("duration_s", 0.0)
    if traj_path.exists():
        with h5py.File(traj_path, "r") as f:
            if "t_wall" in f and len(f["t_wall"]) > 0:
                duration = max(duration, float(f["t_wall"][-1]))
    return {"metadata": meta, "commands": commands, "duration_s": duration}


def reconstruct_command_timeline(commands: list, timing_jitter_ms: float,
                                 duration_jitter_pct: float,
                                 rng: np.random.Generator) -> list:
    timeline = []
    for c in commands:
        t = float(c.get("t", 0.0))
        if timing_jitter_ms > 0:
            t += rng.uniform(-timing_jitter_ms, timing_jitter_ms) / 1000.0
            t = max(0.0, t)
        timeline.append((t, c.get("type"), c.get("payload", {})))
    timeline.sort(key=lambda x: x[0])
    if duration_jitter_pct > 0:
        timeline = _apply_duration_jitter(timeline, duration_jitter_pct, rng)
        timeline.sort(key=lambda x: x[0])
    return timeline


def _apply_duration_jitter(timeline: list, pct: float,
                           rng: np.random.Generator) -> list:
    new_timeline = list(timeline)
    open_presses = {}
    for i, (t, event_t, payload) in enumerate(new_timeline):
        if event_t == "arrow_press":
            open_presses[payload.get("target_joint")] = (i, t)
        elif event_t == "arrow_release":
            key = payload.get("target_joint")
            if key in open_presses:
                press_i, press_t = open_presses.pop(key)
                duration = t - press_t
                if duration > 0:
                    scale = 1.0 + rng.uniform(-pct, pct)
                    new_timeline[i] = (press_t + duration * scale, event_t, payload)
        elif event_t == "rail_press":
            open_presses["rail"] = (i, t)
        elif event_t == "rail_release":
            if "rail" in open_presses:
                press_i, press_t = open_presses.pop("rail")
                duration = t - press_t
                if duration > 0:
                    scale = 1.0 + rng.uniform(-pct, pct)
                    new_timeline[i] = (press_t + duration * scale, event_t, payload)
    return new_timeline


class ReplayExecutor:
    SPEED_LEVELS = {
        1: {"joint_deg_s":   2.0, "rail_mm_s":   5.0},
        2: {"joint_deg_s":   5.0, "rail_mm_s":  10.0},
        3: {"joint_deg_s":  10.0, "rail_mm_s":  20.0},
        4: {"joint_deg_s":  20.0, "rail_mm_s":  40.0},
        5: {"joint_deg_s":  30.0, "rail_mm_s":  60.0},
        6: {"joint_deg_s":  45.0, "rail_mm_s":  90.0},
        7: {"joint_deg_s":  60.0, "rail_mm_s": 120.0},
        8: {"joint_deg_s":  90.0, "rail_mm_s": 180.0},
        9: {"joint_deg_s": 120.0, "rail_mm_s": 250.0},
    }
    CONTROL_HZ = 100.0
    DT = 1.0 / CONTROL_HZ

    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()
        self._running = True
        self.joint_ids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.act_ids   = [self.model.actuator(n).id for n in ACT_NAMES]
        self.dof_dir = {"rail": 0, "joint1": 0, "joint2": 0, "joint3": 0,
                        "joint4": 0, "joint5": 0, "joint6": 0}
        self.speed_level = 5
        with self.lock:
            self.data.ctrl[self.act_ids[0]] = 0.35
            for i in range(6):
                self.data.ctrl[self.act_ids[1 + i]] = 0.0
        threading.Thread(target=self._sim_loop, daemon=True).start()
        threading.Thread(target=self._control_loop, daemon=True).start()

    def _sim_loop(self):
        while self._running:
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)

    def _control_loop(self):
        next_t = time.time()
        while self._running:
            self._apply_held()
            next_t += self.DT
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    def _apply_held(self):
        level = self.SPEED_LEVELS[self.speed_level]
        joint_step_rad = np.deg2rad(level["joint_deg_s"]) * self.DT
        rail_step_m    = (level["rail_mm_s"] / 1000.0) * self.DT
        with self.lock:
            if self.dof_dir["rail"] != 0:
                cur = self.data.ctrl[self.act_ids[0]]
                new = np.clip(cur + self.dof_dir["rail"] * rail_step_m, 0.0, 0.7)
                self.data.ctrl[self.act_ids[0]] = float(new)
            for i, name in enumerate(JOINT_NAMES, start=1):
                d = self.dof_dir[name]
                if d == 0:
                    continue
                cur = self.data.ctrl[self.act_ids[i]]
                lo, hi = self.model.jnt_range[self.joint_ids[i-1]]
                new = np.clip(cur + d * joint_step_rad, lo, hi)
                self.data.ctrl[self.act_ids[i]] = float(new)

    def execute_timeline(self, timeline: list, recorder: Recorder):
        t_start = time.time()
        idx = 0
        max_t = max((t for t, *_ in timeline), default=0.0)
        deadline = t_start + max_t + 2.0
        while time.time() < deadline and self._running:
            now_t = time.time() - t_start
            while idx < len(timeline) and timeline[idx][0] <= now_t:
                _, event_t, payload = timeline[idx]
                self._apply_event(event_t, payload, recorder)
                idx += 1
            time.sleep(0.005)
        for k in self.dof_dir:
            self.dof_dir[k] = 0

    def _apply_event(self, event_t: str, payload: dict, recorder: Recorder):
        if event_t == "speed_change":
            lvl = int(payload.get("level", 5))
            if lvl in self.SPEED_LEVELS:
                self.speed_level = lvl
        elif event_t == "arrow_press":
            j = payload.get("target_joint")
            if j in self.dof_dir:
                self.dof_dir[j] = int(payload.get("direction", 0))
        elif event_t == "arrow_release":
            j = payload.get("target_joint")
            if j in self.dof_dir:
                self.dof_dir[j] = 0
        elif event_t == "rail_press":
            self.dof_dir["rail"] = int(payload.get("dir", 0))
        elif event_t == "rail_release":
            self.dof_dir["rail"] = 0
        elif event_t == "all_stop":
            for k in self.dof_dir:
                self.dof_dir[k] = 0
        elif event_t == "home":
            with self.lock:
                self.data.ctrl[self.act_ids[0]] = 0.35
                for i in range(6):
                    self.data.ctrl[self.act_ids[1 + i]] = 0.0
        elif event_t == "preset":
            vals = payload.get("values", [])
            if len(vals) == 7:
                with self.lock:
                    self.data.ctrl[self.act_ids[0]] = vals[0] / 1000.0
                    for i in range(6):
                        self.data.ctrl[self.act_ids[1 + i]] = np.deg2rad(vals[i+1])
        elif event_t == "set_dof":
            dof_idx = payload.get("dof_idx")
            v = payload.get("display_value")
            if dof_idx is not None:
                with self.lock:
                    if dof_idx == 0:
                        self.data.ctrl[self.act_ids[0]] = v / 1000.0
                    else:
                        self.data.ctrl[self.act_ids[dof_idx]] = np.deg2rad(v)
        recorder.log_command(f"replayed_{event_t}", payload)

    def stop(self):
        self._running = False


def _stop_recorder_silent(recorder: Recorder, kept: bool) -> Optional[Path]:
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


def run_cycle(parent_meta: dict, timeline: list, cycle_index: int,
              augmentation_config: dict, seed: int, render: bool,
              scene_xml: str) -> Optional[Path]:
    rng = np.random.default_rng(seed)
    executor = ReplayExecutor(scene_xml=scene_xml)

    scene_summary = {}
    if (augmentation_config.get("object_pose_jitter_mm", 0) > 0
        or augmentation_config.get("object_rotation_jitter_deg", 0) > 0
        or augmentation_config.get("initial_joint_jitter_deg", 0) > 0):
        with executor.lock:
            scene_summary = randomize_scene(
                executor.model, executor.data,
                pos_jitter_mm=augmentation_config.get("object_pose_jitter_mm", 0),
                rot_jitter_deg=augmentation_config.get("object_rotation_jitter_deg", 0),
                initial_joint_jitter_deg=augmentation_config.get("initial_joint_jitter_deg", 0),
                rng=rng,
            )

    if render:
        def viewer_loop():
            with mujoco.viewer.launch_passive(executor.model, executor.data) as v:
                while v.is_running() and executor._running:
                    with executor.lock:
                        v.sync()
                    time.sleep(0.016)
        threading.Thread(target=viewer_loop, daemon=True).start()
        time.sleep(0.4)

    recorder = Recorder(
        model=executor.model, data=executor.data, lock=executor.lock,
        interface="replay_augment", scene_xml=scene_xml,
    )
    recorder.start()
    recorder.session.parent_session_id = parent_meta.get("session_id", "")
    recorder.session.cycle_index = cycle_index
    recorder.session.task_label = parent_meta.get("task_label", "")
    aug_logged = dict(augmentation_config)
    aug_logged["seed"] = seed
    aug_logged["scene_perturbations"] = scene_summary
    recorder.session.augmentation_config = aug_logged

    print(f"\n[Cycle {cycle_index}] executing timeline "
          f"({len(timeline)} events)...")
    try:
        executor.execute_timeline(timeline, recorder)
    except KeyboardInterrupt:
        print("\n[Cycle] Interrupted")
    time.sleep(0.5)
    saved_dir = _stop_recorder_silent(recorder, kept=True)
    executor.stop()
    return saved_dir


def batch_annotate(saved_dirs: list):
    print("\n" + "=" * 70)
    print("  BATCH ANNOTATION")
    print("=" * 70)
    print(f"  {sum(1 for d in saved_dirs if d)} cycle recordings created.")
    print("-" * 70)
    for i, d in enumerate(saved_dirs):
        if d is None:
            continue
        meta_path = d / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n[{i+1}/{len(saved_dirs)}] {d.name}  "
              f"cycle: {meta.get('cycle_index','')}  "
              f"duration: {meta.get('duration_s',0):.1f}s")
        try:
            outcome = input("     Outcome [s/f/d/Enter]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if outcome.startswith("d"):
            print(f"     x Deleting {d.name}")
            shutil.rmtree(d); continue
        if outcome.startswith("s"):
            meta["outcome"] = "success"
        elif outcome.startswith("f"):
            meta["outcome"] = "failure"
        try:
            note = input("     Notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
        if note:
            meta["notes"] = note
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def resolve_session(arg: str) -> Optional[Path]:
    p = Path(arg)
    if p.is_dir():
        return p
    p2 = RECORDINGS_ROOT / arg
    if p2.is_dir():
        return p2
    if arg.isdigit():
        sessions = sorted([d for d in RECORDINGS_ROOT.iterdir()
                           if d.is_dir() and d.name != "trash"])
        idx = int(arg)
        if 0 <= idx < len(sessions):
            return sessions[idx]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-record a session N times with augmentation."
    )
    parser.add_argument("session", help="Source session (name, path, or index)")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pos-jitter-mm", type=float, default=DEFAULT_POS_JITTER_MM)
    parser.add_argument("--rot-jitter-deg", type=float, default=DEFAULT_ROT_JITTER_DEG)
    parser.add_argument("--timing-jitter-ms", type=float, default=DEFAULT_TIMING_JITTER_MS)
    parser.add_argument("--duration-jitter-pct", type=float, default=DEFAULT_DURATION_JITTER_PCT)
    parser.add_argument("--joint-jitter-deg", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    args = parser.parse_args()

    source = resolve_session(args.session)
    if source is None:
        print(f"Could not find session: {args.session}")
        sys.exit(1)

    session_data = load_session(source)
    parent_meta = session_data["metadata"]
    print(f"\nSource: {source.name}")
    print(f"  Task: {parent_meta.get('task_label','(none)')}")
    print(f"  Duration: {session_data['duration_s']:.1f}s")
    print(f"  Commands: {len(session_data['commands'])}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Augmentation (applied cycles 2+):")
    print(f"    Object pos:      +/-{args.pos_jitter_mm}mm")
    print(f"    Object rot:      +/-{args.rot_jitter_deg} deg")
    print(f"    Timing:          +/-{args.timing_jitter_ms}ms")
    print(f"    Duration:        +/-{args.duration_jitter_pct*100:.0f}%")
    print(f"    Joint:           +/-{args.joint_jitter_deg} deg")

    base_seed = args.seed if args.seed is not None else int(time.time())
    print(f"  Base seed: {base_seed}")

    try:
        ans = input("\nProceed? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans == "n":
        print("Cancelled."); return

    saved_dirs = []
    for cycle in range(1, args.cycles + 1):
        if cycle == 1:
            aug_config = {
                "object_pose_jitter_mm": 0.0, "object_rotation_jitter_deg": 0.0,
                "command_timing_jitter_ms": 0.0, "command_duration_jitter_pct": 0.0,
                "initial_joint_jitter_deg": 0.0,
            }
            cycle_seed = base_seed
            rng = np.random.default_rng(cycle_seed)
            timeline = reconstruct_command_timeline(
                session_data["commands"], 0.0, 0.0, rng)
        else:
            aug_config = {
                "object_pose_jitter_mm": args.pos_jitter_mm,
                "object_rotation_jitter_deg": args.rot_jitter_deg,
                "command_timing_jitter_ms": args.timing_jitter_ms,
                "command_duration_jitter_pct": args.duration_jitter_pct,
                "initial_joint_jitter_deg": args.joint_jitter_deg,
            }
            cycle_seed = base_seed + cycle * 1000
            rng = np.random.default_rng(cycle_seed)
            timeline = reconstruct_command_timeline(
                session_data["commands"],
                args.timing_jitter_ms, args.duration_jitter_pct, rng)

        saved = run_cycle(
            parent_meta=parent_meta, timeline=timeline, cycle_index=cycle,
            augmentation_config=aug_config, seed=cycle_seed,
            render=not args.no_render,
            scene_xml=parent_meta.get("scene_xml", SCENE_XML),
        )
        saved_dirs.append(saved)
        print(f"[Cycle {cycle}] saved: {saved.name if saved else '(failed)'}")

    if not args.no_annotate:
        batch_annotate(saved_dirs)

    n = sum(1 for d in saved_dirs if d)
    print(f"\nOK {n} sessions saved. Inspect with: python replay.py")


if __name__ == "__main__":
    main()
