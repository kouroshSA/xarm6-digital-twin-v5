# recording.py
import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import mujoco


RECORDINGS_ROOT = Path("recordings")
TRASH_DIR = RECORDINGS_ROOT / "trash"
DEFAULT_STATE_HZ = 60.0

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ACT_NAMES   = ["act_rail", "act1", "act2", "act3", "act4", "act5", "act6"]


@dataclass
class SessionMetadata:
    session_id: str
    started_at_iso: str
    ended_at_iso: str = ""
    duration_s: float = 0.0
    interface: str = ""
    task_label: str = ""
    outcome: str = ""
    demonstrator_id: str = ""
    notes: str = ""
    scene_xml: str = ""
    state_hz: float = DEFAULT_STATE_HZ
    n_commands: int = 0
    n_state_samples: int = 0
    kept: bool = False
    parent_session_id: str = ""
    cycle_index: int = 0
    augmentation_config: dict = field(default_factory=dict)


class Recorder:
    """
    Thread-safe recording of MuJoCo state + command events.
    Produces a session folder with metadata.json, commands.jsonl, trajectory.h5.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        lock: threading.Lock,
        interface: str,
        scene_xml: str = "envs/basic_scene.xml",
        state_hz: float = DEFAULT_STATE_HZ,
    ):
        self.model = model
        self.data  = data
        self.lock  = lock
        self.interface = interface
        self.scene_xml = scene_xml
        self.state_hz  = state_hz

        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.act_ids   = [model.actuator(n).id for n in ACT_NAMES]
        self.ee_site   = model.site("end_effector").id

        self._recording = False
        self._session: Optional[SessionMetadata] = None
        self._session_dir: Optional[Path] = None
        self._commands_file = None
        self._state_buffer = []
        self._cmd_lock = threading.Lock()
        self._state_thread: Optional[threading.Thread] = None
        self._start_wall_time = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def session(self) -> Optional[SessionMetadata]:
        return self._session

    def start(self) -> SessionMetadata:
        if self._recording:
            return self._session
        session_id = uuid.uuid4().hex[:8]
        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = RECORDINGS_ROOT / f"{timestamp}_session_{session_id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._session = SessionMetadata(
            session_id=session_id,
            started_at_iso=datetime.now().isoformat(),
            interface=self.interface,
            scene_xml=self.scene_xml,
            state_hz=self.state_hz,
        )

        self._commands_file = open(
            self._session_dir / "commands.jsonl", "w", buffering=1
        )
        self._state_buffer = []
        self._start_wall_time = time.time()
        self._recording = True
        self._state_thread = threading.Thread(
            target=self._state_sampler, daemon=True
        )
        self._state_thread.start()
        print(f"[Recorder] REC  session={session_id}")
        return self._session

    def stop_and_prompt(self, prompt: bool = True,
                        auto_task_label: str = "") -> Optional[Path]:
        if not self._recording:
            return None
        self._recording = False
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)

        self._session.ended_at_iso = datetime.now().isoformat()
        self._session.duration_s = time.time() - self._start_wall_time
        self._session.n_state_samples = len(self._state_buffer)
        if auto_task_label and not self._session.task_label:
            self._session.task_label = auto_task_label

        if self._commands_file is not None:
            self._commands_file.close()
            self._commands_file = None

        self._write_trajectory()
        if prompt:
            self._prompt_metadata()
        self._write_metadata()

        kept = True
        if prompt:
            try:
                ans = input("\nKeep this recording? [Y/n]: ").strip().lower()
                kept = (ans != "n")
            except (EOFError, KeyboardInterrupt):
                kept = True

        self._session.kept = kept
        self._write_metadata()

        if not kept:
            print(f"[Recorder] x Discarded {self._session_dir}")
            self._cleanup_session_dir()
            path = None
        else:
            print(f"[Recorder] OK Saved   {self._session_dir}")
            path = self._session_dir

        self._session = None
        self._session_dir = None
        return path

    def _cleanup_session_dir(self):
        try:
            for f in self._session_dir.glob("*"):
                f.unlink()
            self._session_dir.rmdir()
        except Exception as e:
            print(f"[Recorder] cleanup failed: {e}")

    def log_command(self, event_type: str, payload: dict):
        if not self._recording:
            return
        record = {
            "t": time.time() - self._start_wall_time,
            "type": event_type,
            "payload": payload,
        }
        with self._cmd_lock:
            self._commands_file.write(json.dumps(record) + "\n")
            self._session.n_commands += 1

    def _state_sampler(self):
        period = 1.0 / self.state_hz
        next_t = time.time()
        while self._recording:
            self._sample_one()
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    def _sample_one(self):
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            t_sim    = float(self.data.time)
            rail_m   = float(self.data.qpos[self.rail_jid])
            joints_r = np.array(
                [self.data.qpos[jid] for jid in self.joint_ids],
                dtype=np.float32
            )
            ee_pos   = self.data.site_xpos[self.ee_site].copy().astype(np.float32)
            ee_mat   = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
            ctrl     = np.array(
                [self.data.ctrl[a] for a in self.act_ids],
                dtype=np.float32
            )

        from transforms3d.euler import mat2euler
        ee_rpy = np.array(mat2euler(ee_mat, axes='sxyz'), dtype=np.float32)

        self._state_buffer.append({
            "t_wall":    time.time() - self._start_wall_time,
            "t_sim":     t_sim,
            "rail_mm":   rail_m * 1000.0,
            "joints_deg": np.rad2deg(joints_r),
            "ee_pos_mm": ee_pos * 1000.0,
            "ee_rpy_deg": np.rad2deg(ee_rpy),
            "ctrl":      ctrl,
        })

    def _write_trajectory(self):
        if not self._state_buffer:
            return
        path = self._session_dir / "trajectory.h5"
        b = self._state_buffer
        with h5py.File(path, "w") as f:
            f.create_dataset("t_wall",     data=np.array([s["t_wall"] for s in b], dtype=np.float64), compression="gzip")
            f.create_dataset("t_sim",      data=np.array([s["t_sim"] for s in b], dtype=np.float64), compression="gzip")
            f.create_dataset("rail_mm",    data=np.array([s["rail_mm"] for s in b], dtype=np.float32), compression="gzip")
            f.create_dataset("joints_deg", data=np.stack([s["joints_deg"] for s in b]).astype(np.float32), compression="gzip")
            f.create_dataset("ee_pos_mm",  data=np.stack([s["ee_pos_mm"] for s in b]).astype(np.float32), compression="gzip")
            f.create_dataset("ee_rpy_deg", data=np.stack([s["ee_rpy_deg"] for s in b]).astype(np.float32), compression="gzip")
            f.create_dataset("ctrl",       data=np.stack([s["ctrl"] for s in b]).astype(np.float32), compression="gzip")
            f.attrs["state_hz"] = self.state_hz
            f.attrs["n_samples"] = len(b)
            f.attrs["joint_names"] = JOINT_NAMES
            f.attrs["actuator_names"] = ACT_NAMES
        self._state_buffer = []

    def _write_metadata(self):
        if self._session is None or self._session_dir is None:
            return
        with open(self._session_dir / "metadata.json", "w") as f:
            json.dump(self._session.__dict__, f, indent=2)

    def _prompt_metadata(self):
        print("\n" + "-" * 60)
        print(f"Session {self._session.session_id} ended  "
              f"({self._session.duration_s:.1f}s, "
              f"{self._session.n_commands} commands, "
              f"{self._session.n_state_samples} state samples)")
        if self._session.task_label:
            print(f"Auto task label: '{self._session.task_label}'")
        print("-" * 60)
        print("Optional metadata - press Enter to skip any field.\n")
        try:
            if not self._session.task_label:
                task = input("Task label (e.g. red_cube_to_red_bin): ").strip()
                if task:
                    self._session.task_label = task
            outcome = input("Outcome [s=success / f=failure / blank]: ").strip().lower()
            if outcome.startswith("s"):
                self._session.outcome = "success"
            elif outcome.startswith("f"):
                self._session.outcome = "failure"
            who = input("Demonstrator ID: ").strip()
            if who:
                self._session.demonstrator_id = who
            notes = input("Notes: ").strip()
            if notes:
                self._session.notes = notes
        except (EOFError, KeyboardInterrupt):
            print("\n[Recorder] Metadata prompt aborted.")


# ============================================================
# Soft delete / restore utilities
# ============================================================

def soft_delete_session(session_dir: Path) -> bool:
    if not session_dir.exists() or not session_dir.is_dir():
        print(f"[Recorder] Session not found: {session_dir}")
        return False
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRASH_DIR / session_dir.name
    if dest.exists():
        print(f"[Recorder] Already in trash: {dest}")
        return False
    session_dir.rename(dest)
    print(f"[Recorder] -> trash: {dest}")
    return True


def restore_session(session_name: str) -> bool:
    src = TRASH_DIR / session_name
    if not src.exists():
        print(f"[Recorder] Not in trash: {session_name}")
        return False
    dest = RECORDINGS_ROOT / session_name
    if dest.exists():
        print(f"[Recorder] Cannot restore - {dest} already exists")
        return False
    src.rename(dest)
    print(f"[Recorder] Restored: {dest}")
    return True


def purge_trash() -> int:
    if not TRASH_DIR.exists():
        return 0
    count = 0
    for d in TRASH_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
            count += 1
    return count
