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
    llm_model: str = ""
    llm_prompt: str = ""
    has_llm_log: bool = False
    parent_session_id: str = ""
    cycle_index: int = 0
    augmentation_config: dict = field(default_factory=dict)


class Recorder:

    def __init__(self, model, data, lock, interface, scene_xml="envs/lab_scene.xml",
                 state_hz=DEFAULT_STATE_HZ,
                 enable_frames: bool = False,
                 frame_hz: float = 10.0,
                 frame_width: int = 320,
                 frame_height: int = 240,
                 frame_camera_lookat=(0.0, 0.15, 1.0),
                 frame_camera_distance: float = 2.8,
                 frame_camera_azimuth: float = 135.0,
                 frame_camera_elevation: float = -20.0):
        self.model = model; self.data = data; self.lock = lock
        self.interface = interface; self.scene_xml = scene_xml
        self.state_hz = state_hz
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.act_ids   = [model.actuator(n).id for n in ACT_NAMES]
        self.ee_site   = model.site("end_effector").id

        # Auto-discover free bodies (cubes + tubes etc.) so the recorder
        # works on any scene without hardcoding object names.
        self.free_body_ids = []
        self.free_body_names = []
        for i in range(model.nbody):
            jnt_adr = model.body_jntadr[i]
            if jnt_adr >= 0 and model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_body_ids.append(i)
                self.free_body_names.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                )

        # Equality constraints (welds) — capture activation over time so the
        # downstream pipeline knows when something was being grasped.
        self.eq_ids = list(range(model.neq))
        self.eq_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            for i in self.eq_ids
        ]

        # Optional frame rendering — disabled by default since image data is
        # large. Use enable_frames=True + tune frame_hz/resolution to taste.
        self.enable_frames = enable_frames
        self.frame_hz = frame_hz
        self.frame_width = frame_width
        self.frame_height = frame_height
        # Subsample ratio: render every Nth state sample.
        self._frame_subsample = max(1, int(round(state_hz / max(frame_hz, 1e-6))))
        self._renderer = None
        self._frame_cam = None
        self._frame_buffer = []
        if enable_frames:
            try:
                self._renderer = mujoco.Renderer(
                    model, height=frame_height, width=frame_width
                )
                self._frame_cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(model, self._frame_cam)
                self._frame_cam.lookat[:]   = frame_camera_lookat
                self._frame_cam.distance    = frame_camera_distance
                self._frame_cam.azimuth     = frame_camera_azimuth
                self._frame_cam.elevation   = frame_camera_elevation
            except Exception as e:
                print(f"[Recorder] Frame renderer init failed ({e}) -- frames disabled.")
                print("[Recorder] Tip: on Linux, try setting MUJOCO_GL=egl or "
                      "MUJOCO_GL=osmesa for offscreen rendering when the viewer "
                      "is also using the GL context.")
                self.enable_frames = False
                self._renderer = None

        self._recording = False
        self._session = None; self._session_dir = None
        self._commands_file = None; self._state_buffer = []
        self._cmd_lock = threading.Lock()
        self._state_thread = None; self._start_wall_time = 0.0
        self._sample_count = 0  # used to subsample frames against state samples

    @property
    def is_recording(self): return self._recording
    @property
    def session_dir(self): return self._session_dir
    @property
    def session(self): return self._session

    def start(self):
        if self._recording: return self._session
        sid = uuid.uuid4().hex[:8]
        ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_dir = RECORDINGS_ROOT / f"{ts}_session_{sid}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._session = SessionMetadata(
            session_id=sid, started_at_iso=datetime.now().isoformat(),
            interface=self.interface, scene_xml=self.scene_xml,
            state_hz=self.state_hz,
        )
        self._commands_file = open(self._session_dir / "commands.jsonl",
                                   "w", buffering=1)
        self._state_buffer = []
        self._start_wall_time = time.time()
        self._recording = True
        self._state_thread = threading.Thread(target=self._state_sampler, daemon=True)
        self._state_thread.start()
        print(f"[Recorder] REC  session={sid}")
        return self._session

    def stop_and_prompt(self, prompt=True, auto_task_label=""):
        if not self._recording: return None
        self._recording = False
        if self._state_thread is not None:
            self._state_thread.join(timeout=1.0)
        self._session.ended_at_iso = datetime.now().isoformat()
        self._session.duration_s = time.time() - self._start_wall_time
        self._session.n_state_samples = len(self._state_buffer)
        if auto_task_label and not self._session.task_label:
            self._session.task_label = auto_task_label
        if self._commands_file is not None:
            self._commands_file.close(); self._commands_file = None
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
        self._session = None; self._session_dir = None
        return path

    def _cleanup_session_dir(self):
        try:
            for f in self._session_dir.glob("*"):
                f.unlink()
            self._session_dir.rmdir()
        except Exception as e:
            print(f"[Recorder] cleanup failed: {e}")

    def log_command(self, event_type, payload):
        if not self._recording: return
        record = {
            "t": time.time() - self._start_wall_time,
            "type": event_type, "payload": payload,
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
        rendered_frame = None
        wall_t = time.time() - self._start_wall_time
        do_render = (self.enable_frames and self._renderer is not None and
                     (self._sample_count % self._frame_subsample == 0))

        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            t_sim = float(self.data.time)
            rail_m = float(self.data.qpos[self.rail_jid])
            joints_r = np.array([self.data.qpos[jid] for jid in self.joint_ids],
                                dtype=np.float32)
            ee_pos = self.data.site_xpos[self.ee_site].copy().astype(np.float32)
            ee_mat = self.data.site_xmat[self.ee_site].reshape(3,3).copy()
            ctrl = np.array([self.data.ctrl[a] for a in self.act_ids],
                            dtype=np.float32)

            # Free-body poses (cubes + tubes etc.) — capture full 7D pose
            body_poses = np.zeros((len(self.free_body_ids), 7), dtype=np.float32)
            for k, bid in enumerate(self.free_body_ids):
                body_poses[k, 0:3] = self.data.xpos[bid]
                body_poses[k, 3:7] = self.data.xquat[bid]

            # Equality (weld) activation states
            weld_active = np.array(
                [int(self.data.eq_active[e]) for e in self.eq_ids],
                dtype=np.uint8,
            )

            # Render a frame if it's that sample's turn. Renderer accesses
            # the scene through model/data; safest to do it while holding
            # the sim lock so mj_step doesn't mutate state mid-render.
            if do_render:
                try:
                    self._renderer.update_scene(self.data, camera=self._frame_cam)
                    rendered_frame = self._renderer.render().astype(np.uint8)
                except Exception as e:
                    print(f"[Recorder] frame render failed: {e}")
                    rendered_frame = None

        from transforms3d.euler import mat2euler
        ee_rpy = np.array(mat2euler(ee_mat, axes='sxyz'), dtype=np.float32)
        self._state_buffer.append({
            "t_wall":     wall_t,
            "t_sim":      t_sim,
            "rail_mm":    rail_m * 1000.0,
            "joints_deg": np.rad2deg(joints_r),
            "ee_pos_mm":  ee_pos * 1000.0,
            "ee_rpy_deg": np.rad2deg(ee_rpy),
            "ctrl":       ctrl,
            "body_poses": body_poses,
            "weld_active": weld_active,
        })
        if rendered_frame is not None:
            self._frame_buffer.append({"t_wall": wall_t, "image": rendered_frame})
        self._sample_count += 1

    def _write_trajectory(self):
        if not self._state_buffer: return
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
            # New (per-sample world state of free bodies + weld activations)
            f.create_dataset("body_poses", data=np.stack([s["body_poses"] for s in b]).astype(np.float32), compression="gzip")
            f.create_dataset("weld_active", data=np.stack([s["weld_active"] for s in b]).astype(np.uint8), compression="gzip")
            f.attrs["state_hz"] = self.state_hz
            f.attrs["n_samples"] = len(b)
            f.attrs["joint_names"] = JOINT_NAMES
            f.attrs["actuator_names"] = ACT_NAMES
            # Body / weld metadata so consumers can pair indices with names
            f.attrs["body_names"] = self.free_body_names
            f.attrs["eq_names"]   = self.eq_names

            # Frames (only if enabled and any frames captured)
            if self.enable_frames and self._frame_buffer:
                g = f.create_group("frames")
                imgs = np.stack([fr["image"] for fr in self._frame_buffer]).astype(np.uint8)
                g.create_dataset("images", data=imgs, compression="gzip",
                                 compression_opts=9, chunks=(1, self.frame_height,
                                                              self.frame_width, 3))
                g.create_dataset("t_wall",
                                 data=np.array([fr["t_wall"] for fr in self._frame_buffer],
                                               dtype=np.float64),
                                 compression="gzip")
                g.attrs["frame_hz"] = self.frame_hz
                g.attrs["width"]    = self.frame_width
                g.attrs["height"]   = self.frame_height
                g.attrs["n_frames"] = len(self._frame_buffer)
        self._state_buffer = []
        self._frame_buffer = []

    def _write_metadata(self):
        if self._session is None or self._session_dir is None: return
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
                t = input("Task label: ").strip()
                if t: self._session.task_label = t
            o = input("Outcome [s=success / f=failure / blank]: ").strip().lower()
            if o.startswith("s"): self._session.outcome = "success"
            elif o.startswith("f"): self._session.outcome = "failure"
            w = input("Demonstrator ID: ").strip()
            if w: self._session.demonstrator_id = w
            n = input("Notes: ").strip()
            if n: self._session.notes = n
        except (EOFError, KeyboardInterrupt):
            print("\n[Recorder] Metadata prompt aborted.")


# ============================================================
# LLM-specific session log
# ============================================================

class LLMSessionLog:
    """Captures LLM interaction alongside the Recorder's state/command logs."""

    def __init__(self, recorder: Recorder, model: str, prompt: str):
        self.recorder = recorder
        self.model = model
        self.prompt = prompt
        self._file = None
        self._open()
        recorder.session.llm_model = model
        recorder.session.llm_prompt = prompt
        recorder.session.has_llm_log = True
        if not recorder.session.task_label:
            slug = "_".join(prompt.lower().split()[:5])[:40]
            slug = "".join(c for c in slug if c.isalnum() or c == "_")
            recorder.session.task_label = slug or "llm_task"

    def _open(self):
        if self.recorder.session_dir is None: return
        path = self.recorder.session_dir / "llm_session.jsonl"
        self._file = open(path, "w", buffering=1)

    def log_prompt(self):
        self._write({"event": "user_prompt", "model": self.model, "prompt": self.prompt})

    def log_response(self, raw_text, latency_s, input_tokens=0, output_tokens=0):
        self._write({
            "event": "llm_response", "raw_text": raw_text,
            "latency_s": latency_s,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })

    def log_parsed(self, commands):
        self._write({"event": "parsed_commands", "commands": commands})

    def log_parse_error(self, error):
        self._write({"event": "parse_error", "error": error})

    def log_dispatch(self, action, params, result):
        self._write({"event": "dispatch", "action": action,
                     "params": params, "result": result})

    def close(self):
        if self._file is not None:
            self._file.close(); self._file = None

    def _write(self, record):
        if self._file is None: return
        record["t"] = time.time() - self.recorder._start_wall_time
        self._file.write(json.dumps(record, default=str) + "\n")


# ============================================================
# Soft delete / restore
# ============================================================

def soft_delete_session(session_dir):
    if not session_dir.exists() or not session_dir.is_dir():
        print(f"[Recorder] Session not found: {session_dir}"); return False
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRASH_DIR / session_dir.name
    if dest.exists():
        print(f"[Recorder] Already in trash: {dest}"); return False
    session_dir.rename(dest)
    print(f"[Recorder] -> trash: {dest}")
    return True


def restore_session(session_name):
    src = TRASH_DIR / session_name
    if not src.exists():
        print(f"[Recorder] Not in trash: {session_name}"); return False
    dest = RECORDINGS_ROOT / session_name
    if dest.exists():
        print(f"[Recorder] Cannot restore - {dest} already exists"); return False
    src.rename(dest)
    print(f"[Recorder] Restored: {dest}")
    return True


def purge_trash():
    if not TRASH_DIR.exists(): return 0
    count = 0
    for d in TRASH_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d); count += 1
    return count
