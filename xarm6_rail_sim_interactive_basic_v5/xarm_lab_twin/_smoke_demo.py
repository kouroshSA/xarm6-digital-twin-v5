"""Quick visual smoke test: launches viewer, drives arm through presets while
recording, then saves the session. No interactive prompts."""
import threading
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from recording import Recorder

SCENE_XML = "envs/basic_scene.xml"
ACT_NAMES = ["act_rail", "act1", "act2", "act3", "act4", "act5", "act6"]

PRESETS = [
    ("home",            [0.35, 0.0, 0.0,                0.0, 0.0,              0.0, 0.0]),
    ("above red cube",  [0.15, 0.0, np.deg2rad(20),     0.0, np.deg2rad(60),  0.0, np.deg2rad(20)]),
    ("above blue cube", [0.55, 0.0, np.deg2rad(20),     0.0, np.deg2rad(60),  0.0, np.deg2rad(20)]),
    ("above blue bin",  [0.55, 0.0, np.deg2rad(-10),    0.0, np.deg2rad(50),  0.0, np.deg2rad(50)]),
    ("home",            [0.35, 0.0, 0.0,                0.0, 0.0,              0.0, 0.0]),
]

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)
lock  = threading.Lock()
running = [True]
act_ids = [model.actuator(n).id for n in ACT_NAMES]

# Initial home pose
with lock:
    for i, v in enumerate(PRESETS[0][1]):
        data.ctrl[act_ids[i]] = v

def sim_loop():
    while running[0]:
        with lock:
            mujoco.mj_step(model, data)
        time.sleep(0.002)

threading.Thread(target=sim_loop, daemon=True).start()

recorder = Recorder(model, data, lock,
                    interface="smoke_demo", scene_xml=SCENE_XML)

def drive():
    time.sleep(1.0)  # let viewer settle
    recorder.start()
    recorder.session.task_label = "smoke_demo_preset_tour"
    for name, vals in PRESETS:
        print(f"  -> {name}")
        recorder.log_command("preset", {"name": name})
        with lock:
            for i, v in enumerate(vals):
                data.ctrl[act_ids[i]] = v
        # Hold long enough for motion to settle visually
        time.sleep(2.5)
    # Stop and save without interactive prompt
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
    recorder._session.kept = True
    recorder._write_metadata()
    print(f"\nSaved: {recorder._session_dir}")
    print("Close the viewer window to exit.")

threading.Thread(target=drive, daemon=True).start()

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        with lock:
            v.sync()
        time.sleep(0.016)

running[0] = False
print("Done.")
