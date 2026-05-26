# realtime_keyboard.py
import threading
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

from recording import Recorder, JOINT_NAMES, ACT_NAMES

SCENE_XML = "envs/basic_scene.xml"

SPEED_LEVELS = {
    1: {"joint_deg_s":   2.0, "rail_mm_s":   5.0, "label": "very slow"},
    2: {"joint_deg_s":   5.0, "rail_mm_s":  10.0, "label": "slow"},
    3: {"joint_deg_s":  10.0, "rail_mm_s":  20.0, "label": "moderate"},
    4: {"joint_deg_s":  20.0, "rail_mm_s":  40.0, "label": "moderate+"},
    5: {"joint_deg_s":  30.0, "rail_mm_s":  60.0, "label": "default"},
    6: {"joint_deg_s":  45.0, "rail_mm_s":  90.0, "label": "fast"},
    7: {"joint_deg_s":  60.0, "rail_mm_s": 120.0, "label": "faster"},
    8: {"joint_deg_s":  90.0, "rail_mm_s": 180.0, "label": "very fast"},
    9: {"joint_deg_s": 120.0, "rail_mm_s": 250.0, "label": "max"},
}
DEFAULT_SPEED_LEVEL = 5
CONTROL_HZ = 100.0
DT = 1.0 / CONTROL_HZ


@dataclass
class KeyState:
    rail:    int = 0
    joint1:  int = 0
    joint2:  int = 0
    joint3:  int = 0
    joint4:  int = 0
    joint5:  int = 0
    joint6:  int = 0


class RealtimeKeyboardController:

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()
        self._running = True

        self.joint_ids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = self.model.joint("rail").id
        self.act_ids   = [self.model.actuator(n).id for n in ACT_NAMES]

        self.keys = KeyState()
        self.speed_level = DEFAULT_SPEED_LEVEL
        self.modifier_shift = False
        self.modifier_alt   = False
        self.modifier_ctrl  = False

        self.recorder = Recorder(
            self.model, self.data, self.lock,
            interface="realtime_keyboard",
            scene_xml=SCENE_XML,
        )

        # Home: rail at 350mm, joints at zero
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
            next_t += DT
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    def _apply_held(self):
        level = SPEED_LEVELS[self.speed_level]
        joint_step_rad = np.deg2rad(level["joint_deg_s"]) * DT
        rail_step_m    = (level["rail_mm_s"] / 1000.0) * DT
        with self.lock:
            if self.keys.rail != 0:
                cur = self.data.ctrl[self.act_ids[0]]
                new = np.clip(cur + self.keys.rail * rail_step_m, 0.0, 0.7)
                self.data.ctrl[self.act_ids[0]] = float(new)
            for i, attr in enumerate(
                ["joint1","joint2","joint3","joint4","joint5","joint6"],
                start=1
            ):
                direction = getattr(self.keys, attr)
                if direction == 0:
                    continue
                cur = self.data.ctrl[self.act_ids[i]]
                lo, hi = self.model.jnt_range[self.joint_ids[i-1]]
                new = np.clip(cur + direction * joint_step_rad, lo, hi)
                self.data.ctrl[self.act_ids[i]] = float(new)

    def _is_arrow(self, key):
        return key in (keyboard.Key.up, keyboard.Key.down,
                       keyboard.Key.left, keyboard.Key.right)

    def _handle_arrow(self, key, pressed: bool):
        if self.modifier_shift:
            vert_attr, horz_attr = "joint4", "joint3"
        elif self.modifier_alt:
            vert_attr, horz_attr = "joint6", "joint5"
        else:
            vert_attr, horz_attr = "joint2", "joint1"

        attr = None; direction = 0
        if key == keyboard.Key.up:    attr, direction = vert_attr, +1
        elif key == keyboard.Key.down:  attr, direction = vert_attr, -1
        elif key == keyboard.Key.right: attr, direction = horz_attr, +1
        elif key == keyboard.Key.left:  attr, direction = horz_attr, -1
        if attr is None:
            return

        new_dir = direction if pressed else 0
        setattr(self.keys, attr, new_dir)

        self.recorder.log_command(
            "arrow_press" if pressed else "arrow_release",
            {"key": str(key).split(".")[-1].strip("'"),
             "modifier": ("shift" if self.modifier_shift
                          else "alt" if self.modifier_alt else "none"),
             "target_joint": attr,
             "direction": new_dir}
        )

    def _on_press(self, key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.modifier_shift = True; return
        if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.modifier_alt = True; return
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.modifier_ctrl = True; return

        if self.modifier_ctrl and isinstance(key, keyboard.KeyCode) and key.char:
            if key.char in "123456789":
                lvl = int(key.char)
                self.speed_level = lvl
                info = SPEED_LEVELS[lvl]
                print(f"  [speed] level {lvl} - {info['label']}  "
                      f"({info['joint_deg_s']} deg/s joints, "
                      f"{info['rail_mm_s']}mm/s rail)")
                self.recorder.log_command("speed_change", {"level": lvl})
                return

        if self._is_arrow(key):
            self._handle_arrow(key, pressed=True); return

        if isinstance(key, keyboard.KeyCode):
            if key.char == "]":
                self.keys.rail = +1
                self.recorder.log_command("rail_press", {"dir": +1}); return
            if key.char == "[":
                self.keys.rail = -1
                self.recorder.log_command("rail_press", {"dir": -1}); return
            c = (key.char or "").lower()
            if c == "h":
                self._go_home(); return
            if c == "r":
                self._toggle_record(); return

        if key == keyboard.Key.space:
            self._all_stop(); return

        if key == keyboard.Key.esc:
            print("\n[realtime_keyboard] Escape pressed - shutting down...")
            self._running = False
            return False

    def _on_release(self, key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.modifier_shift = False; return
        if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
            self.modifier_alt = False; return
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.modifier_ctrl = False; return
        if self._is_arrow(key):
            self._handle_arrow(key, pressed=False); return
        if isinstance(key, keyboard.KeyCode) and key.char in ("[", "]"):
            self.keys.rail = 0
            self.recorder.log_command("rail_release", {}); return

    def _go_home(self):
        print("  [home] rail -> 350mm, all joints -> zero")
        with self.lock:
            self.data.ctrl[self.act_ids[0]] = 0.35
            for i in range(6):
                self.data.ctrl[self.act_ids[1 + i]] = 0.0
        self.recorder.log_command("home", {})

    def _all_stop(self):
        self.keys = KeyState()
        print("  [all-stop] released all DOFs")
        self.recorder.log_command("all_stop", {})

    def _toggle_record(self):
        if self.recorder.is_recording:
            print("  [recording] stopping...")
            self.recorder.stop_and_prompt(prompt=True)
        else:
            self.recorder.start()

    def run(self):
        self._print_help()
        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.start()
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as v:
                while v.is_running() and self._running:
                    with self.lock:
                        v.sync()
                    time.sleep(0.016)
        finally:
            self._running = False
            listener.stop()
            if self.recorder.is_recording:
                print("\n[realtime_keyboard] Auto-stopping recording...")
                self.recorder.stop_and_prompt(prompt=True)

    def _print_help(self):
        lvl = SPEED_LEVELS[self.speed_level]
        print("\n" + "=" * 64)
        print(" xArm6 + Rail - Realtime Keyboard Control")
        print("=" * 64)
        print(" Arrow keys      -> J1 (left/right)  J2 (up/down)")
        print(" Shift + arrows  -> J3 (left/right)  J4 (up/down)")
        print(" Alt   + arrows  -> J5 (left/right)  J6 (up/down)")
        print(" [ / ]           -> Rail backward / forward")
        print(" Ctrl + 1...9    -> Set speed level")
        print(" Space           -> All-stop")
        print(" H               -> Home pose")
        print(" R               -> Toggle recording")
        print(" Esc             -> Quit")
        print("-" * 64)
        print(f" Speed: level {self.speed_level} ({lvl['label']}) - "
              f"{lvl['joint_deg_s']} deg/s joints, {lvl['rail_mm_s']}mm/s rail")
        print("=" * 64 + "\n")


def main():
    ctrl = RealtimeKeyboardController()
    ctrl.run()
    print("\n[realtime_keyboard] Done.")


if __name__ == "__main__":
    main()
