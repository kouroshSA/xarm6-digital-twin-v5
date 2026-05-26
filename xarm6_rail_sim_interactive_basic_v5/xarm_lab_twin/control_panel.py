# control_panel.py
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import ttk

try:
    from transforms3d.euler import mat2euler
    HAS_TRANSFORMS3D = True
except ImportError:
    HAS_TRANSFORMS3D = False
    print("[Warning] pip install transforms3d for RPY readout")

from recording import Recorder

SCENE_XML = "envs/basic_scene.xml"

ACT_NAMES  = ["act_rail", "act1", "act2", "act3", "act4", "act5", "act6"]
DOF_LABELS = [
    "Rail     (mm)",
    "J1  base rot  (deg)",
    "J2  shoulder  (deg)",
    "J3  upper arm (deg)",
    "J4  elbow     (deg)",
    "J5  forearm   (deg)",
    "J6  wrist     (deg)",
]

PRESETS = {
    "Home":            [350,   0,   0,   0,   0,   0,   0],
    "Above red cube":  [150,   0,  20,   0,  60,   0,  20],
    "Above green cube":[350,   0,  20,   0,  60,   0,  20],
    "Above blue cube": [550,   0,  20,   0,  60,   0,  20],
    "Above red bin":   [150,   0, -10,   0,  50,   0,  50],
    "Above green bin": [350,   0, -10,   0,  50,   0,  50],
    "Above blue bin":  [550,   0, -10,   0,  50,   0,  50],
    "Rail start":      [  0,   0,   0,   0,   0,   0,   0],
    "Rail end":        [700,   0,   0,   0,   0,   0,   0],
}


def dof_to_ctrl(dof_idx, v):
    return v / 1000.0 if dof_idx == 0 else np.deg2rad(v)


def ctrl_to_dof(dof_idx, c):
    return c * 1000.0 if dof_idx == 0 else np.rad2deg(c)


class SimController:
    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()
        self._running = True
        self.act_ids = [self.model.actuator(n).id for n in ACT_NAMES]
        self.ee_site = self.model.site("end_effector").id
        threading.Thread(target=self._sim_loop, daemon=True).start()

    def _sim_loop(self):
        while self._running:
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)

    def set_dof(self, dof_idx, display_value):
        with self.lock:
            self.data.ctrl[self.act_ids[dof_idx]] = dof_to_ctrl(dof_idx, display_value)

    def set_all_dofs(self, vals):
        with self.lock:
            for i, v in enumerate(vals):
                self.data.ctrl[self.act_ids[i]] = dof_to_ctrl(i, v)

    def get_all_dofs_display(self):
        with self.lock:
            return [ctrl_to_dof(i, self.data.ctrl[self.act_ids[i]])
                    for i in range(len(ACT_NAMES))]

    def get_ee_pose(self):
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site].copy()
            mat = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
        rpy = np.rad2deg(mat2euler(mat, axes='sxyz')) if HAS_TRANSFORMS3D else None
        return {"pos_m": pos, "rpy_deg": rpy}

    def get_rail_mm(self):
        with self.lock:
            return self.data.ctrl[self.act_ids[0]] * 1000.0

    def stop(self):
        self._running = False


class ControlPanel:

    def __init__(self, controller: SimController):
        self.ctrl = controller
        self.recorder = Recorder(
            controller.model, controller.data, controller.lock,
            interface="control_panel", scene_xml=SCENE_XML
        )
        self.root = tk.Tk()
        self.root.title("xArm6 + Rail Sim - Manual Control")
        self.root.resizable(False, False)
        self._slider_vars  = []
        self._suppress_cmd = False
        self._build_ui()
        self._poll_fk()
        self._poll_rec_status()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        slider_frame = ttk.LabelFrame(
            self.root, text="DOF control  [Rail in mm | Joints in degrees]")
        slider_frame.pack(fill="x", **pad)

        display_limits = [
            (0.0,    700.0),
            (-180.0, 180.0), (-120.0, 120.0), (-180.0,  70.0),
            (-110.0, 110.0), (-150.0, 150.0), (-110.0, 110.0),
        ]
        for i, (label, (lo, hi)) in enumerate(zip(DOF_LABELS, display_limits)):
            if i == 1:
                ttk.Separator(slider_frame, orient="horizontal").pack(
                    fill="x", padx=6, pady=2)
            row = ttk.Frame(slider_frame); row.pack(fill="x", padx=6, pady=2)
            color = "darkblue" if i == 0 else "black"
            ttk.Label(row, text=label, width=20, anchor="w",
                      foreground=color).pack(side="left")
            var = tk.DoubleVar(value=350.0 if i == 0 else 0.0)
            self._slider_vars.append(var)
            slider = ttk.Scale(
                row, from_=lo, to=hi, orient="horizontal",
                variable=var, length=300,
                command=lambda val, idx=i: self._on_slider(idx, float(val))
            )
            slider.pack(side="left", padx=4)
            unit = "mm" if i == 0 else "deg"
            val_lbl = ttk.Label(row, text=f"  0{unit}", width=8, anchor="e")
            val_lbl.pack(side="left")
            var.trace_add("write", lambda *_, lbl=val_lbl, v=var, u=unit:
                          lbl.config(text=f"{v.get():+7.1f}{u}"))
            ttk.Label(row, text=f"[{lo:.0f} ... {hi:.0f}]",
                      foreground="gray",
                      font=("TkDefaultFont", 8)).pack(side="left", padx=4)

        fk_frame = ttk.LabelFrame(self.root, text="End-effector position (FK)")
        fk_frame.pack(fill="x", **pad)
        self._fk_labels = {}
        fk_inner = ttk.Frame(fk_frame); fk_inner.pack(pady=4)
        for key, unit in [("X","m"),("Y","m"),("Z","m"),
                          ("Roll","deg"),("Pitch","deg"),("Yaw","deg")]:
            col = ttk.Frame(fk_inner); col.pack(side="left", padx=12)
            ttk.Label(col, text=f"{key} ({unit})",
                      font=("TkDefaultFont", 9, "bold")).pack()
            lbl = ttk.Label(col, text="-", font=("TkFixedFont", 11)); lbl.pack()
            self._fk_labels[key] = lbl
        rail_frame = ttk.Frame(fk_frame); rail_frame.pack(pady=(0, 4))
        ttk.Label(rail_frame, text="Rail position:",
                  font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=6)
        self._rail_label = ttk.Label(rail_frame, text="- mm",
                                     font=("TkFixedFont", 11),
                                     foreground="darkblue")
        self._rail_label.pack(side="left")

        preset_frame = ttk.LabelFrame(self.root, text="Preset poses")
        preset_frame.pack(fill="x", **pad)
        names = list(PRESETS.keys())
        row1 = ttk.Frame(preset_frame); row1.pack(padx=6, pady=2)
        row2 = ttk.Frame(preset_frame); row2.pack(padx=6, pady=2)
        row3 = ttk.Frame(preset_frame); row3.pack(padx=6, pady=2)
        for name in names[:3]:
            ttk.Button(row1, text=name,
                       command=lambda n=name: self._go_preset(n)).pack(side="left", padx=3)
        for name in names[3:6]:
            ttk.Button(row2, text=name,
                       command=lambda n=name: self._go_preset(n)).pack(side="left", padx=3)
        for name in names[6:]:
            ttk.Button(row3, text=name,
                       command=lambda n=name: self._go_preset(n)).pack(side="left", padx=3)

        rec_frame = ttk.LabelFrame(self.root, text="Recording")
        rec_frame.pack(fill="x", **pad)
        rec_inner = ttk.Frame(rec_frame); rec_inner.pack(pady=4)
        self._rec_btn = ttk.Button(
            rec_inner, text="Start recording",
            command=self._toggle_record, width=24
        )
        self._rec_btn.pack(side="left", padx=8)
        self._rec_status = ttk.Label(
            rec_inner, text="(idle)", foreground="gray",
            font=("TkFixedFont", 10)
        )
        self._rec_status.pack(side="left", padx=8)

        act_frame = ttk.Frame(self.root); act_frame.pack(pady=6)
        ttk.Button(act_frame, text="Home",
                   command=self._home).pack(side="left", padx=6)
        ttk.Button(act_frame, text="Sync from sim",
                   command=self._sync).pack(side="left", padx=6)

        ttk.Label(self.root,
                  text="MuJoCo viewer: mouse drag = orbit, right-drag = pan, "
                       "scroll = zoom, F = pause, V = contacts",
                  foreground="gray", font=("TkDefaultFont", 8)
                  ).pack(pady=(0, 6))

    def _on_slider(self, dof_idx, value):
        if self._suppress_cmd:
            return
        self.ctrl.set_dof(dof_idx, value)
        self.recorder.log_command(
            "set_dof", {"dof_idx": dof_idx, "display_value": value}
        )

    def _go_preset(self, name):
        vals = PRESETS[name]
        self.ctrl.set_all_dofs(vals)
        self._suppress_cmd = True
        try:
            for var, v in zip(self._slider_vars, vals):
                var.set(v)
        finally:
            self._suppress_cmd = False
        self.recorder.log_command("preset", {"name": name, "values": vals})

    def _home(self):
        self._go_preset("Home")

    def _sync(self):
        vals = self.ctrl.get_all_dofs_display()
        self._suppress_cmd = True
        try:
            for var, v in zip(self._slider_vars, vals):
                var.set(v)
        finally:
            self._suppress_cmd = False

    def _toggle_record(self):
        if self.recorder.is_recording:
            self._rec_btn.config(text="Start recording")
            self._rec_status.config(text="(saving...)", foreground="orange")
            self.root.update_idletasks()
            def do_stop():
                self.recorder.stop_and_prompt(prompt=True)
                self._rec_status.config(text="(idle)", foreground="gray")
            threading.Thread(target=do_stop, daemon=True).start()
        else:
            self.recorder.start()
            self._rec_btn.config(text="Stop recording")
            self._rec_status.config(text="RECORDING", foreground="red")

    def _poll_rec_status(self):
        if self.recorder.is_recording and self.recorder._session is not None:
            elapsed = time.time() - self.recorder._start_wall_time
            n_cmd = self.recorder._session.n_commands
            self._rec_status.config(
                text=f"REC  {elapsed:.1f}s  {n_cmd} cmds",
                foreground="red"
            )
        self.root.after(500, self._poll_rec_status)

    def _poll_fk(self):
        pose = self.ctrl.get_ee_pose()
        pos = pose["pos_m"]; rpy = pose["rpy_deg"]
        self._fk_labels["X"].config(text=f"{pos[0]:.4f}")
        self._fk_labels["Y"].config(text=f"{pos[1]:.4f}")
        self._fk_labels["Z"].config(text=f"{pos[2]:.4f}")
        if rpy is not None:
            self._fk_labels["Roll"].config( text=f"{rpy[0]:.1f}")
            self._fk_labels["Pitch"].config(text=f"{rpy[1]:.1f}")
            self._fk_labels["Yaw"].config(  text=f"{rpy[2]:.1f}")
        self._rail_label.config(text=f"{self.ctrl.get_rail_mm():.1f} mm")
        self.root.after(100, self._poll_fk)

    def run(self):
        self.root.mainloop()


def main():
    controller = SimController(SCENE_XML)

    def launch_viewer():
        with mujoco.viewer.launch_passive(
                controller.model, controller.data) as v:
            while v.is_running():
                with controller.lock:
                    v.sync()
                time.sleep(0.016)

    threading.Thread(target=launch_viewer, daemon=True).start()
    time.sleep(0.5)
    controller.set_all_dofs(PRESETS["Home"])
    panel = ControlPanel(controller)
    panel.run()
    controller.stop()


if __name__ == "__main__":
    main()
