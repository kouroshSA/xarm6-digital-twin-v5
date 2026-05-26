# terminal_control.py
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time

SCENE_XML   = "envs/basic_scene.xml"
JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]
ACT_NAMES   = ["act_rail","act1","act2","act3","act4","act5","act6"]

model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)
lock  = threading.Lock()

act_ids   = [model.actuator(n).id for n in ACT_NAMES]
joint_ids = [model.joint(n).id for n in JOINT_NAMES]
SITE_ID   = model.site("end_effector").id

PRESETS = {
    "home":         [350,   0,   0,   0,   0,   0,   0],
    "red_cube":     [150,   0,  20,   0,  60,   0,  20],
    "green_cube":   [350,   0,  20,   0,  60,   0,  20],
    "blue_cube":    [550,   0,  20,   0,  60,   0,  20],
    "red_bin":      [150,   0, -10,   0,  50,   0,  50],
    "green_bin":    [350,   0, -10,   0,  50,   0,  50],
    "blue_bin":     [550,   0, -10,   0,  50,   0,  50],
}

HELP = """
Commands:
  rail <mm>                     - Move rail (0-700mm)
  j <1-6> <angle_deg>           - Set one joint angle
  all <rail_mm> <j1..j6_deg>    - Set all 7 DOFs at once
  preset <name>                 - Load preset (home/red_cube/green_cube/etc.)
  fk                            - Print current pose
  home                          - Home position
  presets                       - List preset names
  help                          - Show this message
  quit                          - Exit
"""


def apply_all(vals):
    with lock:
        data.ctrl[act_ids[0]] = vals[0] / 1000.0
        for i in range(1, 7):
            data.ctrl[act_ids[i]] = np.deg2rad(vals[i])


def fk_readout():
    with lock:
        mujoco.mj_forward(model, data)
        pos  = data.site_xpos[SITE_ID].copy()
        rail = data.ctrl[act_ids[0]] * 1000.0
        joints = [np.rad2deg(data.qpos[jid]) for jid in joint_ids]
    print(f"  Rail:      {rail:.1f} mm")
    print(f"  EE pos(m): x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
    print(f"  Joints(deg): {' '.join(f'{a:+7.1f}' for a in joints)}")


def sim_loop():
    while True:
        with lock:
            mujoco.mj_step(model, data)
        time.sleep(0.002)


def main():
    threading.Thread(target=sim_loop, daemon=True).start()
    apply_all(PRESETS["home"])

    print("[xArm6+Rail Terminal Control]")
    print(HELP)

    def run_viewer():
        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running():
                with lock:
                    v.sync()
                time.sleep(0.016)
    threading.Thread(target=run_viewer, daemon=True).start()
    time.sleep(0.3)

    while True:
        try:
            raw = input("xarm6> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        parts = raw.split()
        cmd   = parts[0].lower()

        if cmd == "quit":
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "presets":
            print(f"  Available: {', '.join(PRESETS)}")
        elif cmd == "home":
            apply_all(PRESETS["home"]); print("  Home.")
        elif cmd == "fk":
            fk_readout()
        elif cmd == "rail":
            if len(parts) < 2:
                print("  Usage: rail <mm>")
            else:
                try:
                    mm = float(np.clip(float(parts[1]), 0, 700))
                    with lock:
                        data.ctrl[act_ids[0]] = mm / 1000.0
                    print(f"  Rail -> {mm:.1f} mm")
                except ValueError:
                    print(f"  Bad number: {parts[1]}")
        elif cmd == "j":
            if len(parts) < 3:
                print("  Usage: j <1-6> <angle_deg>")
            else:
                try:
                    idx   = int(parts[1])
                    angle = float(parts[2])
                except ValueError:
                    print("  Bad numbers"); continue
                if 1 <= idx <= 6:
                    with lock:
                        data.ctrl[act_ids[idx]] = np.deg2rad(angle)
                    print(f"  Joint {idx} -> {angle:.1f} deg")
                else:
                    print("  Joint index must be 1-6.")
        elif cmd == "all":
            if len(parts) < 8:
                print("  Usage: all <rail_mm> <j1..j6_deg>  (7 numbers)")
            else:
                try:
                    vals = [float(p) for p in parts[1:8]]
                    apply_all(vals)
                    print(f"  All set: rail={vals[0]:.0f}mm joints={vals[1:]}")
                except ValueError:
                    print("  All 7 values must be numeric")
        elif cmd == "preset":
            if len(parts) < 2 or parts[1] not in PRESETS:
                print(f"  Unknown. Try: {', '.join(PRESETS)}")
            else:
                apply_all(PRESETS[parts[1]])
                print(f"  Preset '{parts[1]}' applied.")
        else:
            print(f"  Unknown command. Type 'help'.")

    print("[xArm6+Rail] Exiting.")


if __name__ == "__main__":
    main()
