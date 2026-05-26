# xArm6 + 700mm Rail — Interactive Simulation (No LLM) — v5
## Claude Code Instructions — Standalone Manual Control with Recording, Playback & Augmentation

> **This is a standalone document.** It supersedes v2–v4 of the basic series and does not require any prior version. Follow this doc end-to-end and you'll have a complete working system.
>
> **What's new in v5 vs v4:**
> - **Cubes-and-bins benchmark scene** — three RGB cubes, three matching bins, on a bench with a 700mm rail. A canonical pick-and-place environment for development before customizing to your real lab.
> - **Official xArm6 MJCF import** as an alternative to the boxed-geometry stand-in (boxes remain as a no-dependency default).
> - **Platform setup notes surfaced at the top** — macOS Accessibility permission and Linux/Wayland keyboard caveats are no longer buried at the bottom.
> - **Data Collection Workflow section** — how to actually use this for generating VLA training data, including the suggested ordering and quality filters.
> - All v3 recording + v4 augmentation infrastructure consolidated into one doc.

---

> **Hardware target:** UFACTORY xArm6 (6 rotational DOF) on a 700mm linear rail (1 prismatic DOF).
> **Architecture:** Direct Python ↔ MuJoCo. No ROS2. Format designed for later ROS2 bag conversion.

---

## Quick Visual Overview

```
                 Linear rail (700mm, X axis)
                  ────●─────────────────────
                      │
                      ▼  xArm6 mounted on carriage
                      ●─┐
                        ├ J1 base rot
                        ├ J2 shoulder
                        ├ J3 upper arm
                        ├ J4 elbow
                        ├ J5 forearm
                        └ J6 wrist
                          └ gripper

  Bench top (z ≈ 0.75 m)
  ┌─────────────────────────────────────────┐
  │   ▢ red bin   ▢ green bin   ▢ blue bin │  (y ≈ 0.35 m)
  │                                          │
  │   ■ red cube  ■ green cube  ■ blue cube │  (y ≈ 0.15 m)
  └─────────────────────────────────────────┘
        x = -0.20         0.00         +0.20
```

Task vocabulary this scene supports: pick a cube, place it in a bin, sort cubes by color, move cubes between bins, etc. Replace cubes/bins with real lab objects (pipettes, racks, tubes) when you're ready to customize for your specific lab — the architecture works identically.

---

## PLATFORM SETUP (READ FIRST)

Skipping this section will cost you hours debugging silent failures. Do it now.

### macOS — Accessibility permission required

`pynput` (used for global keyboard listening in `realtime_keyboard.py`) cannot receive key events without explicit OS permission.

1. Open **System Settings → Privacy & Security → Accessibility**.
2. Add (or enable) your terminal app — Terminal, iTerm2, VS Code, whichever you'll launch the script from.
3. If you switch terminals later, grant the new one too. The permission is per-app, not global.

If you skip this, arrow keys will silently do nothing and the script will appear hung. There is no error message.

### Linux — Wayland vs X11

Global keyboard listeners need an X server. On Wayland (default in Ubuntu 22.04+ and Fedora), `pynput`'s global listener often fails silently.

- **Ubuntu/Debian:** at login, select "Ubuntu on Xorg" instead of "Ubuntu" — this is XWayland.
- **Verify:** `echo $XDG_SESSION_TYPE` — should print `x11`, not `wayland`.
- **If stuck on Wayland:** use `control_panel.py` (Tkinter, no global listener needed) or `terminal_control.py` instead of `realtime_keyboard.py`.

### Windows

Generally works without configuration. Some antivirus products flag global keyboard listeners as suspicious — whitelist Python if needed.

### General — MuJoCo viewer headless support

If you're running on a remote server without a display, the viewer windows won't open. All scripts accept `--no-render` or equivalent to skip the viewer.

---

## Phase 0 — Install Dependencies

Python 3.10 or newer is required.

```bash
python --version    # Should be 3.10+

pip install mujoco numpy transforms3d pynput h5py

# Linux only — Tkinter for control_panel.py:
sudo apt-get install python3-tk
```

Optional but recommended:

```bash
# For importing the official xArm6 MJCF (instead of boxy stand-in)
pip install dm-control

# For better IK in LLM doc (irrelevant here, but install now if you plan to follow LLM v5)
pip install pin pink-ik
```

Total install: ~150MB plus MuJoCo's binary which is included in the `mujoco` package.

---

## Phase 1 — Project Structure

```
xarm_lab_twin/
├── envs/
│   ├── basic_scene.xml          # Scene with arm, rail, cubes, bins
│   ├── scene_randomizer.py      # Object pose perturbation for augmentation
│   └── assets/                  # Optional: scanned mesh, custom textures
├── recording.py                 # Recording backend (shared infrastructure)
├── realtime_keyboard.py         # Realtime keyboard control + recording
├── control_panel.py             # Tkinter GUI with sliders + recording
├── terminal_control.py          # Scripted terminal commands
├── replay.py                    # Playback + delete tool
├── replay_augment.py            # Re-record with variations
└── recordings/                  # Auto-created on first record
    └── <timestamped session folders>
```

Create the directories:

```bash
mkdir -p xarm_lab_twin/envs/assets
cd xarm_lab_twin
```

---

## Phase 2 — Scene XML (`envs/basic_scene.xml`)

This scene includes the xArm6, 700mm rail, bench, three RGB cubes, three matching bins, and the contact-exclusion rules needed for clean simulation.

```xml
<!-- envs/basic_scene.xml -->
<mujoco model="xarm6_rail_cubes_bins">

  <compiler angle="radian" meshdir="assets/"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map zfar="30"/>
  </visual>

  <default>
    <default class="arm_link">
      <!-- Arm geoms: contype=2 collide with world (contype=1) but not each other -->
      <geom contype="2" conaffinity="1"/>
    </default>
    <default class="cube">
      <geom type="box" size="0.015 0.015 0.015" mass="0.05"
            friction="1.0 0.05 0.0001" solref="0.005 1" solimp="0.95 0.99 0.001"/>
    </default>
    <default class="bin_wall">
      <geom type="box" rgba="0.8 0.8 0.8 0.5" mass="0"
            contype="1" conaffinity="1"/>
    </default>
  </default>

  <asset>
    <!-- Materials for cubes -->
    <material name="red_mat"   rgba="0.9 0.2 0.2 1"/>
    <material name="green_mat" rgba="0.2 0.8 0.3 1"/>
    <material name="blue_mat"  rgba="0.2 0.4 0.9 1"/>
    <!-- Materials for bin floor markers -->
    <material name="red_bin_mat"   rgba="0.9 0.2 0.2 0.6"/>
    <material name="green_bin_mat" rgba="0.2 0.8 0.3 0.6"/>
    <material name="blue_bin_mat"  rgba="0.2 0.4 0.9 0.6"/>
  </asset>

  <worldbody>

    <!-- Lighting -->
    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="0.5 0.5 1.8" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.1"
          rgba="0.7 0.7 0.65 1" contype="1" conaffinity="1"/>

    <!-- Bench -->
    <body name="bench" pos="0 0 0.375">
      <geom name="bench_top" type="box" size="0.75 0.45 0.375"
            rgba="0.75 0.65 0.5 1" contype="1" conaffinity="1" mass="0"/>
    </body>

    <!-- ============================================================
         RAIL — 700mm prismatic joint along X axis
         Mounted at y=-0.05 (slightly behind bench center)
    ============================================================ -->
    <body name="rail_track" pos="0.0 -0.05 0.75">

      <geom name="rail_geom" type="box" size="0.35 0.025 0.015"
            rgba="0.4 0.4 0.45 1" contype="0" conaffinity="0" mass="0"/>

      <body name="rail_carriage" pos="-0.35 0.0 0.02">
        <joint name="rail" type="slide" axis="1 0 0"
               range="0.0 0.7" damping="200"/>
        <geom name="carriage_geom" type="box" size="0.06 0.04 0.015"
              rgba="0.55 0.55 0.6 1" contype="1" conaffinity="1" mass="2.0"/>

        <!-- ============================================================
             xARM6 — boxy stand-in (replace with official MJCF if desired,
             see Phase 2b below)
        ============================================================ -->
        <body name="xarm_base" pos="0 0 0.02">
          <geom name="base_link" type="cylinder" size="0.06 0.04"
                rgba="0.3 0.3 0.3 1" mass="0.5" class="arm_link"/>
          <joint name="joint1" type="hinge" axis="0 0 1"
                 range="-3.14159 3.14159" damping="10"/>

          <body name="link1" pos="0 0 0.09">
            <geom name="link1_geom" type="box" size="0.055 0.055 0.11"
                  rgba="0.85 0.85 0.85 1" mass="2.16" class="arm_link"/>
            <joint name="joint2" type="hinge" axis="0 1 0"
                   range="-2.0944 2.0944" damping="10"/>

            <body name="link2" pos="0 0 0.22">
              <geom name="link2_geom" type="box" size="0.045 0.045 0.105"
                    rgba="0.85 0.85 0.85 1" mass="1.71" class="arm_link"/>
              <joint name="joint3" type="hinge" axis="0 1 0"
                     range="-3.14159 1.2217" damping="8"/>

              <body name="link3" pos="0 0 0.20">
                <geom name="link3_geom" type="box" size="0.04 0.04 0.09"
                      rgba="0.8 0.8 0.8 1" mass="1.38" class="arm_link"/>
                <joint name="joint4" type="hinge" axis="1 0 0"
                       range="-1.9199 1.9199" damping="6"/>

                <body name="link4" pos="0 0 0.175">
                  <geom name="link4_geom" type="box" size="0.035 0.035 0.075"
                        rgba="0.75 0.75 0.75 1" mass="1.05" class="arm_link"/>
                  <joint name="joint5" type="hinge" axis="0 1 0"
                         range="-2.6180 2.6180" damping="4"/>

                  <body name="link5" pos="0 0 0.13">
                    <geom name="link5_geom" type="box" size="0.03 0.03 0.055"
                          rgba="0.7 0.7 0.7 1" mass="0.72" class="arm_link"/>
                    <joint name="joint6" type="hinge" axis="1 0 0"
                           range="-1.9199 1.9199" damping="3"/>

                    <body name="link6" pos="0 0 0.075">
                      <geom name="link6_geom" type="cylinder"
                            size="0.025 0.03"
                            rgba="0.6 0.6 0.6 1" mass="0.25" class="arm_link"/>
                      <site name="end_effector" pos="0 0 0.09"
                            size="0.015" rgba="1 0.3 0.3 1" type="sphere"/>
                      <body name="gripper" pos="0 0 0.06">
                        <geom name="gripper_geom" type="box"
                              size="0.018 0.038 0.025"
                              rgba="0.35 0.35 0.35 1" mass="0.15"
                              class="arm_link"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

      </body>
    </body>

    <!-- ============================================================
         CUBES — three RGB cubes near the rail (y=0.15)
    ============================================================ -->
    <body name="red_cube" pos="-0.20 0.15 0.78">
      <geom name="red_cube_geom" class="cube" material="red_mat"/>
      <joint type="free" name="red_cube_joint"/>
    </body>
    <body name="green_cube" pos="0.00 0.15 0.78">
      <geom name="green_cube_geom" class="cube" material="green_mat"/>
      <joint type="free" name="green_cube_joint"/>
    </body>
    <body name="blue_cube" pos="0.20 0.15 0.78">
      <geom name="blue_cube_geom" class="cube" material="blue_mat"/>
      <joint type="free" name="blue_cube_joint"/>
    </body>

    <!-- ============================================================
         BINS — three open-top boxes farther from the rail (y=0.35)
         Each bin is a 4-wall + 1-floor static structure with a colored
         floor marker for visual identification. Internal dim: 80×80×60mm.
    ============================================================ -->
    <body name="red_bin" pos="-0.20 0.35 0.75">
      <!-- Floor marker (colored) -->
      <geom name="red_bin_floor" type="box" size="0.040 0.040 0.001"
            pos="0 0 0.001" material="red_bin_mat"
            contype="1" conaffinity="1" mass="0"/>
      <!-- 4 walls -->
      <geom name="red_bin_w_front"  class="bin_wall"
            size="0.040 0.002 0.030" pos="0 -0.040 0.030"/>
      <geom name="red_bin_w_back"   class="bin_wall"
            size="0.040 0.002 0.030" pos="0  0.040 0.030"/>
      <geom name="red_bin_w_left"   class="bin_wall"
            size="0.002 0.040 0.030" pos="-0.040 0 0.030"/>
      <geom name="red_bin_w_right"  class="bin_wall"
            size="0.002 0.040 0.030" pos=" 0.040 0 0.030"/>
    </body>

    <body name="green_bin" pos="0.00 0.35 0.75">
      <geom name="green_bin_floor" type="box" size="0.040 0.040 0.001"
            pos="0 0 0.001" material="green_bin_mat"
            contype="1" conaffinity="1" mass="0"/>
      <geom name="green_bin_w_front" class="bin_wall"
            size="0.040 0.002 0.030" pos="0 -0.040 0.030"/>
      <geom name="green_bin_w_back"  class="bin_wall"
            size="0.040 0.002 0.030" pos="0  0.040 0.030"/>
      <geom name="green_bin_w_left"  class="bin_wall"
            size="0.002 0.040 0.030" pos="-0.040 0 0.030"/>
      <geom name="green_bin_w_right" class="bin_wall"
            size="0.002 0.040 0.030" pos=" 0.040 0 0.030"/>
    </body>

    <body name="blue_bin" pos="0.20 0.35 0.75">
      <geom name="blue_bin_floor" type="box" size="0.040 0.040 0.001"
            pos="0 0 0.001" material="blue_bin_mat"
            contype="1" conaffinity="1" mass="0"/>
      <geom name="blue_bin_w_front" class="bin_wall"
            size="0.040 0.002 0.030" pos="0 -0.040 0.030"/>
      <geom name="blue_bin_w_back"  class="bin_wall"
            size="0.040 0.002 0.030" pos="0  0.040 0.030"/>
      <geom name="blue_bin_w_left"  class="bin_wall"
            size="0.002 0.040 0.030" pos="-0.040 0 0.030"/>
      <geom name="blue_bin_w_right" class="bin_wall"
            size="0.002 0.040 0.030" pos=" 0.040 0 0.030"/>
    </body>

  </worldbody>

  <!-- Suppress collisions between adjacent arm links -->
  <contact>
    <exclude body1="rail_carriage" body2="xarm_base"/>
    <exclude body1="xarm_base"     body2="link1"/>
    <exclude body1="link1"         body2="link2"/>
    <exclude body1="link2"         body2="link3"/>
    <exclude body1="link3"         body2="link4"/>
    <exclude body1="link4"         body2="link5"/>
    <exclude body1="link5"         body2="link6"/>
    <exclude body1="link6"         body2="gripper"/>
  </contact>

  <!-- ============================================================
       ACTUATORS
       Index 0 = rail (meters)
       Index 1–6 = joints 1–6 (radians)
  ============================================================ -->
  <actuator>
    <position name="act_rail" joint="rail"   kp="2000" kv="200"/>
    <position name="act1"     joint="joint1" kp="500"  kv="50"/>
    <position name="act2"     joint="joint2" kp="500"  kv="50"/>
    <position name="act3"     joint="joint3" kp="400"  kv="40"/>
    <position name="act4"     joint="joint4" kp="300"  kv="30"/>
    <position name="act5"     joint="joint5" kp="200"  kv="20"/>
    <position name="act6"     joint="joint6" kp="150"  kv="15"/>
  </actuator>

</mujoco>
```

Verify the scene loads:

```bash
python -c "
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('envs/basic_scene.xml')
mujoco.viewer.launch(m)
"
```

You should see the arm sitting upright on its carriage, three RGB cubes in a row on the bench, and three matching bins behind them.

### Customizing this scene for your real lab

The cubes-and-bins layout is a generic benchmark, not your actual workspace. When your real lab bench is scanned, replace the bench geometry and free bodies:

1. Export your scan as OBJ at `envs/assets/lab_bench.obj`
2. In the XML `<asset>` block, add: `<mesh name="lab_bench" file="lab_bench.obj"/>`
3. Replace the `<body name="bench">` box with: `<geom type="mesh" mesh="lab_bench" mass="0" contype="1" conaffinity="1"/>`
4. Replace the cube/bin bodies with real instrument placeholders (pipettes, tube racks, etc.) — `<geom>` shapes approximating each object, with free joints
5. Verify position match: the rail should land where the scan shows the rail mount, the arm should sit at the right z-height
6. Update preset poses (defined in the Python scripts below) to match new object positions

The XML structure and all Python code work identically with any scene — only the visible geometry changes.

---

## Phase 2b — (Optional) Use the Official xArm6 MJCF

The boxy stand-in is functional but visually approximate and has approximate inertial properties. For more realistic dynamics, use the official xArm6 MJCF (XML version of the URDF).

```bash
# Clone the MuJoCo Menagerie which includes xArm6
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie

# The xArm files will be at ~/mujoco_menagerie/ufactory_xarm7/  (xArm7 included)
# For xArm6 specifically, check the menagerie or use xArm-Developer/xarm_ros:
# https://github.com/xArm-Developer/xarm_ros/tree/master/xarm_description
```

Then in your scene XML, replace the `<body name="xarm_base">` block with:

```xml
<!-- Include the xArm6 MJCF — adjust path as needed -->
<include file="xarm6.xml"/>
```

The official MJCF defines its own joint actuators with manufacturer-spec gear ratios and inertias. You may need to adjust the `ACT_NAMES` list in the Python scripts if the actuator names differ; check the included XML and update accordingly.

**Recommendation:** Start with the boxy stand-in (zero dependencies, works immediately). Switch to the official MJCF once everything is validated end-to-end — the migration is a one-time XML edit.

---

## Phase 3 — Scene Randomizer (`envs/scene_randomizer.py`)

Used by the augmentation tool to perturb cube and bin poses between replay cycles, generating spatial variants for training data.

```python
# envs/scene_randomizer.py
"""
Perturb free-body positions and orientations in a MuJoCo scene before
a replay cycle. Generates spatial variations for data augmentation.
"""
import numpy as np
import mujoco
from typing import Optional


DEFAULT_POS_JITTER_MM = 20.0
DEFAULT_ROT_JITTER_DEG = 45.0
DEFAULT_INITIAL_JOINT_JITTER_DEG = 0.0

# Free bodies eligible for perturbation. Bins are static (not free joints)
# so they're naturally excluded.
PERTURBABLE_BODIES = {"red_cube", "green_cube", "blue_cube"}


def randomize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos_jitter_mm: float = DEFAULT_POS_JITTER_MM,
    rot_jitter_deg: float = DEFAULT_ROT_JITTER_DEG,
    initial_joint_jitter_deg: float = DEFAULT_INITIAL_JOINT_JITTER_DEG,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    Apply randomization to the scene. Returns a summary dict for logging.

    pos_jitter_mm: max absolute position perturbation (±N mm per axis)
    rot_jitter_deg: max absolute yaw rotation around vertical (±N deg)
    initial_joint_jitter_deg: per-joint angle perturbation (0 = off)
    rng: numpy RNG (pass np.random.default_rng(seed) for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    summary = {
        "perturbed_bodies": {},
        "joint_jitter_applied": [],
    }

    pos_jitter_m = pos_jitter_mm / 1000.0

    for body_name in PERTURBABLE_BODIES:
        try:
            body_id = model.body(body_name).id
        except KeyError:
            continue

        jnt_adr = model.body_jntadr[body_id]
        if jnt_adr < 0:
            continue
        jnt_type = model.jnt_type[jnt_adr]
        if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_adr = model.jnt_qposadr[jnt_adr]

        dx = rng.uniform(-pos_jitter_m, pos_jitter_m)
        dy = rng.uniform(-pos_jitter_m, pos_jitter_m)
        # Z is not perturbed — keeps objects resting on the bench surface
        dyaw = rng.uniform(-rot_jitter_deg, rot_jitter_deg)

        data.qpos[qpos_adr + 0] += dx
        data.qpos[qpos_adr + 1] += dy

        # Yaw rotation as quaternion multiplication
        yaw_rad = np.deg2rad(dyaw)
        cos_h = np.cos(yaw_rad / 2)
        sin_h = np.sin(yaw_rad / 2)
        qw = data.qpos[qpos_adr + 3]
        qx = data.qpos[qpos_adr + 4]
        qy = data.qpos[qpos_adr + 5]
        qz = data.qpos[qpos_adr + 6]
        # Hamilton product (current) * (delta around Z axis)
        new_qw = qw * cos_h - qz * sin_h
        new_qx = qx * cos_h + qy * sin_h
        new_qy = qy * cos_h - qx * sin_h
        new_qz = qz * cos_h + qw * sin_h
        data.qpos[qpos_adr + 3] = new_qw
        data.qpos[qpos_adr + 4] = new_qx
        data.qpos[qpos_adr + 5] = new_qy
        data.qpos[qpos_adr + 6] = new_qz

        summary["perturbed_bodies"][body_name] = {
            "dx_mm": float(dx * 1000.0),
            "dy_mm": float(dy * 1000.0),
            "dyaw_deg": float(dyaw),
        }

    if initial_joint_jitter_deg > 0:
        joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6"]
        jitter_rad = np.deg2rad(initial_joint_jitter_deg)
        for name in joint_names:
            jid = model.joint(name).id
            qadr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            delta = rng.uniform(-jitter_rad, jitter_rad)
            new_val = np.clip(data.qpos[qadr] + delta, lo, hi)
            data.qpos[qadr] = new_val
            summary["joint_jitter_applied"].append({
                "joint": name,
                "delta_deg": float(np.rad2deg(delta)),
            })

    mujoco.mj_forward(model, data)
    return summary
```

---

## Phase 4 — Recording Backend (`recording.py`)

Shared backend used by all interactive scripts. Captures state trajectory at viewer rate (~60Hz), sparse command events, and session metadata.

```python
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
    # Augmentation fields (empty for original sessions)
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
        print(f"[Recorder] ● REC  session={session_id}")
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
            print(f"[Recorder] ✗ Discarded {self._session_dir}")
            self._cleanup_session_dir()
            path = None
        else:
            print(f"[Recorder] ✓ Saved   {self._session_dir}")
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
        print("\n" + "─" * 60)
        print(f"Session {self._session.session_id} ended  "
              f"({self._session.duration_s:.1f}s, "
              f"{self._session.n_commands} commands, "
              f"{self._session.n_state_samples} state samples)")
        if self._session.task_label:
            print(f"Auto task label: '{self._session.task_label}'")
        print("─" * 60)
        print("Optional metadata — press Enter to skip any field.\n")
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
    print(f"[Recorder] → trash: {dest}")
    return True


def restore_session(session_name: str) -> bool:
    src = TRASH_DIR / session_name
    if not src.exists():
        print(f"[Recorder] Not in trash: {session_name}")
        return False
    dest = RECORDINGS_ROOT / session_name
    if dest.exists():
        print(f"[Recorder] Cannot restore — {dest} already exists")
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
```

---

## Phase 5 — Realtime Keyboard Control (`realtime_keyboard.py`)

The primary tool for fluid demonstrations. Hybrid keybinding lets you drive multiple joints simultaneously at a persistent speed level.

**Key bindings:**

| Key | Action |
|---|---|
| `↑` / `↓` | Joint 2 (shoulder) ± |
| `←` / `→` | Joint 1 (base rotation) ± |
| `Shift + ↑/↓` | Joint 4 (elbow) ± |
| `Shift + ←/→` | Joint 3 (upper arm) ± |
| `Alt + ↑/↓` | Joint 6 (wrist) ± |
| `Alt + ←/→` | Joint 5 (forearm) ± |
| `[` / `]` | Rail backward / forward |
| `Ctrl + 1` … `Ctrl + 9` | Set speed level 1–9 (persistent) |
| `Space` | All-stop (release all DOFs) |
| `H` | Go to home pose |
| `R` | Toggle recording |
| `Esc` | Quit |

```python
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
                print(f"  [speed] level {lvl} — {info['label']}  "
                      f"({info['joint_deg_s']}°/s joints, "
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
            print("\n[realtime_keyboard] Escape pressed — shutting down...")
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
        print("  [home] rail → 350mm, all joints → zero")
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
        print("\n" + "═" * 64)
        print(" xArm6 + Rail — Realtime Keyboard Control")
        print("═" * 64)
        print(" Arrow keys      → J1 (←→)  J2 (↑↓)")
        print(" Shift + arrows  → J3 (←→)  J4 (↑↓)")
        print(" Alt   + arrows  → J5 (←→)  J6 (↑↓)")
        print(" [ / ]           → Rail backward / forward")
        print(" Ctrl + 1…9      → Set speed level")
        print(" Space           → All-stop")
        print(" H               → Home pose")
        print(" R               → Toggle recording")
        print(" Esc             → Quit")
        print("─" * 64)
        print(f" Speed: level {self.speed_level} ({lvl['label']}) — "
              f"{lvl['joint_deg_s']}°/s joints, {lvl['rail_mm_s']}mm/s rail")
        print("═" * 64 + "\n")


def main():
    ctrl = RealtimeKeyboardController()
    ctrl.run()
    print("\n[realtime_keyboard] Done.")


if __name__ == "__main__":
    main()
```

Run:
```bash
python realtime_keyboard.py
```

---

## Phase 6 — Tkinter Control Panel (`control_panel.py`)

GUI with sliders for each DOF, FK readout, preset poses, and record toggle. Works on Wayland (no global key listener required).

```python
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
    "J1  base rot  (°)",
    "J2  shoulder  (°)",
    "J3  upper arm (°)",
    "J4  elbow     (°)",
    "J5  forearm   (°)",
    "J6  wrist     (°)",
]

# Preset poses calibrated to reach cube-and-bin positions
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
        self.root.title("xArm6 + Rail Sim — Manual Control")
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
            unit = "mm" if i == 0 else "°"
            val_lbl = ttk.Label(row, text=f"  0{unit}", width=8, anchor="e")
            val_lbl.pack(side="left")
            var.trace_add("write", lambda *_, lbl=val_lbl, v=var, u=unit:
                          lbl.config(text=f"{v.get():+7.1f}{u}"))
            ttk.Label(row, text=f"[{lo:.0f} … {hi:.0f}]",
                      foreground="gray",
                      font=("TkDefaultFont", 8)).pack(side="left", padx=4)

        fk_frame = ttk.LabelFrame(self.root, text="End-effector position (FK)")
        fk_frame.pack(fill="x", **pad)
        self._fk_labels = {}
        fk_inner = ttk.Frame(fk_frame); fk_inner.pack(pady=4)
        for key, unit in [("X","m"),("Y","m"),("Z","m"),
                          ("Roll","°"),("Pitch","°"),("Yaw","°")]:
            col = ttk.Frame(fk_inner); col.pack(side="left", padx=12)
            ttk.Label(col, text=f"{key} ({unit})",
                      font=("TkDefaultFont", 9, "bold")).pack()
            lbl = ttk.Label(col, text="—", font=("TkFixedFont", 11)); lbl.pack()
            self._fk_labels[key] = lbl
        rail_frame = ttk.Frame(fk_frame); rail_frame.pack(pady=(0, 4))
        ttk.Label(rail_frame, text="Rail position:",
                  font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=6)
        self._rail_label = ttk.Label(rail_frame, text="— mm",
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
            rec_inner, text="● Start recording",
            command=self._toggle_record, width=24
        )
        self._rec_btn.pack(side="left", padx=8)
        self._rec_status = ttk.Label(
            rec_inner, text="(idle)", foreground="gray",
            font=("TkFixedFont", 10)
        )
        self._rec_status.pack(side="left", padx=8)

        act_frame = ttk.Frame(self.root); act_frame.pack(pady=6)
        ttk.Button(act_frame, text="⌂  Home",
                   command=self._home).pack(side="left", padx=6)
        ttk.Button(act_frame, text="⟳  Sync from sim",
                   command=self._sync).pack(side="left", padx=6)

        ttk.Label(self.root,
                  text="MuJoCo viewer: mouse drag = orbit · right-drag = pan · "
                       "scroll = zoom · F = pause · V = contacts",
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
            self._rec_btn.config(text="● Start recording")
            self._rec_status.config(text="(saving...)", foreground="orange")
            self.root.update_idletasks()
            def do_stop():
                self.recorder.stop_and_prompt(prompt=True)
                self._rec_status.config(text="(idle)", foreground="gray")
            threading.Thread(target=do_stop, daemon=True).start()
        else:
            self.recorder.start()
            self._rec_btn.config(text="■ Stop recording")
            self._rec_status.config(text="● RECORDING", foreground="red")

    def _poll_rec_status(self):
        if self.recorder.is_recording and self.recorder._session is not None:
            elapsed = time.time() - self.recorder._start_wall_time
            n_cmd = self.recorder._session.n_commands
            self._rec_status.config(
                text=f"● REC  {elapsed:.1f}s  {n_cmd} cmds",
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
```

---

## Phase 7 — Terminal Mode (`terminal_control.py`)

Minimalist precise-coordinate control. No GUI, no global listeners. Useful for scripted commands and debugging.

```python
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
  rail <mm>                     — Move rail (0–700mm)
  j <1-6> <angle_deg>           — Set one joint angle
  all <rail_mm> <j1..j6_deg>    — Set all 7 DOFs at once
  preset <name>                 — Load preset (home/red_cube/green_cube/etc.)
  fk                            — Print current pose
  home                          — Home position
  presets                       — List preset names
  help                          — Show this message
  quit                          — Exit
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
    print(f"  Joints(°): {' '.join(f'{a:+7.1f}' for a in joints)}")


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
                    print(f"  Rail → {mm:.1f} mm")
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
                    print(f"  Joint {idx} → {angle:.1f}°")
                else:
                    print("  Joint index must be 1–6.")
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
```

---

## Phase 8 — Replay Tool (`replay.py`)

List, inspect, replay, and delete sessions.

```python
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
    print("─" * 120)
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
```

---

## Phase 9 — Augmentation Tool (`replay_augment.py`)

Re-record a session N cycles with optional scene randomization and command jitter.

```python
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
    print("\n" + "═" * 70)
    print("  BATCH ANNOTATION")
    print("═" * 70)
    print(f"  {sum(1 for d in saved_dirs if d)} cycle recordings created.")
    print("─" * 70)
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
            print(f"     ✗ Deleting {d.name}")
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
    print(f"    Object pos:      ±{args.pos_jitter_mm}mm")
    print(f"    Object rot:      ±{args.rot_jitter_deg}°")
    print(f"    Timing:          ±{args.timing_jitter_ms}ms")
    print(f"    Duration:        ±{args.duration_jitter_pct*100:.0f}%")
    print(f"    Joint:           ±{args.joint_jitter_deg}°")

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
    print(f"\n✓ {n} sessions saved. Inspect with: python replay.py")


if __name__ == "__main__":
    main()
```

---

## Phase 10 — Data Collection Workflow

This is the practical guide for going from "I have working scripts" to "I have a useful training dataset." Read this before recruiting undergrads.

### Step 1 — Build manual dexterity (1 day)

Spend a few hours just driving the arm with `realtime_keyboard.py`. **Don't record yet.** Goals:

- Memorize key bindings (arrows = J1/J2; Shift+arrows = J3/J4; Alt+arrows = J5/J6)
- Develop a feel for which speed level fits which motion (3–4 for fine grasps, 6–7 for rail traversal)
- Practice the basic task: rail to a cube, lower J2/J4 to approach, position the gripper around the cube, lift, traverse rail, lower over the bin, release

Until you can pick a cube and drop it in a bin smoothly in under 30 seconds without using all-stop more than twice, you're not ready to record demonstrations. Practice first.

### Step 2 — Define your task list (30 minutes)

Before recording anything, list the specific tasks you want demonstrated. For the cubes-and-bins scene, a starting set might be:

- `red_cube_to_red_bin`
- `green_cube_to_green_bin`
- `blue_cube_to_blue_bin`
- `red_cube_to_green_bin`
- `green_cube_to_blue_bin`
- `blue_cube_to_red_bin`
- `red_and_green_to_blue_bin`  (two-cube composite)
- `sort_cubes_by_color`        (three-cube composite, color-matched)

Use these exact strings as `task_label` values when prompted. Consistency is essential — `red_to_red` and `red_cube_to_red_bin` will be treated as different tasks.

### Step 3 — Record demonstrations (focused sessions)

Plan ~5–10 minutes per demonstration *including setup and metadata*. For 8 tasks × 10 demos each = 80 demos = ~10 hours.

Practical rules:

- **One task per session.** Don't combine "pick red" and "pick green" in one recording.
- **Start recording before motion, stop after motion completes.** A few seconds of pre/post idle is fine.
- **Use the metadata prompt every time.** Skip nothing. Task label and outcome are essential.
- **Discard liberally.** If a grasp was sloppy or the motion was jerky, mark outcome as failure or discard during the prompt. Better 50 clean demos than 100 noisy ones.
- **Vary your starting rail position.** Default home rail=350mm. Sometimes start at 100mm or 600mm. This pre-augments your dataset with rail diversity.

### Step 4 — Augment after collecting (overnight)

Once you have 5–10 clean demos of a task, augment each:

```bash
for session in recordings/2026-*; do
    python replay_augment.py "$session" --cycles 10 --no-render --no-annotate
done
```

This runs unattended. 5 demos × 10 cycles = 50 augmented samples per task. Over 8 tasks, that's 400 augmented samples added to ~50 originals = ~450 total samples per training run.

### Step 5 — Annotate augmented batches (1 hour)

The augmentation runs produce many failures because perturbing object positions by ±20mm often causes the gripper to miss. Walk through with the batch annotation prompt (or run `python replay.py` and label one at a time). For each:

- Successful augmentations → mark `s` (success). These are clean training examples.
- Failed augmentations → mark `f` (failure). **Keep about 25%.** These are valuable negative examples for VLA training.
- Garbage (LLM error, arm collision, totally off-trajectory) → mark `d` (delete).

Don't keep all failures — your dataset becomes mostly negatives. Don't keep zero failures — your dataset never sees "this doesn't work."

### Step 6 — Onboard undergrads

Once your own dataset works well, hand off to undergrads. Give them this checklist:

1. macOS Accessibility permission set (Phase 0 of this doc)
2. Task list of acceptable `task_label` values (Step 2 above)
3. 1-hour practice session before recording
4. Same recording/annotation discipline

A productive undergrad collects ~50 demonstrations per 4-hour session. Three undergrads × 4 hours/week × 8 weeks = ~4800 demonstrations. With augmentation: ~40,000+ training samples.

### Quality filters worth applying later

When preparing the dataset for VLA training, exclude:

- Sessions with duration > 2× median for their task label (probably involved corrections)
- Sessions with > 30 commands per task (excessive fiddling)
- Sessions marked outcome="failure" beyond the 25% retention target

These are heuristics that scale better than visual inspection of every recording. `metadata.json` makes this trivial to filter via simple scripts.

---

## File Format Reference

### `metadata.json`

```json
{
  "session_id": "a1b2c3d4",
  "started_at_iso": "2026-05-25T14:32:07.123456",
  "ended_at_iso":   "2026-05-25T14:33:42.789012",
  "duration_s": 95.7,
  "interface": "realtime_keyboard",
  "task_label": "red_cube_to_red_bin",
  "outcome": "success",
  "demonstrator_id": "kk",
  "notes": "Smooth approach, single grasp attempt",
  "scene_xml": "envs/basic_scene.xml",
  "state_hz": 60.0,
  "n_commands": 47,
  "n_state_samples": 5742,
  "kept": true,
  "parent_session_id": "",
  "cycle_index": 0,
  "augmentation_config": {}
}
```

### `commands.jsonl`

```
{"t": 0.832, "type": "speed_change", "payload": {"level": 3}}
{"t": 1.205, "type": "arrow_press",   "payload": {"key": "right", "modifier": "none", "target_joint": "joint1", "direction": 1}}
{"t": 2.890, "type": "arrow_release", "payload": {"key": "right", "modifier": "none", "target_joint": "joint1", "direction": 0}}
```

### `trajectory.h5`

HDF5 with columns `t_wall`, `t_sim`, `rail_mm`, `joints_deg(N,6)`, `ee_pos_mm(N,3)`, `ee_rpy_deg(N,3)`, `ctrl(N,7)`. Attributes: `state_hz`, `n_samples`, `joint_names`, `actuator_names`.

Load in Python:

```python
import h5py, pandas as pd
with h5py.File("recordings/<session>/trajectory.h5", "r") as f:
    df = pd.DataFrame({
        "t":           f["t_wall"][:],
        "rail_mm":     f["rail_mm"][:],
        **{f"j{i+1}_deg": f["joints_deg"][:, i] for i in range(6)},
        "ee_x_mm":     f["ee_pos_mm"][:, 0],
        "ee_y_mm":     f["ee_pos_mm"][:, 1],
        "ee_z_mm":     f["ee_pos_mm"][:, 2],
    })
```

---

## Quick Reference

```bash
# Manual control (primary tool)
python realtime_keyboard.py

# GUI control (Wayland-safe, slider-based)
python control_panel.py

# Terminal (scripted/precise)
python terminal_control.py

# List sessions
python replay.py

# Replay session 0
python replay.py 0

# Replay slowly
python replay.py 0 --speed 0.3

# Delete session 0
python replay.py 0 --delete

# Augment session 0 with 10 cycles
python replay_augment.py 0 --cycles 10

# Augment headlessly (overnight batch)
python replay_augment.py 0 --cycles 30 --no-render --no-annotate --seed 42

# Empty trash
python replay.py --purge-trash
```

---

## Future-Proofing

The recording format maps cleanly to ROS2 conventions, so later migration is straightforward:

| This format | ROS2 equivalent |
|---|---|
| `commands.jsonl` | `/user_input` topic in `rosbag2` |
| `trajectory.h5` columns `joints_deg`, `rail_mm` | `/joint_states` (`sensor_msgs/JointState`) |
| `trajectory.h5` columns `ee_pos_mm`, `ee_rpy_deg` | `/tf` end-effector frame |
| `trajectory.h5` column `ctrl` | `/joint_commands` |

A future `convert_to_rosbag.py` script can walk each session folder and produce a `rosbag2` directory.

For VLA training data preparation, a separate script can iterate the recordings folder, filter by `metadata.json`, sample frames at e.g. 10Hz, render RGB images via `mujoco.Renderer`, and emit (image, instruction, action_chunk) triplets in a format your training framework expects (LeRobot dataset format, OpenVLA dataset format, etc.).

Both of these are future additions. The current architecture intentionally keeps recording format independent of any specific training framework so you can target whichever ends up being right.
