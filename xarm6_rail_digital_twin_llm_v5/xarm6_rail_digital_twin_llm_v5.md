# xArm6 + 700mm Rail — Digital Twin Training System — v5
## Claude Code Instructions — Standalone LLM Agentic Version with Recording, Playback, Augmentation & Prompt Variation

> **This is a standalone document.** It supersedes v2–v4 of the LLM series and does not require any prior version. Follow this doc end-to-end and you'll have a complete working LLM-agentic system.
>
> **What's new in v5 vs v4:**
> - **Cubes-and-bins benchmark scene** — three RGB cubes, three matching bins. A canonical pick-and-place environment for development before customizing to your real lab.
> - **`pink` differential IK** as the recommended solver with the iterative Jacobian as a no-dependency fallback. Substantially faster and more reliable convergence.
> - **Official xArm6 MJCF import** as an alternative to the boxed-geometry stand-in.
> - **Platform setup notes surfaced at the top** — macOS Accessibility permission and Linux/Wayland keyboard caveats.
> - **Data Collection Workflow section** — how to use this for VLA training data generation, including LLM-specific tips on cost, model routing, and variant quality.
> - All v3 recording + v4 augmentation + v4 prompt variation infrastructure consolidated into one doc.

---

> **Hardware target:** UFACTORY xArm6 (6 rotational DOF) on a 700mm linear rail (1 prismatic DOF).
> **Architecture:** Direct Python ↔ MuJoCo ↔ xArm SDK. Claude API for high-level task reasoning. No ROS2 (format ROS2-bag-convertible later).

---

## Quick Visual Overview

```
              NL prompt → Claude (Haiku/Sonnet/Opus)
                              ↓ JSON commands
                  SimXArmAPI (XArmAPI-compatible)
                              ↓
                       MuJoCo physics
                              ↓
                       Recorder logs:
                         metadata.json
                         commands.jsonl
                         trajectory.h5
                         llm_session.jsonl

Scene (initial layout — randomizable via scene_randomizer.py):

   Rail (700mm, X axis)
    ────●─────────────────────
         │ xArm6 on carriage

  Bench top
  ┌────────────────────────────────────────┐
  │  ▢ red bin   ▢ green bin   ▢ blue bin │  y ≈ 0.35
  │  ■ red cube  ■ green cube  ■ blue cube │  y ≈ 0.15
  └────────────────────────────────────────┘
       x = -0.20      0.00      +0.20
```

LLM commands the arm via SDK-style calls (`set_rail`, `move_to`, `gripper_close`, etc). The same SDK interface works on both sim and real hardware — just swap `SimXArmAPI` for `RealXArmAPI`.

---

## PLATFORM SETUP (READ FIRST)

### macOS — Accessibility permission

`pynput` (used in the basic manual scripts) needs Accessibility permission. Not strictly required for the LLM doc since the LLM drives the arm, but worth setting now so the basic doc's manual recording works too.

System Settings → Privacy & Security → Accessibility → enable your terminal app.

### Linux — Wayland vs X11

Same caveat as basic: global keyboard listeners need X. The LLM doc itself doesn't need a keyboard listener (LLM drives motion), so Wayland is fine here. Only matters if you also use `realtime_keyboard.py` for manual demos.

### Windows

Generally works. Whitelist Python if antivirus flags the API client.

### NVIDIA drivers

The Anthropic API runs in the cloud — no local GPU needed. MuJoCo's CPU physics is fine for development. If you later move to GPU-accelerated physics (MJX), drivers matter; for v5, they don't.

### API key

```bash
# Linux / macOS
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Get a key at https://console.anthropic.com — the `anthropic` Python SDK reads `ANTHROPIC_API_KEY` from the environment automatically. Persist this in your shell rc file (`.bashrc`, `.zshrc`) so you don't re-export it each session.

---

## Phase 0 — Install Dependencies

Python 3.10 or newer is required.

```bash
python --version    # Should be 3.10+

# Core
pip install mujoco numpy transforms3d h5py
pip install anthropic

# xArm SDK (for real hardware mode; sim works without it)
pip install xarm-python-sdk

# Recommended IK solver — pink (differential IK on MuJoCo)
pip install pin
pip install pink-ik

# Linux only — Tkinter for control_panel.py if you also use the basic doc:
sudo apt-get install python3-tk

# Optional — only if you also use realtime_keyboard.py from basic doc:
pip install pynput

# Scan processing (only when you replace cubes/bins with a real scan)
pip install open3d
```

**On `pink`:** This is the recommended IK library — fast, MuJoCo-native, well-maintained. The doc's code falls back to a slower iterative Jacobian solver if `pink` is not installed, so you can defer this dependency, but install it before running anything serious. With `pink`, IK converges in ~10ms; without it, ~100ms per call.

If `pip install pink-ik` fails on your system, the iterative fallback will run automatically — you'll see a warning at startup and operation will still work, just slower.

Total install: ~200MB.

---

## Phase 1 — Project Structure

```
xarm_lab_twin/
├── envs/
│   ├── lab_scene.xml           # Scene: arm, rail, cubes, bins
│   ├── scene_randomizer.py     # Object pose perturbation
│   └── assets/                 # Optional: scanned mesh, custom textures
├── sim/
│   ├── __init__.py
│   ├── mujoco_env.py           # SimXArmAPI (XArmAPI-compatible)
│   ├── fk_validator.py         # FK + collision validation
│   └── ik_solver.py            # pink-based IK with iterative fallback
├── agent/
│   ├── __init__.py
│   ├── llm_brain.py            # Claude API agentic loop
│   ├── object_registry.py      # Cube/bin metadata
│   └── prompt_variants.py      # Variant generation via Claude
├── hardware/
│   ├── __init__.py
│   └── real_arm.py             # Real XArmAPI wrapper
├── scripts/
│   ├── scan_to_mesh.py         # Scan → OBJ conversion
│   ├── run_task.py             # Single task execution
│   └── run_task_augmented.py   # Multi-cycle with prompt variants
├── recording.py                # Recording backend (shared)
├── replay.py                   # Playback + delete tool
└── recordings/                 # Auto-created on first run
```

Create directories and `__init__.py` files:

```bash
mkdir -p xarm_lab_twin/envs/assets xarm_lab_twin/sim xarm_lab_twin/agent xarm_lab_twin/hardware xarm_lab_twin/scripts
cd xarm_lab_twin
touch sim/__init__.py agent/__init__.py hardware/__init__.py
```

---

## Phase 2 — Scene XML (`envs/lab_scene.xml`)

Identical scene structure to the basic v5 doc, with the LLM-specific filename `lab_scene.xml`.

```xml
<!-- envs/lab_scene.xml -->
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
    <material name="red_mat"   rgba="0.9 0.2 0.2 1"/>
    <material name="green_mat" rgba="0.2 0.8 0.3 1"/>
    <material name="blue_mat"  rgba="0.2 0.4 0.9 1"/>
    <material name="red_bin_mat"   rgba="0.9 0.2 0.2 0.6"/>
    <material name="green_bin_mat" rgba="0.2 0.8 0.3 0.6"/>
    <material name="blue_bin_mat"  rgba="0.2 0.4 0.9 0.6"/>
  </asset>

  <worldbody>

    <light pos="0 0 2.5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="0.5 0.5 1.8" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <geom name="floor" type="plane" size="3 3 0.1"
          rgba="0.7 0.7 0.65 1" contype="1" conaffinity="1"/>

    <body name="bench" pos="0 0 0.375">
      <geom name="bench_top" type="box" size="0.75 0.45 0.375"
            rgba="0.75 0.65 0.5 1" contype="1" conaffinity="1" mass="0"/>
    </body>

    <body name="rail_track" pos="0.0 -0.05 0.75">
      <geom name="rail_geom" type="box" size="0.35 0.025 0.015"
            rgba="0.4 0.4 0.45 1" contype="0" conaffinity="0" mass="0"/>

      <body name="rail_carriage" pos="-0.35 0.0 0.02">
        <joint name="rail" type="slide" axis="1 0 0"
               range="0.0 0.7" damping="200"/>
        <geom name="carriage_geom" type="box" size="0.06 0.04 0.015"
              rgba="0.55 0.55 0.6 1" contype="1" conaffinity="1" mass="2.0"/>

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
                      <geom name="link6_geom" type="cylinder" size="0.025 0.03"
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

    <!-- Cubes -->
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

    <!-- Bins -->
    <body name="red_bin" pos="-0.20 0.35 0.75">
      <geom name="red_bin_floor" type="box" size="0.040 0.040 0.001"
            pos="0 0 0.001" material="red_bin_mat"
            contype="1" conaffinity="1" mass="0"/>
      <geom name="red_bin_w_front" class="bin_wall"
            size="0.040 0.002 0.030" pos="0 -0.040 0.030"/>
      <geom name="red_bin_w_back"  class="bin_wall"
            size="0.040 0.002 0.030" pos="0  0.040 0.030"/>
      <geom name="red_bin_w_left"  class="bin_wall"
            size="0.002 0.040 0.030" pos="-0.040 0 0.030"/>
      <geom name="red_bin_w_right" class="bin_wall"
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

Verify:
```bash
python -c "
import mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path('envs/lab_scene.xml')
mujoco.viewer.launch(m)
"
```

You should see arm + rail + three RGB cubes + three RGB bins. Orbit the view, zoom, confirm everything looks right.

### Customizing for your real lab

When your bench scan is ready (see `scripts/scan_to_mesh.py` below):

1. Place scanned mesh at `envs/assets/lab_bench.obj`
2. Add `<mesh name="lab_bench" file="lab_bench.obj"/>` to `<asset>`
3. Replace the bench box with `<geom type="mesh" mesh="lab_bench" ...>`
4. Replace cube and bin bodies with real instrument placeholders
5. Update the object registry (`agent/object_registry.py`) to match
6. The XML structure and Python code work identically — only the geometry and registry change

### Optional: Official xArm6 MJCF

For more realistic dynamics, replace the boxy stand-in with the manufacturer's MJCF:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
# See ~/mujoco_menagerie/ufactory_xarm7/  (and similar for xArm6)
```

Then in `lab_scene.xml`, replace the `<body name="xarm_base">` block with `<include file="xarm6.xml"/>` (adjusting path and actuator names as needed). Start with the boxy stand-in to validate everything; switch later.

---

## Phase 3 — Scan Processing (`scripts/scan_to_mesh.py`)

When you replace cubes/bins with a real lab bench scan:

```python
# scripts/scan_to_mesh.py
import open3d as o3d
import numpy as np
import sys

def process_scan(input_path: str, output_obj: str, voxel_size: float = 0.002):
    print(f"Loading scan: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_obj, mesh)
    print(f"Saved: {output_obj}  ({len(mesh.triangles)} triangles)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/scan_to_mesh.py <input.ply> <output.obj>")
        sys.exit(1)
    process_scan(sys.argv[1], sys.argv[2])
```

```bash
python scripts/scan_to_mesh.py raw_scan.ply envs/assets/lab_bench.obj
```

---

## Phase 4 — Scene Randomizer (`envs/scene_randomizer.py`)

Used by the augmentation tool to perturb cube poses between cycles.

```python
# envs/scene_randomizer.py
"""
Perturb free-body positions and orientations in a MuJoCo scene before
a replay or run cycle. Used to generate spatial variations for training data.
"""
import numpy as np
import mujoco
from typing import Optional

DEFAULT_POS_JITTER_MM = 20.0
DEFAULT_ROT_JITTER_DEG = 45.0
DEFAULT_INITIAL_JOINT_JITTER_DEG = 0.0

PERTURBABLE_BODIES = {"red_cube", "green_cube", "blue_cube"}


def randomize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos_jitter_mm: float = DEFAULT_POS_JITTER_MM,
    rot_jitter_deg: float = DEFAULT_ROT_JITTER_DEG,
    initial_joint_jitter_deg: float = DEFAULT_INITIAL_JOINT_JITTER_DEG,
    rng: Optional[np.random.Generator] = None,
) -> dict:
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
        if model.jnt_type[jnt_adr] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        qpos_adr = model.jnt_qposadr[jnt_adr]

        dx = rng.uniform(-pos_jitter_m, pos_jitter_m)
        dy = rng.uniform(-pos_jitter_m, pos_jitter_m)
        dyaw = rng.uniform(-rot_jitter_deg, rot_jitter_deg)

        data.qpos[qpos_adr + 0] += dx
        data.qpos[qpos_adr + 1] += dy

        yaw_rad = np.deg2rad(dyaw)
        cos_h = np.cos(yaw_rad / 2); sin_h = np.sin(yaw_rad / 2)
        qw = data.qpos[qpos_adr + 3]; qx = data.qpos[qpos_adr + 4]
        qy = data.qpos[qpos_adr + 5]; qz = data.qpos[qpos_adr + 6]
        data.qpos[qpos_adr + 3] = qw * cos_h - qz * sin_h
        data.qpos[qpos_adr + 4] = qx * cos_h + qy * sin_h
        data.qpos[qpos_adr + 5] = qy * cos_h - qx * sin_h
        data.qpos[qpos_adr + 6] = qz * cos_h + qw * sin_h

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
                "joint": name, "delta_deg": float(np.rad2deg(delta))
            })

    mujoco.mj_forward(model, data)
    return summary
```

---

## Phase 5 — IK Solver (`sim/ik_solver.py`)

Two implementations: `pink`-based (fast, recommended) and iterative Jacobian (fallback). The wrapper picks the best available automatically.

```python
# sim/ik_solver.py
"""
IK solver for xArm6 (rail held fixed).

Prefers pink (https://github.com/stephane-caron/pink), a differential
IK library built on Pinocchio with native MuJoCo support. Falls back
to a damped Jacobian pseudoinverse if pink is unavailable.
"""
import threading
import numpy as np
import mujoco
from typing import Optional

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# Try to import pink
try:
    import pink
    from pink import solve_ik
    from pink.tasks import FrameTask
    HAS_PINK = True
except ImportError:
    HAS_PINK = False
    print("[ik_solver] pink not available — using iterative Jacobian fallback. "
          "Install with: pip install pin pink-ik  for faster IK.")


class IKSolver:
    """
    Wraps either pink or the iterative Jacobian fallback.

    The interface is the same regardless of backend:
        solver = IKSolver(model, data, lock)
        new_q = solver.solve(target_pos_m, current_qpos_snapshot)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 lock: threading.Lock):
        self.model = model
        self.data  = data
        self.lock  = lock
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.ee_site   = model.site("end_effector").id
        self.backend = "pink" if HAS_PINK else "jacobian"

        if HAS_PINK:
            self._init_pink()

    def _init_pink(self):
        """Build a pink Configuration from the MuJoCo model."""
        # pink's MuJoCo support uses an internal configuration object
        # that tracks joint values. We build it once and update qpos per call.
        try:
            self._pink_config = pink.Configuration(self.model, self.data)
            # FrameTask targets the end-effector site
            self._pink_task = FrameTask(
                "end_effector",
                position_cost=1.0,
                orientation_cost=0.0,   # Position-only IK for now
            )
        except Exception as e:
            print(f"[ik_solver] pink init failed ({e}) — using fallback")
            self.backend = "jacobian"

    def solve(
        self,
        target_pos_m: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for the 6 rotational joints to reach target_pos_m.

        Operates on a snapshot of qpos so the live sim state is NOT
        corrupted — caller must hold the lock when calling, and we
        save+restore qpos around the solve.

        Returns: joint angles array (6,) in radians, or None if failed.
        """
        # Snapshot
        joint_qpos_backup = np.array(
            [self.data.qpos[jid] for jid in self.joint_ids]
        )
        rail_qpos_backup = float(self.data.qpos[self.rail_jid])

        try:
            if self.backend == "pink":
                result = self._solve_pink(target_pos_m, max_iter, tol)
            else:
                result = self._solve_jacobian(target_pos_m, max_iter, tol)
            return result
        finally:
            # Always restore live state
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = joint_qpos_backup[i]
            self.data.qpos[self.rail_jid] = rail_qpos_backup
            mujoco.mj_forward(self.model, self.data)

    def _solve_pink(self, target_pos_m: np.ndarray,
                    max_iter: int, tol: float) -> Optional[np.ndarray]:
        """Pink-based differential IK."""
        try:
            # Update pink's config from current qpos
            self._pink_config.update(self.data.qpos)

            # Set target pose — position only, keep current orientation
            from pink.utils import SE3
            mujoco.mj_forward(self.model, self.data)
            current_rot = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
            target_se3 = SE3(rotation=current_rot, translation=target_pos_m)
            self._pink_task.set_target(target_se3)

            # Iterate
            dt = 0.01
            for _ in range(max_iter):
                velocity = solve_ik(self._pink_config, [self._pink_task],
                                    dt, solver="quadprog")
                self._pink_config.integrate_inplace(velocity, dt)

                # Sync back to MuJoCo data for FK check
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = self._pink_config.q[jid]
                mujoco.mj_forward(self.model, self.data)

                err = np.linalg.norm(
                    target_pos_m - self.data.site_xpos[self.ee_site]
                )
                if err < tol:
                    return np.array(
                        [self.data.qpos[jid] for jid in self.joint_ids]
                    )
            return None  # Did not converge
        except Exception as e:
            print(f"[ik_solver] pink failed: {e} — using fallback once")
            return self._solve_jacobian(target_pos_m, max_iter, tol)

    def _solve_jacobian(self, target_pos_m: np.ndarray,
                        max_iter: int, tol: float) -> Optional[np.ndarray]:
        """Damped least-squares Jacobian IK fallback."""
        q = np.array([self.data.qpos[jid] for jid in self.joint_ids])

        for _ in range(max_iter):
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q[i]
            mujoco.mj_forward(self.model, self.data)

            pos_cur = self.data.site_xpos[self.ee_site].copy()
            err = target_pos_m - pos_cur
            if np.linalg.norm(err) < tol:
                return q

            nv = self.model.nv
            jacp = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site)
            # Columns 1–6 = rotational joints (col 0 = rail)
            J = jacp[:, 1:7]
            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
            dq = J_pinv @ err
            dq = np.clip(dq, -np.deg2rad(3.0), np.deg2rad(3.0))
            q += dq

            for i, jid in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)

        return None
```

**A note on `pink` integration:** The `pink` library has gone through several API revisions. The code above targets the modern API but if you get import errors or unexpected behavior, the iterative Jacobian fallback will engage automatically. The fallback is slower but functionally equivalent. Don't let `pink` issues block you — get the system working with the fallback, then optimize.

---

## Phase 6 — FK Validator (`sim/fk_validator.py`)

Validates candidate joint configurations: checks limits, runs FK, performs collision detection. Shares state with the live simulation so collision checks reflect current rail position and object placements.

```python
# sim/fk_validator.py
import mujoco
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    achieved_pos: Optional[np.ndarray] = None
    position_error_mm: Optional[float] = None
    has_collision: bool = False


class FKValidator:
    """
    Validates against the LIVE sim state — shares model + data + lock.
    Snapshots qpos before testing candidate, restores after.
    """

    POSITION_TOLERANCE_MM = 5.0
    ARM_GEOM_NAMES = {
        "base_link", "link1_geom", "link2_geom", "link3_geom",
        "link4_geom", "link5_geom", "link6_geom", "gripper_geom",
        "carriage_geom"
    }

    def __init__(self, model: mujoco.MjModel,
                 data: mujoco.MjData,
                 lock: threading.Lock):
        self.model = model
        self.data  = data
        self.lock  = lock
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.ee_site   = model.site("end_effector").id

    def validate(self, joint_angles_rad: np.ndarray,
                 target_pos_m: np.ndarray,
                 rail_pos_m: Optional[float] = None) -> ValidationResult:
        # Joint limit check (no state change)
        for i, jid in enumerate(self.joint_ids):
            lo, hi = self.model.jnt_range[jid]
            if not (lo <= joint_angles_rad[i] <= hi):
                return ValidationResult(
                    is_valid=False,
                    reason=(f"Joint {i+1} angle "
                            f"{np.rad2deg(joint_angles_rad[i]):.1f}° "
                            f"outside [{np.rad2deg(lo):.1f}°, "
                            f"{np.rad2deg(hi):.1f}°]")
                )

        if rail_pos_m is not None and not (0.0 <= rail_pos_m <= 0.7):
            return ValidationResult(
                is_valid=False,
                reason=f"Rail position {rail_pos_m*1000:.1f}mm outside 0–700mm"
            )

        with self.lock:
            backup_joints = np.array(
                [self.data.qpos[jid] for jid in self.joint_ids]
            )
            backup_rail = float(self.data.qpos[self.rail_jid])

            try:
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = joint_angles_rad[i]
                if rail_pos_m is not None:
                    self.data.qpos[self.rail_jid] = rail_pos_m
                mujoco.mj_forward(self.model, self.data)

                achieved = self.data.site_xpos[self.ee_site].copy()
                error_mm = np.linalg.norm(achieved - target_pos_m) * 1000.0

                if error_mm > self.POSITION_TOLERANCE_MM:
                    return ValidationResult(
                        is_valid=False,
                        reason=(f"FK position error {error_mm:.1f}mm > "
                                f"tolerance {self.POSITION_TOLERANCE_MM}mm"),
                        achieved_pos=achieved,
                        position_error_mm=error_mm
                    )

                mujoco.mj_collision(self.model, self.data)
                if self.data.ncon > 0:
                    arm_hits = []
                    for i in range(self.data.ncon):
                        c = self.data.contact[i]
                        g1 = self.model.geom(c.geom1).name
                        g2 = self.model.geom(c.geom2).name
                        if self.ARM_GEOM_NAMES & {g1, g2}:
                            other = ({g1, g2} - self.ARM_GEOM_NAMES)
                            if other:
                                arm_hits.append((g1, g2))
                    if arm_hits:
                        return ValidationResult(
                            is_valid=False,
                            reason=(f"Collision: {len(arm_hits)} contacts "
                                    f"({arm_hits[:3]})"),
                            achieved_pos=achieved,
                            position_error_mm=error_mm,
                            has_collision=True
                        )

                return ValidationResult(
                    is_valid=True, reason="OK",
                    achieved_pos=achieved,
                    position_error_mm=error_mm
                )

            finally:
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = backup_joints[i]
                self.data.qpos[self.rail_jid] = backup_rail
                mujoco.mj_forward(self.model, self.data)
```

---

## Phase 7 — Simulation Wrapper (`sim/mujoco_env.py`)

`SimXArmAPI` exposes the same interface as the real `XArmAPI`, with added `set_rail_position()` for the linear axis.

```python
# sim/mujoco_env.py
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from typing import Optional

try:
    from transforms3d.euler import mat2euler
    HAS_TRANSFORMS3D = True
except ImportError:
    HAS_TRANSFORMS3D = False

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]
ACT_NAMES   = ["act_rail","act1","act2","act3","act4","act5","act6"]
RAIL_ACT    = 0


class SimXArmAPI:
    """
    Drop-in simulation replacement for xarm.wrapper.XArmAPI.

    Methods accept **kwargs to absorb extra LLM-emitted parameters
    (e.g. speed_mm_s) that don't match the SDK signature exactly.
    """

    def __init__(self, scene_xml: str, render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()
        self._running = True

        self.act_ids   = [self.model.actuator(n).id for n in ACT_NAMES]
        self.joint_ids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = self.model.joint("rail").id
        self.ee_site   = self.model.site("end_effector").id

        from sim.fk_validator import FKValidator
        from sim.ik_solver import IKSolver
        self.validator = FKValidator(self.model, self.data, self.lock)
        self.ik_solver = IKSolver(self.model, self.data, self.lock)

        threading.Thread(target=self._sim_loop, daemon=True).start()
        if render:
            self._launch_viewer()

    def _sim_loop(self):
        while self._running:
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)

    def _launch_viewer(self):
        def _run():
            with mujoco.viewer.launch_passive(self.model, self.data) as v:
                while v.is_running():
                    with self.lock:
                        v.sync()
                    time.sleep(0.016)
        threading.Thread(target=_run, daemon=True).start()
        time.sleep(0.4)

    def motion_enable(self, enable: bool = True) -> int:  return 0
    def set_mode(self, mode: int) -> int:                 return 0
    def set_state(self, state: int) -> int:               return 0
    def disconnect(self):                                  self._running = False

    # ---- rail control ----
    def set_rail_position(self, position_mm: float,
                          speed_mm_s: float = 50.0,
                          wait: bool = True, **kwargs) -> int:
        pos_m = float(np.clip(position_mm / 1000.0, 0.0, 0.7))
        with self.lock:
            self.data.ctrl[self.act_ids[RAIL_ACT]] = pos_m
        if wait:
            self._wait_rail_settled(pos_m)
        return 0

    def get_rail_position(self) -> tuple[int, float]:
        with self.lock:
            pos_m = self.data.qpos[self.rail_jid]
        return 0, pos_m * 1000.0

    def _wait_rail_settled(self, target_m: float,
                           tol: float = 0.002, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                current = self.data.qpos[self.rail_jid]
            if abs(current - target_m) < tol:
                return
            time.sleep(0.05)

    # ---- arm control ----
    def set_position(self, x: float, y: float, z: float,
                     roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                     speed: float = 100.0, wait: bool = True, **kwargs) -> int:
        target_pos = np.array([x, y, z]) / 1000.0

        with self.lock:
            joint_angles = self.ik_solver.solve(target_pos)
        if joint_angles is None:
            print(f"[SimXArm] IK failed for target "
                  f"({x:.1f}, {y:.1f}, {z:.1f}) mm")
            return 1

        result = self.validator.validate(joint_angles, target_pos)
        if not result.is_valid:
            print(f"[SimXArm] Validation failed: {result.reason}")
            return 2

        self._execute_joint_angles(joint_angles)
        return 0

    def set_servo_angle(self, angle, speed: float = 30.0,
                        wait: bool = True, **kwargs) -> int:
        if len(angle) != 6:
            print(f"[SimXArm] Expected 6 joint angles, got {len(angle)}")
            return 1
        self._execute_joint_angles(np.deg2rad(angle))
        return 0

    def get_position(self) -> tuple[int, list]:
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site].copy()
            mat = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
        result = list(pos * 1000.0)
        if HAS_TRANSFORMS3D:
            result += list(np.rad2deg(mat2euler(mat, axes='sxyz')))
        else:
            result += [0.0, 0.0, 0.0]
        return 0, result

    def get_servo_angle(self) -> tuple[int, list]:
        with self.lock:
            angles = [np.rad2deg(self.data.qpos[jid]) for jid in self.joint_ids]
        return 0, angles

    def open_lite6_gripper(self) -> int:
        print("[SimXArm] Gripper open");  return 0

    def close_lite6_gripper(self) -> int:
        print("[SimXArm] Gripper close"); return 0

    def _execute_joint_angles(self, angles_rad: np.ndarray):
        with self.lock:
            for i, angle in enumerate(angles_rad):
                self.data.ctrl[self.act_ids[1 + i]] = float(angle)
```

---

## Phase 8 — Real Hardware Wrapper (`hardware/real_arm.py`)

Same interface as `SimXArmAPI`, routes to real `XArmAPI`. Rail control varies by firmware — verify your specific install.

```python
# hardware/real_arm.py
from xarm.wrapper import XArmAPI

class RealXArmAPI:
    def __init__(self, ip: str):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self._rail_pos_mm = 0.0

    def set_rail_position(self, position_mm: float,
                          speed_mm_s: float = 50.0,
                          wait: bool = True, **kwargs) -> int:
        try:
            ret = self.arm.set_linear_track_pos(
                position_mm, speed=speed_mm_s, wait=wait)
            self._rail_pos_mm = position_mm
            return ret if isinstance(ret, int) else 0
        except AttributeError:
            print(f"[RealArm] set_linear_track_pos unavailable — "
                  f"check firmware/SDK version")
            self._rail_pos_mm = position_mm
            return 0

    def get_rail_position(self) -> tuple[int, float]:
        try:
            ret = self.arm.get_linear_track_pos()
            if isinstance(ret, tuple):
                return ret
            return 0, float(ret)
        except AttributeError:
            return 0, self._rail_pos_mm

    def set_position(self, x, y, z, roll=0, pitch=0, yaw=0,
                     speed=100, wait=True, **kwargs) -> int:
        return self.arm.set_position(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, wait=wait
        )

    def set_servo_angle(self, angle, speed=30, wait=True, **kwargs) -> int:
        return self.arm.set_servo_angle(angle=angle, speed=speed, wait=wait)

    def get_position(self):     return self.arm.get_position()
    def get_servo_angle(self):  return self.arm.get_servo_angle()
    def open_lite6_gripper(self):  return self.arm.open_lite6_gripper()
    def close_lite6_gripper(self): return self.arm.close_lite6_gripper()
    def motion_enable(self, enable=True): return self.arm.motion_enable(enable=enable)
    def set_mode(self, mode):             return self.arm.set_mode(mode)
    def set_state(self, state):           return self.arm.set_state(state)

    def disconnect(self):
        self.arm.disconnect()
```

> **Rail SDK note:** `set_linear_track_pos()` exists in recent xArm SDK versions but the exact name and signature can vary. Before running on hardware, run `dir(self.arm)` in a REPL to confirm available methods.

---

## Phase 9 — Object Registry (`agent/object_registry.py`)

Maps semantic names to physical poses and grasp configurations. This is the "world knowledge" the LLM consults when planning.

```python
# agent/object_registry.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GraspConfig:
    approach_direction: list      # Unit vector toward object
    grip_orientation_rpy: list    # Gripper RPY at grasp (degrees)
    grip_depth: float             # 0..1
    approach_standoff_mm: float


@dataclass
class LabObject:
    name: str
    aliases: list
    position_xyz_m: list
    grasp: GraspConfig
    safety_notes: str
    optimal_rail_mm: float = 350.0
    is_container: bool = False    # True for bins (place destinations)
    last_updated: str = ""


class ObjectRegistry:

    def __init__(self, registry_path: str = "agent/objects.json"):
        self.path = Path(registry_path)
        self.objects: dict = {}
        if self.path.exists():
            self.load()

    def register(self, obj: LabObject):
        self.objects[obj.name] = obj
        self.save()

    def find(self, query: str) -> Optional[LabObject]:
        q = query.lower().strip()
        for obj in self.objects.values():
            if q == obj.name.lower():
                return obj
            if any(q in alias.lower() for alias in obj.aliases):
                return obj
        return None

    def to_llm_context(self) -> str:
        cubes  = [o for o in self.objects.values() if not o.is_container]
        bins   = [o for o in self.objects.values() if o.is_container]
        lines  = []

        if cubes:
            lines.append("## Cubes (graspable)\n")
            for obj in cubes:
                x, y, z = obj.position_xyz_m
                lines.append(
                    f"- **{obj.name}**  aliases: {', '.join(obj.aliases)}\n"
                    f"  Position: x={x*1000:.0f}mm  y={y*1000:.0f}mm  z={z*1000:.0f}mm\n"
                    f"  Optimal rail: {obj.optimal_rail_mm:.0f}mm\n"
                )
        if bins:
            lines.append("\n## Bins (place destinations)\n")
            for obj in bins:
                x, y, z = obj.position_xyz_m
                lines.append(
                    f"- **{obj.name}**  aliases: {', '.join(obj.aliases)}\n"
                    f"  Position: x={x*1000:.0f}mm  y={y*1000:.0f}mm  z={z*1000:.0f}mm\n"
                    f"  Optimal rail: {obj.optimal_rail_mm:.0f}mm\n"
                )
        return "\n".join(lines)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {k: asdict(v) for k, v in self.objects.items()}, indent=2))

    def load(self):
        data = json.loads(self.path.read_text())
        for k, v in data.items():
            v["grasp"] = GraspConfig(**v["grasp"])
            self.objects[k] = LabObject(**v)


def build_default_registry() -> ObjectRegistry:
    """
    Default registry matching the cubes-and-bins scene.
    Replace with lab-specific objects when ready.
    """
    reg = ObjectRegistry()

    # Standard grasp config for top-down cube pickup
    cube_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=40.0,
    )
    bin_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0,                  # No grip — bins are place targets
        approach_standoff_mm=60.0,       # Stop higher to release into bin
    )

    # Cubes — match XML positions exactly
    reg.register(LabObject(
        name="red_cube",
        aliases=["red cube", "red block", "red"],
        position_xyz_m=[-0.20, 0.15, 0.78],
        optimal_rail_mm=150.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
    ))
    reg.register(LabObject(
        name="green_cube",
        aliases=["green cube", "green block", "green"],
        position_xyz_m=[0.00, 0.15, 0.78],
        optimal_rail_mm=350.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
    ))
    reg.register(LabObject(
        name="blue_cube",
        aliases=["blue cube", "blue block", "blue"],
        position_xyz_m=[0.20, 0.15, 0.78],
        optimal_rail_mm=550.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
    ))

    # Bins — place destinations
    reg.register(LabObject(
        name="red_bin",
        aliases=["red bin", "red container", "red box"],
        position_xyz_m=[-0.20, 0.35, 0.75],
        optimal_rail_mm=150.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
    ))
    reg.register(LabObject(
        name="green_bin",
        aliases=["green bin", "green container", "green box"],
        position_xyz_m=[0.00, 0.35, 0.75],
        optimal_rail_mm=350.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
    ))
    reg.register(LabObject(
        name="blue_bin",
        aliases=["blue bin", "blue container", "blue box"],
        position_xyz_m=[0.20, 0.35, 0.75],
        optimal_rail_mm=550.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
    ))

    return reg
```

---

## Phase 10 — Recording Backend (`recording.py`)

Identical to the basic v5 doc's `recording.py`, with the addition of an `LLMSessionLog` class. The state trajectory format is interchangeable between manual and LLM recordings.

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
    # LLM fields
    llm_model: str = ""
    llm_prompt: str = ""
    has_llm_log: bool = False
    # Augmentation fields
    parent_session_id: str = ""
    cycle_index: int = 0
    augmentation_config: dict = field(default_factory=dict)


class Recorder:

    def __init__(self, model, data, lock, interface, scene_xml="envs/lab_scene.xml",
                 state_hz=DEFAULT_STATE_HZ):
        self.model = model; self.data = data; self.lock = lock
        self.interface = interface; self.scene_xml = scene_xml
        self.state_hz = state_hz
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.act_ids   = [model.actuator(n).id for n in ACT_NAMES]
        self.ee_site   = model.site("end_effector").id
        self._recording = False
        self._session = None; self._session_dir = None
        self._commands_file = None; self._state_buffer = []
        self._cmd_lock = threading.Lock()
        self._state_thread = None; self._start_wall_time = 0.0

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
        print(f"[Recorder] ● REC  session={sid}")
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
            print(f"[Recorder] ✗ Discarded {self._session_dir}")
            self._cleanup_session_dir()
            path = None
        else:
            print(f"[Recorder] ✓ Saved   {self._session_dir}")
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
        from transforms3d.euler import mat2euler
        ee_rpy = np.array(mat2euler(ee_mat, axes='sxyz'), dtype=np.float32)
        self._state_buffer.append({
            "t_wall": time.time() - self._start_wall_time,
            "t_sim": t_sim, "rail_mm": rail_m * 1000.0,
            "joints_deg": np.rad2deg(joints_r),
            "ee_pos_mm": ee_pos * 1000.0,
            "ee_rpy_deg": np.rad2deg(ee_rpy),
            "ctrl": ctrl,
        })

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
            f.attrs["state_hz"] = self.state_hz
            f.attrs["n_samples"] = len(b)
            f.attrs["joint_names"] = JOINT_NAMES
            f.attrs["actuator_names"] = ACT_NAMES
        self._state_buffer = []

    def _write_metadata(self):
        if self._session is None or self._session_dir is None: return
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
    print(f"[Recorder] → trash: {dest}")
    return True


def restore_session(session_name):
    src = TRASH_DIR / session_name
    if not src.exists():
        print(f"[Recorder] Not in trash: {session_name}"); return False
    dest = RECORDINGS_ROOT / session_name
    if dest.exists():
        print(f"[Recorder] Cannot restore — {dest} already exists"); return False
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
```

---

## Phase 11 — LLM Brain (`agent/llm_brain.py`)

```python
# agent/llm_brain.py
import anthropic
import json
import re
import time
from typing import Optional
from agent.object_registry import ObjectRegistry
from recording import Recorder, LLMSessionLog


MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}

DEFAULT_MODEL = "haiku"


def resolve_model(short_or_full: str) -> str:
    return MODELS.get(short_or_full, short_or_full)


def prompt_model_choice() -> str:
    print("\nChoose Claude model for this session:")
    print("  1) Haiku  4.5  — fastest, cheapest, default for routine tasks")
    print("  2) Sonnet 4.6  — balanced")
    print("  3) Opus   4.7  — most capable, slowest, for novel/complex tasks")
    try:
        choice = input("\nSelection [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    if choice in ("", "1"): return "haiku"
    if choice == "2": return "sonnet"
    if choice == "3": return "opus"
    print(f"Unrecognized '{choice}' — using haiku")
    return "haiku"


SYSTEM_PROMPT_TEMPLATE = """\
You are the control brain for a UFACTORY xArm6 mounted on a 700mm linear rail \
in a benchmark pick-and-place environment.

## Your 7 degrees of freedom
- Rail: 0–700mm linear axis along X. Move rail FIRST to get the arm near the target.
- Joints 1–6: rotational axes of the xArm6.

## Command vocabulary (output as JSON array)
- set_rail        params: position_mm, speed_mm_s
- move_to         params: x, y, z, roll, pitch, yaw, speed_mm_s (mm, deg)
- set_joints      params: angles_deg (6 floats)
- gripper_open    params: {{}}
- gripper_close   params: {{}}
- get_pose        params: {{}}
- search_workspace  params: object_name
- wait            params: seconds
- done            params: message

## Motion planning rules
1. ALWAYS call set_rail FIRST. Use optimal_rail_mm from the registry as your target.
2. For pick-and-place: rail to object → move_to above → lower → gripper_close →
   move_to lift height → rail to destination → move_to above → lower → gripper_open.
3. Keep speed_mm_s <= 100. Use 50–80 for grasps, 100 for transit.
4. Object coordinates from the registry are in millimeters in world frame.
   The arm base translates along the rail; pass world coordinates to move_to.
5. If a task is ambiguous, output done() with a message asking for clarification.

## Output format
JSON array ONLY — no prose, no markdown fences. Example for "put red cube in red bin":
[
  {{"action": "set_rail",      "params": {{"position_mm": 150, "speed_mm_s": 100}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 830, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 795, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 50}}}},
  {{"action": "gripper_close", "params": {{}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 150, "z": 870, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 350, "z": 870, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 80}}}},
  {{"action": "move_to",       "params": {{"x": -200, "y": 350, "z": 810, "roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": 50}}}},
  {{"action": "gripper_open",  "params": {{}}}},
  {{"action": "done",          "params": {{"message": "Red cube placed in red bin"}}}}
]

## Current scene registry
{registry_context}
"""


class LLMBrain:

    def __init__(self, arm_api, registry: ObjectRegistry,
                 recorder: Optional[Recorder] = None,
                 model: str = DEFAULT_MODEL):
        self.arm = arm_api
        self.registry = registry
        self.recorder = recorder
        self.client = anthropic.Anthropic()
        self.model_full = resolve_model(model)
        self.model_short = model if model in MODELS else "custom"
        self.history = []
        print(f"[LLMBrain] Using model: {self.model_short} ({self.model_full})")

    def execute_task(self, task_prompt: str, dry_run: bool = False) -> dict:
        llm_log = None
        if self.recorder is not None and self.recorder.is_recording:
            llm_log = LLMSessionLog(self.recorder, self.model_full, task_prompt)
            llm_log.log_prompt()

        system = SYSTEM_PROMPT_TEMPLATE.format(
            registry_context=self.registry.to_llm_context()
        )
        self.history.append({"role": "user", "content": task_prompt})

        t_start = time.time()
        try:
            response = self.client.messages.create(
                model=self.model_full, max_tokens=2048,
                system=system, messages=self.history,
            )
        except anthropic.APIError as e:
            if llm_log:
                llm_log.log_response(f"API ERROR: {e}", time.time() - t_start)
                llm_log.close()
            print(f"[LLMBrain] Claude API error: {e}")
            return {"commands": [], "results": [], "raw": str(e), "error": True}
        latency = time.time() - t_start
        raw = response.content[0].text
        in_tok  = getattr(response.usage, "input_tokens", 0)
        out_tok = getattr(response.usage, "output_tokens", 0)
        if llm_log:
            llm_log.log_response(raw, latency, in_tok, out_tok)
        print(f"[LLMBrain] Response in {latency:.1f}s "
              f"({in_tok}→{out_tok} tokens)")
        self.history.append({"role": "assistant", "content": raw})

        try:
            commands = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"(\[.*\])", raw, re.DOTALL)
            if m:
                commands = json.loads(m.group(1))
            else:
                if llm_log:
                    llm_log.log_parse_error(f"No JSON array:\n{raw}")
                    llm_log.close()
                raise ValueError(f"LLM returned non-JSON:\n{raw}")
        if llm_log:
            llm_log.log_parsed(commands)

        results = []
        if not dry_run:
            results = self._run(commands, llm_log)

        if llm_log:
            llm_log.close()
        return {"commands": commands, "results": results, "raw": raw,
                "latency_s": latency,
                "input_tokens": in_tok, "output_tokens": out_tok}

    def _run(self, commands, llm_log=None):
        results = []
        for cmd in commands:
            action = cmd["action"]
            params = cmd.get("params", {})
            result = self._dispatch(action, params)
            results.append({"action": action, "result": result})
            if llm_log:
                llm_log.log_dispatch(action, params, result)
            if self.recorder and self.recorder.is_recording:
                self.recorder.log_command("llm_dispatch",
                    {"action": action, "params": params, "result": result})
            print(f"[Agent] {action}({params}) → {result}")
            if result != 0 and action not in ("done", "wait", "get_pose"):
                print("[Agent] Command failed — halting sequence")
                break
        return results

    def _move_to(self, p):
        speed = p.get("speed_mm_s", p.get("speed", 100.0))
        return self.arm.set_position(
            x=p["x"], y=p["y"], z=p["z"],
            roll=p.get("roll", 0.0), pitch=p.get("pitch", 0.0),
            yaw=p.get("yaw", 0.0),
            speed=speed, wait=p.get("wait", True),
        )

    def _set_rail(self, p):
        return self.arm.set_rail_position(
            position_mm=p["position_mm"],
            speed_mm_s=p.get("speed_mm_s", 50.0),
            wait=p.get("wait", True),
        )

    def _set_joints(self, p):
        return self.arm.set_servo_angle(
            angle=p["angles_deg"],
            speed=p.get("speed_deg_s", p.get("speed", 30.0)),
            wait=p.get("wait", True),
        )

    def _get_pose(self, p):
        ee = self.arm.get_position()
        rail = self.arm.get_rail_position()
        print(f"[Agent] Pose: ee={ee}  rail={rail}")
        return 0

    def _search(self, name):
        obj = self.registry.find(name)
        if obj is None:
            print(f"[Agent] '{name}' not in registry"); return 1
        print(f"[Agent] Found '{name}' at {obj.position_xyz_m}  "
              f"rail: {obj.optimal_rail_mm}mm")
        return 0

    def _dispatch(self, action, params):
        d = {
            "set_rail":         self._set_rail,
            "move_to":          self._move_to,
            "set_joints":       self._set_joints,
            "gripper_open":     lambda p: self.arm.open_lite6_gripper(),
            "gripper_close":    lambda p: self.arm.close_lite6_gripper(),
            "get_pose":         self._get_pose,
            "wait":             lambda p: time.sleep(p.get("seconds", 1)) or 0,
            "done":             lambda p: print(f"[Done] {p.get('message','')}") or 0,
            "search_workspace": lambda p: self._search(p["object_name"]),
        }
        h = d.get(action)
        if h is None:
            print(f"[Agent] Unknown action: {action}"); return -1
        return h(params)
```

---

## Phase 12 — Prompt Variants (`agent/prompt_variants.py`)

```python
# agent/prompt_variants.py
import anthropic
import json
import os
import re
import subprocess
import tempfile
from typing import List


VARIANT_GEN_SYSTEM_PROMPT = """\
You are generating paraphrased variants of a robot pick-and-place command.
The robot is an xArm6 on a linear rail in a benchmark scene with three RGB
cubes (red, green, blue) and three matching bins.

Given an original command, produce N paraphrased variants that:
1. Preserve the EXACT task intent — same source cube, same destination bin,
   same action verb category. "Put red cube in red bin" can become "place
   the red block in the red container" but NOT "put green cube in red bin".
2. Vary surface form — synonyms, word order, formality, contractions.
3. Cover a range of phrasings: formal ("Could you please place the red cube
   into the red container?") to casual ("red in red plz").
4. Are distinct — no two identical, none merely punctuation-different.

Output ONLY a JSON array of strings. No prose, no markdown fences.
"""


def generate_variants(original_prompt: str, n_variants: int,
                      model: str = "claude-haiku-4-5-20251001") -> List[str]:
    client = anthropic.Anthropic()
    user_msg = (
        f"Original command: \"{original_prompt}\"\n\n"
        f"Generate exactly {n_variants} paraphrased variants. "
        f"Output as a JSON array of strings."
    )
    response = client.messages.create(
        model=model, max_tokens=1024,
        system=VARIANT_GEN_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\])", raw, re.DOTALL)
        if not m:
            raise ValueError(f"Variant generator returned non-JSON:\n{raw}")
        variants = json.loads(m.group(1))
    if not isinstance(variants, list):
        raise ValueError(f"Expected JSON array, got: {type(variants)}")
    variants = [str(v).strip() for v in variants if isinstance(v, str) and v.strip()]
    if len(variants) < n_variants:
        raise ValueError(
            f"Asked for {n_variants}, got {len(variants)}:\n{variants}"
        )
    return variants[:n_variants]


def preview_and_confirm(original: str, variants: List[str]) -> List[str]:
    print("\n" + "═" * 70)
    print("  VARIANT PROMPTS")
    print("═" * 70)
    print(f"\n  Original (cycle 1):\n    \"{original}\"")
    print(f"\n  Variants (cycles 2..{len(variants)+1}):")
    for i, v in enumerate(variants, start=2):
        print(f"    {i}: \"{v}\"")
    print("\n" + "─" * 70)

    while True:
        try:
            ans = input("Accept? [Y/n/e=edit in $EDITOR]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if ans in ("", "y"):
            return variants
        if ans == "n":
            return []
        if ans == "e":
            edited = _edit_in_editor(original, variants)
            if edited:
                print("\nEdited list:")
                print(f"  Original: \"{original}\"")
                for i, v in enumerate(edited, start=2):
                    print(f"  {i}: \"{v}\"")
                variants = edited
                continue
            else:
                print("Edit cancelled — keeping previous list.")
                continue
        print("Please answer y, n, or e.")


def _edit_in_editor(original: str, variants: List[str]) -> List[str]:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("# One variant per line. Lines starting with # are comments.\n")
        f.write(f"# Original (DO NOT EDIT — reference only):\n# {original}\n#\n")
        f.write("# Variants — edit, add, remove freely. Save and close to confirm.\n\n")
        for v in variants:
            f.write(v + "\n")
        tmp = f.name
    try:
        subprocess.run([editor, tmp], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Editor failed: {e}")
        os.unlink(tmp); return []
    with open(tmp) as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith("#")]
    os.unlink(tmp)
    return lines
```

---

## Phase 13 — Single-Task Run (`scripts/run_task.py`)

```python
# scripts/run_task.py
import argparse
import sys
import threading
import time

from agent.llm_brain import LLMBrain, prompt_model_choice, MODELS
from agent.object_registry import build_default_registry
from recording import Recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("--mode", choices=["sim","real"], default="sim")
    parser.add_argument("--ip", default="192.168.1.100")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    args = parser.parse_args()

    model_short = args.model if args.model else prompt_model_choice()

    if args.mode == "sim":
        from sim.mujoco_env import SimXArmAPI
        arm = SimXArmAPI(scene_xml="envs/lab_scene.xml",
                         render=not args.no_render)
        print("[System] SIMULATION mode")
    else:
        from hardware.real_arm import RealXArmAPI
        arm = RealXArmAPI(ip=args.ip)
        print(f"[System] REAL HARDWARE mode — {args.ip}")

    recorder = None
    if not args.no_record:
        recorder = Recorder(
            model=arm.model if hasattr(arm, "model") else None,
            data=arm.data if hasattr(arm, "data") else None,
            lock=arm.lock if hasattr(arm, "lock") else threading.Lock(),
            interface="llm_brain", scene_xml="envs/lab_scene.xml",
        )
        if hasattr(arm, "model"):
            recorder.start()
        else:
            print("[System] Recording requires sim mode (or real-hardware state poller).")
            recorder = None

    registry = build_default_registry()
    brain = LLMBrain(arm=arm, registry=registry, recorder=recorder,
                     model=model_short)

    print(f"\n[Task] {args.task}\n")
    try:
        result = brain.execute_task(args.task, dry_run=args.dry_run)
    except Exception as e:
        print(f"[System] Task failed: {e}")
        result = {"commands": [], "results": [], "error": True}

    print("\n[Planned sequence]")
    for i, cmd in enumerate(result.get("commands", [])):
        print(f"  {i+1:2d}. {cmd['action']}  {cmd.get('params', {})}")
    if not args.dry_run and result.get("results"):
        print("\n[Execution results]")
        for r in result["results"]:
            ok = "✓" if r["result"] == 0 else "✗"
            print(f"  {ok}  {r['action']}  →  {r['result']}")

    if recorder is not None:
        time.sleep(1.5)
        recorder.stop_and_prompt(prompt=True, auto_task_label=args.task)

    arm.disconnect()


if __name__ == "__main__":
    main()
```

---

## Phase 14 — Augmented Run (`scripts/run_task_augmented.py`)

```python
# scripts/run_task_augmented.py
import argparse
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from agent.llm_brain import (
    LLMBrain, prompt_model_choice, MODELS, resolve_model,
)
from agent.object_registry import build_default_registry
from agent.prompt_variants import generate_variants, preview_and_confirm
from envs.scene_randomizer import (
    randomize_scene, DEFAULT_POS_JITTER_MM, DEFAULT_ROT_JITTER_DEG,
)
from recording import Recorder


def run_one_cycle(task_prompt, cycle_index, parent_session_id, original_prompt,
                  model_short, pos_jitter_mm, rot_jitter_deg, seed,
                  render, no_record, dry_run) -> Optional[Path]:
    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=render)

    scene_summary = {}
    if cycle_index > 1 and (pos_jitter_mm > 0 or rot_jitter_deg > 0):
        rng = np.random.default_rng(seed)
        with arm.lock:
            scene_summary = randomize_scene(
                arm.model, arm.data,
                pos_jitter_mm=pos_jitter_mm,
                rot_jitter_deg=rot_jitter_deg,
                initial_joint_jitter_deg=0.0,
                rng=rng,
            )

    recorder = None
    if not no_record:
        recorder = Recorder(
            model=arm.model, data=arm.data, lock=arm.lock,
            interface="llm_brain_augmented", scene_xml="envs/lab_scene.xml",
        )
        recorder.start()
        recorder.session.parent_session_id = parent_session_id
        recorder.session.cycle_index = cycle_index
        recorder.session.augmentation_config = {
            "parent_prompt": original_prompt,
            "variant_prompt": task_prompt,
            "object_pose_jitter_mm": pos_jitter_mm if cycle_index > 1 else 0.0,
            "object_rotation_jitter_deg": rot_jitter_deg if cycle_index > 1 else 0.0,
            "scene_perturbations": scene_summary,
            "seed": seed if cycle_index > 1 else 0,
        }

    registry = build_default_registry()
    brain = LLMBrain(arm=arm, registry=registry, recorder=recorder,
                     model=model_short)
    print(f"\n  [Cycle {cycle_index}] prompt: \"{task_prompt}\"")
    try:
        result = brain.execute_task(task_prompt, dry_run=dry_run)
    except Exception as e:
        print(f"  [Cycle {cycle_index}] task error: {e}")
        result = {"commands": [], "results": [], "error": True}
    n_cmds = len(result.get("commands", []))
    print(f"  [Cycle {cycle_index}] {n_cmds} commands, "
          f"{result.get('latency_s', 0):.1f}s latency")

    saved_path = None
    if recorder is not None:
        time.sleep(1.5)
        saved_path = _stop_silent(recorder)
    arm.disconnect()
    return saved_path


def _stop_silent(recorder):
    if not recorder.is_recording: return None
    recorder._recording = False
    if recorder._state_thread is not None:
        recorder._state_thread.join(timeout=1.0)
    recorder._session.ended_at_iso = datetime.now().isoformat()
    recorder._session.duration_s = time.time() - recorder._start_wall_time
    recorder._session.n_state_samples = len(recorder._state_buffer)
    if recorder._commands_file is not None:
        recorder._commands_file.close(); recorder._commands_file = None
    recorder._write_trajectory()
    recorder._session.kept = True
    recorder._write_metadata()
    return recorder._session_dir


def batch_annotate(saved_dirs):
    import json
    print("\n" + "═" * 70)
    print("  BATCH ANNOTATION")
    print("═" * 70)
    print(f"  {sum(1 for d in saved_dirs if d)} recordings created.")
    print("─" * 70)
    for i, d in enumerate(saved_dirs):
        if d is None: continue
        meta_path = d / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        variant_prompt = meta.get("augmentation_config", {}).get("variant_prompt", "")
        cycle = meta.get("cycle_index", "")
        print(f"\n[{i+1}/{len(saved_dirs)}] {d.name}")
        print(f"     cycle: {cycle}  prompt: \"{variant_prompt}\"")
        try:
            outcome = input("     Outcome [s/f/d/Enter]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if outcome.startswith("d"):
            print(f"     ✗ Deleting {d.name}"); shutil.rmtree(d); continue
        if outcome.startswith("s"):   meta["outcome"] = "success"
        elif outcome.startswith("f"): meta["outcome"] = "failure"
        try:
            note = input("     Notes (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
        if note: meta["notes"] = note
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Original task command")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None)
    parser.add_argument("--variant-model", default="haiku")
    parser.add_argument("--pos-jitter-mm", type=float, default=DEFAULT_POS_JITTER_MM)
    parser.add_argument("--rot-jitter-deg", type=float, default=DEFAULT_ROT_JITTER_DEG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")
    parser.add_argument("--auto-accept-variants", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_short = args.model if args.model else prompt_model_choice()

    n_variants = max(args.cycles - 1, 0)
    if n_variants > 0:
        print(f"\n[Augment] Generating {n_variants} variants via "
              f"{args.variant_model}...")
        try:
            variants = generate_variants(
                original_prompt=args.task, n_variants=n_variants,
                model=resolve_model(args.variant_model),
            )
        except Exception as e:
            print(f"[Augment] Variant generation failed: {e}")
            sys.exit(1)
        if args.auto_accept_variants:
            print(f"[Augment] Auto-accepted {len(variants)} variants.")
        else:
            variants = preview_and_confirm(args.task, variants)
            if not variants:
                print("[Augment] Variants rejected — exiting."); return
    else:
        variants = []

    base_seed = args.seed if args.seed is not None else int(time.time())
    print(f"\n[Augment] Base seed: {base_seed}")
    print(f"[Augment] Cycles: {args.cycles}")
    print(f"[Augment] Robot model: {model_short}")
    print(f"[Augment] Pos jitter: ±{args.pos_jitter_mm}mm  "
          f"Rot jitter: ±{args.rot_jitter_deg}°")

    parent_id = uuid.uuid4().hex[:8]
    all_prompts = [args.task] + variants[:args.cycles - 1]
    saved_dirs = []

    for i, prompt in enumerate(all_prompts, start=1):
        cycle_seed = base_seed + i * 1000
        saved = run_one_cycle(
            task_prompt=prompt, cycle_index=i,
            parent_session_id=parent_id, original_prompt=args.task,
            model_short=model_short,
            pos_jitter_mm=args.pos_jitter_mm,
            rot_jitter_deg=args.rot_jitter_deg,
            seed=cycle_seed,
            render=not args.no_render,
            no_record=args.no_record,
            dry_run=args.dry_run,
        )
        saved_dirs.append(saved)

    if not args.no_record and not args.no_annotate:
        batch_annotate(saved_dirs)

    n = sum(1 for d in saved_dirs if d)
    print(f"\n✓ {n} sessions saved.")
    print(f"  Parent ID: {parent_id}")
    print(f"  Inspect: python replay.py")


if __name__ == "__main__":
    main()
```

---

## Phase 15 — Replay Tool (`replay.py`)

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

SCENE_XML = "envs/lab_scene.xml"


def list_sessions(include_trash=False, filter_outcome=None):
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
        mp = path / "metadata.json"
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        if filter_outcome and meta.get("outcome", "") != filter_outcome:
            continue
        sessions.append((path, meta, in_trash))
    return sessions


def print_session_list(sessions):
    if not sessions:
        print("No sessions found."); return
    print(f"\n{'#':>3}  {'status':>6}  {'name':<48}  task                  model      outcome   dur")
    print("─" * 130)
    for i, (p, m, t) in enumerate(sessions):
        tag = "TRASH" if t else ("kept" if m.get("kept") else "draft")
        task    = (m.get("task_label") or "")[:21]
        model   = (m.get("llm_model") or "—")[:10]
        outcome = (m.get("outcome") or "")[:9]
        dur     = m.get("duration_s", 0)
        cycle   = m.get("cycle_index", 0)
        cs = f" c{cycle}" if cycle else ""
        print(f"[{i:>2d}] {tag:>6}  {p.name:<48}  {task:<21} {model:<10} {outcome:<9} {dur:>5.1f}s{cs}")
    print()


def resolve_session(arg, sessions):
    if arg is None: return None
    if arg.isdigit():
        idx = int(arg)
        if 0 <= idx < len(sessions):
            return sessions[idx][0]
    p = Path(arg)
    if p.is_dir(): return p
    for root in (RECORDINGS_ROOT, TRASH_DIR):
        p2 = root / arg
        if p2.is_dir(): return p2
    return None


def print_llm_session(session_dir):
    log_path = session_dir / "llm_session.jsonl"
    if not log_path.exists(): return False
    print("\n" + "═" * 70)
    print(" LLM SESSION LOG")
    print("═" * 70)
    with open(log_path) as f:
        for line in f:
            ev = json.loads(line)
            t = ev.get("t", 0); kind = ev.get("event", "?")
            if kind == "user_prompt":
                print(f"\n[{t:>6.2f}s]  USER  (model: {ev['model']})")
                print(f"             '{ev['prompt']}'")
            elif kind == "llm_response":
                print(f"\n[{t:>6.2f}s]  CLAUDE  "
                      f"(latency: {ev['latency_s']:.1f}s, "
                      f"tokens: {ev['input_tokens']}→{ev['output_tokens']})")
                for ln in ev["raw_text"].split("\n")[:20]:
                    print(f"             {ln}")
            elif kind == "parsed_commands":
                print(f"\n[{t:>6.2f}s]  PARSED  ({len(ev['commands'])} commands)")
                for i, c in enumerate(ev["commands"]):
                    print(f"             {i+1:>2d}. {c['action']}  {c.get('params', {})}")
            elif kind == "parse_error":
                print(f"\n[{t:>6.2f}s]  PARSE ERROR  {ev['error']}")
            elif kind == "dispatch":
                ok = "✓" if ev["result"] == 0 else "✗"
                print(f"[{t:>6.2f}s]  DISPATCH  {ok}  "
                      f"{ev['action']}({ev['params']})  →  {ev['result']}")
    print("\n" + "═" * 70)
    return True


def replay_trajectory(session_dir, speed=1.0, loop=False):
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
          f"{t_wall[-1]:.1f}s, {speed}x)\n")

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
                if wait > 0: time.sleep(wait)
                with lock:
                    data.ctrl[act_ids[0]] = rail_mm[i] / 1000.0
                    for j in range(6):
                        data.ctrl[act_ids[1 + j]] = np.deg2rad(joints_deg[i, j])
            if not loop: break
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
            with lock: v.sync()
            time.sleep(0.016)


def confirm(prompt, magic_word):
    try:
        ans = input(f"{prompt}\nType '{magic_word}' to confirm: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans == magic_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--include-trash", action="store_true")
    parser.add_argument("--outcome")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--restore")
    parser.add_argument("--purge-trash", action="store_true")
    args = parser.parse_args()

    if args.purge_trash:
        if not TRASH_DIR.exists(): print("Trash empty."); return
        n = sum(1 for d in TRASH_DIR.iterdir() if d.is_dir())
        if n == 0: print("Trash empty."); return
        if confirm(f"Permanently delete {n} session(s)?", "purge"):
            print(f"Purged {purge_trash()}.")
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
        if confirm(f"Move '{target.name}' to trash?", "delete"):
            soft_delete_session(target)
        else:
            print("Cancelled.")
        return

    if args.plan_only:
        had = print_llm_session(target)
        if not had:
            print("(No LLM session log — manual recording.)")
        return

    print_llm_session(target)
    replay_trajectory(target, speed=args.speed, loop=args.loop)

    print()
    try:
        ans = input("Delete this session? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans == "y":
        if confirm(f"Move '{target.name}' to trash?", "delete"):
            soft_delete_session(target)


if __name__ == "__main__":
    main()
```

---

## Phase 16 — Data Collection Workflow

### Step 1 — Validate the system (1 day)

```bash
# 1. Verify scene loads
python -c "import mujoco, mujoco.viewer; mujoco.viewer.launch(mujoco.MjModel.from_xml_path('envs/lab_scene.xml'))"

# 2. Single task with Haiku
python scripts/run_task.py "Put the red cube in the red bin" --model haiku

# 3. Watch the MuJoCo viewer. The arm should:
#    - Move rail to ~150mm (above red cube)
#    - Lower over red cube, close gripper
#    - Lift, traverse to bin position, lower, release
```

If the arm makes incorrect moves: check `recordings/<latest>/llm_session.jsonl` to see exactly what Claude planned. Common issues:

- IK fails on some targets → install `pink` (Phase 0), or accept the slower fallback
- Cube positions don't match XML → check `agent/object_registry.py` against the scene XML
- Arm reaches wrong height → adjust `optimal_rail_mm` in registry, or the heights in the system prompt example

### Step 2 — Build a task list

```python
# Suggested initial set
TASKS = [
    "Put the red cube in the red bin",
    "Put the green cube in the green bin",
    "Put the blue cube in the blue bin",
    "Put the red cube in the green bin",
    "Put the green cube in the blue bin",
    "Put the blue cube in the red bin",
    "Move the red cube to the right",
    "Move the blue cube to the left",
    "Sort the cubes by color",
    "Put all cubes in the green bin",
]
```

Document these in a `tasks.txt` file so undergrads use consistent phrasing. Inconsistent task labels create dataset fragmentation.

### Step 3 — Augmented data generation (overnight)

```bash
# Generate 15 cycles per task — original + 14 variants
for task in "Put the red cube in the red bin" \
            "Put the green cube in the green bin" \
            "Put the blue cube in the blue bin"; do
    python scripts/run_task_augmented.py "$task" \
        --cycles 15 --model haiku --no-render \
        --auto-accept-variants --seed $RANDOM
done
```

A 10-task set × 15 cycles × Haiku ≈ $0.15 in API cost. Runs in ~30 minutes per task on a typical laptop.

### Step 4 — Annotate

```bash
# Walk through all recordings
python replay.py  # List them
python replay.py 0  # Watch each one
```

For each cycle, mark `success`, `failure`, or `delete`. Keep all successes, ~25% of failures (these are valuable negative examples), delete obvious garbage.

### Step 5 — Quality filter for training

When preparing data for VLA training:

```python
# Filter script (example)
from pathlib import Path
import json

useful = []
for s in Path("recordings").iterdir():
    if not s.is_dir() or s.name == "trash": continue
    meta = json.loads((s / "metadata.json").read_text())
    if not meta.get("kept"): continue
    if meta.get("outcome") == "success":
        useful.append(s)
    elif meta.get("outcome") == "failure":
        # Keep ~25% of failures
        if hash(meta["session_id"]) % 4 == 0:
            useful.append(s)

print(f"Training set: {len(useful)} sessions")
```

---

## Cost & Latency Reference

Typical per-task numbers for the cube-and-bin tasks in this scene:

| Model | Latency | Per-task cost | Plan quality |
|---|---|---|---|
| Haiku 4.5 | 1.5–3s | ~$0.001 | Usually good for templated pick-place |
| Sonnet 4.6 | 3–5s | ~$0.01 | Better for unusual phrasings |
| Opus 4.7 | 5–10s | ~$0.05 | Best for multi-step or novel tasks |

After 1000 task executions: Haiku ~$1, Sonnet ~$10, Opus ~$50.

**Recommendation:** Use Haiku as default. Escalate to Sonnet for tasks where Haiku produces visibly wrong plans. Use Opus only when Sonnet also struggles — typically multi-cube sorting or implicit-instruction tasks.

---

## File Format Reference

### `metadata.json`

```json
{
  "session_id": "a1b2c3d4",
  "interface": "llm_brain",
  "task_label": "put_the_red_cube_in_the",
  "outcome": "success",
  "scene_xml": "envs/lab_scene.xml",
  "state_hz": 60.0,
  "n_commands": 9,
  "n_state_samples": 5200,
  "kept": true,
  "llm_model": "claude-haiku-4-5-20251001",
  "llm_prompt": "Put the red cube in the red bin",
  "has_llm_log": true,
  "parent_session_id": "",
  "cycle_index": 0,
  "augmentation_config": {}
}
```

### `llm_session.jsonl`

```
{"event": "user_prompt", "model": "claude-haiku-4-5-20251001", "prompt": "Put the red cube in the red bin", "t": 0.002}
{"event": "llm_response", "raw_text": "[{\"action\": ...}]", "latency_s": 1.8, "input_tokens": 612, "output_tokens": 387, "t": 1.823}
{"event": "parsed_commands", "commands": [{"action": "set_rail", "params": {"position_mm": 150, "speed_mm_s": 100}}, ...], "t": 1.825}
{"event": "dispatch", "action": "set_rail", "params": {"position_mm": 150, "speed_mm_s": 100}, "result": 0, "t": 1.831}
```

### `trajectory.h5`

Same schema as the basic v5 doc. HDF5 columns: `t_wall`, `t_sim`, `rail_mm`, `joints_deg(N,6)`, `ee_pos_mm(N,3)`, `ee_rpy_deg(N,3)`, `ctrl(N,7)`. Format is identical between manual and LLM recordings — downstream training pipelines see one unified dataset.

---

## Quick Reference

```bash
# Single LLM task
python scripts/run_task.py "Put the red cube in the red bin"
python scripts/run_task.py "Put the red cube in the red bin" --model sonnet

# Plan only (no motion)
python scripts/run_task.py "Put red in red" --dry-run

# Augmented run (variants + scene jitter)
python scripts/run_task_augmented.py "Put red in red" --cycles 10

# Headless batch (overnight)
python scripts/run_task_augmented.py "Put red in red" \
    --cycles 30 --no-render --auto-accept-variants --seed 1001

# Real hardware
python scripts/run_task.py "Put red in red" --mode real --ip 192.168.1.100

# Replay
python replay.py                       # List
python replay.py 0                     # Replay session 0
python replay.py 0 --plan-only         # Inspect LLM log without viewer
python replay.py 0 --speed 0.3         # Slow playback

# Delete
python replay.py 0 --delete            # Move to trash
python replay.py --purge-trash         # Empty trash

# Filter
python replay.py --outcome success
python replay.py --outcome failure
```

---

## Honest Caveats

**LLM non-determinism limits cycle-1 reproducibility.** Cycle 1 of an augmented run uses the original prompt and no scene jitter, but the LLM still plans freshly each cycle. Identical prompts may produce slightly different plans across runs. Treat cycle 1 as "baseline-similar," not "exactly the same."

**IK quality affects success rate.** With `pink` installed, IK converges reliably for most targets. With the iterative fallback, you may see more "Validation failed" errors on edge poses. If you can't install `pink`, consider relaxing the `POSITION_TOLERANCE_MM` in `FKValidator` from 5.0 to 8.0 to accept slightly looser IK.

**The system prompt is tuned for the cube-and-bin scene.** When you customize the scene for your real lab, also update the example in the system prompt (`SYSTEM_PROMPT_TEMPLATE` in `agent/llm_brain.py`) to use lab-appropriate coordinates and an actual lab task. Outdated examples in the system prompt are a common cause of confused LLM plans.

**Variant prompts can drift semantically.** Always preview before letting an augmented run execute. The few seconds saves hours of corrupted training data.

**Failed augmented cycles are training-relevant.** Don't discard them all. Keep ~25% of failures — they teach a future fine-tuned model the boundary between "works" and "doesn't work."

**Real-hardware recording requires additional code.** The `Recorder` class samples MuJoCo's `mjData` directly. For real hardware, you'd need a separate sampler that polls the xArm SDK at 60Hz. This is left as a future addition since it requires hardware to test against.

**For Isaac Sim migration:** the architecture is sim-agnostic at the data layer. Only `SimXArmAPI`, `FKValidator`, `IKSolver`, and `scene_randomizer.py` would need to be reimplemented against Isaac's API. The recording format, LLM brain, prompt variants, and augmentation pipeline carry over unchanged. A future migration is a few hundred lines of Python, not a project rewrite.
