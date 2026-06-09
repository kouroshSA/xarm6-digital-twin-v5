# VR Integration for `xarm6-digital-twin-v5` — Claude Code Build Spec

**Goal.** Add Meta Quest 3 teleoperation of the MuJoCo digital twin (no physical
robot). The operator drives the simulated xArm6 with the Touch controllers
(move EE, jog the rail, open/close gripper, toggle recording, reset scene) and
**sees the twin** through the headset in two selectable display modes:

1. **Mono mode** — MuJoCo's built-in OpenGL renderer streamed as a single flat
   video panel (also viewable on a plain monitor / browser tab). This is the
   simple "monitoring" view.
2. **Stereo mode** — a true stereo camera **pair** rendered inside MuJoCo
   (left/right eyes offset by IPD), with head tracking, presented per-eye in a
   WebXR `immersive-vr` session for genuine depth immersion in the twin.

Both modes share one transport, one input path, and one teleop receiver. The
only difference is how many cameras are rendered and how the client lays the
frame(s) out.

This is **sim-only** and depends on **nothing from GR00T**. It reuses the
existing `SimXArmAPI` → `IKSolver` → `ctrl` → MuJoCo → `Recorder` path; the only
new thing is the *source* of the EE target (a human hand instead of the LLM).

---

## 0. Ground rules (read before writing any code)

These are facts about the existing repo. Do **not** re-derive or "fix" them.

- **Working directory** is `xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/`.
  Every script runs from there (`cd .../xarm_lab_twin && python scripts/...`).
  Scripts add the project root to `sys.path` at the top — copy that idiom.
- **Conda env** is `xarm6sim` (Python 3.11). Install new deps into it.
- **Physics runs continuously** in a daemon thread (`SimXArmAPI._sim_loop`)
  that calls `mujoco.mj_step` while holding `self.lock`. **Every** read or write
  you do against `arm.data` / `arm.model` (rendering, writing `ctrl`, writing
  mocap pose, calling IK) **must hold `arm.lock`**, or you will race the
  stepper and get garbage / segfaults.
- **Construct the arm with `render=False` in VR mode.** The desktop passive
  viewer (`mujoco.viewer.launch_passive`, GLFW context) and an offscreen
  `mujoco.Renderer` (EGL context) contend for GL on the same process/GPU. In VR
  the headset *is* the viewer, so do **not** launch the passive viewer. See §7.
- **No `<camera>` elements exist** in `envs/lab_scene.xml` today — all current
  views are free cameras set up in code. Stereo mode adds cameras (§4).
- **Units at the public API:** `SimXArmAPI.set_position` takes **mm** and
  **degrees** (extrinsic XYZ Euler; `roll=180,pitch=0,yaw=0` = gripper pointing
  straight down). `IKSolver.solve` takes **meters** and a 3×3 rotation matrix.
- **Don't touch** `agent/`, `auto_play.py`, the LLM loop, or grading. VR is
  orthogonal to all of it.

### Existing symbols you will call (already implemented, verified)

```python
# sim/mujoco_env.py
arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=False)
arm.model, arm.data, arm.lock          # MjModel, MjData, threading.Lock
arm.ee_site                            # site id of "end_effector"
arm.ik_solver                          # IKSolver instance
arm.set_position(x, y, z, roll, pitch, yaw, speed=, wait=)   # mm, deg -> int rc
arm.get_position()                     # -> (rc, [x_mm,y_mm,z_mm,roll,pitch,yaw])
arm.set_rail_position(position_mm, speed_mm_s=, wait=)       # 0..700 mm
arm.get_rail_position()                # -> (rc, mm)
arm.open_lite6_gripper()               # release weld
arm.close_lite6_gripper()              # grasp nearest body in reach (magnetic weld)
arm.reset_scene()                      # re-randomize / reset bodies

# sim/ik_solver.py  (caller must hold arm.lock)
arm.ik_solver.solve(target_pos_m, target_rot=R3x3, seed_q=None)  # -> (6,) rad or None

# recording.py
rec = Recorder(model=arm.model, data=arm.data, lock=arm.lock,
               interface="vr_teleop", scene_xml="envs/lab_scene.xml",
               enable_frames=False)
rec.start()
rec.log_command(event_type: str, payload: dict)   # call on every teleop command
rec.stop(kept=True, task_label="...")              # writes commands.jsonl + trajectory.h5
rec.is_recording()
```

The `Recorder` runs its own state sampler thread; you only need to call
`log_command` when a discrete command is issued (target sent, gripper toggled,
etc.). The continuous joint/EE trajectory is sampled automatically.

---

## 1. New dependencies

Add to the `xarm6sim` env:

```bash
conda activate xarm6sim
pip install fastapi "uvicorn[standard]" websockets pillow
# already present: mujoco, numpy, transforms3d, h5py
# optional low-latency upgrade (Phase 2, see §10): pip install aiortc av
```

Create a `vr/requirements-vr.txt` listing these so the addition is reproducible.

Set the offscreen GL backend for the server process (DGX Spark is headless GPU):

```bash
export MUJOCO_GL=egl      # use osmesa only if egl is unavailable
```

Document this in `vr/README.md` and set it inside `scripts/run_vr.py` *before*
`import mujoco` if not already in the environment.

---

## 2. File layout to create

```
xarm_lab_twin/
├── vr/
│   ├── __init__.py
│   ├── transforms.py          # XR<->twin coordinate math, clutch, smoothing
│   ├── stereo_renderer.py     # mono + stereo offscreen rendering -> JPEG frames
│   ├── teleop_receiver.py     # pose/buttons -> IK/ctrl, gripper, record, rail
│   ├── server.py              # FastAPI: serves client + WebSocket (frames<->pose)
│   ├── config.py              # IPD, scale, origin offset, rates, ports, workspace clamp
│   ├── requirements-vr.txt
│   ├── README.md              # setup, HTTPS/cert, Quest pairing, controls
│   └── static/
│       ├── index.html         # WebXR client (entry button + canvas)
│       └── xr-client.js       # immersive-vr loop: draw eyes, stream controller pose
├── scripts/
│   └── run_vr.py              # entrypoint wiring arm + recorder + renderer + server
└── envs/
    └── lab_scene.xml          # EDIT: add vr_head mocap body + cam_left/cam_right
```

---

## 3. Coordinate frames & mapping (`vr/transforms.py`)

This is the subtle part — get the conventions right.

- **WebXR world** (`local-floor` reference space): right-handed, **+X right,
  +Y up, −Z forward**, meters, origin on the floor where the session started.
- **MuJoCo twin**: right-handed, **Z-up**, meters. Bench top ≈ `z=0.76`; the
  workspace sits in front of the arm (positive‑Y-ish). EE targets at the public
  API are mm.

**Basis change XR→twin** (rotate −90° about X so XR-up becomes twin-up):

```
twin_x =  s * xr_x
twin_y = -s * xr_z
twin_z =  s * xr_y
```

`s` is `config.WORLD_SCALE` (default `1.0`; lets you shrink/grow the mapping if
the operator wants the bench within easy arm's reach). After the basis change,
add `config.TWIN_ORIGIN_OFFSET_M` (a 3-vector) so the operator's comfortable
standing/hand origin lands near the bench workspace. Both are tunable in
`config.py`.

Provide these functions:

- `xr_to_twin_pos(p_xr) -> p_twin_m` — basis change + scale + offset.
- `xr_to_twin_quat(q_xr) -> R_twin_3x3` — rotate the controller/head orientation
  through the same −90°-about-X basis change; return a 3×3 matrix
  (`set_position` and `IKSolver.solve` want a rotation matrix / Euler, not a
  quaternion). Use `transforms3d.quaternions`/`euler` (already a dependency).
- A `Clutch` helper: the EE only follows the controller **while the grip
  (squeeze) button is held**. On grip-press, record the offset between current
  controller pose and current EE pose; while held, `EE_target = controller_pose
  + frozen_offset`. On release, freeze the arm. This is the standard VR teleop
  "clutch" so the operator can reposition their hand without dragging the arm.
- An exponential **smoother** on the target (`alpha ≈ 0.3` at 60–72 Hz) to damp
  controller jitter before IK.
- A **workspace clamp**: clamp the twin-frame target into a safe AABB
  (`config.WORKSPACE_AABB_MM`) so a wild hand motion can't fling an IK target to
  infinity. (Failure is free in sim, but clamping keeps the demo smooth.)

---

## 4. Scene edit — add the stereo head cameras (`envs/lab_scene.xml`)

Stereo immersion needs two cameras that move with the operator's head. Add a
**mocap body** named `vr_head` carrying two child cameras. Mocap bodies are
driven directly via `data.mocap_pos` / `data.mocap_quat`, so the receiver can
write the headset pose into them every frame (head tracking → the twin viewpoint
moves).

Insert into the `<worldbody>` (IPD default 63 mm → ±0.0315 m on local X):

```xml
<body name="vr_head" mocap="true" pos="0 -0.6 1.4">
  <!-- MuJoCo cameras look down their local -Z, +X right, +Y up.
       quat below makes -Z point along world +Y (toward the bench) and
       +Y point along world +Z (up) when mocap_quat is identity, so an
       identity headset orientation looks "forward at the bench, upright".
       Adjust if your operator's start facing differs. -->
  <camera name="cam_left"  pos="-0.0315 0 0" quat="0.7071 0.7071 0 0" fovy="90"/>
  <camera name="cam_right" pos=" 0.0315 0 0" quat="0.7071 0.7071 0 0" fovy="90"/>
</body>
```

Notes for Claude Code:
- Keep `fovy` around 90–100 for VR; the client corrects aspect per eye.
- Confirm there is no existing body named `vr_head`; if the `<worldbody>` uses a
  different indentation style, match it.
- The mono renderer does **not** use these cameras — it uses a free camera (§5).

---

## 5. Rendering (`vr/stereo_renderer.py`)

One module, two entry points. Use a **single** `mujoco.Renderer` reused across
calls (avoid multiple GL contexts). Default frame size `640×480` per eye
(tunable in `config.py`). Encode each frame to JPEG with Pillow (quality ~80).

**Critical locking pattern** — hold the lock only around `update_scene` (which
reads `data`), release before `render()` (which reads the renderer's internal
scene copy):

```python
class TwinRenderer:
    def __init__(self, model, data, lock, w=640, h=480):
        self.model, self.data, self.lock = model, data, lock
        self.r = mujoco.Renderer(model, height=h, width=w)
        # free camera for mono mode, framed like the existing passive viewer
        self.free_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.free_cam)
        self.free_cam.lookat[:] = (0.0, 0.15, 1.0)
        self.free_cam.distance, self.free_cam.azimuth, self.free_cam.elevation = 2.8, 135.0, -20.0

    def _render(self, camera):
        with self.lock:
            self.r.update_scene(self.data, camera=camera)  # reads data -> lock
        return self.r.render()                              # ndarray HxWx3 -> no lock

    def mono_frame(self):           # -> JPEG bytes
        return _jpeg(self._render(self.free_cam))

    def stereo_frames(self):        # -> (left_jpeg, right_jpeg)
        return _jpeg(self._render("cam_left")), _jpeg(self._render("cam_right"))
```

- `camera=` accepts either a `MjvCamera` (free) or a **camera name string**
  (`"cam_left"`). Verify your installed MuJoCo build accepts the name form; if
  not, resolve the camera id and pass that.
- Run rendering in its **own thread/loop** at `config.RENDER_HZ` (start at 30;
  the sim stepper is independent). Push the latest frame(s) into an
  `asyncio`-friendly slot the server reads (e.g. a single-slot buffer / latest-
  wins; do **not** queue/backlog frames — always send the newest).
- Stereo sends **two** JPEGs per tick (or one side-by-side image — pick one and
  keep the client in sync; two messages with an `eye` tag is simpler to reason
  about).

---

## 6. Input + teleop receiver (`vr/teleop_receiver.py`)

Consumes controller/head state coming up the WebSocket (§8) and drives the arm.
Run the control update at `config.CONTROL_HZ` (start at 60).

**Touch controller → action mapping** (WebXR `XRInputSource.gamepad`):

| Input (right controller unless noted)        | Action                                              |
|----------------------------------------------|-----------------------------------------------------|
| **grip / squeeze** (`buttons[1]`) held       | **clutch**: arm follows controller while held       |
| **trigger** (`buttons[0]`) press             | **toggle gripper** (`close_lite6_gripper` / `open`) |
| **A** (`buttons[4]`) press                   | **toggle recording** (`rec.start()` / `rec.stop()`) |
| **B** (`buttons[5]`) press                   | **`reset_scene()`**                                 |
| **left** thumbstick X (`axes[2]`)            | **jog rail** ± along 0–700 mm                        |
| head pose (`XRViewerPose.transform`)         | drive `vr_head` mocap (stereo viewpoint)            |

Edge-detect button presses (act on rising edge, not while-held, except grip).

**Control loop, per tick:**

1. Read latest head pose → under `arm.lock`, write `data.mocap_pos[vr_head]` and
   `data.mocap_quat[vr_head]` (map via `transforms`). This moves the stereo eyes.
2. If grip held (clutch engaged):
   - `p_twin = transforms.xr_to_twin_pos(controller_pos)` (+ frozen clutch offset)
   - `R_twin = transforms.xr_to_twin_quat(controller_quat)`
   - smooth + clamp `p_twin`.
   - **Servo the arm.** Two selectable paths (`config.SERVO_MODE`):
     - `"direct"` (default, smooth, low-latency): under `arm.lock`,
       `q = arm.ik_solver.solve(p_twin_m, target_rot=R_twin, seed_q=<current q>)`;
       if not `None`, write the six joint targets into `arm.data.ctrl` at the
       arm actuator indices (mirror `SimXArmAPI._execute_joint_angles` — read
       how it indexes `act_ids[1:]`). This bypasses the collision validator for
       continuous servoing, which is fine in sim.
     - `"validated"` (safer, slightly jerkier): call
       `arm.set_position(x_mm, y_mm, z_mm, roll, pitch, yaw, speed=400, wait=False)`
       at the tick rate. Reuses the existing IK + `FKValidator` + pacing path.
   - `rec.log_command("ee_target", {"pos_mm": [...], "rpy_deg": [...], "mode": SERVO_MODE})`.
3. Trigger edge → toggle a `gripper_closed` flag; call the matching gripper
   method; `rec.log_command("gripper", {"closed": bool})`.
4. A edge → if `not rec.is_recording()`: `rec.start()`; else
   `rec.stop(kept=True, task_label=<prompt or timestamp>)` then create a fresh
   `Recorder` for the next take. Surface recording state to the client (so the
   headset can show a red dot) via an outbound status message.
5. B edge → `arm.reset_scene()`; `rec.log_command("reset_scene", {})`.
6. Left thumbstick X → integrate into a rail target, `arm.set_rail_position(mm,
   wait=False)`, clamp 0–700.

Keep the receiver framework-agnostic: it exposes `handle_input(state: dict)` and
`tick()`; the server calls them. No asyncio inside the receiver itself.

---

## 7. GL context & process model (do this right)

- **VR run constructs `SimXArmAPI(..., render=False)`** — no passive viewer.
- Server process exports `MUJOCO_GL=egl` before importing mujoco.
- One `mujoco.Renderer` instance, used from one render thread. Do not create a
  renderer per request.
- If you ever want the desktop passive viewer *and* VR at once for debugging,
  run them as **separate processes** sharing nothing but the scene file — do not
  mix GLFW + EGL contexts in one process.

---

## 8. Server & transport (`vr/server.py`)

FastAPI + a single WebSocket. Keep the MVP on **WebSocket-JPEG** (simple, robust
on LAN); WebRTC is a later upgrade (§10).

- `GET /` → serve `vr/static/index.html`.
- `GET /static/*` → static assets.
- `GET /config.json` → expose `{mode, ipd, render_hz, control_hz}` to the client.
- `WS /ws`:
  - **Down (server→headset):** newest rendered frame(s). For stereo send two
    binary messages tagged `eye:0` / `eye:1` (1-byte prefix + JPEG), or a small
    JSON header then the binary. For mono send one. Always latest-wins; never
    backlog.
  - **Up (headset→server):** ~60–72 Hz JSON with head pose, both controllers'
    grip poses, buttons, axes. Forward each into `receiver.handle_input(state)`.
  - Periodic **status** message down (recording on/off, gripper state, current
    rail mm, IK-fail flag) so the client can render a HUD.

Run the render loop and control tick as asyncio tasks (or background threads
bridged with `run_in_executor`). Bind `0.0.0.0` so the headset can reach it.

### HTTPS / secure-context gotcha (must handle)

`navigator.xr.requestSession('immersive-vr')` **requires a secure context**.
The headset connects to the workstation by LAN IP (not `localhost`), so plain
`http://` will be rejected. Provide **both** paths in `vr/README.md`:

1. **Self-signed TLS (recommended):**
   ```bash
   openssl req -x509 -newkey rsa:2048 -nodes -keyout vr/key.pem -out vr/cert.pem -days 365 \
     -subj "/CN=<workstation-ip>" -addext "subjectAltName=IP:<workstation-ip>"
   uvicorn vr.server:app --host 0.0.0.0 --port 8443 \
     --ssl-keyfile vr/key.pem --ssl-certfile vr/cert.pem
   ```
   Then open `https://<workstation-ip>:8443` in the Quest Browser and accept the
   cert warning once.
2. **Dev flag fallback:** on the Quest, `chrome://flags` →
   "Insecure origins treated as secure" → add `http://<workstation-ip>:<port>`.

Headset and workstation must be on the **same LAN/subnet**.

---

## 9. WebXR client (`vr/static/index.html` + `xr-client.js`)

A self-contained WebXR page (no Unity, no Open Teach dependency). It does both
**display** and **input** — the same page that shows the twin also streams the
controller poses back.

Behaviour:

1. Landing page with an **"Enter VR"** button (WebXR requires a user gesture).
   Detect support via `navigator.xr.isSessionSupported('immersive-vr')`.
2. On enter: `requestSession('immersive-vr')` with
   `requiredFeatures: ['local-floor']`. Set up a WebGL layer.
3. Open the WebSocket to `/ws`. Decode incoming JPEG frames to `ImageBitmap`
   (`createImageBitmap(blob)`), upload as a WebGL texture.
4. **Per `XRFrame`:**
   - Get `viewerPose` from the `local-floor` reference space.
   - **Mono mode:** draw the single latest frame on a head-locked quad filling
     the view (a simple billboard a couple meters in front), same texture both
     eyes.
   - **Stereo mode:** for each `view` in `viewerPose.views`, draw the matching
     eye's texture (`eye:0`→left view, `eye:1`→right view) to that view's
     viewport. This is what produces depth.
   - Read `session.inputSources`: for each, get `gripSpace` pose relative to
     `local-floor`, plus `gamepad.buttons` / `gamepad.axes`. Package
     `{head:{pos,quat}, right:{pos,quat,buttons,axes}, left:{...}}` and send up
     the WebSocket (throttle to ≤72 Hz).
   - Render a small HUD (recording dot, gripper open/closed, rail mm) from the
     server status messages.
5. Handle `sessionend` (stop streaming, close WS) and reconnect cleanly.

Keep `xr-client.js` dependency-free (raw WebGL + WebXR). Document the controls
in the HUD and in `vr/README.md`.

---

## 10. Entry point (`scripts/run_vr.py`)

Wire it all together. Mirror `scripts/run_task.py`'s header (sys.path insert,
`from env_loader import load_env; load_env()`).

```
python scripts/run_vr.py [--mode mono|stereo] [--servo direct|validated]
                         [--no-record] [--port 8443] [--scale 1.0]
                         [--cert vr/cert.pem --key vr/key.pem]
```

Steps:
1. `os.environ.setdefault("MUJOCO_GL", "egl")` **before** importing mujoco.
2. `from env_loader import load_env; load_env()`.
3. `arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=False)`.
4. `rec = Recorder(model=arm.model, data=arm.data, lock=arm.lock,
   interface="vr_teleop", scene_xml="envs/lab_scene.xml", enable_frames=False)`
   (created but **not** auto-started; the A button starts/stops takes).
5. Build `TwinRenderer`, `TeleopReceiver`, then launch the FastAPI app with
   uvicorn (TLS if cert provided). Print the `https://<ip>:<port>` URL to open
   on the headset.
6. Clean shutdown on Ctrl-C: stop render/control loops, `rec.stop()` if
   recording, `arm.disconnect()`.

---

## 11. Acceptance tests / definition of done

Verify in this order; each builds on the last.

1. **Headless render smoke test:** a tiny script builds the arm with
   `render=False`, instantiates `TwinRenderer`, and writes `mono.jpg`,
   `left.jpg`, `right.jpg` to disk. Confirm the arm/bench are visible and that
   `left.jpg` ≠ `right.jpg` (proves stereo offset). No GL errors with
   `MUJOCO_GL=egl`.
2. **Transform unit tests:** `xr_to_twin_pos` maps a known XR point to the
   expected twin point; round-trip a quaternion through `xr_to_twin_quat` and
   check the resulting matrix orients the gripper as expected (e.g. controller
   pointing down → `roll≈180`).
3. **Server up, client loads:** open `https://<ip>:<port>` in the Quest Browser,
   "Enter VR" succeeds, and the twin is visible (mono first).
4. **Head tracking (stereo):** moving/turning the head moves the twin viewpoint
   smoothly; left/right images differ → depth is perceivable.
5. **Clutch + servo:** holding grip makes the EE follow the controller; releasing
   freezes it. Motion is smooth in `direct` mode. IK failures are rare and the
   HUD flags them rather than crashing.
6. **Gripper:** trigger grabs a tube/cube within reach (weld activates, console
   logs `grasped <body>`), trigger again releases it.
7. **Rail:** left thumbstick jogs the arm along 0–700 mm; clamps at the ends.
8. **Recording round-trip:** A starts a take, perform a pick-and-place, A stops
   it. A session dir appears under the recordings root with `commands.jsonl`
   (containing `ee_target` / `gripper` events) and a non-empty `trajectory.h5`.
   **Replay the recorded session with the existing `scripts/`/`replay.py` path
   and confirm the arm reproduces the motion** — this proves VR demos are
   pipeline-identical to autonomous episodes and will feed the LeRobot→GR00T
   path later.
9. **Reset:** B re-randomizes/repositions the scene without restarting.

---

## 12. Out of scope (do NOT build now)

- GR00T / LeRobot conversion — VR demos already land as standard
  `trajectory.h5`; conversion is a separate later step.
- Real-hardware teleop (`--mode real`) — same receiver will later point at
  `RealXArmAPI`, but not in this task.
- WebRTC transport — keep WebSocket-JPEG for the MVP. Leave a `# Phase 2:
  aiortc` note where the frame encode/send happens so the upgrade path is
  obvious. WebRTC matters once depth-feedback latency does.
- Bare hand-tracking — controllers only; they're more reliable for precision.

---

## 13. Style / repo conventions

- Match existing code: type hints, module-top comment, `print("[VR] ...")`
  logging consistent with `[SimXArm]` / `[Recorder]` tags.
- All `arm.data` / `arm.model` access under `arm.lock` — no exceptions.
- Tunables (IPD, scale, origin offset, rates, ports, workspace AABB, servo mode)
  live in `vr/config.py`, not scattered as literals.
- Add a `vr/README.md`: install, `MUJOCO_GL`, cert generation, Quest pairing
  steps, the full control map, and the run command.
- Update the top-level `CLAUDE.md` with a short "VR teleop" section pointing at
  `scripts/run_vr.py` and `vr/README.md`.
