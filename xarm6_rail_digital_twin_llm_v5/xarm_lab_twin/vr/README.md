# VR teleoperation of the xArm6 digital twin

Drive the simulated xArm6 with Meta Quest 3 Touch controllers and see the
MuJoCo twin through the headset — no physical robot, no GR00T. Sim-only WebXR
teleop that reuses the existing `SimXArmAPI → IKSolver → ctrl → MuJoCo →
Recorder` path; the only new thing is the source of the EE target (your hand
instead of the LLM).

Two display modes:

- **`mono`** — MuJoCo's renderer streamed as one flat panel (also viewable in
  a plain browser tab for monitoring).
- **`stereo`** — a true stereo camera pair rendered inside MuJoCo (left/right
  eyes offset by IPD), head-tracked, presented per-eye in a WebXR
  `immersive-vr` session for genuine depth.

Both modes share one transport (WebSocket-JPEG), one input path, and one
teleop receiver. Only the number of rendered cameras and the client layout
differ.

---

## 1. Install

```bash
conda activate xarm6sim
pip install -r vr/requirements-vr.txt
# already present: mujoco, numpy, transforms3d, h5py
```

The server renders offscreen, so set the EGL GL backend (the DGX Spark is a
headless GPU). `scripts/run_vr.py` sets this for you, but you can export it:

```bash
export MUJOCO_GL=egl      # use osmesa only if egl is unavailable
```

> The desktop passive viewer (GLFW) and the offscreen renderer (EGL) contend
> for GL in one process, so the VR run builds the arm with `render=False` and
> never launches the passive viewer. The headset *is* the viewer.

---

## 2. HTTPS / secure-context requirement

`navigator.xr.requestSession('immersive-vr')` requires a **secure context**.
The headset connects to the host by LAN IP (not `localhost`), so plain
`http://` is not a secure context for entering VR. Two options:

### a) Dev-flag + plain HTTP (recommended — most reliable on Quest)

Run without `--cert/--key`, then on the **Quest Browser** (inside the headset):
`chrome://flags` → search **"Insecure origins treated as secure"** → add
`http://<host-ip>:<port>` (exact scheme + IP + port, no trailing slash) →
set **Enabled** → **Relaunch** the browser. Then open
`http://<host-ip>:<port>` and tap **Enter VR**.

> **Why this is preferred over self-signed TLS:** Chromium **disables WebXR on
> pages reached by clicking through a self-signed certificate warning** — the
> page loads but `isSessionSupported('immersive-vr')` returns `false`
> ("immersive-vr unavailable"). The flag grants a genuinely trustworthy
> context, so WebXR stays enabled. (See `note-to-claude-code.md` for the full
> debugging story.)

### b) Self-signed TLS

```bash
# from xarm_lab_twin/
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout vr/key.pem -out vr/cert.pem -days 365 \
  -subj "/CN=<host-ip>" \
  -addext "subjectAltName=IP:<host-ip>"

python scripts/run_vr.py --mode stereo --cert vr/cert.pem --key vr/key.pem --port 8443
```

Open `https://<host-ip>:8443` in the Quest Browser and accept the warning. If
"Enter VR" then reports unavailable, fall back to option (a).

### Test WebXR *inside the headset*, not on a desktop

`navigator.xr` only reports VR support when a VR runtime is present. A plain
desktop/laptop browser shows "VR support not detected" unless it's driving a
headset through a PCVR runtime (SteamVR / Oculus OpenXR). **Meta Quest Link is
x86-Windows-only — it does not exist for Linux or ARM**, so on a Linux host the
*only* path is the Quest's **own standalone browser** over Wi-Fi (which is
exactly what this stack targets). Sanity-check with
`https://immersive-web.github.io/webxr-samples/` opened *in the Quest browser* —
it should say "✅ VR support detected".

### Networking: native Linux vs WSL2

- **Native Linux host (Mint/Ubuntu, etc.):** the server binds straight to the
  LAN IP — **no port forwarding needed**. Find the IP with `hostname -I` or
  `ip -4 addr`, then point the Quest at `http://<that-ip>:8443`. If a firewall
  is on: `sudo ufw allow 8443/tcp`. Host and headset must share the LAN/subnet
  (avoid "Guest" Wi-Fi — it usually isolates clients).
- **WSL2 host (Windows):** WSL sits behind a NAT, so the WSL IP is unreachable
  from the Quest. You must forward the Windows LAN port into WSL (admin
  PowerShell): `netsh interface portproxy add v4tov4 listenport=8443
  listenaddress=0.0.0.0 connectport=8443 connectaddress=<wsl-ip>` plus a
  firewall rule, then the Quest uses the **Windows** LAN IP. This is the messy
  path — prefer a native Linux host.

The flat mono preview works over plain `http://` in any browser tab without a secure
context — only *entering VR* needs one.

---

## 3. Quest pairing / connect

1. Put the Quest on the same Wi-Fi as the workstation.
2. Find the workstation IP (`ip addr` / `hostname -I`); `run_vr.py` also prints
   the URL banner on startup.
3. In the Quest Browser, open the printed `https://<ip>:<port>` (or `http://`
   with the dev flag).
4. Press **Enter VR** (WebXR needs a user gesture). The twin appears; the HUD
   strip shows recording / gripper / rail state.

---

## 4. Controls

| Input (right controller unless noted) | Action |
|---|---|
| **Grip / squeeze** (hold) | **Clutch** — the arm follows your hand while held; release to freeze. Engaging never jumps the arm (the controller↔EE offset is frozen on press). |
| **Trigger** (press) | Toggle gripper (magnetic weld close / open). |
| **A** (press) | Toggle recording a take (start / stop). |
| **B** (press) | Reset / re-randomize the scene. |
| **Left thumbstick X** | Jog the rail along 0–700 mm. |
| **Head pose** | Moves the stereo viewpoint (stereo mode). |

Tip: point the controller straight down → the gripper points straight down
(roll≈180), the canonical grasp pose. Controller laser forward → gripper
points forward at the bench.

---

## 5. Run

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
conda activate xarm6sim

# Mono monitoring panel, validated servo, HTTPS:
python scripts/run_vr.py --mode mono --servo validated \
    --cert vr/cert.pem --key vr/key.pem

# Stereo immersion, direct (low-latency) servo:
python scripts/run_vr.py --mode stereo --servo direct \
    --cert vr/cert.pem --key vr/key.pem
```

Flags: `--mode {mono,stereo}` `--servo {direct,validated}` `--no-record`
`--port N` `--scale S` `--cert PEM --key PEM` `--scene PATH`.

- **`direct`** servo (default): solves IK once per control tick and writes
  joint ctrl directly — smooth and low-latency, bypasses the collision
  validator (fine for continuous servoing in sim).
- **`validated`** servo: routes each target through `set_position` (IK +
  `FKValidator` + pacing) — safer, slightly jerkier.

All tunables (IPD, world scale, origin offset, loop rates, ports, workspace
AABB, servo mode, frame size) live in [`config.py`](config.py).

---

## 6. Recordings

The **A** button starts/stops takes via the existing `Recorder`, so VR demos
land as standard sessions under `recordings/<timestamp>_session_<id>/`
(`metadata.json`, `commands.jsonl` with `ee_target` / `gripper` / `reset_scene`
events, and a 60 Hz `trajectory.h5`) — pipeline-identical to autonomous
episodes. Replay one with:

```bash
python replay.py            # list sessions
python replay.py <index>    # replay; the arm reproduces the recorded motion
```

This is what feeds the LeRobot → GR00T export later (out of scope here).

---

## 7. Architecture / files

```
vr/
├── config.py            # all tunables
├── transforms.py        # XR<->twin coordinate math, clutch, smoother, clamp
├── stereo_renderer.py   # mono + stereo offscreen rendering -> JPEG (latest-wins)
├── teleop_receiver.py   # pose/buttons -> IK/ctrl, gripper, record, rail, head
├── server.py            # FastAPI: client + WebSocket (frames<->pose)
├── static/{index.html,xr-client.js}   # dependency-free WebXR client
├── smoke_test.py        # acceptance test 1 (headless render)
└── test_transforms.py   # acceptance test 2 (coordinate math)
scripts/run_vr.py        # entrypoint
envs/lab_scene.xml       # vr_head mocap + cam_left/cam_right (added)
```

The scene gained a `vr_head` mocap body carrying `cam_left`/`cam_right`
(IPD ±0.0315 m). The receiver writes the headset pose into `vr_head`'s
`mocap_pos`/`mocap_quat` each frame so head motion moves the stereo viewpoint.

---

## 8. Tests

```bash
# Coordinate math (no GL needed):
python -m vr.test_transforms

# Headless render smoke test (writes vr/{mono,left,right}.jpg):
MUJOCO_GL=egl python -m vr.smoke_test
```

---

## 9. Phase 2 (not built): WebRTC

The MVP uses WebSocket-JPEG (simple, robust on LAN). For lower-latency depth
feedback, swap the frame encode/send for an `aiortc` WebRTC video track — see
the `# Phase 2: aiortc` note in `server.py`'s sender loop. Install with
`pip install aiortc av`.
