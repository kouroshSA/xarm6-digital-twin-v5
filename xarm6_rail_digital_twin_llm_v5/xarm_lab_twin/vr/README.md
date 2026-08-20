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

## 2. Secure-context requirement (the #1 source of "Enter VR" failures)

`navigator.xr.requestSession('immersive-vr')` requires a **secure context**.
The headset connects to the host by LAN IP (not `localhost`), so plain
`http://<host-ip>:<port>` is not a secure context and **Enter VR** reports
"immersive-vr unavailable" even though the page loads fine.

Three options, best first.

### a) `adb reverse` + `http://localhost` (recommended)

This is the method **Meta officially documents**, and it is the least fiddly:
`localhost` is a secure context *by definition*, so forwarding the server port
onto the headset's own loopback enables WebXR with **no certificate and no
browser flags**.

```bash
sudo apt install android-tools-adb          # once, on the host

# On the Quest: Settings -> System -> Developer -> enable Developer Mode
#   (needs a personal Meta account set up via the Meta Horizon phone app)
# Connect the headset by USB-C, then accept "Allow USB debugging" in-headset.

adb devices                                 # confirm the headset is listed
adb reverse tcp:8443 tcp:8443

python scripts/run_vr.py --mode stereo --servo direct --port 8443
```

Then in the **Quest Browser** open `http://localhost:8443` and tap **Enter VR**.

Why prefer this:

- Nothing to configure per headset — matters if you have more than one.
- No cert to regenerate when the host IP changes or you move machines.
- Immune to LAN/subnet problems, guest-network client isolation, and
  multi-homed hosts (see §2.1).
- Re-run `adb reverse tcp:8443 tcp:8443` after replugging; that is the only
  recurring step.

Cost: a USB cable. `adb reverse` over Wi-Fi (`adb tcpip 5555` + `adb connect`)
has historically been unreliable on Quest hardware — prefer USB, and use (b) or
(c) when you need to be untethered.

### b) Dev-flag + plain HTTP (untethered, per-headset setup)

Run without `--cert/--key`, then on the **Quest Browser** (inside the headset):
`chrome://flags` → search **"Insecure origins treated as secure"** → add
`http://<host-ip>:<port>` (exact scheme + IP + port, no trailing slash) →
set **Enabled** → **Relaunch** the browser. Then open
`http://<host-ip>:<port>` and tap **Enter VR**.

Must be repeated on every headset, and redone whenever the host IP changes.

### c) A *genuinely trusted* certificate (untethered, robust)

> **Do not use a plain self-signed cert.** Chromium **disables WebXR on pages
> reached by clicking through a certificate warning** — the page loads but
> `isSessionSupported('immersive-vr')` returns `false`, i.e. the exact symptom
> you were trying to fix. The certificate must chain to a CA the headset
> actually trusts. (See `note-to-claude-code.md` for the full debugging story.)

Two ways to get one:

- **`mkcert`** — create a local CA, install its root certificate on each Quest,
  then issue a host cert:
  ```bash
  mkcert -install
  mkcert -cert-file vr/cert.pem -key-file vr/key.pem <host-ip>
  python scripts/run_vr.py --mode stereo --cert vr/cert.pem --key vr/key.pem
  ```
- **Tailscale** — `tailscale cert <host>.<tailnet>.ts.net` issues a publicly
  trusted certificate; join both headsets to the tailnet. This also works
  off-site and sidesteps LAN addressing entirely.

### 2.1 Multi-homed hosts

If the host has **two interfaces on the same subnet** — e.g. a Livox lidar NIC
statically addressed `192.168.1.50/24` alongside Wi-Fi on `192.168.1.0/24` —
routing is ambiguous and replies to the Quest may leave via the wrong NIC. Check
with `ip -4 -br addr` and `ip -4 route`. Either move the lidar NIC to its own
subnet (e.g. `192.168.50.0/24`) or use option (a), which does not depend on LAN
routing at all.

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

**Tethered (option 2a — recommended):**

1. Enable Developer Mode on the Quest, connect USB-C, accept the debugging prompt.
2. `adb reverse tcp:8443 tcp:8443` on the host.
3. Start the server, then open `http://localhost:8443` in the Quest Browser.
4. Press **Enter VR** (WebXR needs a user gesture). The twin appears; the HUD
   strip shows recording / gripper / rail state.

**Untethered (options 2b / 2c):**

1. Put the Quest on the same Wi-Fi as the workstation — **not** a Guest network
   (those usually isolate clients from each other).
2. Find the workstation IP (`hostname -I`); `run_vr.py` also prints the URL
   banner on startup.
3. In the Quest Browser open `http://<ip>:<port>` (dev flag) or
   `https://<ip>:<port>` (trusted cert).
4. Press **Enter VR**.

If **Enter VR** is unavailable, read the client's diagnostic line
(`xr=… secureContext=… error=…`) — `secureContext=false` means the transport is
the problem, not the code.

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

# --- Recommended: adb reverse, no TLS needed (see 2a) ---
adb reverse tcp:8443 tcp:8443

# Mono monitoring panel first, to verify the feed:
python scripts/run_vr.py --mode mono --servo validated

# Stereo immersion, direct (low-latency) servo:
python scripts/run_vr.py --mode stereo --servo direct

# --- Untethered with a trusted cert (see 2c) ---
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
