# Note to Claude Code — VR teleop handoff

**Read this first.** It captures where the VR-teleoperation work stands so a
fresh Claude Code session on the new Linux machine can pick up without
re-deriving the whole history. Written 2026-06-09.

---

## TL;DR

- A complete **Meta Quest 3 → MuJoCo digital-twin VR teleop** feature was
  built and lives in `xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/` +
  `scripts/run_vr.py`. It is **sim-only**, reuses the existing
  `SimXArmAPI → IKSolver → ctrl → MuJoCo → Recorder` path, and is committed to
  `main`.
- **Software is done and verified** (render smoke test, transform unit tests,
  and a headless server+receiver integration test all pass).
- **What's NOT yet done:** the live on-headset acceptance tests (Enter VR, head
  tracking, clutch/servo feel, gripper, rail, record→replay). These need the
  Quest 3 physically connected.
- **Why we switched machines:** the previous host was **Windows-on-ARM + WSL2**,
  which made networking painful (WSL NAT bridge) and made PCVR/Link impossible
  (Meta Quest Link is x86-Windows-only). The **new Linux Mint (Intel + NVIDIA
  8 GB)** host removes the networking pain entirely and is the intended
  deployment shape.

---

## What the feature does

Operator wears a Quest 3, opens a web page served by this repo **in the Quest's
own browser**, taps "Enter VR", and drives the simulated xArm6 with the Touch
controllers while seeing the twin in the headset. Two display modes:
**mono** (one flat panel, also viewable in any browser tab) and **stereo**
(per-eye cameras offset by IPD, head-tracked, true depth). Transport is
WebSocket-JPEG; input is streamed back up the same socket.

Full design + controls: `xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/README.md`.
Original build spec: `vr-instruction.md` (repo root).

### Files (all committed)
```
xarm_lab_twin/vr/
  config.py            # all tunables (IPD, scale, rates, ports, workspace AABB, servo mode)
  transforms.py        # XR<->twin coord math, Clutch, Smoother, workspace clamp
  stereo_renderer.py   # mono + stereo offscreen render -> JPEG (latest-wins, EGL)
  teleop_receiver.py   # pose/buttons -> IK/ctrl, gripper, record, rail, head mocap
  server.py            # FastAPI: client + WebSocket (frames down / pose up)
  static/index.html, static/xr-client.js   # dependency-free WebXR client
  smoke_test.py        # acceptance test 1 (headless render)
  test_transforms.py   # acceptance test 2 (coordinate math)
  integration_test.py  # headless server+receiver integration (no headset)
  README.md, requirements-vr.txt
xarm_lab_twin/scripts/run_vr.py             # entrypoint
xarm_lab_twin/envs/lab_scene.xml            # EDITED: added vr_head mocap + cam_left/cam_right
CLAUDE.md                                   # added a "VR teleop" section
```

---

## Verified so far (run these to re-confirm on the new box)

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
conda activate xarm6sim
python -m vr.test_transforms                 # 8/8 PASS (pure math, no GL)
MUJOCO_GL=egl python -m vr.smoke_test         # writes vr/{mono,left,right}.jpg, PASS; left!=right
MUJOCO_GL=egl python -m vr.integration_test   # ALL PASS: clutch servo moves EE, gripper, rail,
                                              # head mocap, reset, server serves frames over WS
```
On the old machine all three passed. The integration test even runs a real
uvicorn server in-thread and pulls a live frame over the WebSocket.

---

## Setup on the NEW Linux Mint machine (Intel + NVIDIA 8 GB)

1. **Base sim env** (same as the rest of the repo — see the top-level
   `README.md` and `xarm_lab_twin/README.md`): conda env **`xarm6sim`**,
   Python 3.11, with `mujoco`, `numpy`, `transforms3d`, `h5py`. Recreate it if
   it's a fresh box.
2. **VR extras:**
   ```bash
   conda activate xarm6sim
   pip install -r xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/vr/requirements-vr.txt
   # = fastapi, uvicorn[standard], websockets, pillow
   ```
3. **NVIDIA + EGL (the GPU helps here).** Offscreen rendering uses
   `MUJOCO_GL=egl`. Make sure the proprietary NVIDIA driver + libEGL are
   installed (Mint: Driver Manager). Quick check:
   ```bash
   MUJOCO_GL=egl python -c "import mujoco,numpy as np; \
     m=mujoco.MjModel.from_xml_path('xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/envs/lab_scene.xml'); \
     d=mujoco.MjData(m); mujoco.mj_forward(m,d); \
     r=mujoco.Renderer(m,120,160); r.update_scene(d); print('EGL render OK', r.render().shape)"
   ```
   `run_vr.py` sets `MUJOCO_GL=egl` itself before importing mujoco. The 8 GB GPU
   is plenty; you can comfortably raise `FRAME_WIDTH/HEIGHT` and `RENDER_HZ` in
   `vr/config.py` for sharper/smoother stereo than the 640×480@30 defaults.
4. **API key:** `.env` is gitignored and will NOT be in the clone. The VR path
   does **not** need an Anthropic key (no LLM). Only the LLM scripts do.

---

## Run + connect the Quest (native Linux — the easy path)

On native Linux there is **no WSL bridge** — the server binds to the LAN IP
directly.

1. Find the host LAN IP: `hostname -I` (pick the `192.168.x.x` one).
2. If a firewall is enabled: `sudo ufw allow 8443/tcp`.
3. Start the server:
   ```bash
   cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
   conda activate xarm6sim
   python scripts/run_vr.py --mode stereo --servo direct --port 8443
   ```
   (Start with `--mode mono` first if you want to verify the feed in a normal
   browser tab at `http://<host-ip>:8443` before going immersive.)
4. **On the Quest (in its own standalone Browser, NOT a desktop browser):**
   - Same Wi-Fi as the host, **not** a Guest network.
   - One-time secure-context setup (recommended path): in the Quest browser open
     `chrome://flags` → "Insecure origins treated as secure" → add
     `http://<host-ip>:8443` → **Enabled** → **Relaunch**.
   - Go to `http://<host-ip>:8443` → tap **Enter VR (stereo)**.
   - (Alternative: generate a self-signed cert and use `--cert/--key` + https,
     but see the gotcha below — the flag path is more reliable.)

**Controls:** grip (hold) = clutch (arm follows hand), trigger = gripper,
A = record take, B = reset scene, left stick X = jog rail, head = stereo
viewpoint.

---

## Remaining acceptance tests (need the headset) — do these next

From `vr-instruction.md` §11, items 3–9, in order:
3. Server up, client loads, **Enter VR succeeds**, twin visible (mono first).
4. **Head tracking (stereo):** turning head moves the viewpoint; L/R images
   differ → depth.
5. **Clutch + servo:** holding grip makes the EE follow the controller; release
   freezes. Smooth in `direct` mode; HUD flags IK failures.
6. **Gripper:** trigger grabs a tube/cube in reach (weld), trigger again releases.
7. **Rail:** left stick jogs along 0–700 mm and clamps.
8. **Recording round-trip:** A starts/stops a take → a `recordings/<...>` dir
   with `commands.jsonl` (ee_target/gripper events) + non-empty `trajectory.h5`;
   then **`python replay.py <idx>` reproduces the motion** (proves VR demos are
   pipeline-identical to autonomous episodes). The recorder is the standard one,
   so this should "just work" — verify it.
9. **Reset:** B re-randomizes the scene without restarting.

---

## Key learnings / gotchas (don't relearn these)

1. **Self-signed cert click-through DISABLES WebXR.** Chromium treats a page
   reached by "proceed past the cert warning" as not-trustworthy, so
   `isSessionSupported('immersive-vr')` returns false → the button shows
   "immersive-vr unavailable" even though the page loaded fine. **Use the
   `chrome://flags` "Insecure origins treated as secure" + plain HTTP path**, or
   a properly trusted cert. The client now prints a diagnostic status line
   (`xr=… secureContext=… error=…`) to make this obvious.
2. **Test WebXR inside the headset, not on a desktop.** A desktop/laptop browser
   reports "VR support not detected" unless it's driving a headset via a PCVR
   runtime. On Linux there is no such runtime for Quest, so the standalone Quest
   browser is the only path (and is the design).
3. **Meta Quest Link / Air Link is x86-Windows-only.** No Linux client, no ARM
   client. PCVR/Link is NOT an option on this project — ignore that route.
4. **GL contexts are thread-affine.** `TwinRenderer` creates its
   `mujoco.Renderer` lazily on whatever thread first renders (the render-loop
   thread under the server; the main thread in the smoke test) and closes it on
   that same thread. Creating it on one thread and rendering on another gives
   `EGL_BAD_ACCESS` spam and blank frames. Already handled — don't "fix" it by
   moving renderer construction back into `__init__`.
5. **Build the arm with `render=False` for VR** — no GLFW passive viewer (it
   contends with the EGL offscreen renderer). The headset is the viewer.
6. **All `arm.data`/`arm.model` access holds `arm.lock`** — the physics stepper
   runs in a daemon thread; racing it segfaults.
7. **Quest account:** the headset needs a *personal* Meta account (set up via
   the Meta Horizon phone app). An institutional/MDM-locked unit can block this;
   unrelated to our code.

---

## Git state

Committed to `main` and pushed to `origin`
(`github.com/kouroshSA/xarm6-digital-twin-v5`): all of `vr/`, `scripts/run_vr.py`,
the `lab_scene.xml` edit, the `CLAUDE.md` VR section, the `.gitignore` additions
(`vr/*.pem`, `vr/*.jpg`), `vr-instruction.md`, and this note. The TLS cert/key
and render-test JPEGs are gitignored and intentionally NOT in the repo —
regenerate the cert on the new host if you go the TLS route.
