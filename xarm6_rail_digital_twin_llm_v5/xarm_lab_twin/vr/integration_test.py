# vr/integration_test.py
"""Headless integration test for the VR teleop stack (no headset required).

Part 1 — receiver logic: feeds synthetic controller/head state and asserts
  clutch servoing moves the EE, the gripper toggles, the rail jogs, head pose
  drives the vr_head mocap, and reset works.
Part 2 — transport: starts the real uvicorn server in a thread, fetches
  /config.json, connects the WebSocket, receives a rendered frame, and pushes
  an input message.

Run:  MUJOCO_GL=egl python -m vr.integration_test
"""
import asyncio
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import websockets

from sim.mujoco_env import SimXArmAPI, RAIL_ACT
from vr import config, transforms
from vr.stereo_renderer import TwinRenderer
from vr.teleop_receiver import TeleopReceiver
from vr.server import create_app, start_control_loop

PORT = 8459
_fails = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fails.append(name)


def _down_quat():
    # controller pointing straight down -> gripper down (roll~180)
    return [float(np.sin(np.deg2rad(-45.0))), 0.0, 0.0,
            float(np.cos(np.deg2rad(-45.0)))]


def part1_receiver(arm):
    print("\n--- Part 1: receiver logic ---")
    rec = TeleopReceiver(arm, recorder=None, recorder_factory=None)

    # --- clutch servo: engage, then move the controller +0.12 m in XR x ---
    q = _down_quat()
    p0 = [0.0, 0.0, 0.0]
    # engage tick
    rec.handle_input({"right": {"pos": p0, "quat": q, "buttons": [False, True], "axes": []}})
    rec.tick(1 / 60.0)
    _, ee_start = arm.get_position()
    ee_x0 = ee_start[0]
    # hold grip, move controller +x; many ticks for the smoother to converge
    moved = [0.12, 0.0, 0.0]
    for _ in range(120):
        rec.handle_input({"right": {"pos": moved, "quat": q,
                                    "buttons": [False, True], "axes": []}})
        rec.tick(1 / 60.0)
        time.sleep(0.005)
    _, ee_end = arm.get_position()
    ee_x1 = ee_end[0]
    check("clutch did not crash / IK mostly OK", not rec.ik_fail)
    check(f"clutch servo moved EE in +x (dx={ee_x1 - ee_x0:.1f}mm)",
          (ee_x1 - ee_x0) > 20.0)

    # release clutch
    rec.handle_input({"right": {"pos": moved, "quat": q,
                                "buttons": [False, False], "axes": []}})
    rec.tick(1 / 60.0)
    check("clutch released", not rec.clutch.engaged)

    # --- gripper toggle on trigger rising edge ---
    g0 = rec.gripper_closed
    rec.handle_input({"right": {"pos": moved, "quat": q,
                                "buttons": [True, False], "axes": []}})
    rec.tick(1 / 60.0)
    check("gripper toggled on trigger edge", rec.gripper_closed != g0)
    # falling edge must NOT toggle again
    rec.handle_input({"right": {"pos": moved, "quat": q,
                                "buttons": [False, False], "axes": []}})
    rec.tick(1 / 60.0)
    g_after = rec.gripper_closed
    rec.handle_input({"right": {"pos": moved, "quat": q,
                                "buttons": [True, False], "axes": []}})
    rec.tick(1 / 60.0)
    check("gripper toggles back on second press", rec.gripper_closed != g_after)

    # --- rail jog via left thumbstick X ---
    rail0 = rec.rail_mm
    for _ in range(30):
        rec.handle_input({
            "right": {"pos": moved, "quat": q, "buttons": [False, False], "axes": []},
            "left": {"pos": [0, 0, 0], "quat": [0, 0, 0, 1],
                     "buttons": [], "axes": [0.0, 0.0, 1.0, 0.0]},
        })
        rec.tick(1 / 30.0)
    check(f"rail jogged (+) from {rail0:.0f} to {rec.rail_mm:.0f}mm",
          rec.rail_mm > rail0 + 5.0)
    with arm.lock:
        ctrl_rail_mm = float(arm.data.ctrl[arm.act_ids[RAIL_ACT]]) * 1000.0
    check("rail ctrl reflects jog target", abs(ctrl_rail_mm - rec.rail_mm) < 1.0)
    check("rail clamped <= 700", rec.rail_mm <= config.RAIL_MAX_MM + 1e-6)

    # --- head pose drives vr_head mocap (first sample -> base viewpoint) ---
    rec.handle_input({"head": {"pos": [0.0, 1.5, 0.0], "quat": [0, 0, 0, 1]},
                      "right": {"pos": moved, "quat": q, "buttons": [False, False], "axes": []}})
    rec.tick(1 / 60.0)
    mid = rec._mocap_id
    check("vr_head mocap resolved", mid is not None)
    if mid is not None:
        with arm.lock:
            mp = arm.data.mocap_pos[mid].copy()
        check(f"head mocap at base viewpoint (got {np.round(mp,3)})",
              np.allclose(mp, config.VR_HEAD_BASE_M, atol=1e-6))

    # --- reset scene ---
    rec.handle_input({"right": {"pos": moved, "quat": q,
                                "buttons": [False, False, False, False, False, True],
                                "axes": []}})
    rec.tick(1 / 60.0)
    check("reset_scene ran without error & cleared gripper flag",
          not rec.gripper_closed)

    return rec


async def _ws_roundtrip():
    got_frame = False
    sent_ok = False
    uri = f"ws://127.0.0.1:{PORT}/ws"
    async with websockets.connect(uri, max_size=None) as wsc:
        # push one input message
        await wsc.send(json.dumps({
            "head": {"pos": [0, 1.4, 0], "quat": [0, 0, 0, 1]},
            "right": {"pos": [0, 0, 0], "quat": [0, 0, 0, 1],
                      "buttons": [False, False], "axes": []},
        }))
        sent_ok = True
        for _ in range(40):
            try:
                msg = await asyncio.wait_for(wsc.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, (bytes, bytearray)):
                got_frame = (len(msg) > 100 and msg[0] in (0, 1, 2))
                if got_frame:
                    break
    return got_frame, sent_ok


def part2_transport(arm, renderer):
    print("\n--- Part 2: server transport ---")
    receiver = TeleopReceiver(arm, recorder=None, recorder_factory=None)
    start_control_loop(receiver)
    app = create_app(arm, renderer, receiver)

    server = __import__("uvicorn").Server(
        __import__("uvicorn").Config(app, host="127.0.0.1", port=PORT,
                                     log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 10:
        time.sleep(0.05)
    check("uvicorn server started", server.started)

    # /config.json
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/config.json",
                                    timeout=5) as r:
            cfg = json.loads(r.read())
        check(f"/config.json returns mode={cfg.get('mode')}",
              cfg.get("mode") in ("mono", "stereo"))
    except Exception as e:
        check(f"/config.json reachable ({e})", False)

    # index page
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5) as r:
            html = r.read().decode()
        check("GET / serves the WebXR client", "xr-client.js" in html)
    except Exception as e:
        check(f"GET / reachable ({e})", False)

    got_frame, sent_ok = asyncio.run(_ws_roundtrip())
    check("WS accepted an input message", sent_ok)
    check("WS delivered a rendered frame (eye-prefixed JPEG)", got_frame)

    server.should_exit = True
    th.join(timeout=3)


def main():
    print("[itest] building arm (render=False)")
    arm = SimXArmAPI(scene_xml="envs/lab_scene.xml", render=False)
    renderer = TwinRenderer(arm.model, arm.data, arm.lock, mode="stereo")
    renderer.start()
    try:
        time.sleep(0.4)
        part1_receiver(arm)
        part2_transport(arm, renderer)
    finally:
        renderer.stop()
        arm.disconnect()

    print(f"\n{'ALL PASS' if not _fails else 'FAILURES: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
