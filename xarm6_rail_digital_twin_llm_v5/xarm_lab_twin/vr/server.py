# vr/server.py
"""FastAPI app + single WebSocket transport for VR teleop.

MVP transport is WebSocket-JPEG: simple and robust on a LAN. WebRTC is a
later upgrade (see the `# Phase 2: aiortc` note in the sender loop).

Routes:
  GET  /              -> vr/static/index.html  (the WebXR client)
  GET  /static/*      -> static assets
  GET  /config.json   -> {mode, ipd, render_hz, control_hz}
  WS   /ws            -> frames down (binary, latest-wins) + status (text);
                         controller/head state up (text JSON).

Frame wire format (down): one binary message per frame, 1-byte eye prefix +
JPEG payload:  0x00 = mono, 0x01 = left eye, 0x02 = right eye.
Status (down): text JSON ({"type":"status", ...}).
Input  (up):   text JSON, see vr/teleop_receiver.py for the schema.

The render loop lives in TwinRenderer's own thread; the control tick runs in
a daemon thread started here (independent of any connection). The WebSocket
handler only moves bytes.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vr import config

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Frame eye-prefix bytes.
EYE_MONO = b"\x00"
EYE_LEFT = b"\x01"
EYE_RIGHT = b"\x02"


def start_control_loop(receiver) -> threading.Thread:
    """Spawn the control tick in a daemon thread at config.CONTROL_HZ.

    Runs regardless of client connection; receiver.tick() returns early when
    no input has arrived yet, so this is harmless before/after a session.
    """
    period = 1.0 / max(config.CONTROL_HZ, 1e-6)

    def loop():
        prev = time.time()
        next_t = prev
        while True:
            now = time.time()
            dt = now - prev
            prev = now
            try:
                receiver.tick(dt)
            except Exception as e:  # noqa: BLE001 - never let the loop die
                print(f"[VR] control tick error: {e}")
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"[VR] control loop started ({config.CONTROL_HZ:.0f}Hz, "
          f"servo={config.SERVO_MODE})")
    return t


def create_app(arm, renderer, receiver) -> FastAPI:
    app = FastAPI(title="xArm6 VR teleop")
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    @app.get("/config.json")
    async def client_config():
        return JSONResponse({
            "mode": renderer.mode,
            "ipd": config.IPD_M,
            "render_hz": config.RENDER_HZ,
            "control_hz": config.CONTROL_HZ,
        })

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        print("[VR] client connected")
        # Reset relative head tracking so each new session re-zeroes on the
        # operator's current head position.
        receiver._head_xr0 = None

        async def up():
            """headset -> server: controller/head state."""
            try:
                while True:
                    msg = await ws.receive_text()
                    try:
                        receiver.handle_input(json.loads(msg))
                    except (ValueError, TypeError) as e:
                        print(f"[VR] bad input message: {e}")
            except WebSocketDisconnect:
                pass

        async def down():
            """server -> headset: newest frame(s) + periodic status."""
            last_seq = -1
            last_status = 0.0
            frame_period = 1.0 / max(config.RENDER_HZ, 1e-6)
            status_period = 1.0 / max(config.STATUS_HZ, 1e-6)
            try:
                while True:
                    seq, mono, left, right = renderer.latest.get()
                    if seq != last_seq:
                        last_seq = seq
                        # Phase 2: aiortc — replace these send_bytes() with a
                        # WebRTC video track for lower-latency depth feedback.
                        if mono is not None:
                            await ws.send_bytes(EYE_MONO + mono)
                        elif left is not None and right is not None:
                            await ws.send_bytes(EYE_LEFT + left)
                            await ws.send_bytes(EYE_RIGHT + right)
                    now = time.time()
                    if now - last_status >= status_period:
                        last_status = now
                        await ws.send_text(json.dumps(receiver.status()))
                    await asyncio.sleep(frame_period)
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass
            except Exception:
                # Socket closed underneath us (client vanished mid-send) — the
                # up() task already noticed; exit quietly.
                pass

        # Run both directions; when either finishes (almost always the client
        # disconnecting in up()), cancel the other so down() never tries to
        # send on an already-closed socket.
        tasks = [asyncio.ensure_future(up()), asyncio.ensure_future(down())]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:  # noqa: BLE001
            print(f"[VR] websocket error: {e}")
        finally:
            print("[VR] client disconnected")

    return app
