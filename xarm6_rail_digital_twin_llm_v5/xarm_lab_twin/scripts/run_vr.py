# scripts/run_vr.py
"""Entry point for Meta Quest 3 teleoperation of the MuJoCo digital twin.

Wires SimXArmAPI (render=False) + Recorder + TwinRenderer + TeleopReceiver
together and serves the WebXR client over HTTP(S) with a single WebSocket.

    python scripts/run_vr.py [--mode mono|stereo] [--servo direct|validated]
                             [--no-record] [--port 8443] [--scale 1.0]
                             [--cert vr/cert.pem --key vr/key.pem]

The headset opens https://<workstation-ip>:<port> in the Quest Browser; see
vr/README.md for cert generation and the secure-context requirement.
"""
import argparse
import os
import sys

# Make project root importable when running as `python scripts/run_vr.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headless GPU: pick the EGL offscreen GL backend BEFORE mujoco is imported
# anywhere (the renderer needs it; the desktop viewer is never launched here).
os.environ.setdefault("MUJOCO_GL", "egl")

from env_loader import load_env
load_env()

import uvicorn

from recording import Recorder
from sim.mujoco_env import SimXArmAPI
from vr import config
from vr.server import create_app, start_control_loop
from vr.stereo_renderer import TwinRenderer
from vr.teleop_receiver import TeleopReceiver


def _local_ip() -> str:
    """Best-effort LAN IP for the URL banner (no external traffic sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="xArm6 digital-twin VR teleop")
    ap.add_argument("--mode", choices=["mono", "stereo"], default=config.MODE)
    ap.add_argument("--servo", choices=["direct", "validated"],
                    default=config.SERVO_MODE)
    ap.add_argument("--no-record", action="store_true",
                    help="Disable recording entirely (A button becomes a no-op).")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--scale", type=float, default=config.WORLD_SCALE,
                    help="XR->twin world scale (shrink <1 to bring the bench "
                         "within easy arm's reach).")
    ap.add_argument("--scene", default="envs/lab_scene.xml")
    ap.add_argument("--cert", default=None, help="TLS cert PEM (enables HTTPS).")
    ap.add_argument("--key", default=None, help="TLS key PEM (enables HTTPS).")
    ap.add_argument("--task-label", default="vr_teleop",
                    help="Label stamped on recorded sessions.")
    args = ap.parse_args()

    # Push CLI choices into config so every module reads one source of truth.
    config.MODE = args.mode
    config.SERVO_MODE = args.servo
    config.WORLD_SCALE = args.scale
    config.PORT = args.port
    config.RECORD = not args.no_record

    print(f"[VR] building twin (render=False, MUJOCO_GL={os.environ.get('MUJOCO_GL')})")
    arm = SimXArmAPI(scene_xml=args.scene, render=False)

    # Recorder is created but NOT auto-started; the A button starts/stops takes.
    def make_recorder():
        return Recorder(model=arm.model, data=arm.data, lock=arm.lock,
                        interface="vr_teleop", scene_xml=args.scene,
                        enable_frames=False)

    recorder = make_recorder() if config.RECORD else None

    renderer = TwinRenderer(arm.model, arm.data, arm.lock, mode=args.mode)
    receiver = TeleopReceiver(arm, recorder=recorder,
                              recorder_factory=make_recorder if config.RECORD else None,
                              task_label=args.task_label)

    renderer.start()
    start_control_loop(receiver)

    app = create_app(arm, renderer, receiver)

    use_tls = bool(args.cert and args.key)
    scheme = "https" if use_tls else "http"
    ip = _local_ip()
    print("\n" + "=" * 64)
    print(f"[VR] mode={args.mode}  servo={args.servo}  "
          f"record={'on' if config.RECORD else 'off'}  scale={args.scale}")
    print(f"[VR] Open on the headset:  {scheme}://{ip}:{args.port}")
    if not use_tls:
        print("[VR] NOTE: immersive-vr needs a secure context. Either pass "
              "--cert/--key for HTTPS,")
        print("[VR]       or whitelist this origin via the Quest's "
              "chrome://flags 'Insecure origins treated as secure'.")
    print("=" * 64 + "\n")

    uv_kwargs = dict(host=config.HOST, port=args.port, log_level="warning")
    if use_tls:
        uv_kwargs.update(ssl_certfile=args.cert, ssl_keyfile=args.key)

    server = uvicorn.Server(uvicorn.Config(app, **uv_kwargs))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[VR] shutting down…")
        renderer.stop()
        if receiver.rec is not None and receiver.rec.is_recording:
            path = receiver.rec.stop(kept=True, task_label=args.task_label)
            print(f"[VR] saved in-progress take -> {path}")
        arm.disconnect()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # WSLg / EGL teardown can deadlock on interpreter shutdown; data is already
    # persisted (mirrors replay.py's epilogue).
    os._exit(rc)
