#!/usr/bin/env python3
"""Open the scene in the interactive MuJoCo viewer. No LLM, no API key, no motion.

Every other entry point either drives the arm or needs an Anthropic key, which
makes "just look at the scene" awkward — and the interactive viewer is not the
same renderer as the offscreen EGL path used by `Recorder` and the VR server, so
a shading artifact can appear in one and not the other. This exists to inspect
the scene as a human sees it.

    python scripts/view_scene.py                      # default (mesh) scene
    python scripts/view_scene.py --scene primitive
    python scripts/view_scene.py --settle 3           # let physics settle first

Do NOT set MUJOCO_GL=egl for this: the interactive viewer needs a windowed GL
context. (The VR server is the opposite case — it runs render=False precisely so
GLFW and EGL do not contend for GL in one process.)

Viewer controls: left-drag orbit, right-drag pan, scroll zoom, double-click to
select a body, and Ctrl+right-drag to apply a force. Press Tab for the panels;
the Rendering panel toggles shadows, reflections and the headlight, which is the
fastest way to attribute a shading artifact to a specific setting.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())

SCENES = {"primitive": "envs/lab_scene_primitive.xml",
          "meshes": "envs/lab_scene.xml"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="meshes", choices=["meshes", "primitive"])
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds of physics to run before you look (objects drop "
                         "from their spawn height onto the bench)")
    args = ap.parse_args()

    if os.environ.get("MUJOCO_GL") == "egl":
        print("[view_scene] MUJOCO_GL=egl is set, which gives an offscreen context "
              "and no window. Unsetting it for this process.")
        os.environ.pop("MUJOCO_GL")

    from sim.mujoco_env import SimXArmAPI

    path = SCENES[args.scene]
    print(f"[view_scene] opening {path} in the interactive viewer")
    arm = SimXArmAPI(scene_xml=path, render=True)

    if args.settle:
        print(f"[view_scene] settling physics for {args.settle}s")
        time.sleep(args.settle)

    print("[view_scene] viewer is open. Close the window or press Ctrl+C here to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[view_scene] closing")
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
