# vr/smoke_test.py
"""Headless render smoke test (acceptance test 1).

Builds the arm with render=False, instantiates TwinRenderer, and writes
mono.jpg / left.jpg / right.jpg to vr/ (or --out-dir). Confirms:
  * EGL offscreen rendering works (no GL errors),
  * the arm/bench are visible (non-trivial image),
  * left.jpg != right.jpg (proves the stereo IPD offset).

Run from the working directory (xarm_lab_twin/):
    MUJOCO_GL=egl python -m vr.smoke_test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Headless GPU: pick EGL before mujoco is imported anywhere.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from sim.mujoco_env import SimXArmAPI
from vr.stereo_renderer import TwinRenderer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--scene", default="envs/lab_scene.xml")
    args = ap.parse_args()

    print("[smoke] building arm (render=False) ...")
    arm = SimXArmAPI(scene_xml=args.scene, render=False)
    try:
        # Let the sim settle a couple of steps so bodies are placed.
        import time
        time.sleep(0.3)

        rend = TwinRenderer(arm.model, arm.data, arm.lock)

        mono = rend.mono_frame()
        left, right = rend.stereo_frames()

        for name, data in (("mono.jpg", mono), ("left.jpg", left),
                           ("right.jpg", right)):
            path = os.path.join(args.out_dir, name)
            with open(path, "wb") as f:
                f.write(data)
            print(f"[smoke] wrote {path}  ({len(data)} bytes)")

        ok = True
        # Non-trivial frames.
        for name, data in (("mono", mono), ("left", left), ("right", right)):
            if len(data) < 1000:
                print(f"[smoke] FAIL: {name} JPEG suspiciously small ({len(data)}B)")
                ok = False

        # Stereo offset: the two eye JPEGs must differ.
        if left == right:
            print("[smoke] FAIL: left.jpg == right.jpg (no stereo offset!)")
            ok = False
        else:
            # quantify pixel difference for confidence
            from PIL import Image
            import io
            li = np.asarray(Image.open(io.BytesIO(left)).convert("RGB"), float)
            ri = np.asarray(Image.open(io.BytesIO(right)).convert("RGB"), float)
            mad = float(np.mean(np.abs(li - ri)))
            print(f"[smoke] stereo mean abs pixel diff = {mad:.3f}")
            if mad < 0.05:
                print("[smoke] FAIL: stereo images nearly identical")
                ok = False

        print("[smoke] PASS" if ok else "[smoke] FAIL")
        return 0 if ok else 1
    finally:
        arm.disconnect()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # WSLg / EGL teardown can deadlock on interpreter shutdown; the data is
    # already on disk, so exit hard (mirrors replay.py's epilogue).
    os._exit(rc)
