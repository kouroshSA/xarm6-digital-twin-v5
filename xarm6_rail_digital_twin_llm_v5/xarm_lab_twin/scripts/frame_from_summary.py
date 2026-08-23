#!/usr/bin/env python3
"""Render a still of the scene as a run left it, from a saved summary JSON.

Offscreen and after the fact, deliberately. Grabbing a screenshot inside a
run would open a second GL context while the viewer holds one, which is the
`mj_copyDataVisual` race that has been killing runs under `--render`. Saving
qpos and rendering later costs nothing and cannot race anything.

    python scripts/frame_from_summary.py run.json out.png
    python scripts/frame_from_summary.py --dir summaries/ --out frames/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def render(summary_path: Path, out_path: Path,
           azimuth: float = -128.0, elevation: float = -20.0,
           distance: float = 0.95, lookat=(0.05, -0.25, 0.80)) -> str:
    payload = json.loads(Path(summary_path).read_text())
    qpos = payload.get("final_qpos")
    if not qpos:
        return f"{summary_path.name}: no final_qpos recorded"
    model = mujoco.MjModel.from_xml_path(payload.get("scene_xml", "envs/lab_scene.xml"))
    data = mujoco.MjData(model)
    n = min(len(qpos), model.nq)
    data.qpos[:n] = np.asarray(qpos[:n])
    mujoco.mj_forward(model, data)

    r = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(data, cam)
    Image.fromarray(r.render()).save(out_path)
    return f"{out_path.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--dir", help="render every *.json in this directory")
    ap.add_argument("--out-dir", default="frames")
    args = ap.parse_args()

    if args.dir:
        outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(args.dir).glob("*.json")):
            print("  " + render(f, outd / (f.stem + ".png")))
        return 0
    if not (args.summary and args.out):
        ap.error("need SUMMARY OUT, or --dir")
    print(render(Path(args.summary), Path(args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
