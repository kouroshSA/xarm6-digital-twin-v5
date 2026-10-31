#!/usr/bin/env python3
"""Render the per-command snapshots a run saved with ``run_task.py --snapshot-dir``.

Offscreen and after the fact, like ``frame_from_summary.py``: a second GL
context beside the live viewer races it. Each snapshot becomes one PNG of four
views (overview plus close-ups of the incubator, the cabinet and the bench, for
the simready demo room), titled with the command and its return code.

    python scripts/render_snapshots.py SNAPDIR             # every snapshot
    python scripts/render_snapshots.py SNAPDIR --failed    # only rc != 0
    python scripts/render_snapshots.py SNAPDIR --only 007 012
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
from PIL import Image, ImageDraw  # noqa: E402

from sim.render_options import scene_option  # noqa: E402

# (label, lookat, distance, azimuth, elevation)
VIEWS = [
    ("overview", (-0.20, 0.10, 0.95), 3.0, 70.0, -45.0),
    # The door views look in from the side away from the rail: straight on from
    # the bench, the arm parked at home hid both doors.
    ("incubator", (0.00, 0.60, 1.05), 1.40, 125.0, -25.0),
    ("cabinet", (-0.85, -0.30, 1.10), 1.60, 160.0, -25.0),
    ("bench", (0.10, -0.35, 0.80), 1.30, 90.0, -40.0),
]
W, H = 640, 480


def render(snap: Path, out: Path, models: dict) -> str:
    s = json.loads(snap.read_text())
    xml = s.get("scene_xml")
    if xml not in models:
        m = mujoco.MjModel.from_xml_path(xml)
        models[xml] = (m, mujoco.Renderer(m, H, W))
    model, r = models[xml]
    data = mujoco.MjData(model)
    q = s["final_qpos"]
    n = min(len(q), model.nq)
    data.qpos[:n] = np.asarray(q[:n])
    mujoco.mj_forward(model, data)
    opt = scene_option()
    tiles = []
    for label, lookat, dist, az, el in VIEWS:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(data, cam, scene_option=opt)
        img = Image.fromarray(r.render())
        ImageDraw.Draw(img).text((8, 8), label, fill=(255, 255, 0))
        tiles.append(img)
    sheet = Image.new("RGB", (2 * W, 2 * H + 24))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * W, 24 + (i // 2) * H))
    title = f"#{s['seq']} {s['action']} {s['params']} -> rc={s['result']}"
    ImageDraw.Draw(sheet).text((8, 6), title[:200], fill=(255, 255, 255))
    sheet.save(out)
    return out.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapdir")
    ap.add_argument("--out-dir", help="default: SNAPDIR/png")
    ap.add_argument("--failed", action="store_true", help="only rc != 0")
    ap.add_argument("--only", nargs="*", help="sequence-number prefixes")
    args = ap.parse_args()
    snaps = sorted(Path(args.snapdir).glob("*.json"))
    if args.failed:
        snaps = [p for p in snaps if not p.stem.endswith("_rc0")]
    if args.only:
        snaps = [p for p in snaps if any(p.name.startswith(o) for o in args.only)]
    outd = Path(args.out_dir or Path(args.snapdir) / "png")
    outd.mkdir(parents=True, exist_ok=True)
    models: dict = {}
    for p in snaps:
        print("  " + render(p, outd / (p.stem + ".png"), models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
