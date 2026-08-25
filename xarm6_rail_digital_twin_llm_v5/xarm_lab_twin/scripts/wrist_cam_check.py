#!/usr/bin/env python3
"""Look through the wrist camera — in the twin, on the real device, or both.

    python scripts/wrist_cam_check.py                     # sim only
    python scripts/wrist_cam_check.py --real              # real D435i only
    python scripts/wrist_cam_check.py --both --out /tmp/w # both, save PNGs

Sim frames come from ``envs/lab_scene.xml``; real frames come from whatever
D435i is plugged into this machine. The camera does not have to be on the robot
for ``--real`` to work -- with no arm attached the frames simply carry no pose,
which is the honest answer rather than a fabricated one.

With ``--both`` the two backends are compared on the things that must agree if
the twin is worth trusting: image size, intrinsics, and the deprojection of the
same pixel at the same depth. What is *not* compared is the picture -- the sim
scene and whatever the real camera is pointed at are different worlds. This
checks the camera model, not the content.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from perception import d435i_calib as calib  # noqa: E402
from perception.rgbd import RGBDFrame  # noqa: E402


def _report(label: str, frame: RGBDFrame) -> None:
    k = frame.intrinsics
    stats = frame.depth_stats()
    print(f"\n{label}")
    print(f"  colour     : {frame.color.shape} {frame.color.dtype}")
    print(f"  depth      : {frame.depth.shape} {frame.depth.dtype}, "
          f"{stats['valid_frac'] * 100:.1f}% valid")
    if stats["valid_frac"] > 0:
        print(f"  depth range: {stats['min_m']:.3f} - {stats['max_m']:.3f} m "
              f"(median {stats['median_m']:.3f})")
    print(f"  intrinsics : fx={k.fx:.3f} fy={k.fy:.3f} cx={k.cx:.3f} cy={k.cy:.3f}")
    print(f"  FOV        : {k.fovx_deg:.2f} x {k.fovy_deg:.2f} deg")

    centre = frame.deproject_pixel(k.cx, k.cy)
    if centre is None:
        print("  centre px  : no valid depth")
    else:
        print(f"  centre px  : {np.round(centre * 1000, 1).tolist()} mm "
              f"in the camera frame")
        if frame.cam_to_world is not None:
            w = frame.pixel_to_world(k.cx, k.cy)
            print(f"               {np.round(w * 1000, 1).tolist()} mm in the "
                  f"world frame")
    if frame.cam_to_world is None:
        print("  pose       : none (no arm attached) — pixel_to_world would raise")


def _save(frame: RGBDFrame, out_dir: str, tag: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    Image.fromarray(frame.color).save(os.path.join(out_dir, f"{tag}_color.png"))

    # Depth as a viewable image: NaN (no return) renders black, everything else
    # spans the frame's own valid range so near/far detail survives.
    d = frame.depth.copy()
    valid = np.isfinite(d)
    vis = np.zeros(d.shape, dtype=np.uint8)
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        span = max(hi - lo, 1e-6)
        vis[valid] = (255 * (1.0 - (d[valid] - lo) / span)).astype(np.uint8)
    Image.fromarray(vis).save(os.path.join(out_dir, f"{tag}_depth.png"))
    print(f"  saved      : {out_dir}/{tag}_color.png, {tag}_depth.png")


def capture_sim(pose: str) -> RGBDFrame:
    import mujoco

    from perception.sim_camera import SimWristCamera

    model = mujoco.MjModel.from_xml_path("envs/lab_scene.xml")
    data = mujoco.MjData(model)
    if pose == "bench":
        # Same look-at-the-bench pose the projection test uses, so the two
        # scripts show the same view and disagree visibly if one drifts.
        for name, deg in (("joint1", 0.0), ("joint2", -35.0), ("joint3", -60.0),
                          ("joint4", 0.0), ("joint5", 95.0), ("joint6", 0.0)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = np.radians(deg)
    mujoco.mj_forward(model, data)

    cam = SimWristCamera.from_model(model, data)
    frame = cam.capture()
    cam.close()
    return frame


def capture_real() -> RGBDFrame:
    from perception.realsense_camera import RealSenseWristCamera

    cam = RealSenseWristCamera(arm=None)
    # The D400 auto-exposure needs a few frames to settle; the first capture is
    # usually dark and its depth sparse. Cheap to discard, confusing not to.
    for _ in range(10):
        cam.capture()
    frame = cam.capture()
    cam.close()
    return frame


def compare(sim: RGBDFrame, real: RGBDFrame) -> int:
    """Do the two backends model the same camera? Returns a failure count."""
    print("\n--- sim vs real ---")
    failures = 0

    if sim.color.shape != real.color.shape:
        print(f"  FAIL image size: sim {sim.color.shape} vs real {real.color.shape}")
        failures += 1
    else:
        print(f"  OK   image size {sim.color.shape[1]}x{sim.color.shape[0]}")

    worst = max(abs(getattr(sim.intrinsics, f) - getattr(real.intrinsics, f))
                for f in ("fx", "fy", "cx", "cy"))
    if worst > 1.0:
        print(f"  FAIL intrinsics differ by up to {worst:.3f} px — the sim "
              f"camera is not modelling this device")
        failures += 1
    else:
        print(f"  OK   intrinsics agree to {worst:.4f} px")

    # The one that actually matters: the same pixel at the same depth must
    # become the same 3D point, or nothing learned in sim transfers.
    probes = [(80.0, 60.0), (320.0, 240.0), (560.0, 420.0)]
    worst_mm = 0.0
    for u, v in probes:
        a = sim.deproject_pixel(u, v, depth_m=0.5)
        b = real.deproject_pixel(u, v, depth_m=0.5)
        worst_mm = max(worst_mm, float(np.max(np.abs(a - b))) * 1000.0)
    if worst_mm > 1.0:
        print(f"  FAIL deprojection differs by up to {worst_mm:.3f} mm at 0.5 m")
        failures += 1
    else:
        print(f"  OK   deprojection agrees to {worst_mm:.4f} mm at 0.5 m")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true", help="capture from the D435i")
    ap.add_argument("--both", action="store_true",
                    help="capture from both and compare the camera models")
    ap.add_argument("--out", metavar="DIR", help="save colour + depth PNGs here")
    ap.add_argument("--pose", choices=("bench", "home"), default="bench",
                    help="sim arm pose (default: bench, which sees the objects)")
    args = ap.parse_args()

    print(calib.summary())

    sim = real = None
    if not args.real or args.both:
        sim = capture_sim(args.pose)
        _report("SIM  (envs/lab_scene.xml)", sim)
        if args.out:
            _save(sim, args.out, "sim")

    if args.real or args.both:
        try:
            real = capture_real()
        except Exception as exc:  # noqa: BLE001
            print(f"\nREAL capture failed: {type(exc).__name__}: {exc}")
            print("  Is the D435i plugged in? `rs-enumerate-devices -s` lists it.")
            return 1
        _report("REAL (D435i on this machine)", real)
        if args.out:
            _save(real, args.out, "real")

    if sim is not None and real is not None:
        failures = compare(sim, real)
        print()
        print(f"{failures} mismatch(es)" if failures
              else "sim and real agree on the camera model")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
