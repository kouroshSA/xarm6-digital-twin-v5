#!/usr/bin/env python3
"""Survey the bench with the wrist camera from several known poses.

    python scripts/wrist_survey.py --dry-run          # coverage on paper, no motion
    python scripts/wrist_survey.py --pass nadir       # one pass, for real
    python scripts/wrist_survey.py --all --out /tmp/s # every pass

One frame from one pose is a guess about a 3D world. Several frames from poses
whose camera matrices are all known is a measurement -- the same object seen
from two viewpoints fixes its position by triangulation, WITHOUT trusting the
depth stream. That matters here: on 2026-08-31 the wrist depth put the benchtop
at world z~803 against a touched 750, and a second capture minutes later at the
same arm pose disagreed with the first. Colour plus a known pose is currently
the more trustworthy pair of the two, so every frame is saved with its pose.

What is saved per stop, into ``--out``:

    <pass>_<nn>_color.png     the colour frame
    <pass>_<nn>_depth.npy     metres, NaN where invalid (kept, but see above)
    manifest.json             commanded pose, measured pose, rail, cam_to_world,
                              intrinsics, and the predicted bench aim point

``--dry-run`` runs the whole geometry without a robot or a camera: it composes
each commanded pose through the same hand-eye chain ``cam_to_world`` uses and
reports where the optical axis meets the benchtop, plus the footprint there. Use
it to argue about coverage before anything moves, which is cheaper than
discovering a gap from the images afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from arm_backend import BENCH_TOP_Z_MM, RAIL_LIMITS_MM  # noqa: E402
from perception import d435i_calib as calib  # noqa: E402

#: Tool z of the nadir pass, in world mm. 300 mm of standoff over a 750 mm
#: benchtop: high enough that one frame covers a useful patch, low enough that
#: a 30 mm cube is still tens of pixels across.
NADIR_Z_MM = 1050.0

#: The working strip. Objects sit ~200 mm in front of the rail centreline
#: (y=-50), which is y=-250; nothing is ever placed ON the centreline because
#: the track is there.
WORK_Y_MM = -250.0


def _pose_at_rail(rail_mm: float, dx: float = 0.0) -> float:
    """World x that puts the tool straight out in front of the base.

    The base sits at world x = -350 + rail, so commanding this x keeps the arm
    in the same comfortable base-frame pose at every rail stop instead of
    reaching further and further sideways as the carriage moves.
    """
    return -350.0 + rail_mm + dx


def nadir_pass(step_mm: float = 100.0) -> list[dict]:
    """Straight down, stepping along the rail. The detail pass."""
    out = []
    r = RAIL_LIMITS_MM[0]
    while r <= RAIL_LIMITS_MM[1] + 1e-6:
        out.append(dict(rail=r, x=_pose_at_rail(r), y=WORK_Y_MM, z=NADIR_Z_MM,
                        roll=180.0, pitch=0.0, yaw=0.0))
        r += step_mm
    return out


def raised_pass(step_mm: float = 175.0, pitch_deg: float = 25.0) -> list[dict]:
    """Higher, tilted back toward the rail, to take in the whole front strip.

    yaw=-90 puts the tilt axis across the bench rather than along the rail, so
    `pitch` swings the view between the rail centreline and the bench front
    edge. With yaw=0 the same pitch would tilt ALONG the rail, which the nadir
    pass already covers by stepping.
    """
    out = []
    r = RAIL_LIMITS_MM[0]
    while r <= RAIL_LIMITS_MM[1] + 1e-6:
        out.append(dict(rail=r, x=_pose_at_rail(r), y=WORK_Y_MM - 60.0,
                        z=NADIR_Z_MM + 120.0,
                        roll=180.0, pitch=pitch_deg, yaw=-90.0))
        r += step_mm
    return out


def oblique_pass(from_rail: float, pitch_deg: float = 35.0) -> list[dict]:
    """Park at one end of the rail and look back along it, for a side view.

    A nadir pass sees lids and tops; it cannot tell a tall object from a flat
    patch of the same colour, which is exactly the confusion that made a red
    TAPE SQUARE read as a red cube on 2026-08-31. An oblique view separates
    them, and gives triangulation a wide baseline against the nadir frames.
    """
    # Sign check, because this was wrong the first time and the dry run caught
    # it: with roll=180 and yaw=0 the optical axis is (-sin pitch, 0, -cos
    # pitch), so POSITIVE pitch looks toward world -x. Parked at the North end
    # (rail 0, base at world x=-350) we want to look South, toward +x, so the
    # pitch must be negative there. The first version had it the other way and
    # aimed 240 mm off the North end of the bench at empty floor.
    look_along = -1.0 if from_rail <= sum(RAIL_LIMITS_MM) / 2 else 1.0
    out = []
    for dy in (0.0, -120.0):
        out.append(dict(rail=from_rail, x=_pose_at_rail(from_rail),
                        y=WORK_Y_MM + dy, z=NADIR_Z_MM + 150.0,
                        roll=180.0, pitch=pitch_deg * look_along, yaw=0.0))
    return out


def cam_pose_for(pose: dict, tcp_z_mm: float = 217.0) -> np.ndarray:
    """4x4 camera-optical -> world for a COMMANDED pose.

    Deliberately the same composition as
    ``RealSenseWristCamera.cam_to_world``: rpy -> flange rotation, back the TCP
    offset out to reach the flange, then the fixed hand-eye transform. Predicted
    here from the commanded pose rather than a measured one, which is the whole
    point of --dry-run. A second, independent euler chain would be defect class
    #1, so this must stay a mirror of that method.
    """
    r_wf = calib._euler_to_mat(math.radians(pose["roll"]),
                               math.radians(pose["pitch"]),
                               math.radians(pose["yaw"]))
    t_wf = (np.array([pose["x"], pose["y"], pose["z"]], float)
            - r_wf @ np.array([0.0, 0.0, tcp_z_mm])) / 1000.0
    r_fc, t_fc = calib.flange_to_color_optical()
    m = np.eye(4)
    m[:3, :3] = r_wf @ r_fc
    m[:3, 3] = t_wf + r_wf @ t_fc
    return m


def bench_aim(m: np.ndarray, bench_z_mm: float = BENCH_TOP_Z_MM):
    """Where the optical axis meets the benchtop, and the footprint there.

    Returns (x_mm, y_mm, range_mm, width_mm, height_mm) or None if the camera
    is not looking at the bench at all -- which is a pose worth catching on
    paper rather than in a blank image.
    """
    origin, axis = m[:3, 3] * 1000.0, m[:3, 2]
    if abs(axis[2]) < 1e-6 or (bench_z_mm - origin[2]) / axis[2] <= 0:
        return None
    t = (bench_z_mm - origin[2]) / axis[2]
    hit = origin + t * axis
    k = calib.COLOR_INTRINSICS
    return (hit[0], hit[1], t, t * k.width / k.fx, t * k.height / k.fy)


def _png_writer():
    """Resolve a PNG writer up front, or refuse the survey now rather than later.

    This used to fall back to `imageio = None` and then skip the write, so a
    missing dependency would have driven the arm through every stop, printed a
    valid-depth percentage per frame, written a manifest naming colour files
    that were never created, and exited 0. The colour frames are the part of
    this survey we actually trust -- losing them silently while reporting
    success is defect class #2 with the robot already moving.
    """
    try:
        import imageio.v2 as imageio
        return lambda path, img: imageio.imwrite(path, img)
    except ImportError:
        pass
    try:
        import cv2
        return lambda path, img: cv2.imwrite(path, img[:, :, ::-1])
    except ImportError:
        pass
    raise SystemExit("wrist_survey needs imageio or opencv to save colour "
                     "frames; refusing to move the arm and throw them away")


def describe(name: str, poses: list[dict]) -> None:
    print(f"\n  {name}  ({len(poses)} stops)")
    print(f"  {'rail':>6} {'world x':>8} {'y':>7} {'z':>7} "
          f"{'pitch':>6} {'yaw':>5}   {'aims at (x, y)':>20} "
          f"{'range':>7} {'covers':>15}")
    for p in poses:
        a = bench_aim(cam_pose_for(p))
        if a is None:
            aim = "  NOT ON THE BENCH"
        else:
            aim = (f"({a[0]:7.0f}, {a[1]:6.0f})  {a[2]:6.0f}  "
                   f"{a[3]:5.0f} x {a[4]:4.0f} mm")
        print(f"  {p['rail']:6.0f} {p['x']:8.0f} {p['y']:7.0f} {p['z']:7.0f} "
              f"{p['pitch']:6.1f} {p['yaw']:5.0f}   {aim}")
    xs = [bench_aim(cam_pose_for(p)) for p in poses]
    xs = [a for a in xs if a]
    if xs:
        lo_x = min(a[0] - a[3] / 2 for a in xs); hi_x = max(a[0] + a[3] / 2 for a in xs)
        lo_y = min(a[1] - a[4] / 2 for a in xs); hi_y = max(a[1] + a[4] / 2 for a in xs)
        print(f"  union of footprints: x {lo_x:.0f}..{hi_x:.0f}  "
              f"y {lo_y:.0f}..{hi_y:.0f} mm")


PASSES = {"nadir": nadir_pass, "raised": raised_pass}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.229")
    ap.add_argument("--pass", dest="which", choices=list(PASSES) + ["oblique"],
                    action="append")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--oblique-from-rail", type=float, default=None,
                    help="rail mm to park at for the oblique pass (0 or 700)")
    ap.add_argument("--out", default="/tmp/wrist_survey")
    ap.add_argument("--dry-run", action="store_true",
                    help="geometry only: no robot, no camera, nothing moves")
    ap.add_argument("--speed", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=1.2,
                    help="seconds to let the arm stop shaking before capturing")
    args = ap.parse_args()

    which = args.which or (["nadir", "raised", "oblique"] if args.all else ["nadir"])
    plans = {}
    for name in which:
        if name == "oblique":
            if args.oblique_from_rail is None:
                print("  oblique pass needs --oblique-from-rail (0 or 700); "
                      "skipping it")
                continue
            plans[name] = oblique_pass(args.oblique_from_rail)
        else:
            plans[name] = PASSES[name]()

    if args.dry_run:
        print("DRY RUN -- predicted coverage, nothing moves.")
        for name, poses in plans.items():
            describe(name, poses)
        print("\n  Aim points are computed through the same hand-eye chain as\n"
              "  cam_to_world, so they inherit its calibration. They say where\n"
              "  the camera is POINTED, not that the calibration is right.")
        return 0

    from hardware.real_arm import RealXArmAPI
    from perception.realsense_camera import RealSenseWristCamera

    os.makedirs(args.out, exist_ok=True)
    # MERGE with whatever is already in this directory. Writing a fresh
    # manifest per invocation silently destroyed the nadir pass's poses the
    # moment the next pass ran: 15 images on disk, 7 of them described. The
    # images survive that, but a frame whose cam_to_world is gone cannot be
    # triangulated against anything, which is the entire reason for the survey.
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        print(f"  merging into {len(manifest)} existing stop(s) in {manifest_path}")
    arm = RealXArmAPI(args.ip, effector="standard", ft_sensor=False)
    cam = RealSenseWristCamera(arm=arm)
    write_png = _png_writer()          # resolved BEFORE the arm moves

    try:
        for name, poses in plans.items():
            print(f"\n=== pass {name}: {len(poses)} stops ===")
            for i, p in enumerate(poses):
                if arm.get_rail_position()[1] != p["rail"]:
                    arm.set_rail_position(p["rail"], speed=80, wait=True)
                rc = arm.set_position(x=p["x"], y=p["y"], z=p["z"],
                                      roll=p["roll"], pitch=p["pitch"],
                                      yaw=p["yaw"], speed=args.speed, wait=True)
                if rc != 0:
                    print(f"  [{i:02d}] REFUSED rc={rc}; skipping this stop")
                    continue
                time.sleep(args.settle)
                frame = cam.capture(align=True)
                code, measured = arm.get_position()
                m = cam.cam_to_world()
                stem = os.path.join(args.out, f"{name}_{i:02d}")
                write_png(stem + "_color.png", frame.color)
                np.save(stem + "_depth.npy", frame.depth)
                valid = float(np.isfinite(frame.depth).mean())
                manifest = [e for e in manifest
                            if not (e["pass_name"] == name and e["index"] == i)]
                manifest.append(dict(
                    pass_name=name, index=i, commanded=p,
                    measured_world=[float(v) for v in measured[:6]],
                    rail_mm=float(arm.get_rail_position()[1]),
                    cam_to_world=(m.tolist() if m is not None else None),
                    intrinsics=frame.intrinsics.tolist()
                    if hasattr(frame.intrinsics, "tolist") else None,
                    depth_valid_frac=valid, files=os.path.basename(stem)))
                a = bench_aim(m) if m is not None else None
                print(f"  [{i:02d}] rail {p['rail']:4.0f}  depth {valid*100:4.1f}% valid"
                      + (f"  aims ({a[0]:.0f}, {a[1]:.0f})" if a else ""))
    finally:
        manifest.sort(key=lambda e: (e["pass_name"], e["index"]))
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n  manifest now describes {len(manifest)} stop(s) in {args.out}")
        try:
            arm.go_home(wait=True)
        except Exception as exc:                            # noqa: BLE001
            print(f"  could not park the arm: {exc}")
        arm.arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
