"""Calibrate the MaxArm's wrist D435 (eye-in-hand). See hardware/maxarm/hand_eye.py.

    python scripts/maxarm_hand_eye.py place                      # MOVES: cup 10 mm over the floor spot
    python scripts/maxarm_hand_eye.py collect --tag-size-mm 40   # MOVES: views the tag from a grid
    python scripts/maxarm_hand_eye.py verify --tag-size-mm 40    # MOVES: finds the tag, hovers over it

Run from xarm_lab_twin/ in xarm6sim, with the port's group (`sg dialout -c ...`
until a re-login). The tag (DICT_4X4_50 id 0, 40 mm) is in ~/Models/maxarm_calib/. Every connect reboots the board and its stock program
may home the arm first: keep hands clear when a step starts.

1. `place` leaves the cup hovering 10 mm above the table at (0, -200) and
   records where it really is. Slide the tag under it, centred on the cup, and
   tape it down. Its z is the measured table contact (TABLE_CONTACT_Z_MM).
2. `collect` views the tag from a grid of wrist positions and heights, solves
   cam_to_wrist with perception.extrinsic_calibration, and writes
   hardware/maxarm/hand_eye_calib.json (samples go to --out).
3. Move the tag anywhere else in reach. `verify` finds it with the camera,
   converts it with the calibration, and hovers the cup 10 mm over the centre
   it computed: judge the offset by eye.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from hardware.maxarm import hand_eye as he  # noqa: E402
from hardware.maxarm import kinematics as kin  # noqa: E402

PLACE_XY = (0.0, -200.0)
HOVER_MM = 10.0
DEFAULT_OUT = Path.home() / "Models" / "maxarm_calib"
# The camera does not look at the cup. Measured 2026-11-08 on the lab unit: with
# the wrist straight over the tag at z=210 the tag sits at (+52, +44) mm in the
# optical frame (picture right/down = arm +x/+y), and below z~190 it leaves the
# picture. So the grid is centred where the camera looks at the tag, and kept
# high enough for the colour camera's focus and the depth minimum (0.105 m).
GRID_CENTRE = (50.0, 40.0)
GRID_DX = (-30, -15, 0, 15, 30)
GRID_DY = (-20, 0, 20)
GRID_Z = (140, 160, 180, 200)
WARMUP_FRAMES = 60      # auto-exposure; the wrapper's warm-up left frames black in a dim room
VIEWS_PER_POSE = 3
MAX_REPROJ_PX = 1.0
VIEW_POSE = (50.0, -160.0, 200.0)      # verify: camera over (0, -200) on the table


def _arm():
    from hardware.maxarm.real_maxarm import RealMaxArmAPI
    return RealMaxArmAPI("/dev/ttyUSB0")


def _camera():
    from perception.realsense_camera import RealSenseWristCamera
    cam = RealSenseWristCamera(serial=None, verify_intrinsics=False)
    for _ in range(WARMUP_FRAMES):
        cam.capture()
    serial = cam.profile.get_device().get_info(__import__("pyrealsense2").camera_info.serial_number)
    return cam, serial


def _where(arm):
    return np.array(arm.get_position()[1][:3], dtype=float)


def _see_tag(cam, size_mm, n=VIEWS_PER_POSE):
    """Median tag centre (optical frame, metres) over n frames, or None."""
    from teleop_hand.hand_calib import detect_tag
    hits, frame = [], None
    for _ in range(n + 2):
        frame = cam.capture()
        obs = detect_tag(frame.color, frame.intrinsics.matrix, size_mm / 1000.0, frame.depth)
        if obs is not None and obs.reproj_px <= MAX_REPROJ_PX:
            hits.append(obs)
        if len(hits) == n:
            break
    if len(hits) < n:
        return None, frame
    pos = np.median([h.pos_cam_m for h in hits], axis=0)
    dch = [h.depth_check_mm for h in hits if h.depth_check_mm is not None]
    return {"pos_cam_m": pos.tolist(), "reproj_px": float(np.mean([h.reproj_px for h in hits])),
            "depth_check_mm": float(np.median(dch)) if dch else None,
            "uv": hits[-1].corners_px.mean(axis=0).tolist()}, frame


def cmd_place(a) -> int:
    arm = _arm()
    try:
        x, y = PLACE_XY
        arm.set_position(x, y, kin.TABLE_CONTACT_Z_MM + HOVER_MM, speed=30)
        p = _where(arm)
    finally:
        arm.disconnect()
    tag = [float(p[0]), float(p[1]), kin.TABLE_CONTACT_Z_MM + a.tag_thickness_mm]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "tag_place.json").write_text(json.dumps(
        {"tag_xyz_arm_mm": tag, "cup_readback_mm": p.tolist(),
         "time": dt.datetime.now().isoformat(timespec="seconds")}, indent=2))
    print(f"cup hovering {HOVER_MM:.0f} mm over the table; read-back {np.round(p, 1).tolist()}")
    print(f"tag centre will be taken as {np.round(tag, 1).tolist()} (arm frame, command z)")
    print("Slide the tag under the cup, centred, and tape it down. Then run `collect`.")
    return 0


def cmd_collect(a) -> int:
    tag = json.loads((a.out / "tag_place.json").read_text())["tag_xyz_arm_mm"]
    poses = []
    for i, z in enumerate(GRID_Z):
        for j, dy in enumerate(GRID_DY if i % 2 == 0 else GRID_DY[::-1]):
            for dx in (GRID_DX if j % 2 == 0 else GRID_DX[::-1]):
                p = (tag[0] + GRID_CENTRE[0] + dx, tag[1] + GRID_CENTRE[1] + dy, float(z))
                if not kin.check_pose(p):
                    poses.append(p)
    cam, serial = _camera()
    arm = _arm()
    samples = []
    try:
        for k, p in enumerate(poses, 1):
            arm.set_position(*p, speed=40)
            time.sleep(0.4)
            w = _where(arm)
            obs, frame = _see_tag(cam, a.tag_size_mm)
            if obs is None:
                print(f"  {k:2d}/{len(poses)} {p}: tag not seen")
                continue
            samples.append({"wrist_cmd_mm": list(p), "wrist_readback_mm": w.tolist(), **obs})
            dc = obs["depth_check_mm"]
            print(f"  {k:2d}/{len(poses)} {p}: tag at {np.round(np.array(obs['pos_cam_m']) * 1000, 1).tolist()} mm "
                  f"(reproj {obs['reproj_px']:.2f} px, depth check {'-' if dc is None else f'{dc:+.1f}'} mm)")
        arm.go_home()
    finally:
        arm.disconnect()
        cam.close()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (a.out / f"hand_eye_samples_{stamp}.json").write_text(json.dumps(
        {"tag_xyz_arm_mm": tag, "tag_size_mm": a.tag_size_mm, "camera_serial": serial,
         "samples": samples}, indent=2))
    if len(samples) < 6:
        print(f"only {len(samples)} views saw the tag; not solving")
        return 1
    res = he.solve(tag, [s["wrist_readback_mm"] for s in samples], [s["pos_cam_m"] for s in samples])
    print(f"\n  {res.n} views: rms {res.rms_mm:.2f} mm, max {res.max_mm:.2f} mm, "
          f"spread {res.spread_ratio:.3f} (min {he.MIN_SPREAD_RATIO})")
    print("  cam_to_wrist (rh frame, m):\n" + np.array2string(res.cam_to_base, precision=4))
    if res.spread_ratio < he.MIN_SPREAD_RATIO:
        print("  views too close to a plane; not saving")
        return 1
    he.save(res, {"date": dt.date.today().isoformat(), "camera_serial": serial,
                  "tag": f"DICT_4X4_50 id 0, {a.tag_size_mm} mm",
                  "tag_xyz_arm_mm": tag, "samples_file": str(a.out / f"hand_eye_samples_{stamp}.json"),
                  "table_contact_z_mm": kin.TABLE_CONTACT_Z_MM})
    print(f"  saved {he.CALIB_PATH}")
    return 0


def cmd_verify(a) -> int:
    calib = he.load()
    cam, serial = _camera()
    arm = _arm()
    try:
        arm.set_position(*VIEW_POSE, speed=40)
        time.sleep(0.4)
        w = _where(arm)
        obs, _ = _see_tag(cam, a.tag_size_mm)
        if obs is None:
            print("tag not seen from", VIEW_POSE)
            return 1
        p = he.camera_point_to_arm(obs["pos_cam_m"], w, calib.cam_to_wrist)
        print(f"tag centre per camera + calibration: {np.round(p, 1).tolist()} (arm frame, mm)")
        print(f"  (its z should be the table contact, {kin.TABLE_CONTACT_Z_MM:.0f}: "
              f"off by {p[2] - kin.TABLE_CONTACT_Z_MM:+.1f} mm)")
        target = (p[0], p[1], kin.TABLE_CONTACT_Z_MM + HOVER_MM)
        arm.set_position(*target, speed=30)
        print(f"cup now hovering {HOVER_MM:.0f} mm over {np.round(target[:2], 1).tolist()}: "
              f"how far is it from the tag's centre, and which way?")
    finally:
        arm.disconnect()
        cam.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["place", "collect", "verify"])
    ap.add_argument("--tag-size-mm", type=float, default=40.0)
    ap.add_argument("--tag-thickness-mm", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    return {"place": cmd_place, "collect": cmd_collect, "verify": cmd_verify}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
