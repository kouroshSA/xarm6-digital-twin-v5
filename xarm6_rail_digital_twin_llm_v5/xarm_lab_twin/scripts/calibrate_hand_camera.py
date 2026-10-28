#!/usr/bin/env python3
"""Calibrate the hand-tracking camera's pose in the robot WORLD frame.

    python scripts/calibrate_hand_camera.py --ip 192.168.1.229 --tag-size-mm 60 \
        --tag-offset-mm 0 0 40
    python scripts/calibrate_hand_camera.py --fake          # rehearsal, no hardware

Run in the `xarm6sim_hand` env (OpenCV with aruco, pyrealsense2, mujoco).

SETUP
  1. Print an ArUco tag, dictionary DICT_4X4_50, id 0
     (`python scripts/calibrate_hand_camera.py --print-tag tag.png`), on a flat
     rigid plate. MEASURE THE PRINTED BLACK-SQUARE SIDE with calipers -- a wrong
     --tag-size-mm scales every camera-frame point and the depth cross-check
     is what flags it.
  2. Fix the plate to the gripper, facing the camera. --tag-offset-mm is the
     tag CENTRE's offset from the tool point, in the TOOL frame (mm).
  3. Mount the tracker camera where it will live. Calibration is only valid
     for that mounting: bump the camera and redo it.

PROCEDURE
  Move the arm by any means you already trust (UFACTORY Studio jog, teach
  mode). THIS SCRIPT NEVER COMMANDS THE ARM: it only reads joints and rail.
  For each pose press Enter. Collect 8-12 poses, spread over the volume the
  operator's hand will use -- vary HEIGHT and DISTANCE, not just left/right
  (a sweep at one height is a plane, and the solver refuses planes).
    Enter  capture      u  undo last      s  solve + save      q  quit

OUTPUT
  teleop_hand/camera_pose.json (or --out), loaded by
  `run_handteleop_*.py --cam-extrinsic FILE`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from teleop_hand import hand_calib as hc  # noqa: E402
from teleop_hand.camera_pose import HandCameraPose  # noqa: E402

SCENE = "envs/lab_scene.xml"
FRAMES = 15                     # frames medianed per capture
MAX_SCATTER_MM = 3.0            # tag must be still: per-axis spread over FRAMES
MAX_REPROJ_PX = 1.5
MAX_DEPTH_DISAGREE_MM = 25.0


def print_tag(path: str) -> int:
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, hc.DEFAULT_DICT))
    img = cv2.aruco.generateImageMarker(d, hc.DEFAULT_TAG_ID, 800)
    cv2.imwrite(path, cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255))
    print(f"wrote {path}  (DICT_4X4_50 id {hc.DEFAULT_TAG_ID}). Print at 100% scale, measure the "
          "black square, pass it as --tag-size-mm.")
    return 0


class Rig:
    """The arm's world-frame tool pose (real joints through the twin's FK) and
    the camera. Read-only on the arm."""

    def __init__(self, backend, arm, source):
        from teleop_kin.real_teach import RealMirror
        self.backend, self.arm, self.source = backend, arm, source
        self.mirror = RealMirror(arm, backend, use_rail=bool(getattr(backend, "_rail_trustworthy", False)))
        if not self.mirror.use_rail:
            print("[calib] rail not homed/readable: the world frame assumes the rail at the TWIN's "
                  "current position. Fix that or park the rail where the twin is.")

    def tool_pose(self):
        self.mirror._rail_read_t = 0.0
        self.mirror.sync()
        a = self.arm
        with a.lock:
            return (a.data.site_xpos[a.ee_site].copy(),
                    a.data.site_xmat[a.ee_site].reshape(3, 3).copy())

    def capture(self, args):
        """Median of FRAMES tag detections -> (cam_m, world_m, notes) or raises."""
        obs = []
        for _ in range(FRAMES * 3):
            rgb, depth, k = self.source.read()
            o = hc.detect_tag(rgb, k, args.tag_size_mm / 1000.0, depth)
            if o is not None:
                obs.append(o)
            if len(obs) >= FRAMES:
                break
        if len(obs) < FRAMES // 2:
            raise RuntimeError(f"tag seen in only {len(obs)} frames -- is it facing the camera, "
                               "lit, and fully in view?")
        pts = np.array([o.pos_cam_m for o in obs])
        scatter = float(np.max(np.std(pts, axis=0)) * 1000.0)
        reproj = float(np.median([o.reproj_px for o in obs]))
        dchk = [o.depth_check_mm for o in obs if o.depth_check_mm is not None]
        dmed = float(np.median(dchk)) if dchk else None
        if scatter > MAX_SCATTER_MM:
            raise RuntimeError(f"tag moving/noisy (spread {scatter:.1f} mm > {MAX_SCATTER_MM}) -- "
                               "let the arm settle and retry")
        if reproj > MAX_REPROJ_PX:
            raise RuntimeError(f"tag corners fit badly ({reproj:.2f} px) -- glare, blur or a "
                               "warped plate")
        if dmed is not None and abs(dmed) > MAX_DEPTH_DISAGREE_MM:
            raise RuntimeError(f"PnP depth and sensor depth disagree by {dmed:+.0f} mm -- the "
                               "--tag-size-mm is probably wrong")
        pos, rot = self.tool_pose()
        world = hc.tag_world_m(pos, rot, args.tag_offset_mm)
        note = f"spread {scatter:.1f} mm, reproj {reproj:.2f} px" + (
            f", depth check {dmed:+.0f} mm" if dmed is not None else "")
        return np.median(pts, axis=0), world, note


def finish(samples, out: str) -> int:
    try:
        pose, report = hc.solve(samples)
    except ValueError as e:
        print(e)
        return 1
    print(report)
    print("\n" + hc.describe_pose(pose))
    pose.save(out)
    print(f"saved {out}   (rms {pose.rms_mm:.1f} mm)")
    if pose.rms_mm > 8.0:
        print("WARNING: rms above 8 mm -- check --tag-size-mm / --tag-offset-mm, then redo.")
    return 0


def build(args):
    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(scene_xml=SCENE, render=False)
    if args.fake:
        from teleop_kin.real_teach import FakeBackend
        return FakeBackend(), arm, None
    from hardware.real_arm import RealXArmAPI
    from teleop_hand.tracker import RealSenseSource
    be = RealXArmAPI(args.ip, effector="none", ft_sensor=False)
    return be, arm, RealSenseSource(serial=args.serial)


def run_fake(args) -> int:
    """Scripted rehearsal: a hidden camera pose, joints stepped through a
    spread of poses, the same code path as a real session."""
    be, arm, _ = build(args)
    truth = HandCameraPose.looking_at(
        (-1100.0, -100.0, 1300.0), (0.0, -100.0, 950.0))   # east of the rail, looking west
    k = np.array([[610.0, 0, 320], [0, 610.0, 240], [0, 0, 1]])
    fake_cam = type("S", (), {})()
    rig = Rig(be, arm, fake_cam)
    tcp = np.array(args.tag_offset_mm, float)

    def read():
        pos, rot = rig.tool_pose()
        c = hc.tag_world_m(pos, rot, tcp)
        toward = truth.cam_to_world[:3, 3] - c
        return (*hc.render_tag(truth, c, toward, k, args.tag_size_mm / 1000.0), k)

    fake_cam.read = read
    rng = np.random.default_rng(1)
    samples = hc.SampleSet()
    base = np.array([0, -30, -25, 0, 55, 0], float)
    for i in range(40):
        be.joints_deg = base + rng.uniform([-60, -25, -25, -20, -25, -20], [60, 25, 25, 20, 25, 20])
        be.rail_mm = float(rng.uniform(50, 650))
        try:
            cam, world, note = rig.capture(args)
        except RuntimeError as e:
            print(f"  skip: {e}")
            continue
        ok, why = samples.try_add(cam, world)
        if ok:
            print(f"  pose {len(samples)}: {note}")
        if len(samples) >= 10:
            break
    rc = finish(samples, args.out)
    if rc == 0:
        got = HandCameraPose.load(args.out)
        err = np.linalg.norm(got.cam_to_world[:3, 3] - truth.cam_to_world[:3, 3]) * 1000
        print(f"[fake] recovered camera position error vs hidden truth: {err:.1f} mm")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.229")
    ap.add_argument("--serial", default=None, help="tracker D435 serial (default: first found)")
    ap.add_argument("--tag-size-mm", type=float, default=60.0, help="MEASURED black-square side")
    ap.add_argument("--tag-offset-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("X", "Y", "Z"), help="tag centre from the tool point, in the tool frame")
    ap.add_argument("--out", default="teleop_hand/camera_pose.json")
    ap.add_argument("--fake", action="store_true", help="rehearse with a synthetic camera")
    ap.add_argument("--print-tag", metavar="PNG", help="write the tag image and exit")
    args = ap.parse_args()
    if args.print_tag:
        return print_tag(args.print_tag)
    if args.fake:
        return run_fake(args)

    be, arm, src = build(args)
    rig = Rig(be, arm, src)
    samples = hc.SampleSet()
    print(__doc__.split("PROCEDURE")[1].split("OUTPUT")[0])
    try:
        while True:
            cmd = input(f"[{len(samples)} poses, spread {samples.spread():.2f}] > ").strip().lower()
            if cmd == "q":
                return 0
            if cmd == "u":
                samples.undo()
            elif cmd == "s":
                return finish(samples, args.out)
            elif cmd == "":
                try:
                    cam, world, note = rig.capture(args)
                except RuntimeError as e:
                    print("  not kept:", e)
                    continue
                ok, why = samples.try_add(cam, world)
                print(f"  {'kept' if ok else 'not kept: ' + why}  ({note})")
    finally:
        src.close()
        be.disconnect()


if __name__ == "__main__":
    sys.exit(main())
