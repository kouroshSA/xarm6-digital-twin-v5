"""MaxArm: find an object with the wrist camera, pick it with the cup, put it on the tag.

    python scripts/maxarm_pick_place.py look                   # MOVES to the view pose only
    python scripts/maxarm_pick_place.py run                    # MOVES: pick and place
    python scripts/maxarm_pick_place.py run --phrase "a red cube" --size-mm 40

Run from xarm_lab_twin/ in xarm6sim (`sg dialout -c ...` until a re-login).

Perception: MaxArmWristCamera frames (arm frame, metres); Grounding DINO (the
LanguageTargeter's grounder) boxes the phrase; inside the box the object is its
saturated pixels and its TOP FACE the brighter part of them (the side faces in
view are shaded). detect_tag finds the target. `look` saves the annotated
picture so the plan can be checked before anything goes down to the table.

Heights are geometry, not depth: the top of a 40 mm cube is ~135 mm from the
camera even from the highest view (wrist z 225), inside the D435's minimum range,
and only 8% of it returned depth (2026-11-09). So the top-face centre pixel is
put on the plane z = table contact + object height (wrist_camera.pixel_at_height),
and the cup reaches that top at the same commanded z. The object's measured
width on that plane is checked against --size-mm.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from hardware.maxarm import kinematics as kin  # noqa: E402

VIEW_POSE = (75.0, -210.0, 200.0)   # sees the tag and the cubes beside it
CARRY_Z = 150.0
APPROACH_MM = 30.0
PRESS_MM = 2.0          # past the computed top: the cup lip has to seal
PLACE_GAP_MM = 2.0      # release this far above resting height
SIZE_TOL_MM = 8.0
PLACE_TOL_MM = 15.0    # cube centre vs tag centre, judged from the after-look
MIN_SATURATION = 80
MIN_VALUE = 25
HUE_TOL = 12          # OpenCV hue units (0-179) around the box's dominant hue
OUT = Path(__file__).resolve().parents[1] / "hardware" / "maxarm" / "data"   # in the repo


def _connect(view):
    from hardware.maxarm.real_maxarm import RealMaxArmAPI
    from hardware.maxarm.wrist_camera import MaxArmWristCamera
    arm = RealMaxArmAPI("/dev/ttyUSB0")
    arm.go_home()
    arm.set_position(*view, speed=40)
    time.sleep(0.4)
    return arm, MaxArmWristCamera(arm)


def _object_mask(frame, box):
    """The object's pixels inside a grounding box (bool mask), or None.

    Saturated pixels of the box's dominant hue, largest blob: the top face AND
    the side faces in view. Brightness is not used: light across the top face
    (bright to dark corner to corner) split it diagonally (2026-11-09).
    """
    import cv2
    h, w = frame.color.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_RGB2HSV)
    obj = np.zeros((h, w), bool)
    obj[y0:y1, x0:x1] = (hsv[y0:y1, x0:x1, 1] > MIN_SATURATION) & (hsv[y0:y1, x0:x1, 2] > MIN_VALUE)
    if obj.sum() < 200:
        return None
    # Only the box's dominant colour: a box edge overlapping something else
    # saturated (pink paper beside the green cube, 2026-11-09) otherwise joins
    # the object and drags the centre.
    hue = hsv[..., 0].astype(int)
    mode = int(np.argmax(np.bincount(hue[obj], minlength=180)))
    dh = np.abs(hue - mode)
    obj &= np.minimum(dh, 180 - dh) <= HUE_TOL
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
    if n < 2:
        return None
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _top_square(frame, mask, z_top, size_mm):
    """(centre arm mm, fitted dims mm, angle deg) of the top face, from the whole object.

    Every object pixel is put on the top plane. Top-face pixels land on the
    true size x size square; side-face pixels sit lower, so on the top plane
    they land between the object and the point under the camera. So fit a
    rectangle, and on each axis longer than `size_mm` keep the `size_mm` at the
    end AWAY from the camera.
    """
    import cv2
    from hardware.maxarm.wrist_camera import pixel_at_height
    rows, cols = np.nonzero(mask)
    step = max(1, len(rows) // 3000)
    pts = np.array([pixel_at_height(frame, c, r, z_top)[:2]
                    for r, c in zip(rows[::step], cols[::step])], dtype=np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    centre = np.array([cx, cy], dtype=np.float64)
    nadir = np.asarray(frame.cam_to_world)[:2, 3] * 1000.0
    rad = np.radians(ang)
    for d, axis in ((w, np.array([np.cos(rad), np.sin(rad)])),
                    (h, np.array([-np.sin(rad), np.cos(rad)]))):
        if d > size_mm:
            away = np.sign(np.dot(centre - nadir, axis)) or 1.0
            centre += away * (d - size_mm) / 2.0 * axis
    return np.array([centre[0], centre[1], z_top]), np.array([w, h]), ang


def _look(cam, a, targeter=None):
    """(object dict or None, tag arm-mm or None, frame, tag observation)."""
    from perception.language import LanguageTargeter
    from teleop_hand.hand_calib import detect_tag
    targeter = targeter or LanguageTargeter()
    frame = cam.capture()
    dets = [d for d in targeter.grounder.detect(frame.color, [a.phrase, *a.distractors])
            if d.phrase == a.phrase]
    obj = None
    if dets:
        det = max(dets, key=lambda d: d.score)
        mask = _object_mask(frame, det.box)
        if mask is not None:
            z_top = kin.TABLE_CONTACT_Z_MM + a.size_mm
            centre, dims, ang = _top_square(frame, mask, z_top, a.size_mm)
            obj = {"xyz": centre, "score": det.score, "box": det.box, "mask": mask,
                   "px": frame.project_point(np.linalg.inv(frame.cam_to_world) @ np.append(centre / 1000.0, 1.0)),
                   "size_mm": dims, "angle_deg": ang,
                   "rgb": np.median(frame.color[mask], axis=0).astype(int)}
    tag = None
    obs = detect_tag(frame.color, frame.intrinsics.matrix, a.tag_size_mm / 1000.0, frame.depth)
    if obs is not None:
        u, v = obs.corners_px.mean(axis=0)
        win = frame.depth[int(v) - 3:int(v) + 4, int(u) - 3:int(u) + 4]
        if np.isfinite(win).any():
            tag = frame.pixel_to_world(u, v, depth_m=float(np.nanmedian(win))) * 1000.0
    return obj, tag, frame, obs


def _report(obj, tag, a):
    if obj is None:
        print(f"'{a.phrase}' not found")
    else:
        print(f"'{a.phrase}': top centre {np.round(obj['xyz'], 1).tolist()} mm, score "
              f"{obj['score']:.2f}, seen extent {np.round(obj['size_mm'], 0).tolist()} mm, "
              f"colour {obj['rgb'].tolist()}")
    print("tag: " + ("not found" if tag is None else f"{np.round(tag, 1).tolist()} mm"))


def _save(frame, obj, obs, name):
    import cv2
    img = frame.color[:, :, ::-1].copy()
    if obj is not None:
        x0, y0, x1, y1 = (int(v) for v in obj["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
        img[obj["mask"]] = (0.6 * img[obj["mask"]] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
        cv2.circle(img, tuple(int(v) for v in obj["px"]), 5, (0, 0, 255), -1)
    if obs is not None:
        cv2.polylines(img, [obs.corners_px.astype(int)], True, (255, 0, 255), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), img)
    print(f"picture: {path}")


def _plan(obj, tag, a):
    """Wrist targets, or raise with the reason."""
    top_geom = kin.TABLE_CONTACT_Z_MM + a.size_mm
    # Top face plus the sides in view: at least the object's size, and the
    # sides add at most about its height again (camera ~60 mm off to one side).
    if np.any(obj["size_mm"] < a.size_mm - SIZE_TOL_MM) or \
            np.any(obj["size_mm"] > 2.0 * a.size_mm + SIZE_TOL_MM):
        raise SystemExit(f"refused: the object spans {np.round(obj['size_mm'], 0).tolist()} mm "
                         f"on the plane z {top_geom:.0f}; is it really a {a.size_mm:.0f} mm "
                         f"object standing on the table?")
    xyz = obj["xyz"]
    pick = np.array([xyz[0], xyz[1], top_geom - PRESS_MM])
    place = np.array([tag[0], tag[1], top_geom + PLACE_GAP_MM])
    print(f"pick at {np.round(pick, 1).tolist()}, place at {np.round(place, 1).tolist()}")
    for p in (pick, place, (*pick[:2], CARRY_Z), (*place[:2], CARRY_Z)):
        bad = kin.check_pose(tuple(p))
        if bad:
            raise SystemExit(f"refused: {np.round(p, 1).tolist()} " + "; ".join(bad))
    return pick, place


def cmd_look(a) -> int:
    arm, cam = _connect(a.view)
    try:
        obj, tag, frame, obs = _look(cam, a)
        _report(obj, tag, a)
        _save(frame, obj, obs, "pick_place_look.png")
        if obj is not None and tag is not None:
            _plan(obj, tag, a)
    finally:
        arm.disconnect()
        cam.close()
    return 0


def cmd_run(a) -> int:
    from perception.language import LanguageTargeter
    targeter = LanguageTargeter()
    arm, cam = _connect(a.view)
    try:
        obj, tag, frame, obs = _look(cam, a, targeter)
        _report(obj, tag, a)
        _save(frame, obj, obs, "pick_place_before.png")
        if obj is None or tag is None:
            return 1
        pick, place = _plan(obj, tag, a)

        arm.set_position(pick[0], pick[1], CARRY_Z, speed=40)
        arm.set_position(pick[0], pick[1], pick[2] + APPROACH_MM, speed=30)
        arm.set_position(*pick, speed=10)
        arm.close_gripper()
        time.sleep(0.8)
        arm.set_position(pick[0], pick[1], CARRY_Z, speed=20)
        arm.set_position(place[0], place[1], CARRY_Z, speed=30)
        arm.set_position(*place, speed=10)
        arm.open_gripper()
        time.sleep(0.8)
        arm.set_position(place[0], place[1], CARRY_Z, speed=20)

        # Judge the place by looking, not by the commands (no vacuum sensor).
        arm.set_position(*a.view, speed=40)
        time.sleep(0.4)
        obj2, _, frame2, obs2 = _look(cam, a, targeter)
        _save(frame2, obj2, obs2, "pick_place_after.png")
        arm.go_home()
        if obj2 is None:
            print(f"after: '{a.phrase}' not seen; place NOT confirmed")
            return 1
        d = obj2["xyz"][:2] - tag[:2]
        moved = np.linalg.norm(obj2["xyz"][:2] - obj["xyz"][:2])
        print(f"after: '{a.phrase}' top centre {np.round(obj2['xyz'][:2], 1).tolist()} mm, "
              f"{np.linalg.norm(d):.1f} mm from where the tag was ({np.round(d, 1).tolist()}); "
              f"moved {moved:.1f} mm")
        # The commands all returning is not a place (no vacuum sensor): the
        # cube that never left the table printed a distance and exited 0.
        if np.linalg.norm(d) > PLACE_TOL_MM:
            print(f"FAILED: not on the tag (tolerance {PLACE_TOL_MM:.0f} mm)"
                  + ("; it never left the pick spot" if moved < PLACE_TOL_MM else ""))
            return 1
        print("placed")
    finally:
        # The pump stays as it is: if a move failed while holding, switching it
        # off drops the object from carry height (it did, 2026-11-09). The
        # place releases it on the success path; disconnect warns otherwise.
        arm.disconnect()
        cam.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["look", "run"])
    ap.add_argument("--phrase", default="a green cube")
    ap.add_argument("--distractors", nargs="*", default=["a red cube", "a paper marker"])
    ap.add_argument("--size-mm", type=float, default=40.0, help="object height (and width)")
    ap.add_argument("--tag-size-mm", type=float, default=40.0)
    ap.add_argument("--view", type=float, nargs=3, default=VIEW_POSE,
                    help="wrist pose to look from (arm mm); the camera looks ~(-50, -40) mm from it")
    a = ap.parse_args()
    return {"look": cmd_look, "run": cmd_run}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
