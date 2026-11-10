"""MaxArm: lift a Petri-dish lid by its taped centre, set it aside, put it back.

    python scripts/maxarm_lid.py look                  # MOVES to the view pose only
    python scripts/maxarm_lid.py test-pick [--press-mm 6]   # MOVES: lift, check, set back down
    python scripts/maxarm_lid.py run                   # MOVES: lid off, aside, back on

Run from xarm_lab_twin/ in xarm6sim. The lid carries blue tape (a + sign) on
its UNDERSIDE, so the tape marks the lid without touching the cup's seal.

Perception: MaxArmWristCamera frames (arm frame, mm). The tape is the largest
blue blob in the picture; its pixels are put on the lid's top plane and their
mean is the pick point. The plane height comes from DEPTH on the tape when the
sensor returns it (the tape is opaque, the lid around it is clear), else from
--stack-top-mm (lid top above the table while the lid is on its base). Clear
plastic returns no depth and the lid top is near this D435's minimum range, so
the fallback is expected to be used some of the time; the height used is printed.

The lid goes back where it was picked, with the same commands, so the camera's
absolute error cancels there; only the arm's repeatability counts. The lid at
the set-aside spot is looked at again before it is picked back up (it can slide
on release). Success is judged by the after-look: the tape centre within
--back-tol-mm of where it started and at the same height (a lid perched on the
rim sits higher and off-centre), never by the commands returning (no vacuum
sensor).
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

VIEW_POSE = (75.0, -210.0, 200.0)   # the camera looks ~(-50, -40) mm from the wrist
CARRY_Z = 150.0
APPROACH_MM = 30.0
PRESS_MM = 4.0           # past the lid top: the cup lip has to seal (held at 4-5 below contact, 2026-11-09)
PLACE_GAP_MM = 2.0       # release this far above resting height
# Lid top, seated on its base: 61.0-61.6 measured (depth - bias) when it looked
# seated, 65.9 when perched on the rim. The put-back releases 2 mm above the
# seated height (user: never drop it), and anything higher afterwards is perched.
SEATED_TOP_Z_MM = 60.0   # 59.6 twice at the dish's current spot (x 40); 61.0-61.6 at x 0
RELEASE_PRESS_MM = 1.0   # the lid is vented IN CONTACT, cup lightly compressed (see _release_on)
OFFSET_MAX_MM = 15.0     # tape centre vs lid rim centre: more than this is a bad rim fit
LID_THICKNESS_MM = 1.0   # tape is on the underside of the lid top
# MEASURED 2026-11-09: depth put the tape at z 68.8 while the cup, pressing the
# lid hard (commanded 56.8), read back 60.3, so the lid top is near 61: depth on
# the tape through the lid reads ~8 mm too high this close to the D435's minimum
# range. Subtracted from every depth height. Contact-derived, so re-measure if
# the view height or the camera changes.
DEPTH_BIAS_MM = 8.0
ASIDE_DX_MM = -110.0     # "left of the robot" is -x (+x is the arm's right)
MIN_TAPE_PX = 150
MIN_DEPTH_PX = 40
BLUE_HUE = (95, 130)     # OpenCV hue units (0-179); set per lid with --hue
_tape_mask_hue_default = BLUE_HUE
MIN_SATURATION = 70
MIN_VALUE = 40
OUT = Path(__file__).resolve().parents[1] / "hardware" / "maxarm" / "data"


def _set_hue(lo, hi):
    _tape_mask.hue = (int(lo), int(hi))



def _connect(view, floor_z):
    from hardware.maxarm.real_maxarm import RealMaxArmAPI
    from hardware.maxarm.wrist_camera import MaxArmWristCamera
    arm = RealMaxArmAPI("/dev/ttyUSB0", floor_z_mm=floor_z)
    arm.go_home()
    arm.set_position(*view, speed=40)
    time.sleep(0.4)
    _capture.arm = arm
    return arm, MaxArmWristCamera(arm)


POSE_AGREE_MM = 5.0


def _capture(cam, arm, tries=3):
    """A frame whose pose agrees with a second servo read-back (and prints both).

    2026-11-09: two looks from the same wrist pose at the same scene put the lid
    120 mm apart; the frame's cam_to_world comes from one servo read at capture
    time, and a bad read gives a wrong world position with nothing flagging it.
    """
    from hardware.maxarm import hand_eye as he
    for i in range(tries):
        frame = cam.capture()
        _, pose = arm.get_position()
        if frame.cam_to_world is None:
            continue
        again = he.cam_to_arm(pose[:3], cam.hand_eye.cam_to_wrist)
        gap = float(np.linalg.norm((np.asarray(frame.cam_to_world)[:3, 3] - again[:3, 3]) * 1000.0))
        if gap <= POSE_AGREE_MM:
            print(f"frame from wrist {np.round(pose[:3], 1).tolist()}, camera at "
                  f"{np.round(np.asarray(frame.cam_to_world)[:3, 3] * 1000.0, 1).tolist()} mm")
            return frame
        print(f"WARNING: frame pose and a re-read disagree by {gap:.1f} mm; recapturing")
    raise SystemExit("refused: the wrist pose cannot be read consistently; not moving on a guess")


_capture.arm = None


def _tape_mask(frame, hue=None):
    """The largest blob of the tape's hue (bool mask) or None."""
    import cv2
    lo, hi = hue if hue is not None else _tape_mask.hue
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
         & (hsv[..., 1] > MIN_SATURATION) & (hsv[..., 2] > MIN_VALUE))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < MIN_TAPE_PX:
        return None
    return lab == i


BASE_RIM_ABOVE_TABLE_MM = 6.0   # the bare base's rim height, for putting its pixel on a plane
BASE_RADIUS_PX = (110, 200)     # 45 mm at 150-230 mm from the camera, D435 colour fx ~615


def _base_circle_px(color_rgb, radius_px=BASE_RADIUS_PX):
    """(u, v, r) of the strongest circle the size of a dish rim, or None (pixels)."""
    import cv2
    g = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=90, param2=40,
                         minRadius=radius_px[0], maxRadius=radius_px[1])
    if c is None:
        return None
    u, v, r = c[0][0]
    return float(u), float(v), float(r)


def _base_centre(frame, name=None, z_mm=None, what="dish base"):
    """A dish-sized rim's centre in arm mm (on the plane z_mm), or None; draws it on `name`."""
    import cv2
    from hardware.maxarm.wrist_camera import pixel_at_height
    hit = _base_circle_px(frame.color)
    if hit is None:
        print(f"{what}: no rim circle found")
        return None
    u, v, r = hit
    z = kin.TABLE_CONTACT_Z_MM + BASE_RIM_ABOVE_TABLE_MM if z_mm is None else z_mm
    centre = pixel_at_height(frame, u, v, z)
    edge = pixel_at_height(frame, u + r, v, z)
    dia = 2.0 * float(np.linalg.norm(edge[:2] - centre[:2]))
    print(f"{what}: centre {np.round(centre[:2], 1).tolist()} mm, rim diameter {dia:.0f} mm "
          f"(pixel ({u:.0f}, {v:.0f}) r {r:.0f})")
    if name is not None:
        img = frame.color[:, :, ::-1].copy()
        cv2.circle(img, (int(u), int(v)), int(r), (255, 0, 255), 2)
        cv2.circle(img, (int(u), int(v)), 5, (255, 0, 255), -1)
        cv2.imwrite(str(OUT / name), img)
    return centre[:2], dia


def _tape_height(frame, mask):
    """(z of the tape plane in arm mm from depth, number of depth pixels) or (None, n)."""
    rows, cols = np.nonzero(mask)
    d = frame.depth[rows, cols]
    ok = np.isfinite(d) & (d > 0.05)
    if ok.sum() < MIN_DEPTH_PX:
        return None, int(ok.sum())
    step = max(1, ok.sum() // 500)
    zs = [frame.pixel_to_world(c, r, depth_m=float(dd))[2] * 1000.0
          for r, c, dd in zip(rows[ok][::step], cols[ok][::step], d[ok][::step])]
    return float(np.median(zs)), int(ok.sum())


def _look(cam, a, default_top_mm, name):
    """The lid as seen now: dict with the tape centre on the lid top plane, or None."""
    import cv2
    from hardware.maxarm.wrist_camera import pixel_at_height
    frame = _capture(cam, _capture.arm)
    mask = _tape_mask(frame)
    img = frame.color[:, :, ::-1].copy()
    lid = None
    if mask is not None:
        z_tape, n_depth = _tape_height(frame, mask)
        if z_tape is not None and not (kin.TABLE_CONTACT_Z_MM + 3 <= z_tape <= kin.TABLE_CONTACT_Z_MM + 35):
            print(f"depth puts the tape at z {z_tape:.1f}, not a lid on this table; ignoring it")
            z_tape = None
        if z_tape is None:
            z_top = default_top_mm
            source = f"assumed ({n_depth} depth px on the tape)"
        else:
            z_top = z_tape + LID_THICKNESS_MM - a.depth_bias_mm
            source = f"depth {z_tape:.1f} - bias {a.depth_bias_mm:.0f}, {n_depth} px"
        rows, cols = np.nonzero(mask)
        step = max(1, len(rows) // 3000)
        pts = np.array([pixel_at_height(frame, c, r, z_top)[:2]
                        for r, c in zip(rows[::step], cols[::step])])
        centre = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        lid = {"xyz": np.array([centre[0], centre[1], z_top]), "extent_mm": extent,
               "z_source": source, "px": (float(cols.mean()), float(rows.mean())),
               "n_px": int(mask.sum())}
        img[mask] = (0.5 * img[mask] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        cv2.circle(img, (int(lid["px"][0]), int(lid["px"][1])), 6, (0, 0, 255), -1)
        print(f"lid tape: centre {np.round(lid['xyz'], 1).tolist()} mm (top height from "
              f"{source}), tape spans {np.round(extent, 0).tolist()} mm, {lid['n_px']} px")
    else:
        print("no blue tape seen")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    cv2.imwrite(str(path), img)
    print(f"picture: {path}")
    _look.last_frame = frame
    return lid


def _gate(*poses):
    for p in poses:
        bad = kin.check_pose(tuple(p), floor_z_mm=None)
        if bad:
            raise SystemExit(f"refused: {np.round(p, 1).tolist()} " + "; ".join(bad))


def _aside_xy(lid_xy, a):
    """A spot to the robot's left of the dish, clear of it and reachable."""
    if a.aside is not None:
        x, y = a.aside
        if np.hypot(x - lid_xy[0], y - lid_xy[1]) < a.dish_mm + 15:
            raise SystemExit(f"refused: --aside {a.aside} overlaps the dish at {np.round(lid_xy, 1).tolist()}")
        bad = kin.check_pose((x, y, CARRY_Z), floor_z_mm=None)
        if bad:
            raise SystemExit(f"refused: --aside {a.aside}: " + "; ".join(bad))
        return np.array([x, y], dtype=float)
    for dy in (0.0, 20.0, 40.0, 60.0):          # pull in towards the base if needed
        x = lid_xy[0] + ASIDE_DX_MM
        y = lid_xy[1] + dy
        if np.hypot(x - lid_xy[0], y - lid_xy[1]) < a.dish_mm + 15:
            continue
        if not kin.check_pose((x, y, CARRY_Z), floor_z_mm=None):
            return np.array([x, y])
    raise SystemExit("refused: no reachable set-aside spot to the left of the dish")


def _pick(arm, xyz, z_floor, press_mm=PRESS_MM, hold_s=1.5):
    z = max(float(xyz[2]) - press_mm, z_floor)
    arm.set_position(xyz[0], xyz[1], CARRY_Z, speed=40)
    arm.set_position(xyz[0], xyz[1], z + APPROACH_MM, speed=30)
    arm.set_position(xyz[0], xyz[1], z, speed=10)
    _, pose = arm.get_position()
    print(f"cup at commanded z {z:.1f} (lid top {xyz[2]:.1f}, press {press_mm:.0f}); "
          f"read-back {np.round(pose[:3], 1).tolist()}")
    arm.close_gripper()
    time.sleep(hold_s)
    arm.set_position(xyz[0], xyz[1], CARRY_Z, speed=15)


def _place(arm, xy, z_rest_top):
    z = z_rest_top + PLACE_GAP_MM
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=30)
    arm.set_position(xy[0], xy[1], z + APPROACH_MM, speed=30)
    arm.set_position(xy[0], xy[1], z, speed=10)
    arm.open_gripper()
    time.sleep(0.5)
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=20)


def _view_over(xy):
    """A view pose whose camera footprint is centred near xy (camera ~(-50,-40) from the wrist)."""
    return (float(xy[0]) + 50.0, float(xy[1]) + 40.0, VIEW_POSE[2])


def _verify_lift(arm, cam, xyz, a, name):
    """True if the lid is no longer seen at xyz after the pick (held, not left behind).

    A lid left on the dish is seen where it was, at its height, with depth. A
    held lid hangs ~35 mm under the camera: it shows as a blue patch somewhere
    else in the frame, with no depth or a nonsense height (2026-11-09: 5341 px
    at 54 mm from the pick point, "z 48.8", below the table). So "still there"
    needs a blob near the pick point AND at the lid's height from depth; anything
    else is the lid on the cup.
    """
    arm.set_position(xyz[0] + 25.0, xyz[1] + 20.0, VIEW_POSE[2], speed=30)
    time.sleep(0.4)
    seen = _look(cam, a, xyz[2], name)
    if seen is None:
        print("lift check: tape not seen below; the lid came up")
        return True
    off = float(np.linalg.norm(seen["xyz"][:2] - xyz[:2]))
    from_depth = "depth" in seen["z_source"]
    dz = float(seen["xyz"][2] - xyz[2])
    if from_depth and off <= a.aside_tol_mm and abs(dz) <= 6.0 and seen["n_px"] >= 15000:
        print(f"lift check: the lid is still there ({off:.1f} mm from the pick point, "
              f"height {dz:+.1f}); nothing was lifted")
        return False
    print(f"lift check: blue seen {off:.0f} mm from the pick point, height {dz:+.0f}, "
          f"{seen['n_px']} px, {'depth' if from_depth else 'no depth'}: the lid is on the cup")
    return True


def _abort_holding(arm, xyz):
    """Set whatever may be on the cup down where it was picked, then release and home.

    Never release at height: the first run's misjudged lift check opened the
    valve at z 200 and dropped the lid 140 mm (2026-11-09).
    """
    _place(arm, xyz[:2], xyz[2])
    arm.go_home()


def cmd_test_pick(a) -> int:
    """Pick at the tape centre, check the lift by looking, set it straight back down."""
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    default_top = kin.TABLE_CONTACT_Z_MM + a.stack_top_mm
    arm, cam = _connect(a.view, floor)
    try:
        lid = _look(cam, a, default_top, "lid_test_before.png")
        if lid is None:
            return 1
        _gate(lid["xyz"])
        _pick(arm, lid["xyz"], floor, a.press_mm, a.hold_s)
        held = _verify_lift(arm, cam, lid["xyz"], a, "lid_test_lift.png")
        _place(arm, lid["xyz"][:2], lid["xyz"][2])
        arm.set_position(*a.view, speed=40)
        time.sleep(0.4)
        _look(cam, a, default_top, "lid_test_after.png")
        arm.go_home()
        print("HELD" if held else "NOT HELD")
        return 0 if held else 1
    finally:
        arm.disconnect()
        cam.close()


def _hanging(cam, a, name):
    """True if the lid is seen hanging on the cup (near blue patch, no table-height depth)."""
    seen = _look(cam, a, SEATED_TOP_Z_MM, name)
    if seen is None:
        return False
    return not ("depth" in seen["z_source"] and seen["n_px"] >= 15000
                and abs(seen["xyz"][2] - SEATED_TOP_Z_MM) <= 8.0)


def _release_on(arm, cam, a, xy, z_top):
    """Lower the held lid onto z_top at xy, vent until it lets go; True when it did.

    2026-11-09: vented 2 mm ABOVE the base the lid stayed on the cup through three
    vents (residual vacuum holds a 10 g lid; a sideways nudge only shoved the
    base). Vented while RESTING on the table at the set-aside spot it came off
    first time, every time. So the lid is lowered into contact, the cup lightly
    compressed, and vented there: as the vacuum goes the cup springs back off
    the lid instead of carrying it. Pump stays off between tries.
    """
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=30)
    arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    for attempt in range(3):
        z = max(z_top - RELEASE_PRESS_MM - attempt, arm.floor_z_mm)
        arm.set_position(xy[0], xy[1], z, speed=10)
        arm.open_gripper()
        time.sleep(0.5)
        arm.set_position(xy[0], xy[1], z_top + 20.0, speed=10)
        arm.set_position(xy[0] + 25.0, xy[1] + 20.0, VIEW_POSE[2], speed=30)
        time.sleep(0.4)
        if not _hanging(cam, a, f"lid_release_{attempt}.png"):
            print(f"released after {attempt + 1} vent(s), lid set down at z {z:.1f}")
            return True
        print(f"vent {attempt + 1} at z {z:.1f}: the lid still hangs on the cup")
        arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=15)
    return False


def cmd_release(a) -> int:
    """If the lid hangs on the cup, set it on the dish base and free it."""
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    arm, cam = _connect(a.view, floor)
    try:
        if not _hanging(cam, a, "lid_release_look.png"):
            print("nothing hangs on the cup")
            arm.go_home()
            return 0
        base = _base_centre(_look.last_frame, "lid_release_base.png")
        if base is None:
            print("the lid hangs on the cup but no base rim is seen; not releasing blind")
            arm.go_home()
            return 1
        ok = _release_on(arm, cam, a, base[0], SEATED_TOP_Z_MM)
        arm.set_position(*a.view, speed=30)
        time.sleep(0.4)
        _look(cam, a, SEATED_TOP_Z_MM, "lid_release_after.png")
        arm.go_home()
        print("RELEASED" if ok else "STILL HANGING")
        return 0 if ok else 1
    finally:
        arm.disconnect()
        cam.close()


SEAT_TOL_MM = 2.5        # lid centre vs rim centre, same frame
SEAT_HEIGHT_TOL_MM = 1.5  # lid top above the seated height: more is perched on the rim


def _seated(cam, a, name):
    """(ok, lid_centre, rim_centre, above) judged in one frame from the dish view."""
    back = _look(cam, a, SEATED_TOP_Z_MM, name)
    if back is None:
        return False, None, None, None
    rim = _base_centre(_look.last_frame, name.replace(".png", "_rim.png"), what="dish rim")
    if rim is None or abs(rim[1] - a.dish_mm) > 10:
        return False, back["xyz"][:2], None, None
    above = float(back["xyz"][2] - a.seated_z)
    d = back["xyz"][:2] - a.offset - rim[0]
    print(f"seat check: lid centre {np.linalg.norm(d):.1f} mm from the rim centre "
          f"({np.round(d, 1).tolist()}), lid top {above:+.1f} mm vs seated")
    ok = np.linalg.norm(d) <= SEAT_TOL_MM and above <= SEAT_HEIGHT_TOL_MM
    return ok, back["xyz"], rim[0], above


def _reseat(arm, cam, a, lid_xyz, rim_xy, tries=2):
    """Lift a perched lid 20 mm and set it down again, corrected by the measured miss."""
    for i in range(tries):
        d = lid_xyz[:2] - a.offset - rim_xy
        target = lid_xyz[:2] - d          # move the cup by the miss
        print(f"re-seat {i + 1}: lid is {np.round(d, 1).tolist()} mm off; shifting by "
              f"{np.round(-d, 1).tolist()}")
        z = max(float(lid_xyz[2]) - a.press_mm, arm.floor_z_mm)
        arm.set_position(lid_xyz[0], lid_xyz[1], z + APPROACH_MM, speed=30)
        arm.set_position(lid_xyz[0], lid_xyz[1], z, speed=10)
        arm.close_gripper()
        time.sleep(a.hold_s)
        arm.set_position(lid_xyz[0], lid_xyz[1], a.seated_z + 20.0, speed=10)
        arm.set_position(target[0], target[1], a.seated_z + 20.0, speed=10)
        if not _release_on(arm, cam, a, target, a.seated_z):
            return False
        arm.set_position(*a.dish_view, speed=30)
        time.sleep(0.4)
        ok, lid_xyz, rim_xy, above = _seated(cam, a, f"lid_reseat_{i + 1}.png")
        if ok:
            return True
        if lid_xyz is None or rim_xy is None:
            return False
    return False


def cmd_reseat(a) -> int:
    """Judge the lid on the dish from the dish view; if perched, lift and re-seat it."""
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    a.offset = np.zeros(2)        # no aside measurement here: tape centre taken as lid centre
    a.seated_z = SEATED_TOP_Z_MM
    arm, cam = _connect(a.dish_view, floor)
    try:
        ok, lid_xyz, rim_xy, above = _seated(cam, a, "lid_reseat_0.png")
        if ok:
            print("SEATED")
            arm.go_home()
            return 0
        if lid_xyz is None or rim_xy is None:
            print("cannot judge: lid or rim not seen")
            arm.go_home()
            return 1
        ok = _reseat(arm, cam, a, lid_xyz, rim_xy)
        arm.go_home()
        print("SEATED" if ok else "NOT SEATED")
        return 0 if ok else 1
    finally:
        arm.disconnect()
        cam.close()


def cmd_survey(a) -> int:
    """Look from several wrist poses; report tape and rim seen from each. Moves only the view."""
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    views = [tuple(a.view)] + [tuple(v) for v in (a.extra_views or [])]
    arm, cam = _connect(views[0], floor)
    try:
        for i, v in enumerate(views):
            arm.set_position(*v, speed=40)
            time.sleep(0.4)
            print(f"-- view {i}: wrist {v}")
            _look(cam, a, SEATED_TOP_Z_MM, f"lid_survey_{i}.png")
            _base_centre(_look.last_frame, f"lid_survey_{i}_rim.png", what="rim")
        arm.go_home()
    finally:
        arm.disconnect()
        cam.close()
    return 0


def cmd_look(a) -> int:
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    arm, cam = _connect(a.view, floor)
    try:
        lid = _look(cam, a, kin.TABLE_CONTACT_Z_MM + a.stack_top_mm, "lid_look.png")
        if lid is not None:
            aside = _aside_xy(lid["xyz"][:2], a)
            z_aside_top = kin.TABLE_CONTACT_Z_MM + a.lid_height_mm
            print(f"plan: pick at {np.round(lid['xyz'], 1).tolist()} (cup to z "
                  f"{lid['xyz'][2] - PRESS_MM:.1f}), set aside at {np.round(aside, 1).tolist()} "
                  f"releasing at z {z_aside_top + PLACE_GAP_MM:.1f}, back to the same spot")
            _gate(lid["xyz"], (*aside, z_aside_top), (*aside, CARRY_Z), _view_over(aside))
            arm.set_position(*_view_over(aside), speed=40)
            time.sleep(0.4)
            _look(cam, a, z_aside_top, "lid_look_aside.png")
            arm.go_home()
    finally:
        arm.disconnect()
        cam.close()
    return 0


def cmd_run(a) -> int:
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    default_top = kin.TABLE_CONTACT_Z_MM + a.stack_top_mm
    arm, cam = _connect(a.view, floor)
    try:
        lid = _look(cam, a, default_top, "lid_before.png")
        if lid is None:
            return 1
        home_xyz = lid["xyz"].copy()
        aside = _aside_xy(home_xyz[:2], a)
        z_aside_top = kin.TABLE_CONTACT_Z_MM + a.lid_height_mm
        view_aside = _view_over(aside)
        _gate(home_xyz, (*aside, z_aside_top), (*aside, CARRY_Z), view_aside)

        # 1. lid off; make sure it came up before carrying nothing to the left
        _pick(arm, home_xyz, floor, a.press_mm, a.hold_s)
        if not _verify_lift(arm, cam, home_xyz, a, "lid_lift.png"):
            _abort_holding(arm, home_xyz)
            print("FAILED: the lid was not lifted")
            return 1
        # The bare base, where it is NOW (the camera wire can drag it): the first
        # run put the lid back where the lid had been and found the base moved.
        # Measured from the SAME view pose as the pick, so the calibration error
        # is the same for both and cancels; positions from two poses differed by
        # up to 8 mm (2026-11-09).
        arm.set_position(*a.dish_view, speed=30)
        time.sleep(0.4)
        frame_base = _capture(cam, arm)
        _look.last_frame = frame_base
        base = _base_centre(frame_base, "lid_base.png")
        if base is None or abs(base[1] - a.dish_mm) > 10:
            print("WARNING: no credible base rim after the lift; putting the lid back where it was")
            base_xy = home_xyz[:2].copy()
        else:
            base_xy = base[0]
        if not _release_on(arm, cam, a, aside, z_aside_top):
            arm.go_home()
            print("FAILED: the lid would not leave the cup at the set-aside spot; parked at home, still held")
            return 1

        # 2. did it get there? look at the set-aside spot
        arm.set_position(*view_aside, speed=40)
        time.sleep(0.4)
        lid_aside = _look(cam, a, z_aside_top, "lid_aside.png")
        if lid_aside is None or np.linalg.norm(lid_aside["xyz"][:2] - aside) > a.aside_tol_mm:
            where = "not seen" if lid_aside is None else \
                f"{np.linalg.norm(lid_aside['xyz'][:2] - aside):.1f} mm from the set-aside spot"
            arm.set_position(*a.view, speed=40)
            time.sleep(0.4)
            still = _look(cam, a, default_top, "lid_aside_check.png")
            arm.go_home()
            print(f"FAILED: after the set-aside the lid is {where}"
                  + ("; it is still on the dish" if still is not None and
                     np.linalg.norm(still["xyz"][:2] - home_xyz[:2]) < a.back_tol_mm else ""))
            return 1
        print(f"set aside: lid at {np.round(lid_aside['xyz'], 1).tolist()}")
        # Where the cup takes the lid (the tape centre) vs the lid's own centre
        # (its rim), measured in one frame with the lid alone on the table.
        rim = _base_centre(_look.last_frame, "lid_aside_rim.png", z_mm=kin.TABLE_CONTACT_Z_MM,
                           what="lid rim")
        offset = np.zeros(2)
        if rim is not None and abs(rim[1] - a.dish_mm) <= 10 and \
                np.linalg.norm(lid_aside["xyz"][:2] - rim[0]) <= OFFSET_MAX_MM:
            offset = lid_aside["xyz"][:2] - rim[0]
            print(f"tape centre is {np.round(offset, 1).tolist()} mm from the lid centre; "
                  f"the put-back corrects for it")
        else:
            print("WARNING: lid rim not measured; taking the tape centre as the lid centre")
        time.sleep(a.pause_s)

        # 3. pick it up again where it actually is, put it back on the dish
        _pick(arm, lid_aside["xyz"], floor, a.press_mm, a.hold_s)
        if not _verify_lift(arm, cam, lid_aside["xyz"], a, "lid_lift2.png"):
            _abort_holding(arm, lid_aside["xyz"])
            print("FAILED: the lid was not lifted from the set-aside spot")
            return 1
        target = base_xy + offset
        z_seated = min(float(home_xyz[2]), SEATED_TOP_Z_MM)
        a.seated_z = z_seated
        print(f"putting the lid back: base at {np.round(base_xy, 1).tolist()}, cup to "
              f"{np.round(target, 1).tolist()}, lowered to z {z_seated + PLACE_GAP_MM:.1f} "
              f"({PLACE_GAP_MM:.0f} mm above the seated top) before release")
        _gate((*target, z_seated))
        if not _release_on(arm, cam, a, target, z_seated):
            arm.go_home()
            print("FAILED: the lid would not leave the cup over the base; it is parked at home, still held")
            return 1

        # 4. judge by looking, in one frame; a perched lid is lifted and re-set
        a.offset = offset
        arm.set_position(*a.dish_view, speed=40)
        time.sleep(0.4)
        ok, lid_xyz, rim_xy, above = _seated(cam, a, "lid_after.png")
        if not ok and lid_xyz is not None and rim_xy is not None:
            ok = _reseat(arm, cam, a, lid_xyz, rim_xy)
        arm.go_home()
        if not ok:
            print("FAILED: lid not seated on the base")
            return 1
        print("lid back on the dish, seated")
    finally:
        arm.disconnect()
        cam.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["look", "survey", "test-pick", "release", "reseat", "run"])
    ap.add_argument("--extra-views", type=float, nargs=3, action="append", default=None,
                    help="survey: more wrist poses to look from (repeatable)")
    ap.add_argument("--hue", type=int, nargs=2, default=BLUE_HUE, metavar=("LO", "HI"),
                    help="OpenCV hue window (0-179) of this lid's tape; dish 1 light blue ~%d-%d" % BLUE_HUE)
    ap.add_argument("--press-mm", type=float, default=PRESS_MM, help="cup below the lid top at the pick")
    ap.add_argument("--depth-bias-mm", type=float, default=DEPTH_BIAS_MM,
                    help="subtracted from depth heights on the tape (measured against cup contact)")
    ap.add_argument("--hold-s", type=float, default=1.5, help="pump time before lifting")
    ap.add_argument("--aside", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="set-aside spot (arm mm); default: %.0f mm in x from the dish" % ASIDE_DX_MM)
    ap.add_argument("--stack-top-mm", type=float, default=16.0,
                    help="lid top above the table while on its base, used when depth gives nothing")
    ap.add_argument("--lid-height-mm", type=float, default=5.0,
                    help="lid top above the table when the lid stands alone; the set-aside release height")
    ap.add_argument("--dish-mm", type=float, default=90.0)
    ap.add_argument("--floor-margin-mm", type=float, default=3.0,
                    help="lowest cup z allowed, above table contact")
    ap.add_argument("--aside-tol-mm", type=float, default=25.0)
    ap.add_argument("--back-tol-mm", type=float, default=5.0)
    ap.add_argument("--pause-s", type=float, default=1.0)
    ap.add_argument("--view", type=float, nargs=3, default=VIEW_POSE,
                    help="wrist pose the lid is looked for from")
    ap.add_argument("--dish-view", type=float, nargs=3, default=None,
                    help="wrist pose the bare base and the end state are looked at from (default: --view)")
    a = ap.parse_args()
    if a.dish_view is None:
        a.dish_view = a.view
    _set_hue(*a.hue)
    return {"look": cmd_look, "survey": cmd_survey, "test-pick": cmd_test_pick, "release": cmd_release,
            "reseat": cmd_reseat, "run": cmd_run}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
