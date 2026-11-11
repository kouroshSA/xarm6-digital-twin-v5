"""MaxArm: episodes of lid off / set aside / lid back on, for every dish in view.

    python scripts/maxarm_lid_loop.py                     # MOVES: 3 episodes
    python scripts/maxarm_lid_loop.py --episodes 0        # until Ctrl-C
    python scripts/maxarm_lid_loop.py --reseat-tries 5 --pause-s 10

Run from xarm_lab_twin/ in xarm6sim. Builds on scripts/maxarm_lid.py (the
single-dish steps measured on the bench 2026-11-09); the perception here sees
EVERY tape blob in a frame, not the largest, so several dishes can share the
table.

Each episode is new. The arm surveys the table from the view pose (and over
every dish it knew last time, since the dishes may have been moved), takes each
lid that sits on a dish (tape at the seated height), and for each in turn:

  1. looks at it from the dish view and lifts it at its tape centre,
  2. sets it aside (released in contact, never dropped) and looks at it there,
  3. picks it up again and puts it back AT THE COORDINATE IT WAS PICKED FROM:
     the same cup position, so the camera's absolute error cancels and nothing
     is looked up for the put-back (the dish is remembered, not re-seen),
  4. judges the put-back by looking (tape centre vs rim centre, lid height). A
     lid perched on the rim is picked up again and set down shifted by the
     measured miss, up to --reseat-tries times. A lid the operator seats by
     hand meanwhile reads as seated at the next look.

Between episodes the arm is home for --pause-s so the dishes can be moved.
Pictures go to hardware/maxarm/data/loop/. Ctrl-C leaves the arm where it is
and the suction as it was: a held lid is never let go in the air.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from hardware.maxarm import kinematics as kin  # noqa: E402
import maxarm_lid as L  # noqa: E402
from maxarm_lid import (  # noqa: E402
    APPROACH_MM, ASIDE_DX_MM, CARRY_Z, DEPTH_BIAS_MM, LID_THICKNESS_MM, MIN_SATURATION,
    MIN_TAPE_PX, MIN_VALUE, OFFSET_MAX_MM, OUT, PRESS_MM, RELEASE_PRESS_MM, SEAT_HEIGHT_TOL_MM,
    SEAT_TOL_MM, SEATED_TOP_Z_MM, VIEW_POSE, _abort_holding, _capture, _connect, _gate, _pick,
    _set_hue, _tape_height, _view_over,
)

LOOP_OUT = OUT / "loop"
MERGE_MM = 45.0          # the same lid seen from two views, or re-seen from the dish view
SEATED_BAND_MM = 4.0     # tape top within this BELOW the seated height: a lid on its dish
PERCHED_MM = 9.0         # ... or up to this ABOVE it: a lid perched on the rim (64.7 seen) is
                         # still on its dish and gets picked and re-seated, not skipped
GLARE_MAX_Z_MM = kin.TABLE_CONTACT_Z_MM - 1.5   # a blob with depth whose top reads below this
                         # is a glint on the table (48.1-48.9 seen, seven times); a lid lying
                         # on the table reads 51.8-53.5 (six times). Not a lid, not an obstacle.
LID_PX_MIN = 6000        # a lid seen flat with depth. Seen obliquely at a set-aside spot near the
                         # reach limit a lid is ~10k px; at 15000 the lift check ignored one still
                         # lying at the pick point and the arm went on with an empty cup (2026-11-11)
ON_DISH_NODEPTH_PX = 15000   # a blob with NO depth counts as a lid on a dish only this big: a
                         # 6000 px depth-less phantom was "picked" and the empty cup was then
                         # judged to be holding a lid that would not let go (episode 3, 2026-11-11)
SIGHTING_PX_MIN = 3000   # smaller tape-coloured blobs are glints or edges: a 452 px speck with
                         # depth at the seated height became a third dish (2026-11-11), and every
                         # speck counted as an obstacle, so no set-aside spot was clear
RIM_NEAR_PX = 150        # a rim circle this close to the tape blob is that lid's rim
SURVEY_OFFSETS_MM = ((0.0, 0.0), (0.0, 70.0), (0.0, -70.0), (70.0, 0.0), (-70.0, 0.0),
                     (70.0, 70.0), (-70.0, 70.0), (110.0, 90.0), (-110.0, 90.0))   # the far right/left need a shorter reach
VIEW_DEDUPE_MM = 40.0
CHECK_VIEW_DXY = (25.0, 20.0)   # the lift and release checks look from here beside the spot


def _check_view(xy):
    """The wrist pose the lift and release checks look from (maxarm_lid._verify_lift)."""
    return (float(xy[0]) + CHECK_VIEW_DXY[0], float(xy[1]) + CHECK_VIEW_DXY[1], VIEW_POSE[2])


def _view_for_check(xy):
    """A REACHABLE pose to check xy from: the check view, else over it, else beside it.

    2026-11-11: a lid set aside slid 12 mm on release to a spot whose check view was
    one servo pulse out of range; the move was refused and the run stopped holding
    the lid at carry height. The looks are position-based, so any view whose
    footprint covers xy serves.
    """
    for v in (_check_view(xy), _view_over(xy),
              (float(xy[0]) - CHECK_VIEW_DXY[0], float(xy[1]) + CHECK_VIEW_DXY[1], VIEW_POSE[2]),
              (float(xy[0]), float(xy[1]) + 60.0, VIEW_POSE[2])):
        if not kin.check_pose(v, floor_z_mm=None):
            return v
    raise SystemExit(f"refused: no reachable view to check {np.round(xy, 1).tolist()} from")


# -- perception: every lid in a frame ------------------------------------------

def _blobs(frame):
    """Every blob of the tape's hue, largest first (bool masks)."""
    import cv2
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_RGB2HSV)
    m = L.hue_mask(hsv)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    return [lab == i for i in order if stats[i, cv2.CC_STAT_AREA] >= MIN_TAPE_PX]


def _tape_band(frame, mask):
    """(z_tape, n_depth, rows, cols) from depth: the MODAL 6 mm height band of the blob's
    depth pixels and the sampled pixels in it, or (None, n_depth, None, None).

    Glare on the table beside a tape blob passes the hue mask and merges with it; the
    median height of the merged blob was the table's, 12 mm under the lid, and a lid
    sitting on its dish was judged to be hanging on the cup, three times (2026-11-11).
    The tape is one plane, so its band wins the histogram; the glare's band is dropped."""
    rows, cols = np.nonzero(mask)
    d = frame.depth[rows, cols]
    ok = np.isfinite(d) & (d > 0.05)
    n_ok = int(ok.sum())
    if n_ok < L.MIN_DEPTH_PX:
        return None, n_ok, None, None
    step = max(1, n_ok // 1500)
    r, c, dd = rows[ok][::step], cols[ok][::step], d[ok][::step]
    zs = np.array([frame.pixel_to_world(int(cc), int(rr), depth_m=float(z))[2] * 1000.0
                   for rr, cc, z in zip(r, c, dd)])
    h, edges = np.histogram(zs, np.arange(zs.min() - 1.0, zs.max() + 3.0, 2.0))
    k = int(np.argmax(h))
    lo, hi = edges[max(k - 1, 0)], edges[min(k + 2, len(edges) - 1)]
    inb = (zs >= lo) & (zs <= hi)
    if inb.mean() < 0.999:
        print(f"  (depth band {lo:.0f}-{hi:.0f} mm holds {100 * inb.mean():.0f}% of the blob's depth "
              f"pixels; the rest, at other heights, is left out)")
    return float(np.median(zs[inb])), n_ok, r[inb], c[inb]


def _lid_from_mask(frame, mask, default_top, a):
    """The tape blob as a lid: centre on the lid-top plane, whether depth gave the height."""
    from hardware.maxarm.wrist_camera import pixel_at_height
    z_tape, n_depth, rows, cols = _tape_band(frame, mask)
    if z_tape is not None and not (kin.TABLE_CONTACT_Z_MM + 3 <= z_tape <= kin.TABLE_CONTACT_Z_MM + 35):
        z_tape = None            # not on this table: a lid hanging on the cup reads nonsense
    if z_tape is None:
        z_top, depth = float(default_top), False
        rows, cols = np.nonzero(mask)
    else:
        z_top, depth = z_tape + LID_THICKNESS_MM - a.depth_bias_mm, True
    step = max(1, len(rows) // 3000)
    pts = [pixel_at_height(frame, c, r, z_top) for r, c in zip(rows[::step], cols[::step])]
    pts = np.array([p[:2] for p in pts if p is not None])
    if len(pts) == 0:
        return None
    centre = pts.mean(axis=0)
    return {"xyz": np.array([centre[0], centre[1], z_top]), "depth": depth, "n_depth": int(n_depth),
            "n_px": int(mask.sum()), "px": (float(cols.mean()), float(rows.mean())), "mask": mask}


def _look_all(cam, a, default_top, name):
    """(frame, lids) seen now; every tape blob, largest first. Saves the picture."""
    import cv2
    frame = _capture(cam, _capture.arm)
    lids = [l for l in (_lid_from_mask(frame, m, default_top, a) for m in _blobs(frame)) if l is not None]
    img = frame.color[:, :, ::-1].copy()
    for i, l in enumerate(lids):
        img[l["mask"]] = (0.5 * img[l["mask"]] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        u, v = int(l["px"][0]), int(l["px"][1])
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.putText(img, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        src = f"depth ({l['n_depth']} px)" if l["depth"] else "assumed"
        print(f"  lid {i}: tape centre {np.round(l['xyz'], 1).tolist()} mm (height from {src}), "
              f"{l['n_px']} px")
    if not lids:
        print("  no tape seen")
    LOOP_OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(LOOP_OUT / name), img)
    return frame, lids


def _nearest(lids, xy, tol_mm):
    """The lid nearest xy within tol_mm, or None."""
    best = None
    for l in lids:
        d = float(np.linalg.norm(l["xyz"][:2] - np.asarray(xy, dtype=float)))
        if d <= tol_mm and (best is None or d < best[0]):
            best = (d, l)
    return None if best is None else best[1]


def _rim_near(frame, px, z_mm, a, name=None):
    """(centre_xy, diameter) of the dish-sized rim circle nearest pixel px, or None."""
    import cv2
    from hardware.maxarm.wrist_camera import pixel_at_height
    g = cv2.cvtColor(frame.color, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=90, param2=40,
                         minRadius=L.BASE_RADIUS_PX[0], maxRadius=L.BASE_RADIUS_PX[1])
    if c is None:
        print("  rim: no circle found")
        return None
    best = None
    for u, v, r in c[0]:
        centre = pixel_at_height(frame, float(u), float(v), z_mm)
        edge = pixel_at_height(frame, float(u) + float(r), float(v), z_mm)
        if centre is None or edge is None:
            continue
        dia = 2.0 * float(np.linalg.norm(edge[:2] - centre[:2]))
        if abs(dia - a.dish_mm) > 10:
            continue
        d = float(np.hypot(u - px[0], v - px[1]))
        if d <= RIM_NEAR_PX and (best is None or d < best[0]):
            best = (d, centre[:2].copy(), dia, (float(u), float(v), float(r)))
    if best is None:
        print(f"  rim: {len(c[0])} circle(s) but none dish-sized near the tape")
        return None
    _, centre, dia, (u, v, r) = best
    print(f"  rim: centre {np.round(centre, 1).tolist()} mm, diameter {dia:.0f} mm "
          f"(pixel ({u:.0f}, {v:.0f}) r {r:.0f})")
    if name is not None:
        img = frame.color[:, :, ::-1].copy()
        cv2.circle(img, (int(u), int(v)), int(r), (255, 0, 255), 2)
        cv2.circle(img, (int(u), int(v)), 5, (255, 0, 255), -1)
        cv2.imwrite(str(LOOP_OUT / name), img)
    return centre, dia


# -- checks that look --------------------------------------------------------

def _pick_remember(arm, xyz, floor, press_mm, hold_s):
    """maxarm_lid._pick, and the joints READ BACK with the cup sealed on the lid: the
    plate's coordinate in servo values, replayed for the first put-back (bench rule,
    2026-11-11). Returns (joints_deg[3], read-back xyz)."""
    z = max(float(xyz[2]) - press_mm, floor)
    arm.set_position(xyz[0], xyz[1], CARRY_Z, speed=40)
    arm.set_position(xyz[0], xyz[1], z + APPROACH_MM, speed=30)
    arm.set_position(xyz[0], xyz[1], z, speed=10)
    _, pose = arm.get_position()
    _, joints = arm.get_servo_angle()
    joints = [float(j) for j in joints[:3]]
    print(f"cup at commanded z {z:.1f} (lid top {xyz[2]:.1f}, press {press_mm:.0f}); "
          f"read-back {np.round(pose[:3], 1).tolist()}, servo pose {np.round(joints, 2).tolist()} deg")
    arm.close_gripper()
    time.sleep(hold_s)
    arm.set_position(xyz[0], xyz[1], CARRY_Z, speed=15)
    return joints, np.array(pose[:3], float)


def _lifted(arm, cam, a, xyz, name):
    """True if no lid lies at xyz any more after the pick (maxarm_lid._verify_lift, any blob)."""
    arm.set_position(*_view_for_check(xyz[:2]), speed=30)
    time.sleep(0.4)
    _, lids = _look_all(cam, a, xyz[2], name)
    for l in lids:
        off = float(np.linalg.norm(l["xyz"][:2] - xyz[:2]))
        dz = float(l["xyz"][2] - xyz[2])
        # A big tape blob at the pick point is the lid whether or not depth is valid: the
        # orange tape seen obliquely often has none, and a hanging lid is never in the
        # picture (the camera looks beside the cup). Depth, when there, must agree.
        if off <= a.aside_tol_mm and l["n_px"] >= LID_PX_MIN and (not l["depth"] or abs(dz) <= 6.0):
            print(f"lift check: the lid is still there ({off:.1f} mm from the pick point, "
                  f"height {dz:+.1f}{'' if l['depth'] else ' assumed'}); nothing was lifted")
            return False
    print("lift check: no lid lies at the pick point; it came up")
    return True


def _hanging(cam, a, xy, z_top, name):
    """True unless a lid is seen LYING at xy (depth at z_top, flat, big): maxarm_lid._hanging per blob."""
    _, lids = _look_all(cam, a, z_top, name)
    for l in lids:
        # as _lifted: a big blob where the lid was set down is the lid, depth or not (the
        # orange lid at its set-aside spot had no depth and was called hanging, 2026-11-11)
        if (l["n_px"] >= LID_PX_MIN and np.linalg.norm(l["xyz"][:2] - xy) <= a.aside_tol_mm
                and (not l["depth"] or abs(l["xyz"][2] - z_top) <= 8.0)):
            return False
    return True


#: Tape hue windows, OR-ed: blue at the default floors, orange windowed on brightness too
#: (the wooden table shares its hue but is dark; measured 2026-11-11, maxarm_lid.ORANGE_WINDOW).
TAPE_WINDOWS = (L.BLUE_HUE, L.ORANGE_WINDOW)

PEEL_MM = 6.0   # sideways drag of the cup lip across the lid at release, per attempt


def _peel(arm, xy, z, mm):
    """Drag the cup `mm` sideways at height z and back; +x first, -x if that is out of reach."""
    from hardware.maxarm.real_maxarm import MaxArmError
    for dx in (mm, -mm):
        try:
            arm.set_position(xy[0] + dx, xy[1], z, speed=10)
        except MaxArmError as exc:
            print(f"  peel {dx:+.0f} mm refused: {exc}")
            continue
        arm.set_position(xy[0], xy[1], z, speed=10)
        return dx
    return 0.0


def _release_on(arm, cam, a, xy, z_top, tag, peel=True):
    """Lower the held lid onto z_top at xy, vent with the valve HELD OPEN, peel the cup
    off sideways, rise, close the valve, look. Each retry presses 1 mm deeper and peels
    further. (The vendor's one-second vent closes the valve before the cup rises, and the
    cup's spring-back then re-grips a light lid: bench, 2026-11-11.)"""
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=30)
    arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    for attempt in range(3):
        # Press as the pick does (a.press_mm, 4 mm): at RELEASE_PRESS_MM (1 mm) over a dish the
        # cup, which reads ~3 mm high here, never put the lid on its base; it was vented
        # dangling and adhesion took it up again, three times (2026-11-11).
        z = max(z_top - a.press_mm - attempt, arm.floor_z_mm)
        arm.set_position(xy[0], xy[1], z, speed=10)
        arm.vent(True)
        time.sleep(0.6)
        # Peel only on the table. Over a dish the drag moved the lid along x by the peel's
        # length and left it perched on the rim (three put-backs, 2026-11-11); the valve
        # held open through the lift-off is what stops the sticking.
        peeled = _peel(arm, xy, z, PEEL_MM * (attempt + 1)) if peel else 0.0
        if not peel:
            _settle(arm, xy, z_top)       # over a dish: push a perched skirt into the seat
        arm.set_position(xy[0], xy[1], z_top + 20.0, speed=10)
        arm.vent(False)
        arm.set_position(*_view_for_check(xy), speed=30)
        time.sleep(0.4)
        if not _hanging(cam, a, xy, z_top, f"{tag}_release{attempt}.png"):
            print(f"released after {attempt + 1} vent(s) (peel {peeled:+.0f} mm), "
                  f"lid set down at z {z:.1f}")
            return True
        print(f"vent {attempt + 1} at z {z:.1f} (peel {peeled:+.0f} mm): the lid still hangs on the cup")
        arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=15)
    return False


SETTLE_RADII_MM = (3.0, 6.0, 9.0)
SETTLE_PRESS_MM = 2.0


def _settle(arm, xy, z_seated):
    """Vented, pressing SETTLE_PRESS_MM below the seated top, walk the cup round widening
    squares about xy and back to it: a lid perched on the rim (6-7 mm high, 5-9 mm off,
    every put-back on 2026-11-11) is pushed sideways until its skirt drops over the rim.
    The cup follows it down because the servo is commanded below the seat."""
    from hardware.maxarm.real_maxarm import MaxArmError
    z = max(z_seated - SETTLE_PRESS_MM, arm.floor_z_mm)
    arm.set_position(xy[0], xy[1], z, speed=10)
    done = 0
    for r in SETTLE_RADII_MM:
        for dx, dy in ((r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)):
            try:
                arm.set_position(xy[0] + dx, xy[1] + dy, z, speed=10)
                done += 1
            except MaxArmError as exc:
                print(f"  settle step ({dx:+.0f}, {dy:+.0f}) refused: {exc}")
    arm.set_position(xy[0], xy[1], z, speed=10)
    print(f"settled: {done} vented pushes up to {SETTLE_RADII_MM[-1]:.0f} mm, pressing {SETTLE_PRESS_MM:.0f} mm below the seat")


def _release_at_joints(arm, cam, a, joints, xy, z_top, tag):
    """Put the held lid back by replaying the servo pose read at the pick (the cup sealed on
    the seated lid there), vent with the valve held open, peel, rise, look. One attempt."""
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=30)
    arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    arm.set_servo_angle(joints, speed=15)
    _, pose = arm.get_position()
    print(f"cup at the pick's servo pose; read-back {np.round(pose[:3], 1).tolist()}")
    # Let go PLACE_GAP_MM (2 mm) above that pose so the skirt can drop into place: set down
    # pressed, a lid arriving 3 mm off perched on the rim (2026-11-11). The user allows a
    # drop of a couple of millimetres, never more.
    arm.set_position(float(pose[0]), float(pose[1]), float(pose[2]) + L.PLACE_GAP_MM, speed=10)
    arm.vent(True)
    time.sleep(0.6)
    peeled = 0.0                          # no peel over a dish (see _release_on)
    _settle(arm, (float(pose[0]), float(pose[1])), z_top)
    arm.set_position(xy[0], xy[1], z_top + 20.0, speed=10)
    arm.vent(False)
    arm.set_position(*_view_for_check(xy), speed=30)
    time.sleep(0.4)
    if not _hanging(cam, a, xy, z_top, f"{tag}_release0.png"):
        print(f"released at the servo pose (peel {peeled:+.0f} mm)")
        return True
    print(f"servo-pose release (peel {peeled:+.0f} mm): the lid still hangs on the cup")
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=15)
    return False


def _rim_pair(frame, px, a, z_mm, name=None):
    """The lid's rim and the base's rim near the tape, as (lid_centre_xy, base_centre_xy) in
    mm on the base-rim plane, or None if two distinct dish-sized circles are not seen.
    A perched lid shows two circles a few mm apart (its own rim and the base's); seated,
    they coincide and only one is found."""
    import cv2
    from hardware.maxarm.wrist_camera import pixel_at_height
    g = cv2.cvtColor(frame.color, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 1.5)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12, param1=90, param2=40,
                         minRadius=L.BASE_RADIUS_PX[0], maxRadius=L.BASE_RADIUS_PX[1])
    if c is None:
        return None
    found = []
    for u, v, r in c[0]:
        centre = pixel_at_height(frame, float(u), float(v), z_mm)
        edge = pixel_at_height(frame, float(u) + float(r), float(v), z_mm)
        if centre is None or edge is None:
            continue
        dia = 2.0 * float(np.linalg.norm(edge[:2] - centre[:2]))
        if abs(dia - a.dish_mm) > 12 or np.hypot(u - px[0], v - px[1]) > RIM_NEAR_PX:
            continue
        found.append((dia, centre[:2].copy(), (float(u), float(v), float(r))))
    if len(found) < 2:
        return None
    found.sort(key=lambda f: -f[0])
    lid = found[0]
    base = next((f for f in found[1:] if np.linalg.norm(f[1] - lid[1]) >= 2.0), None)
    if base is None:
        return None
    print(f"  rims: lid {lid[0]:.0f} mm at {np.round(lid[1], 1).tolist()}, base {base[0]:.0f} mm at "
          f"{np.round(base[1], 1).tolist()}")
    if name is not None:
        img = frame.color[:, :, ::-1].copy()
        for f, col in ((lid, (255, 0, 255)), (base, (0, 255, 0))):
            u, v, r = f[2]
            cv2.circle(img, (int(u), int(v)), int(r), col, 2)
        cv2.imwrite(str(LOOP_OUT / name), img)
    return lid[1], base[1]


def _seat(cam, a, home_xy, z_seated, tag):
    """(ok, lid_xyz, miss): seated is judged by HEIGHT (the tape's depth: a lid on its
    seat reads within SEAT_HEIGHT_TOL_MM of it, perched 4-7 mm above; a seated lid is
    centred by the base, so no rim check is needed). For a perched lid, `miss` is where
    the lid's rim is relative to the base's rim, from the pair of circles, or None.
    (The tape's offset from the lid's centre rotates with the lid on the cup, so the
    old tape-vs-rim comparison judged a seated lid 10 mm off, 2026-11-11.)"""
    frame, lids = _look_all(cam, a, z_seated, f"{tag}.png")
    lid = _nearest(lids, home_xy, 60.0)
    if lid is None:
        print("seat check: no lid seen near the dish")
        return False, None, None
    if not lid["depth"]:
        print("seat check: the lid's height was not measured (no depth on the tape); cannot judge")
        return False, lid["xyz"], None
    above = float(lid["xyz"][2] - z_seated)
    ok = bool(-3.0 <= above <= SEAT_HEIGHT_TOL_MM)
    miss = None
    if not ok:
        pair = _rim_pair(frame, lid["px"], a, kin.TABLE_CONTACT_Z_MM + L.BASE_RIM_ABOVE_TABLE_MM,
                         f"{tag}_rim.png")
        if pair is not None:
            miss = pair[0] - pair[1]
            if not 2.0 <= float(np.linalg.norm(miss)) <= 25.0:
                miss = None
    print(f"seat check: lid top {above:+.1f} mm vs seated -> {'seated' if ok else 'NOT seated'}"
          + (f"; lid rim is {np.round(miss, 1).tolist()} mm from the base rim" if miss is not None
             else ("" if ok else "; direction of the miss not measured")))
    return ok, lid["xyz"], miss


def _reseat(arm, cam, a, home_xy, lid_xyz, miss, z_seated, tag, tries):
    """Pick the perched lid up again (pump on) and set it down shifted by the measured
    miss, or, when the miss is not known, back at the remembered pick coordinate."""
    for i in range(1, tries + 1):
        if miss is not None:
            target = lid_xyz[:2] - miss
            print(f"re-seat {i}/{tries}: lid rim is {np.round(miss, 1).tolist()} mm off the base; "
                  f"shifting by {np.round(-miss, 1).tolist()}")
        else:
            target = np.array(home_xy, float)
            print(f"re-seat {i}/{tries}: miss unknown; putting it back at the pick coordinate "
                  f"{np.round(target, 1).tolist()}")
        z = max(float(lid_xyz[2]) - a.press_mm, arm.floor_z_mm)
        _gate((*target, z_seated), (*lid_xyz[:2], z))
        arm.set_position(lid_xyz[0], lid_xyz[1], z + APPROACH_MM, speed=30)
        arm.set_position(lid_xyz[0], lid_xyz[1], z, speed=10)
        arm.close_gripper()
        time.sleep(a.hold_s)
        arm.set_position(lid_xyz[0], lid_xyz[1], z_seated + 20.0, speed=10)
        arm.set_position(target[0], target[1], z_seated + 20.0, speed=10)
        if not _release_on(arm, cam, a, target, z_seated, f"{tag}_reseat{i}", peel=False):
            return False, "HELD: the lid would not leave the cup over the dish"
        arm.set_position(*_view_over(home_xy), speed=30)
        time.sleep(0.4)
        ok, lid_xyz, miss = _seat(cam, a, home_xy, z_seated, f"{tag}_reseat{i}_after")
        if ok:
            return True, f"seated after {i} re-seat(s)"
        if lid_xyz is None:
            return False, "cannot judge: lid not seen after the re-seat"
    return False, f"not seated after {tries} re-seat(s)"


# -- planning ------------------------------------------------------------------

def _aside_for(lid_xy, dishes_xy, a):
    """A set-aside spot clear of every dish on the table, left of this one first; None if none."""
    z_aside_top = kin.TABLE_CONTACT_Z_MM + a.lid_height_mm
    if a.aside is not None:
        cands = [tuple(a.aside)]
    else:
        # left of the dish first (ASIDE_DX_MM is -x), then right, then round the ring
        cands = [(lid_xy[0] + dx, lid_xy[1] + dy) for dx in (ASIDE_DX_MM, -ASIDE_DX_MM)
                 for dy in (0.0, 20.0, 40.0, 60.0, -20.0)]
        for r in (abs(ASIDE_DX_MM), abs(ASIDE_DX_MM) + 20.0):
            for deg in (150, 210, 30, 330, 120, 240, 60, 300, 90, 270):
                cands.append((lid_xy[0] - r * np.cos(np.radians(deg)), lid_xy[1] + r * np.sin(np.radians(deg))))
    for x, y in cands:
        if any(np.hypot(x - o[0], y - o[1]) < a.dish_mm + 15 for o in dishes_xy):
            continue
        t = a.aside_tol_mm            # the lid may slide this far on release and still count
        corners = [(x + sx * t, y + sy * t) for sx in (-1, 1) for sy in (-1, 1)]
        if any(kin.check_pose(p, floor_z_mm=None) for p in
               [(x, y, CARRY_Z), (x, y, z_aside_top), _view_over((x, y)), _check_view((x, y))]
               + [_check_view(c) for c in corners] + [(c[0], c[1], z_aside_top) for c in corners]):
            continue
        return np.array([x, y], dtype=float)
    return None


def _survey_views(a, remembered):
    views = [tuple(a.view)]
    views += [_view_over(xy) for xy in remembered]
    views += [tuple(v) for v in (a.extra_views or [])]
    if a.survey:
        views += [(a.view[0] + dx, a.view[1] + dy, a.view[2]) for dx, dy in SURVEY_OFFSETS_MM[1:]]
    out = []
    for v in views:
        if kin.check_pose(v, floor_z_mm=None):
            continue
        if any(np.hypot(v[0] - o[0], v[1] - o[1]) < VIEW_DEDUPE_MM for o in out):
            continue
        out.append(v)
    return out


def _merge(found, lids):
    for l in lids:
        same = _nearest(found, l["xyz"][:2], MERGE_MM)
        if same is None:
            found.append(l)
        elif (l["depth"] and not same["depth"]) or (l["depth"] == same["depth"] and l["n_px"] > same["n_px"]):
            found[[i for i, f in enumerate(found) if f is same][0]] = l



def _return_lid(arm, cam, a, home_xyz, home_joints, z_seated, aside, tag):
    """The lid lies at the set-aside spot: pick it up and put it back at home_xyz.

    Returns (result line, held): held = parked at home with the lid still on the cup.
    """
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    z_aside_top = kin.TABLE_CONTACT_Z_MM + a.lid_height_mm
    # where did it land? and the tape-vs-rim offset for the seat judgement
    arm.set_position(*_view_over(aside), speed=40)
    time.sleep(0.4)
    frame, lids = _look_all(cam, a, z_aside_top, f"{tag}_aside.png")
    lid_aside = _nearest(lids, aside, a.aside_tol_mm)
    if lid_aside is None:
        arm.go_home()
        return f"FAILED: the lid is not within {a.aside_tol_mm:.0f} mm of the set-aside spot", False
    print(f"set aside: lid at {np.round(lid_aside['xyz'], 1).tolist()}")
    offset = np.zeros(2)
    rim = _rim_near(frame, lid_aside["px"], kin.TABLE_CONTACT_Z_MM, a, f"{tag}_aside_rim.png")
    if rim is not None and np.linalg.norm(lid_aside["xyz"][:2] - rim[0]) <= OFFSET_MAX_MM:
        offset = lid_aside["xyz"][:2] - rim[0]
        print(f"tape centre is {np.round(offset, 1).tolist()} mm from the lid centre")
    else:
        print("WARNING: lid rim not measured; taking the tape centre as the lid centre")

    # lid back on, at the remembered pick coordinate
    # Pick height from GEOMETRY (table + --lid-height-mm), not from depth: the tape is on the
    # lid's underside and depth reads it ~1 mm above the table, under the lid's top, and the
    # cup at the floor margin stopped short of the lid and lifted nothing (2026-11-11).
    pick_xyz = np.array([lid_aside["xyz"][0], lid_aside["xyz"][1], z_aside_top])
    _gate(pick_xyz, _view_for_check(pick_xyz[:2]))
    _pick(arm, pick_xyz, floor, a.press_mm, a.hold_s)
    if not _lifted(arm, cam, a, lid_aside["xyz"], f"{tag}_lift2.png"):
        _abort_holding(arm, lid_aside["xyz"])
        return "FAILED: the lid was not lifted from the set-aside spot", False
    target = home_xyz[:2]
    if float(home_xyz[2]) > z_seated + SEAT_HEIGHT_TOL_MM:
        # picked PERCHED (above the seat): its servo pose is no seat to replay, and a lid
        # let go there would drop further than the couple of millimetres allowed
        print(f"the lid was picked {home_xyz[2] - z_seated:+.1f} mm above the seat: putting it "
              f"back pressed at the pick coordinate {np.round(target, 1).tolist()}, not at the servo pose")
        first = _release_on(arm, cam, a, target, z_seated, f"{tag}_back", peel=False)
    else:
        print(f"putting the lid back at the pick's servo pose (xy {np.round(target, 1).tolist()})")
        first = _release_at_joints(arm, cam, a, home_joints, target, z_seated, f"{tag}_back")
    if not first:
        print("second try: pressing release at the pick coordinate")
        if not _release_on(arm, cam, a, target, z_seated, f"{tag}_back2", peel=False):
            arm.go_home()
            return "HELD: the lid would not leave the cup over the dish", True

    # judge by looking; re-seat by the measured miss until seated
    arm.set_position(*_view_over(home_xyz[:2]), speed=40)
    time.sleep(0.4)
    ok, lid_xyz, miss = _seat(cam, a, home_xyz[:2], z_seated, f"{tag}_after")
    why = "seated first time"
    if not ok:
        if lid_xyz is None:
            why = "cannot judge: lid not seen"
        else:
            ok, why = _reseat(arm, cam, a, home_xyz[:2], lid_xyz, miss, z_seated, tag, a.reseat_tries)
            if why.startswith("HELD"):
                arm.go_home()
                return why, True
    return ("SEATED: " if ok else "NOT SEATED: ") + why, False


# -- one episode -----------------------------------------------------------------

def episode(arm, cam, a, ep, remembered):
    """Returns (results, held, dishes_xy): one line per dish; held = parked with a lid on the cup."""
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    z_aside_top = kin.TABLE_CONTACT_Z_MM + a.lid_height_mm
    found = []
    for i, v in enumerate(_survey_views(a, remembered)):
        arm.set_position(*v, speed=40)
        time.sleep(0.4)
        print(f"survey view {i} from wrist {np.round(v, 0).tolist()}:")
        _, lids = _look_all(cam, a, SEATED_TOP_Z_MM, f"e{ep}_survey{i}.png")
        _merge(found, lids)
    on_dish, other, specks = [], [], []
    for l in found:
        if l["n_px"] < SIGHTING_PX_MIN or (l["depth"] and l["xyz"][2] < GLARE_MAX_Z_MM):
            specks.append(l)                 # glints and tape edges: not a dish, not an obstacle
            continue
        seated = (-SEATED_BAND_MM <= l["xyz"][2] - SEATED_TOP_Z_MM <= PERCHED_MM) if l["depth"] \
            else l["n_px"] >= ON_DISH_NODEPTH_PX
        (on_dish if seated else other).append(l)
    if specks:
        print(f"ignoring {len(specks)} tape-coloured blob(s) (under {SIGHTING_PX_MIN} px, or glints "
              f"at table height) at {[np.round(l['xyz'][:2], 0).tolist() for l in specks]}")
    for l in other:
        print(f"skipping tape at {np.round(l['xyz'], 1).tolist()}: not at the seated height "
              f"(a lid alone or upside down, or too small to be sure)")
    on_dish.sort(key=lambda l: -l["xyz"][1])          # nearest the base first
    dishes_xy = [l["xyz"][:2].copy() for l in on_dish + other]   # lids on dishes and lids lying about
    print(f"episode {ep}: {len(on_dish)} lid(s) on dishes at "
          f"{[np.round(l['xyz'][:2], 0).tolist() for l in on_dish]}")
    results = []
    for k, lid in enumerate(on_dish[:a.max_dishes]):
        tag = f"e{ep}_d{k}"
        print(f"--- dish {k}: lid near {np.round(lid['xyz'][:2], 0).tolist()}")
        # Pick and put-back share this view pose, so the camera's error cancels.
        arm.set_position(*_view_over(lid["xyz"][:2]), speed=40)
        time.sleep(0.4)
        _, lids = _look_all(cam, a, SEATED_TOP_Z_MM, f"{tag}_before.png")
        here = _nearest(lids, lid["xyz"][:2], MERGE_MM)
        if here is None:
            results.append((k, "FAILED: lid not seen from the dish view"))
            continue
        if not here["depth"]:
            results.append((k, "FAILED: the lid's height was not measured from the dish view "
                               "(no depth on the tape); not picking blind"))
            continue
        home_xyz = here["xyz"].copy()
        z_seated = min(float(home_xyz[2]), SEATED_TOP_Z_MM)
        aside = _aside_for(home_xyz[:2], dishes_xy, a)
        if aside is None:
            results.append((k, "FAILED: no reachable set-aside spot clear of the other dishes"))
            continue
        _gate(home_xyz, (*aside, z_aside_top), (*aside, CARRY_Z), _view_over(aside),
              _check_view(aside), _check_view(home_xyz), (*home_xyz[:2], z_seated))
        print(f"plan: pick at {np.round(home_xyz, 1).tolist()}, aside at {np.round(aside, 1).tolist()}, "
              f"back to the pick coordinate, released {L.PLACE_GAP_MM:.0f} mm above seated z {z_seated:.1f}")

        # 1. lid off, remembering the servo pose with the cup sealed on the seated lid
        home_joints, _ = _pick_remember(arm, home_xyz, floor, a.press_mm, a.hold_s)
        if not _lifted(arm, cam, a, home_xyz, f"{tag}_lift.png"):
            _abort_holding(arm, home_xyz)
            results.append((k, "FAILED: the lid was not lifted (set back down)"))
            continue
        if not _release_on(arm, cam, a, aside, z_aside_top, f"{tag}_aside"):
            arm.go_home()
            results.append((k, "HELD: the lid would not leave the cup at the set-aside spot"))
            return results, True, dishes_xy

        line, held = _return_lid(arm, cam, a, home_xyz, home_joints, z_seated, aside, tag)
        results.append((k, line))
        if held:
            return results, True, dishes_xy
        arm.go_home()
    return results, False, dishes_xy


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=3, help="0 = until Ctrl-C")
    ap.add_argument("--pause-s", type=float, default=10.0, help="at home between episodes")
    ap.add_argument("--reseat-tries", type=int, default=1,
                    help="re-seats (lift with the pump on, set down shifted by the miss) before "
                         "moving on to the next dish; one, by the bench rule of 2026-11-11")
    ap.add_argument("--max-dishes", type=int, default=4, help="per episode")
    ap.add_argument("--no-survey", dest="survey", action="store_false",
                    help="look only from --view, the remembered dishes and --extra-views")
    ap.add_argument("--extra-views", type=float, nargs=3, action="append", default=None)
    ap.add_argument("--hue", type=int, nargs="+", default=None, metavar="LO HI",
                    help="tape hue windows (OpenCV 0-179), LO HI pairs; default: blue and orange "
                         "(TAPE_WINDOWS)")
    ap.add_argument("--press-mm", type=float, default=PRESS_MM)
    ap.add_argument("--depth-bias-mm", type=float, default=DEPTH_BIAS_MM)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--aside", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="one fixed set-aside spot (only sensible with one dish)")
    ap.add_argument("--lid-height-mm", type=float, default=3.0,
                    help="cup height above table contact at which a lid lying on the table is held "
                         "(set-down released at 56 = contact + 4; a pick at 55 missed it; 2026-11-11)")
    ap.add_argument("--dish-mm", type=float, default=90.0)
    ap.add_argument("--floor-margin-mm", type=float, default=-1.0,
                    help="lowest commanded cup z, relative to table contact (52); -1 lets the soft "
                         "cup press a lid lying on the table by the pick's 4 mm")
    ap.add_argument("--aside-tol-mm", type=float, default=40.0,
                    help="how far from the set-aside spot the lid may land (28 mm seen, 2026-11-11)")
    ap.add_argument("--view", type=float, nargs=3, default=VIEW_POSE)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    if a.hue is None:
        _set_hue(*TAPE_WINDOWS)
    elif len(a.hue) % 2:
        raise SystemExit("--hue takes LO HI pairs")
    else:
        _set_hue(*zip(a.hue[::2], a.hue[1::2]))
    floor = kin.TABLE_CONTACT_Z_MM + a.floor_margin_mm
    arm, cam = _connect(a.view, floor)
    arm.open_gripper()            # pump off: nothing is held at the start of a run
    remembered, ep, tally = [], 0, []
    try:
        while a.episodes <= 0 or ep < a.episodes:
            ep += 1
            print(f"\n=== episode {ep} ===")
            t0 = time.time()
            results, held, remembered = episode(arm, cam, a, ep, remembered)
            print(f"episode {ep} result ({time.time() - t0:.0f} s):")
            for k, line in results:
                print(f"  dish {k}: {line}")
            if not results:
                print("  no lid on a dish was found")
            tally.append(results)
            if held:
                print("STOPPED: parked at home with the lid still on the cup; free it by hand "
                      "or run `python scripts/maxarm_lid.py release`")
                return 1
            if a.episodes > 0 and ep >= a.episodes:
                break
            print(f"next episode in {a.pause_s:.0f} s: move the dishes now if you want")
            time.sleep(a.pause_s)
    except KeyboardInterrupt:
        print("\ninterrupted: the arm stays where it is, suction as it was (a held lid is not let go)")
        return 130
    finally:
        arm.disconnect()
        cam.close()
    n = sum(len(r) for r in tally)
    ok = sum(1 for r in tally for _, line in r if line.startswith("SEATED"))
    print(f"\n{len(tally)} episode(s), {ok}/{n} lids back on seated")
    return 0 if ok == n and n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
