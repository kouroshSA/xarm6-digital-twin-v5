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
SEATED_BAND_MM = 4.0     # tape top within this of the seated height: a lid on its dish
LID_PX_MIN = 15000       # a lid seen flat with depth (maxarm_lid._verify_lift)
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
    lo, hi = L._tape_mask.hue
    hsv = cv2.cvtColor(frame.color, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
         & (hsv[..., 1] > MIN_SATURATION) & (hsv[..., 2] > MIN_VALUE))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    return [lab == i for i in order if stats[i, cv2.CC_STAT_AREA] >= MIN_TAPE_PX]


def _lid_from_mask(frame, mask, default_top, a):
    """The tape blob as a lid: centre on the lid-top plane, whether depth gave the height."""
    from hardware.maxarm.wrist_camera import pixel_at_height
    z_tape, n_depth = _tape_height(frame, mask)
    if z_tape is not None and not (kin.TABLE_CONTACT_Z_MM + 3 <= z_tape <= kin.TABLE_CONTACT_Z_MM + 35):
        z_tape = None            # not on this table: a lid hanging on the cup reads nonsense
    if z_tape is None:
        z_top, depth = float(default_top), False
    else:
        z_top, depth = z_tape + LID_THICKNESS_MM - a.depth_bias_mm, True
    rows, cols = np.nonzero(mask)
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
        print("  no blue tape seen")
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

def _lifted(arm, cam, a, xyz, name):
    """True if no lid lies at xyz any more after the pick (maxarm_lid._verify_lift, any blob)."""
    arm.set_position(*_view_for_check(xyz[:2]), speed=30)
    time.sleep(0.4)
    _, lids = _look_all(cam, a, xyz[2], name)
    for l in lids:
        off = float(np.linalg.norm(l["xyz"][:2] - xyz[:2]))
        dz = float(l["xyz"][2] - xyz[2])
        if l["depth"] and off <= a.aside_tol_mm and abs(dz) <= 6.0 and l["n_px"] >= LID_PX_MIN:
            print(f"lift check: the lid is still there ({off:.1f} mm from the pick point, "
                  f"height {dz:+.1f}); nothing was lifted")
            return False
    print("lift check: no lid lies at the pick point; it came up")
    return True


def _hanging(cam, a, xy, z_top, name):
    """True unless a lid is seen LYING at xy (depth at z_top, flat, big): maxarm_lid._hanging per blob."""
    _, lids = _look_all(cam, a, z_top, name)
    for l in lids:
        if (l["depth"] and l["n_px"] >= LID_PX_MIN and abs(l["xyz"][2] - z_top) <= 8.0
                and np.linalg.norm(l["xyz"][:2] - xy) <= a.aside_tol_mm):
            return False
    return True


def _release_on(arm, cam, a, xy, z_top, tag):
    """Lower the held lid onto z_top at xy and vent IN CONTACT until it lets go (maxarm_lid._release_on)."""
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=30)
    arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    for attempt in range(3):
        z = max(z_top - RELEASE_PRESS_MM - attempt, arm.floor_z_mm)
        arm.set_position(xy[0], xy[1], z, speed=10)
        arm.open_gripper()
        time.sleep(0.5)
        arm.set_position(xy[0], xy[1], z_top + 20.0, speed=10)
        arm.set_position(*_view_for_check(xy), speed=30)
        time.sleep(0.4)
        if not _hanging(cam, a, xy, z_top, f"{tag}_release{attempt}.png"):
            print(f"released after {attempt + 1} vent(s), lid set down at z {z:.1f}")
            return True
        print(f"vent {attempt + 1} at z {z:.1f}: the lid still hangs on the cup")
        arm.set_position(xy[0], xy[1], z_top + APPROACH_MM, speed=30)
    arm.set_position(xy[0], xy[1], CARRY_Z, speed=15)
    return False


def _seat(cam, a, home_xy, offset, z_seated, tag):
    """(ok, lid_xyz, rim_xy, miss) judged in one frame from the dish view."""
    frame, lids = _look_all(cam, a, z_seated, f"{tag}.png")
    lid = _nearest(lids, home_xy, 60.0)
    if lid is None:
        print("seat check: no lid seen near the dish")
        return False, None, None, None
    rim = _rim_near(frame, lid["px"], kin.TABLE_CONTACT_Z_MM + L.BASE_RIM_ABOVE_TABLE_MM, a,
                    f"{tag}_rim.png")
    if rim is None:
        return False, lid["xyz"], None, None
    above = float(lid["xyz"][2] - z_seated)
    d = lid["xyz"][:2] - offset - rim[0]
    ok = bool(np.linalg.norm(d) <= SEAT_TOL_MM and above <= SEAT_HEIGHT_TOL_MM)
    print(f"seat check: lid centre {np.linalg.norm(d):.1f} mm from the rim centre "
          f"({np.round(d, 1).tolist()}), lid top {above:+.1f} mm vs seated -> "
          f"{'seated' if ok else 'NOT seated'}")
    return ok, lid["xyz"], rim[0], d


def _reseat(arm, cam, a, home_xy, lid_xyz, rim_xy, offset, z_seated, tag, tries):
    """Pick the perched lid up again and set it down shifted by the measured miss, until seated."""
    for i in range(1, tries + 1):
        d = lid_xyz[:2] - offset - rim_xy
        target = lid_xyz[:2] - d
        print(f"re-seat {i}/{tries}: lid is {np.round(d, 1).tolist()} mm off; shifting by "
              f"{np.round(-d, 1).tolist()}")
        z = max(float(lid_xyz[2]) - a.press_mm, arm.floor_z_mm)
        _gate((*target, z_seated), (*lid_xyz[:2], z))
        arm.set_position(lid_xyz[0], lid_xyz[1], z + APPROACH_MM, speed=30)
        arm.set_position(lid_xyz[0], lid_xyz[1], z, speed=10)
        arm.close_gripper()
        time.sleep(a.hold_s)
        arm.set_position(lid_xyz[0], lid_xyz[1], z_seated + 20.0, speed=10)
        arm.set_position(target[0], target[1], z_seated + 20.0, speed=10)
        if not _release_on(arm, cam, a, target, z_seated, f"{tag}_reseat{i}"):
            return False, "HELD: the lid would not leave the cup over the dish"
        arm.set_position(*_view_over(home_xy), speed=30)
        time.sleep(0.4)
        ok, lid_xyz, rim_xy, d = _seat(cam, a, home_xy, offset, z_seated, f"{tag}_reseat{i}_after")
        if ok:
            return True, f"seated after {i} re-seat(s)"
        if lid_xyz is None or rim_xy is None:
            return False, "cannot judge: lid or rim not seen after the re-seat"
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



def _return_lid(arm, cam, a, home_xyz, z_seated, aside, tag):
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
    _gate(lid_aside["xyz"], _view_for_check(lid_aside["xyz"][:2]))
    _pick(arm, lid_aside["xyz"], floor, a.press_mm, a.hold_s)
    if not _lifted(arm, cam, a, lid_aside["xyz"], f"{tag}_lift2.png"):
        _abort_holding(arm, lid_aside["xyz"])
        return "FAILED: the lid was not lifted from the set-aside spot", False
    target = home_xyz[:2]
    print(f"putting the lid back at the pick coordinate {np.round(target, 1).tolist()}")
    if not _release_on(arm, cam, a, target, z_seated, f"{tag}_back"):
        arm.go_home()
        return "HELD: the lid would not leave the cup over the dish", True

    # judge by looking; re-seat by the measured miss until seated
    arm.set_position(*_view_over(home_xyz[:2]), speed=40)
    time.sleep(0.4)
    ok, lid_xyz, rim_xy, d = _seat(cam, a, home_xyz[:2], offset, z_seated, f"{tag}_after")
    why = "seated first time"
    if not ok:
        if lid_xyz is None or rim_xy is None:
            why = "cannot judge: lid or rim not seen"
        else:
            ok, why = _reseat(arm, cam, a, home_xyz[:2], lid_xyz, rim_xy, offset, z_seated,
                              tag, a.reseat_tries)
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
    on_dish, other = [], []
    for l in found:
        seated = (abs(l["xyz"][2] - SEATED_TOP_Z_MM) <= SEATED_BAND_MM) if l["depth"] \
            else l["n_px"] >= LID_PX_MIN
        (on_dish if seated else other).append(l)
    for l in other:
        print(f"skipping tape at {np.round(l['xyz'], 1).tolist()}: not at the seated height "
              f"(a lid alone or upside down, or too small to be sure)")
    on_dish.sort(key=lambda l: -l["xyz"][1])          # nearest the base first
    dishes_xy = [l["xyz"][:2].copy() for l in found]
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

        # 1. lid off
        _pick(arm, home_xyz, floor, a.press_mm, a.hold_s)
        if not _lifted(arm, cam, a, home_xyz, f"{tag}_lift.png"):
            _abort_holding(arm, home_xyz)
            results.append((k, "FAILED: the lid was not lifted (set back down)"))
            continue
        if not _release_on(arm, cam, a, aside, z_aside_top, f"{tag}_aside"):
            arm.go_home()
            results.append((k, "HELD: the lid would not leave the cup at the set-aside spot"))
            return results, True, dishes_xy

        line, held = _return_lid(arm, cam, a, home_xyz, z_seated, aside, tag)
        results.append((k, line))
        if held:
            return results, True, dishes_xy
        arm.go_home()
    return results, False, dishes_xy


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", type=int, default=3, help="0 = until Ctrl-C")
    ap.add_argument("--pause-s", type=float, default=10.0, help="at home between episodes")
    ap.add_argument("--reseat-tries", type=int, default=5)
    ap.add_argument("--max-dishes", type=int, default=4, help="per episode")
    ap.add_argument("--no-survey", dest="survey", action="store_false",
                    help="look only from --view, the remembered dishes and --extra-views")
    ap.add_argument("--extra-views", type=float, nargs=3, action="append", default=None)
    ap.add_argument("--hue", type=int, nargs=2, default=L.BLUE_HUE, metavar=("LO", "HI"))
    ap.add_argument("--press-mm", type=float, default=PRESS_MM)
    ap.add_argument("--depth-bias-mm", type=float, default=DEPTH_BIAS_MM)
    ap.add_argument("--hold-s", type=float, default=1.5)
    ap.add_argument("--aside", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="one fixed set-aside spot (only sensible with one dish)")
    ap.add_argument("--lid-height-mm", type=float, default=5.0)
    ap.add_argument("--dish-mm", type=float, default=90.0)
    ap.add_argument("--floor-margin-mm", type=float, default=3.0)
    ap.add_argument("--aside-tol-mm", type=float, default=25.0)
    ap.add_argument("--view", type=float, nargs=3, default=VIEW_POSE)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    _set_hue(*a.hue)
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
