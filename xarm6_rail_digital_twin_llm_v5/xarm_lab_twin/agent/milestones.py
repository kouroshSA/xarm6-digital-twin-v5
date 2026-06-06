# agent/milestones.py
"""
Milestone (process) grading for credit assignment + false-positive detection
(FIX 7).

Terminal grading (`check_outcome`) gives a single bool: it can't say *where* a
run broke, can't tell a 5/6 near-miss from a 1/6 flail, and only heuristically
catches false-positive successes (object bounced into the destination without
ever being grasped). This module decomposes each known task family into
ordered sub-goals checkable from `body_poses` / `weld_active` and emits a
milestone trace per episode.

`grade_milestones` reads ONLY the h5 and never raises -- on a malformed or
unparseable trajectory it returns an empty trace with a note, so callers can
treat the episode as un-graded at the process level and fall back to terminal
grading alone.

The destination geometry is taken from the same `_COLLISION_ENVELOPES` table
the episode loop uses, so milestone AABBs stay consistent with the rest of the
system.
"""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# --- Thresholds (mm). Transparent + conservative; see CLAUDE_CODE_FIXES.md. --
_REACH_MM = 90.0          # EE within this of the source center == "reached"
_PUSH_REACH_MM = 130.0    # looser for push (the EE approaches from a side)
_LIFT_MARGIN_MM = 25.0    # source must rise this far above its start, welded
_CONTACT_MM = 20.0        # source moved this far from start == "contact"
_BENCH_FLOOR_Z_MM = 700.0  # below this == fell to floor
_BENCH_X_MM = 750.0        # |x| beyond this == off the bench
_BENCH_Y_MM = 450.0        # |y| beyond this == off the bench
_TRANSIT_DISTURB_MM = 40.0  # a non-target moved more than this during transit


def _decode_names(raw) -> List[str]:
    out: List[str] = []
    for n in list(raw):
        out.append(n.decode("utf-8", "replace") if isinstance(n, bytes) else str(n))
    return out


def _envelopes() -> Dict[str, Dict[str, float]]:
    """The shared collision-envelope table (import lazily to avoid any
    import-order coupling with episode_loop)."""
    try:
        from agent.episode_loop import _COLLISION_ENVELOPES
        return _COLLISION_ENVELOPES
    except Exception:
        # Conservative local fallback mirroring episode_loop's table.
        return {
            "rack": {"half_xy_mm": 100}, "bin": {"half_xy_mm": 80},
            "cube": {"half_xy_mm": 20}, "tube": {"half_xy_mm": 20},
        }


def _track(body_poses: np.ndarray, body_names: List[str],
           name: str) -> Optional[np.ndarray]:
    """Return a body's (T, 3) xyz track in mm, or None if absent."""
    if name not in body_names:
        return None
    i = body_names.index(name)
    if body_poses.ndim != 3 or i >= body_poses.shape[1]:
        return None
    return body_poses[:, i, 0:3] * 1000.0


def _dest_aabb(registry, dest_name: str
               ) -> Optional[Tuple[float, float, float]]:
    """(cx_mm, cy_mm, half_xy_mm) for a destination bin/rack, or None."""
    if registry is None or not dest_name:
        return None
    try:
        obj = registry.find(dest_name)
    except Exception:
        obj = None
    if obj is None:
        return None
    env = _envelopes().get(getattr(obj, "object_type", ""), {})
    half = env.get("half_xy_mm")
    if half is None:
        return None
    cx = obj.position_xyz_m[0] * 1000.0
    cy = obj.position_xyz_m[1] * 1000.0
    return (cx, cy, float(half))


def _xy_in_aabb(x: float, y: float, aabb: Tuple[float, float, float]) -> bool:
    cx, cy, half = aabb
    return abs(x - cx) <= half and abs(y - cy) <= half


def _parse_placement_pairs(task: str, body_names: List[str]
                           ) -> List[Tuple[str, str]]:
    """Return [(source_body, dest_name), ...] from the outcome_checker spec."""
    try:
        from agent.outcome_checker import expected_outcome
    except Exception:
        return []
    spec = expected_outcome(task)
    if spec is None:
        return []
    _mode, subs = spec
    present = set(body_names)
    pairs: List[Tuple[str, str]] = []
    for s in subs:
        if " in " not in s:
            continue
        src, dst = s.split(" in ", 1)
        src, dst = src.strip(), dst.strip()
        if src in present:
            pairs.append((src, dst))
    return pairs


def _parse_push_targets(task: str, body_names: List[str]) -> List[str]:
    try:
        from agent.outcome_checker import expected_outcome
    except Exception:
        return []
    spec = expected_outcome(task)
    if spec is None:
        return []
    _mode, subs = spec
    present = set(body_names)
    out: List[str] = []
    for s in subs:
        name = s
        for marker in (" fell to floor", " off bench"):
            if name.endswith(marker):
                name = name[: -len(marker)]
        name = name.strip()
        if name in present and name not in out:
            out.append(name)
    return out


def _weld_masks(weld_active: np.ndarray):
    """(any_weld[T] bool, up_indices, down_indices)."""
    if weld_active.ndim != 2 or weld_active.shape[0] < 2:
        T = weld_active.shape[0] if weld_active.ndim >= 1 else 0
        return np.zeros(T, dtype=bool), [], []
    any_weld = weld_active.astype(bool).any(axis=1)
    d = np.diff(any_weld.astype(np.int8))
    up = list(np.where(d == 1)[0] + 1)
    down = list(np.where(d == -1)[0] + 1)
    return any_weld, up, down


def _grade_one_placement(src_track: np.ndarray, ee_mm: np.ndarray,
                         any_weld: np.ndarray, up, down,
                         dest_aabb, nontarget_tracks: List[np.ndarray],
                         prefix: str = "") -> Tuple[List[Dict], bool, bool]:
    """Grade the ordered placement milestones for one source->dest pair.

    Returns (milestones, grasped, final_in_dest)."""
    T = src_track.shape[0]
    src_init = src_track[0]

    def m(name, achieved, t_index=None):
        return {"name": prefix + name, "achieved": bool(achieved),
                "t_index": (int(t_index) if t_index is not None else None)}

    ms: List[Dict] = []

    # reached: EE close to the source's INITIAL position.
    if ee_mm is not None and ee_mm.shape[0] == T:
        d_ee = np.linalg.norm(ee_mm - src_init[None, :], axis=1)
        ri = int(d_ee.argmin())
        ms.append(m("reached", d_ee[ri] <= _REACH_MM, ri if d_ee[ri] <= _REACH_MM else None))
    else:
        ms.append(m("reached", False))

    # grasped: any weld engaged.
    grasped = len(up) > 0
    ms.append(m("grasped", grasped, up[0] if grasped else None))

    # lifted_clear: source rose above its start while a weld was active.
    rose = src_track[:, 2] - src_init[2]
    lifted_mask = any_weld & (rose > _LIFT_MARGIN_MM)
    lifted_idx = int(np.argmax(lifted_mask)) if lifted_mask.any() else None
    ms.append(m("lifted_clear", lifted_mask.any(), lifted_idx))

    # transported: no non-target body displaced beyond threshold during the
    # welded (transit) window.
    if grasped:
        t0 = up[0]
        t1 = down[0] if down else T
        clean = True
        for nt in nontarget_tracks:
            if nt is None or nt.shape[0] != T:
                continue
            seg = nt[t0:t1]
            if seg.shape[0] >= 1:
                disp = np.linalg.norm(seg - nt[t0][None, :], axis=1).max()
                if disp > _TRANSIT_DISTURB_MM:
                    clean = False
                    break
        ms.append(m("transported", clean, t0 if clean else None))
    else:
        ms.append(m("transported", False))

    # aligned: source xy inside the destination AABB while welded.
    aligned = False
    aligned_idx = None
    final_in_dest = False
    if dest_aabb is not None:
        for t in range(T):
            if any_weld[t] and _xy_in_aabb(src_track[t, 0], src_track[t, 1], dest_aabb):
                aligned, aligned_idx = True, t
                break
        # final_in_dest: last sample inside the AABB (weld may be released).
        final_in_dest = _xy_in_aabb(src_track[-1, 0], src_track[-1, 1], dest_aabb)
    ms.append(m("aligned", aligned, aligned_idx))

    # released_in_place: a weld release while the source is inside the AABB.
    released = False
    released_idx = None
    if dest_aabb is not None:
        for t in down:
            tt = min(t, T - 1)
            if _xy_in_aabb(src_track[tt, 0], src_track[tt, 1], dest_aabb):
                released, released_idx = True, tt
                break
    ms.append(m("released_in_place", released, released_idx))

    return ms, grasped, final_in_dest


def grade_milestones(h5_path, task: str, registry=None) -> dict:
    """Return {'milestones': [{'name', 'achieved', 't_index'}],
               'fraction': float, 'implausible_success': bool, 'notes': [str]}.

    Reads only the h5. `fraction` is achieved/total in [0,1].
    `implausible_success` is True when the FINAL state matches the task but the
    milestone trace could not have produced it legitimately (e.g. the target
    ended in the rack but was never welded -> it bounced in). Returns an empty
    trace with a note (and implausible_success=False) on any read/parse issue.
    """
    notes: List[str] = []
    try:
        import h5py
        from pathlib import Path
        path = Path(h5_path)
        if not path.exists():
            return {"milestones": [], "fraction": 0.0,
                    "implausible_success": False,
                    "notes": [f"no trajectory at {path}"]}
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("n_samples", 0)) or (
                f["t_wall"].shape[0] if "t_wall" in f else 0)
            if n < 4:
                return {"milestones": [], "fraction": 0.0,
                        "implausible_success": False,
                        "notes": [f"trajectory too short ({n} samples)"]}
            ee_mm = np.asarray(f["ee_pos_mm"]) if "ee_pos_mm" in f else None
            body_poses = np.asarray(f["body_poses"]) if "body_poses" in f else None
            weld_active = (np.asarray(f["weld_active"])
                           if "weld_active" in f else None)
            body_names = _decode_names(f.attrs.get("body_names", []))
    except Exception as e:
        return {"milestones": [], "fraction": 0.0, "implausible_success": False,
                "notes": [f"unreadable trajectory: {type(e).__name__}: {e}"]}

    if body_poses is None or not body_names or weld_active is None:
        return {"milestones": [], "fraction": 0.0, "implausible_success": False,
                "notes": ["trajectory missing body_poses/weld_active/body_names"]}

    try:
        from agent.outcome_checker import classify_task
        family = classify_task(task)
    except Exception:
        family = "unknown"

    any_weld, up, down = _weld_masks(weld_active)

    milestones: List[Dict] = []
    implausible = False

    if family in ("placement", "sort"):
        pairs = _parse_placement_pairs(task, body_names)
        if not pairs:
            notes.append("no placement source/dest could be parsed from task")
        multi = len(pairs) > 1
        for src, dst in pairs:
            src_track = _track(body_poses, body_names, src)
            if src_track is None:
                continue
            dest_aabb = _dest_aabb(registry, dst)
            if dest_aabb is None:
                notes.append(f"destination '{dst}' geometry unavailable "
                             f"(registry missing) -- alignment unchecked")
            nontarget = [_track(body_poses, body_names, nm)
                         for nm in body_names if nm != src]
            prefix = f"{src}:" if multi else ""
            ms, grasped, final_in_dest = _grade_one_placement(
                src_track, ee_mm, any_weld, up, down, dest_aabb,
                nontarget, prefix=prefix)
            milestones.extend(ms)
            # The target ended in the destination but was never grasped ->
            # it could only have bounced/slid in. Flag the grader.
            if final_in_dest and not grasped:
                implausible = True
                notes.append(f"{src} ended in {dst} but was never welded "
                             f"(implausible success)")

    elif family == "push_off":
        targets = _parse_push_targets(task, body_names)
        if not targets:
            notes.append("no push target could be parsed from task")
        multi = len(targets) > 1
        for tgt in targets:
            tr = _track(body_poses, body_names, tgt)
            if tr is None:
                continue
            init = tr[0]
            prefix = f"{tgt}:" if multi else ""

            def m(name, achieved, t_index=None):
                return {"name": prefix + name, "achieved": bool(achieved),
                        "t_index": (int(t_index) if t_index is not None else None)}

            # reached
            reached_idx = None
            if ee_mm is not None and ee_mm.shape[0] == tr.shape[0]:
                d_ee = np.linalg.norm(ee_mm - init[None, :], axis=1)
                ri = int(d_ee.argmin())
                reached_idx = ri if d_ee[ri] <= _PUSH_REACH_MM else None
            milestones.append(m("reached", reached_idx is not None, reached_idx))

            # contact: target moved from its start.
            disp = np.linalg.norm(tr - init[None, :], axis=1)
            contact_mask = disp > _CONTACT_MM
            contact = bool(contact_mask.any())
            contact_idx = int(np.argmax(contact_mask)) if contact else None
            milestones.append(m("contact", contact, contact_idx))

            # off_bench: ended below the bench floor or beyond a bench edge.
            fx, fy, fz = tr[-1, 0], tr[-1, 1], tr[-1, 2]
            off = (fz < _BENCH_FLOOR_Z_MM or abs(fx) > _BENCH_X_MM
                   or abs(fy) > _BENCH_Y_MM)
            milestones.append(m("off_bench", off, (tr.shape[0] - 1) if off else None))

            # Ended off the bench yet never observed moving -> implausible.
            if off and not contact:
                implausible = True
                notes.append(f"{tgt} ended off-bench but contact was never "
                             f"observed (implausible success)")
    else:
        notes.append(f"no milestone set for task family '{family}'")

    total = len(milestones)
    achieved = sum(1 for ms in milestones if ms["achieved"])
    fraction = (achieved / total) if total else 0.0
    if total:
        notes.append(f"{achieved}/{total} milestones achieved")
    return {"milestones": milestones, "fraction": fraction,
            "implausible_success": implausible, "notes": notes}


def first_unmet_milestone(milestones: List[Dict]) -> Optional[str]:
    """Name of the first milestone in the ordered trace that wasn't achieved,
    or None if all achieved / the trace is empty."""
    for ms in milestones or []:
        if not ms.get("achieved"):
            return ms.get("name")
    return None


# Specific, actionable constraint text per milestone. The failed milestone
# name is a far more useful learned-constraint than the generic
# physical-failure text (FIX 7b).
_MILESTONE_HINTS = {
    "reached": ("The end-effector never got within grasp range of the target. "
                "Set the rail so the target's xy is reachable, then move_to "
                "directly above it before descending."),
    "grasped": ("The target was never welded (grasp never engaged). "
                "gripper_close must fire within ~70mm of the target's centre "
                "-- descend to the registry grip height for that object type "
                "and verify xy matches the target before closing."),
    "lifted_clear": ("The target was grasped but never lifted clear of the "
                     "bench. After gripper_close, raise z by at least 30mm "
                     "while the weld holds before transiting."),
    "transported": ("The transit path disturbed another object. Keep xy "
                    "inside [-750..+750, -450..+450] mm and route around the "
                    "other bodies' positions from the registry while carrying."),
    "aligned": ("The held target never reached the destination's footprint. "
                "Move to the destination's xy (from the registry) before "
                "releasing; for tube-into-rack use place_tube_in_rack."),
    "released_in_place": ("The target was not released inside the destination. "
                          "gripper_open directly over the destination's xy, "
                          "just above its top surface, so it settles inside."),
    "contact": ("The push never contacted the target. Use push_object with "
                "target_name set and approach from the side opposite the "
                "chosen bench edge."),
    "off_bench": ("The target never left the bench. Aim push_object's "
                  "to_x_mm/to_y_mm past the nearest edge (|x|>750 or "
                  "|y|>450 mm)."),
}


def milestone_constraint(milestones: List[Dict]) -> Optional[str]:
    """Build a specific learned-constraint string from the first unmet
    milestone, or None if every milestone was achieved."""
    name = first_unmet_milestone(milestones)
    if name is None:
        return None
    bare = name.split(":", 1)[-1]  # strip per-object prefix for sort tasks
    hint = _MILESTONE_HINTS.get(bare)
    if hint is None:
        return f"Milestone '{name}' was not achieved."
    return f"Milestone '{name}' not achieved -- {hint}"
