# agent/quality.py
"""
Trajectory-derived execution-quality scorer (FIX 6).

Terminal grading (`check_outcome`) collapses a clean placement and an ugly
one that nudged three other objects into the same `SUCCESS`. The episode loop
then pins plans by *fewest commands*, so it cannot prefer the clean plan and
tends to re-pin and repeat messy ones.

This module computes a scalar `quality in [0, 1]` plus a transparent component
breakdown OFFLINE from the recorded `trajectory.h5` -- no extra sim runs, no
LLM cost. The loop uses it to (a) select the best plan, (b) reframe the plans
shown to the planner, and (c) annotate the lesson.

Design notes (read before tuning):
  * Dense reward is mis-specifiable. The weights here are deliberately
    conservative and the components are kept transparent. The `notes` text is
    what actually steers the LLM planner ("you disturbed green_cube 38mm" is
    more actionable than "0.42"), so prefer improving the notes over chasing a
    perfectly tuned scalar.
  * `score_trajectory` NEVER raises on a malformed/short trajectory -- it
    returns `quality=None` with a note so callers treat it as 'unscored'.
  * Components are only included when they can be computed for the task
    family; absent components are dropped from the weighting rather than
    faked (the doc's `ik_fallback_rate` is omitted entirely -- not wired yet).

Reads only the h5: t_wall, joints_deg, ee_pos_mm, body_poses, weld_active,
and the body_names attribute the recorder stamps onto the file.
"""
from typing import Dict, List, Optional, Set

import numpy as np


# --- Normalisation constants (module-level so they're transparent/tunable) ---
# Non-target movement below this is treated as physics settling noise.
_DISTURB_TOL_MM = 5.0
# Total non-target displacement (mm, summed over bodies) that drives the
# disturbance score to 0. Generous so only real knocks are penalised.
_DISTURB_NORM_MM = 150.0
# Generous expected travel; paths at or under these score full marks on
# efficiency (capped at 1.0 so a reckless shortcut can't be rewarded ABOVE a
# clean path).
_EXPECTED_EE_MM = 2500.0
_EXPECTED_JOINT_DEG = 1200.0
# Dimensionless jerk/velocity ratio that drives the smoothness sub-score to 0.
_JERK_RATIO_NORM = 1.5
# Fraction of samples that may be direction reversals before the reversal
# sub-score hits 0.
_REVERSAL_RATE_NORM = 0.30
# Ideal weld transition count for a grasp task (one grasp + one release).
_IDEAL_WELD_TRANSITIONS = 2

# Conservative component weights. Only the components actually present for a
# given task are used; the weights are renormalised over those present.
_WEIGHTS = {
    "disturbance": 0.40,
    "smoothness": 0.25,
    "efficiency": 0.20,
    "grasp_stability": 0.15,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _decode_names(raw) -> List[str]:
    """Decode the h5 body_names attribute into a list of python strings."""
    out: List[str] = []
    for n in list(raw):
        if isinstance(n, bytes):
            out.append(n.decode("utf-8", "replace"))
        else:
            out.append(str(n))
    return out


def _target_bodies(task: str, body_names: List[str]) -> Set[str]:
    """Best-effort set of free-body names the task is *supposed* to move.

    Derived from the existing outcome_checker spec so geometry stays
    consistent with grading. Names not present in this trajectory's
    body_names are dropped. Empty set when the task can't be parsed -- the
    caller then skips the disturbance component rather than penalising
    legitimate motion.
    """
    try:
        from agent.outcome_checker import expected_outcome
    except Exception:
        return set()
    spec = expected_outcome(task)
    if spec is None:
        return set()
    _mode, subs = spec
    present = set(body_names)
    targets: Set[str] = set()
    for s in subs:
        name = s
        for marker in (" fell to floor", " off bench"):
            if name.endswith(marker):
                name = name[: -len(marker)]
        if " in " in name:
            name = name.split(" in ", 1)[0]
        name = name.strip()
        if name in present:
            targets.add(name)
    return targets


def _disturbance(body_poses: np.ndarray, body_names: List[str],
                 targets: Set[str], notes: List[str]) -> Optional[float]:
    """Penalise movement of NON-target bodies.

    body_poses: (T, n_bodies, 7) -- xyz in metres in cols 0:3.
    Returns a score in [0, 1] (1 = nothing else moved), or None if there are
    no non-target bodies to judge.
    """
    if body_poses.ndim != 3 or body_poses.shape[1] != len(body_names):
        return None
    xyz_mm = body_poses[:, :, 0:3] * 1000.0  # (T, B, 3)
    init = xyz_mm[0]                          # (B, 3)
    # Max Euclidean displacement from the start pose, per body.
    disp = np.linalg.norm(xyz_mm - init[None, :, :], axis=2).max(axis=0)  # (B,)

    total = 0.0
    worst_name, worst_mm = None, 0.0
    judged = 0
    for i, name in enumerate(body_names):
        if name in targets:
            continue
        judged += 1
        d = float(disp[i])
        if d > _DISTURB_TOL_MM:
            total += d
            if d > worst_mm:
                worst_mm, worst_name = d, name
    if judged == 0:
        return None
    if worst_name is not None:
        notes.append(f"disturbed {worst_name} {worst_mm:.0f}mm")
    return _clamp01(1.0 - total / _DISTURB_NORM_MM)


def _efficiency(ee_pos_mm: np.ndarray, joints_deg: np.ndarray,
                notes: List[str]) -> Optional[float]:
    """End-effector path length + total joint travel vs. generous expected.

    Capped at 1.0 so a reckless shortcut scores no higher than a clean path.
    """
    if ee_pos_mm.ndim != 2 or ee_pos_mm.shape[0] < 2:
        return None
    ee_path = float(np.linalg.norm(np.diff(ee_pos_mm, axis=0), axis=1).sum())
    joint_travel = float(np.abs(np.diff(joints_deg, axis=0)).sum())
    ee_score = min(1.0, _EXPECTED_EE_MM / max(ee_path, _EXPECTED_EE_MM))
    jt_score = min(1.0, _EXPECTED_JOINT_DEG / max(joint_travel, _EXPECTED_JOINT_DEG))
    notes.append(f"ee path {ee_path:.0f}mm, joint travel {joint_travel:.0f}deg")
    return _clamp01(0.5 * ee_score + 0.5 * jt_score)


def _smoothness(ee_pos_mm: np.ndarray, notes: List[str]) -> Optional[float]:
    """Mean |jerk| (normalised by mean speed) + direction-reversal rate."""
    if ee_pos_mm.ndim != 2 or ee_pos_mm.shape[0] < 4:
        return None
    vel = np.diff(ee_pos_mm, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    mean_speed = float(speed.mean()) + 1e-6
    jerk = np.diff(ee_pos_mm, n=3, axis=0)
    mean_jerk = float(np.linalg.norm(jerk, axis=1).mean())
    jerk_ratio = mean_jerk / mean_speed
    jerk_score = _clamp01(1.0 - jerk_ratio / _JERK_RATIO_NORM)

    # Direction reversals: consecutive velocity vectors pointing "backwards".
    if vel.shape[0] >= 2:
        dots = (vel[1:] * vel[:-1]).sum(axis=1)
        reversals = int((dots < 0).sum())
        reversal_rate = reversals / vel.shape[0]
    else:
        reversals, reversal_rate = 0, 0.0
    reversal_score = _clamp01(1.0 - reversal_rate / _REVERSAL_RATE_NORM)
    if reversals:
        notes.append(f"{reversals} direction reversals")
    return _clamp01(0.6 * jerk_score + 0.4 * reversal_score)


def _grasp_stability(weld_active: np.ndarray, family: str,
                     notes: List[str]) -> Optional[float]:
    """Count weld transitions; >2 (one grasp + one release expected) is
    fumbling. Only meaningful for grasp-based families."""
    if family not in ("placement", "sort"):
        return None
    if weld_active.ndim != 2 or weld_active.shape[0] < 2:
        return None
    # Total 0<->1 transitions across all equality constraints.
    changes = np.abs(np.diff(weld_active.astype(np.int16), axis=0)).sum()
    transitions = int(changes)
    notes.append(f"{transitions} weld transitions")
    excess = max(0, transitions - _IDEAL_WELD_TRANSITIONS)
    if transitions == 0:
        # Placement family but nothing was ever grasped -- weak grasp signal.
        return 0.3
    # Each extra transition beyond the ideal pair costs 0.25.
    return _clamp01(1.0 - 0.25 * excess)


def score_trajectory(h5_path, task: str, registry=None) -> dict:
    """Return {'quality': float in [0,1] | None, 'components': {...},
    'notes': [str]}.

    Reads only the h5 (t_wall, joints_deg, ee_pos_mm, body_poses, weld_active,
    body_names). Never raises on a malformed/short trajectory -- returns
    quality=None with a note instead, so callers can treat it as 'unscored'.

    `registry` is accepted for signature compatibility / future use; the
    current scorer derives target bodies from the task text alone.
    """
    notes: List[str] = []
    try:
        import h5py
        from pathlib import Path
        path = Path(h5_path)
        if not path.exists():
            return {"quality": None, "components": {},
                    "notes": [f"no trajectory at {path}"]}
        with h5py.File(path, "r") as f:
            n = int(f.attrs.get("n_samples", 0)) or (
                f["t_wall"].shape[0] if "t_wall" in f else 0)
            if n < 4:
                return {"quality": None, "components": {},
                        "notes": [f"trajectory too short ({n} samples)"]}
            joints_deg = np.asarray(f["joints_deg"]) if "joints_deg" in f else None
            ee_pos_mm = np.asarray(f["ee_pos_mm"]) if "ee_pos_mm" in f else None
            body_poses = np.asarray(f["body_poses"]) if "body_poses" in f else None
            weld_active = np.asarray(f["weld_active"]) if "weld_active" in f else None
            body_names = _decode_names(f.attrs.get("body_names", []))
    except Exception as e:
        return {"quality": None, "components": {},
                "notes": [f"unreadable trajectory: {type(e).__name__}: {e}"]}

    try:
        from agent.outcome_checker import classify_task
        family = classify_task(task)
    except Exception:
        family = "unknown"

    targets = _target_bodies(task, body_names) if body_names else set()

    components: Dict[str, float] = {}

    if body_poses is not None and body_names:
        d = _disturbance(body_poses, body_names, targets, notes)
        if d is not None:
            components["disturbance"] = d
    if ee_pos_mm is not None and joints_deg is not None:
        e = _efficiency(ee_pos_mm, joints_deg, notes)
        if e is not None:
            components["efficiency"] = e
    if ee_pos_mm is not None:
        s = _smoothness(ee_pos_mm, notes)
        if s is not None:
            components["smoothness"] = s
    if weld_active is not None:
        g = _grasp_stability(weld_active, family, notes)
        if g is not None:
            components["grasp_stability"] = g

    if not components:
        return {"quality": None, "components": {},
                "notes": notes or ["no scorable components"]}

    # Weighted mean over the components that were actually computed.
    total_w = sum(_WEIGHTS[k] for k in components)
    quality = sum(_WEIGHTS[k] * v for k, v in components.items()) / total_w
    return {"quality": _clamp01(quality), "components": components,
            "notes": notes}


def summarize_quality(quality: dict) -> str:
    """One-line human-readable breakdown for lessons / prompts, e.g.
    'quality=0.42 disturbed green_cube 38mm, 2 weld transitions'.
    Returns '' if there's nothing scorable."""
    if not quality or quality.get("quality") is None:
        return ""
    q = quality["quality"]
    notes = quality.get("notes") or []
    note_str = "; ".join(notes)
    return f"quality={q:.2f}" + (f" {note_str}" if note_str else "")
