#!/usr/bin/env python3
"""Convert `recordings/` sessions into a LeRobot dataset.

    python scripts/export_lerobot.py --dry-run              # inspect, needs no lerobot
    python scripts/export_lerobot.py --repo-id ksa/xarm6_rail --out /tmp/ds

This is the integration point with the LeRobot ecosystem, and deliberately the
ONLY one. The alternative -- adopting `lerobot_robot_ufactory`'s `uf_robot`
driver -- would replace `RealXArmAPI`, which carries the measured `BASE_YAW_DEG`
frame conversion, the benchtop floor guards, the rail-trust refusal,
`descend_until_contact`, and the F/T readiness workaround. None of that is in a
LeRobot driver's job description, and all of it was learned the hard way. So we
keep our driver and our recorder, and convert at the data layer.

TWO THINGS THAT WILL SILENTLY CORRUPT THE DATASET IF MISHANDLED, both reconciled
in `session_arrays()`:

1. **`ctrl` is ordered rail-first, `joints_deg` is joints-first.**
   `ACT_NAMES = ["act_rail", "act1".."act6"]` while
   `JOINT_NAMES = ["joint1".."joint6"]`. Concatenating them naively pairs the
   rail with joint 1 in the learned mapping.
2. **They are in different units.** `joints_deg`/`rail_mm` are degrees and
   millimetres; `ctrl` is the raw MuJoCo actuator setpoint -- radians for the
   joints, metres for the rail. A policy whose state is degrees and whose
   action is radians learns a 57x scale factor as if it were physics.

Both are converted here into one canonical layout:

    [j1, j2, j3, j4, j5, j6, rail_mm, gripper]   degrees, mm, 0/1

RATE. State is logged at 60 Hz and frames at 10 Hz; a LeRobot dataset has a
single fps. Frames are the limiting stream, so each frame is paired with the
state sample nearest its timestamp and the dataset runs at the frame rate.

GRIPPER. The action's gripper element is the *state* channel, not a command --
the twin has no actuated fingers, so "is it holding" is the only ground truth
the recorder can see. They differ on every failed grasp. See the note in
`recording.py::_sample_one`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

#: Canonical channel order for both state and action.
CHANNELS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "rail_mm", "gripper"]

#: `ctrl` column order as written by the Recorder (ACT_NAMES).
CTRL_RAIL_COL = 0
CTRL_JOINT_COLS = slice(1, 7)


def session_arrays(session_dir: str, camera: str | None = None):
    """Read one session into LeRobot-shaped arrays. No lerobot import needed.

    Returns a dict with `state`, `action`, `images` (or None), `timestamps`,
    `task`, `fps`, and `notes` describing anything that had to be assumed.
    Returns None when the session has nothing usable.
    """
    import h5py

    h5_path = Path(session_dir) / "trajectory.h5"
    meta_path = Path(session_dir) / "metadata.json"
    if not h5_path.exists():
        return None

    meta = {}
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path))
        except Exception:
            pass
    task = (meta.get("task_label") or meta.get("notes") or "").strip()

    notes = []
    with h5py.File(h5_path, "r") as f:
        if "joints_deg" not in f or "ctrl" not in f:
            return None
        joints = np.asarray(f["joints_deg"])            # (N, 6) degrees
        rail = np.asarray(f["rail_mm"]).reshape(-1, 1)  # (N, 1) mm
        ctrl = np.asarray(f["ctrl"])                    # (N, 7) rail-first, rad/m
        t_state = np.asarray(f["t_wall"])

        if "gripper" in f:
            grip = np.asarray(f["gripper"]).reshape(-1, 1)
        elif "weld_active" in f:
            # Pre-2026-09-16 sessions predate the gripper channel; derive the
            # same quantity from the welds rather than dropping the column.
            grip = (np.asarray(f["weld_active"]).any(axis=1)
                    .astype(np.float32).reshape(-1, 1))
            notes.append("gripper derived from weld_active (pre-2026-09-16 session)")
        else:
            grip = np.zeros((len(joints), 1), dtype=np.float32)
            notes.append("NO gripper signal at all -- column is constant zero")

        # --- images -------------------------------------------------------
        images = None
        t_frame = None
        if "frames" in f:
            g = f["frames"]
            cams = sorted(k for k in g.keys() if k != "t_wall")
            if cams:
                pick = camera or ("cam_wrist_color" if "cam_wrist_color" in cams
                                  else cams[0])
                if pick not in cams:
                    notes.append(f"camera {pick!r} absent; have {cams}")
                else:
                    images = np.asarray(g[pick]["images"])
                    t_frame = np.asarray(g["t_wall"])
                    if pick != "cam_wrist_color":
                        notes.append(f"using camera {pick!r}, not the wrist view")

    # --- reconcile order and units into the canonical layout --------------
    state = np.concatenate([joints, rail, grip], axis=1).astype(np.float32)

    act_joints = np.rad2deg(ctrl[:, CTRL_JOINT_COLS])          # rad -> deg
    act_rail = (ctrl[:, CTRL_RAIL_COL] * 1000.0).reshape(-1, 1)  # m -> mm
    action = np.concatenate([act_joints, act_rail, grip], axis=1).astype(np.float32)

    # --- pair frames with the nearest state sample ------------------------
    if images is not None and len(t_frame):
        idx = np.abs(t_state[None, :] - t_frame[:, None]).argmin(axis=1)
        drift = np.abs(t_state[idx] - t_frame)
        if drift.max() > 0.05:
            notes.append(f"frame/state pairing drifts up to {drift.max()*1000:.0f} ms")
        state, action = state[idx], action[idx]
        timestamps = t_frame
        n = len(t_frame)
        fps = (n - 1) / (t_frame[-1] - t_frame[0]) if n > 1 else 10.0
    else:
        # State-only episode. Usable for a state-conditioned policy; useless for
        # anything vision-based, which is most of what LeRobot is for.
        notes.append("NO IMAGES -- state-only episode")
        timestamps = t_state
        n = len(t_state)
        fps = (n - 1) / (t_state[-1] - t_state[0]) if n > 1 else 60.0

    return dict(state=state, action=action, images=images, timestamps=timestamps,
                task=task, fps=float(fps), notes=notes, n=int(n),
                session=os.path.basename(str(session_dir).rstrip("/")))


def survey(sessions, camera=None):
    """What would be exported, without importing lerobot or writing anything."""
    ok = with_img = no_img = skipped = 0
    tasks, all_notes, frames_total = {}, {}, 0
    for s in sessions:
        d = session_arrays(s, camera=camera)
        if d is None:
            skipped += 1
            continue
        ok += 1
        if d["images"] is not None:
            with_img += 1
            frames_total += len(d["images"])
        else:
            no_img += 1
        tasks[d["task"] or "(no task label)"] = tasks.get(d["task"] or "(no task label)", 0) + 1
        for note in d["notes"]:
            all_notes[note] = all_notes.get(note, 0) + 1
    print(f"\nsessions scanned : {len(sessions)}")
    print(f"  usable         : {ok}")
    print(f"  with images    : {with_img}   ({frames_total:,} frames)")
    print(f"  state-only     : {no_img}")
    print(f"  skipped        : {skipped}")
    print(f"\ntop task labels:")
    for t, c in sorted(tasks.items(), key=lambda kv: -kv[1])[:8]:
        print(f"   {c:5d}  {t[:70]}")
    if all_notes:
        print(f"\ncaveats hit:")
        for note, c in sorted(all_notes.items(), key=lambda kv: -kv[1]):
            print(f"   {c:5d}  {note}")


def export(sessions, repo_id, out_dir, camera=None, fps=10, images_required=True):
    """Write a LeRobot dataset. Imports lerobot, so keep it out of --dry-run."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa

    sample = None
    for s in sessions:
        sample = session_arrays(s, camera=camera)
        if sample is not None and (sample["images"] is not None or not images_required):
            break
    if sample is None:
        raise SystemExit("no usable session found")

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(CHANNELS),),
                              "names": CHANNELS},
        "action":            {"dtype": "float32", "shape": (len(CHANNELS),),
                              "names": CHANNELS},
    }
    if sample["images"] is not None:
        h, w = sample["images"].shape[1:3]
        features["observation.images.wrist"] = {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channel"]}

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=out_dir,
                               robot_type="xarm6_rail", features=features)
    written = 0
    for s in sessions:
        d = session_arrays(s, camera=camera)
        if d is None:
            continue
        if images_required and d["images"] is None:
            continue
        for i in range(d["n"]):
            frame = {"observation.state": d["state"][i],
                     "action": d["action"][i],
                     "task": d["task"] or "manipulate"}
            if d["images"] is not None:
                frame["observation.images.wrist"] = d["images"][i]
            ds.add_frame(frame)
        ds.save_episode()
        written += 1
        print(f"  episode {written:4d}  {d['session']}  {d['n']:5d} steps")
    print(f"\nwrote {written} episode(s) to {out_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--repo-id", default="local/xarm6_rail")
    ap.add_argument("--out", default=None, help="output dir for the dataset")
    ap.add_argument("--camera", default=None,
                    help="which /frames/<camera> to export (default cam_wrist_color)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="only the first N sessions")
    ap.add_argument("--dry-run", action="store_true",
                    help="survey what would be exported; imports no lerobot")
    ap.add_argument("--allow-state-only", action="store_true",
                    help="include sessions with no images (vision policies need images)")
    args = ap.parse_args()

    sessions = sorted(glob.glob(os.path.join(args.recordings, "*/")))
    if args.limit:
        sessions = sessions[:args.limit]
    if not sessions:
        raise SystemExit(f"no sessions under {args.recordings}/")

    if args.dry_run:
        survey(sessions, camera=args.camera)
        return 0
    if not args.out:
        raise SystemExit("--out is required unless --dry-run")
    export(sessions, args.repo_id, args.out, camera=args.camera, fps=args.fps,
           images_required=not args.allow_state_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
