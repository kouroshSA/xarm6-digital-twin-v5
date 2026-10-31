#!/usr/bin/env python3
"""Run the same task as several LIVE run_task.py sessions in parallel.

Testing a task one episode after another is slow; independent sessions run side
by side give the same evidence sooner. Each session gets its own folder (log,
summary JSON, per-command and in-primitive snapshots), its own viewer window,
and one episode by default -- sessions share the room's lessons / reviews /
world_model files, and the end-of-session review that writes them only runs on
sessions of 3+ episodes.

    python scripts/run_parallel_sessions.py "open the incubator door, ..." \\
        --room simready_demo --sessions 3 --speed-tier fast --out /tmp/par_round1

When every session has finished, it renders the screenshots worth looking at
(every command that failed, and each session's final scene) with
scripts/render_snapshots.py and writes OUT/summary.md: per session the commands
and return codes, doors that fell back to the handle, objects put_in disturbed,
the physical outcome and the grader's verdict. Exit code 0 only if every session
passed.

Pass --no-render to skip the viewers (headless, MUJOCO_GL=egl) -- for CI, not
for watching. Extra run_task.py arguments go after "--".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PY = sys.executable


def _session_cmd(args, d: Path, extra) -> list:
    cmd = [PY, "-u", "scripts/run_task.py", args.task, "--room", args.room,
           "--model", args.model, "--speed-tier", args.speed_tier,
           "--loop", "--max-episodes", str(args.episodes),
           "--snapshot-dir", str(d / "snap"), "--summary-json", str(d / "run.json")]
    if args.no_render:
        cmd.append("--no-render")
    return cmd + list(extra)


def _summarise(d: Path) -> dict:
    log = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
    cmds = re.findall(r"^\[Agent\] (\w+)\((.*)\) -> (-?\d+)$", log, re.M)
    pick = lambda pat: [ln for ln in log.splitlines() if re.search(pat, ln)]
    outcome = pick(r"^\[EpisodeLoop\] Physical outcome")
    check = pick(r"^\[EpisodeLoop\] Outcome check")
    passed = None
    summary = d / "run.json"
    if summary.exists():
        try:
            passed = bool(json.loads(summary.read_text()).get("loop", {}).get("success"))
        except (ValueError, AttributeError):
            passed = None
    return {
        "session": d.name, "passed": passed,
        "commands": [(a, int(rc)) for a, _, rc in cmds],
        "failed_commands": [f"{a}({p}) -> {rc}" for a, p, rc in cmds if int(rc) != 0],
        "handle_fallbacks": pick(r"finishing with the handle"),
        "disturbed": pick(r"disturbed "),
        "outcome": outcome[-1] if outcome else "",
        "verdict": check[-1] if check else "",
        "traceback": "Traceback" in log,
    }


def _render(d: Path, s: dict) -> list:
    snap = d / "snap"
    if not snap.is_dir():
        return []
    files = sorted(p.name for p in snap.glob("[0-9][0-9][0-9]_*.json"))
    want = [f[:3] for f in files if not f.endswith("_rc0.json")]
    if files:
        want.append(files[-1][:3])                 # the final scene
    if not want:
        return []
    env = dict(os.environ, MUJOCO_GL="egl")
    subprocess.run([PY, "scripts/render_snapshots.py", str(snap), "--only", *sorted(set(want))],
                   cwd=HERE, env=env, capture_output=True)
    return sorted(str(p) for p in (snap / "png").glob("*.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--room", default="simready_demo")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=1,
                    help="episodes per session (default 1: sessions share the room's memory files)")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--speed-tier", default="fast")
    ap.add_argument("--out", required=True, help="folder; one sub-folder per session")
    ap.add_argument("--stagger", type=float, default=5.0,
                    help="seconds between session starts (spreads the API and GL start-up)")
    ap.add_argument("--no-render", action="store_true", help="headless sessions (no viewers)")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(1, args.sessions + 1):
        d = out / f"s{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if args.no_render:
            env["MUJOCO_GL"] = "egl"
        with open(d / "run.log", "w") as log:
            p = subprocess.Popen(_session_cmd(args, d, extra), cwd=HERE, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        procs.append((d, p))
        print(f"[parallel] session {d.name} started (pid {p.pid}) -> {d}")
        time.sleep(args.stagger)

    t0 = time.time()
    for d, p in procs:
        p.wait()
        print(f"[parallel] session {d.name} exited {p.returncode} after {time.time() - t0:.0f} s")

    rows, lines = [], [f"# Parallel sessions: {args.task}\n",
                       f"room `{args.room}`, model {args.model}, tier {args.speed_tier}, "
                       f"{args.sessions} session(s) x {args.episodes} episode(s)\n"]
    for d, _ in procs:
        s = _summarise(d)
        s["pngs"] = _render(d, s)
        rows.append(s)
        mark = {True: "PASS", False: "FAIL", None: "?"}[s["passed"]]
        lines += [f"## {s['session']}: {mark}",
                  f"- commands: " + ", ".join(f"{a}->{rc}" for a, rc in s["commands"]),
                  f"- failed commands: {s['failed_commands'] or 'none'}",
                  f"- handle fallbacks: {len(s['handle_fallbacks'])}",
                  f"- disturbed: {s['disturbed'] or 'none'}",
                  f"- {s['outcome']}", f"- {s['verdict']}",
                  f"- screenshots: {', '.join(s['pngs']) or 'none'}", ""]
    (out / "summary.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"[parallel] {n_pass}/{len(rows)} passed; see {out / 'summary.md'}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
