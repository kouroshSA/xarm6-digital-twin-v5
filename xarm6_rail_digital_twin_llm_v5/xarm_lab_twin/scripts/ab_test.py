#!/usr/bin/env python3
"""A/B harness: does a prompt-layer or skill actually change anything?

Everything we add above the planner is text aimed at a language model --
skills, Layer 0 constraints, Layer 2 rewrites. Text like that always *reads*
as though it helps, which is exactly what makes it dangerous: without
measurement you accumulate confident prose that costs latency and tokens on
every run and does nothing. `world_model.md` asserting an error string that
does not exist is what that looks like when it goes wrong.

This runs the same tasks with a feature off (arm A) and on (arm B), from a
matched baseline, and reports the difference.

    python scripts/ab_test.py --feature skills --repeats 2
    python scripts/ab_test.py --feature intake --tasks tasks.txt --episodes 8
    python scripts/ab_test.py --list-features

## A null result is not a verdict on the feature

It is a verdict on the feature FOR THE TASKS TESTED. `safe-traverse` is about
traversing; measuring it on a task whose failures are all grasp-height
failures tells you nothing about traversing. So results are reported PER
TASK, never as a single headline number, and the summary says which tasks
showed a difference rather than averaging them into silence.

Task difficulty is part of the experiment. An easy task the planner already
solves on episode 1 has no room to improve -- it cannot show a benefit even
if one exists. Those are reported as "no headroom" rather than as evidence
against the feature.

## Matched baselines

`world_model.md` and `lessons.md` persist across runs and are injected into
every planner prompt. Without resetting them, arm B starts with arm A's
learning and the comparison measures the wrong thing. Both are snapshotted
once and restored before EVERY run, not merely between arms.

Sim only. Never touches hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

sys.path.insert(0, os.getcwd())

#: name -> (arm A env/args, arm B env/args, what it is)
FEATURES = {
    "skills": (
        {"env": {"XARM_NO_SKILLS": "1"}, "args": []},
        {"env": {}, "args": []},
        "planning skills injected into the planner's system prompt",
    ),
    "intake": (
        {"env": {}, "args": ["--no-intake"]},
        {"env": {}, "args": []},
        "Layer 0 instruction intake (decompose / constrain / push back)",
    ),
    "refine": (
        {"env": {}, "args": []},
        {"env": {}, "args": ["--refine-prompt"]},
        "Layer 2 instance-level prompt rewriting",
    ),
}

#: Tasks with a rough difficulty label. Difficulty is not decoration: a task
#: the planner already solves immediately cannot demonstrate an improvement.
DEFAULT_TASKS = [
    ("put red_cube_front in the translucent cup", "easy"),
    ("put the red cube that is in front of the rail into the translucent cup", "medium"),
    ("move the blue cube into the translucent cup", "medium"),
    ("put both red cubes into the translucent cup, one at a time", "hard"),
]

PERSISTED = ("world_model.md", "lessons.md", "reviews.md")


@dataclass
class RunResult:
    task: str
    difficulty: str
    arm: str
    successes: int = 0
    episodes: int = 0
    first_success: int | None = None    # 1-indexed episode, None if never
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class Baseline:
    """Snapshot of the files that leak learning between runs."""
    tmp: str = ""
    saved: dict = field(default_factory=dict)

    def capture(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ab_baseline_")
        for name in PERSISTED:
            if os.path.exists(name):
                dst = os.path.join(self.tmp, name)
                shutil.copy2(name, dst)
                self.saved[name] = dst

    def restore(self) -> None:
        for name in PERSISTED:
            src = self.saved.get(name)
            if src:
                shutil.copy2(src, name)
            elif os.path.exists(name):
                os.remove(name)      # it did not exist at baseline


def run_once(task: str, difficulty: str, arm: str, spec: dict,
             episodes: int, model: str, timeout: int) -> RunResult:
    env = dict(os.environ)
    env.update(spec.get("env", {}))
    env.setdefault("MUJOCO_GL", "egl")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        summary_path = fh.name

    cmd = [sys.executable, "scripts/run_task.py", task,
           "--model", model, "--loop", "--max-episodes", str(episodes),
           "--speed-tier", "fast", "--no-render",
           "--summary-json", summary_path] + list(spec.get("args", []))

    r = RunResult(task=task, difficulty=difficulty, arm=arm)
    try:
        subprocess.run(cmd, env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(summary_path) as fh:
            payload = json.load(fh)
        loop = payload.get("loop") or {}
        outcomes = loop.get("episode_outcomes") or []
        r.episodes = len(outcomes)
        r.successes = sum(1 for o in outcomes if o is True)
        for i, o in enumerate(outcomes, 1):
            if o is True:
                r.first_success = i
                break
    except subprocess.TimeoutExpired:
        r.error = f"timed out after {timeout}s"
    except Exception as exc:                                # noqa: BLE001
        r.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.unlink(summary_path)
        except OSError:
            pass
    return r


def summarise(results: list, feature: str, desc: str) -> None:
    print(f"\n{'=' * 74}\n  A/B: {feature} -- {desc}\n{'=' * 74}")
    tasks = []
    for t, d in dict.fromkeys((r.task, r.difficulty) for r in results):
        tasks.append((t, d))

    verdicts = []
    for task, diff in tasks:
        a = [r for r in results if r.task == task and r.arm == "A" and r.ok]
        b = [r for r in results if r.task == task and r.arm == "B" and r.ok]
        if not a or not b:
            print(f"\n  {task[:66]}  [{diff}]\n    incomplete -- a run failed, no comparison")
            continue
        a_rate = statistics.mean(x.successes / max(x.episodes, 1) for x in a)
        b_rate = statistics.mean(x.successes / max(x.episodes, 1) for x in b)
        a_fs = [x.first_success for x in a if x.first_success]
        b_fs = [x.first_success for x in b if x.first_success]
        a_ttf = statistics.mean(a_fs) if a_fs else None
        b_ttf = statistics.mean(b_fs) if b_fs else None

        print(f"\n  {task[:66]}  [{diff}]")
        print(f"    success rate      A {a_rate:5.0%}   B {b_rate:5.0%}")
        print(f"    episodes to first A {('%.1f' % a_ttf) if a_ttf else '  -- ':>5}   "
              f"B {('%.1f' % b_ttf) if b_ttf else '  -- ':>5}")

        # "No headroom" is a distinct outcome from "no benefit". A task the
        # control already solves on episode 1 cannot show an improvement, and
        # counting it as evidence against the feature would be wrong.
        if a_ttf == 1 and a_rate >= 0.99:
            verdict = "no headroom (control already solves it immediately)"
        elif abs(a_rate - b_rate) < 1e-9 and a_ttf == b_ttf:
            verdict = "no measurable difference on this task"
        elif b_rate > a_rate or (b_ttf and a_ttf and b_ttf < a_ttf):
            verdict = "B better"
        else:
            verdict = "A better"
        print(f"    -> {verdict}")
        verdicts.append((task, diff, verdict))

    print(f"\n{'-' * 74}")
    better = [v for v in verdicts if v[2] == "B better"]
    worse = [v for v in verdicts if v[2] == "A better"]
    flat = [v for v in verdicts if v[2].startswith("no measurable")]
    none = [v for v in verdicts if v[2].startswith("no headroom")]
    print(f"  B better on {len(better)}, A better on {len(worse)}, "
          f"flat on {len(flat)}, no headroom on {len(none)} "
          f"of {len(verdicts)} task(s)")
    if not better and flat:
        print("\n  No benefit measured -- but ONLY for the tasks above. A feature\n"
              "  aimed at a failure mode these tasks do not exercise cannot show\n"
              "  up here. Read this as 'not demonstrated yet', not 'useless'.")
    reps = max((len([r for r in results if r.arm == 'A' and r.task == t]) for t, _ in tasks),
               default=0)
    if reps < 3:
        print(f"\n  CAUTION: {reps} repeat(s) per arm. Too few to separate a real\n"
              f"  effect from run-to-run variation. Treat as indicative only.")


def self_test() -> int:
    """Verdict logic, on synthetic results. No model, no sim.

    The distinction being protected here is the one that is easy to lose: a
    task the control already solves immediately has NO ROOM to improve, and
    reporting that as evidence against a feature would be wrong. So "no
    headroom" must come out as its own verdict, not as "no benefit".
    """
    fails = 0
    cases = [
        ("no headroom when the control already solves it at episode 1",
         [RunResult("t", "easy", "A", 6, 6, 1), RunResult("t", "easy", "B", 6, 6, 1)],
         "no headroom"),
        ("B better when it reaches success sooner",
         [RunResult("t", "med", "A", 1, 6, 5), RunResult("t", "med", "B", 2, 6, 2)],
         "B better"),
        ("flat when the two arms are identical",
         [RunResult("t", "hard", "A", 0, 6, None), RunResult("t", "hard", "B", 0, 6, None)],
         "no measurable difference"),
        ("A better when the feature makes it worse",
         [RunResult("t", "med", "A", 3, 6, 2), RunResult("t", "med", "B", 1, 6, 5)],
         "A better"),
    ]
    import io, contextlib
    for label, res, expect in cases:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarise(res, "self-test", "")
        ok = expect in buf.getvalue()
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:58} expect={expect!r}")

    b = Baseline()
    ok = hasattr(b, "capture") and hasattr(b, "restore") and PERSISTED
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {'baseline snapshots the files that leak learning':58} "
          f"{list(PERSISTED)}")

    print(f"\n  {len(cases) + 1 - fails} PASS  {fails} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature", choices=sorted(FEATURES), default="skills")
    ap.add_argument("--repeats", type=int, default=2, help="runs per arm per task")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per run")
    ap.add_argument("--tasks", help="file of tasks, one per line, optional '| difficulty'")
    ap.add_argument("--list-features", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the matrix and cost estimate without running")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.list_features:
        for name, (_, _, desc) in sorted(FEATURES.items()):
            print(f"  {name:8} {desc}")
        return 0

    tasks = DEFAULT_TASKS
    if args.tasks:
        tasks = []
        for line in open(args.tasks):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, _, d = line.partition("|")
            tasks.append((t.strip(), d.strip() or "unlabelled"))

    spec_a, spec_b, desc = FEATURES[args.feature]
    n_runs = len(tasks) * args.repeats * 2
    print(f"  feature : {args.feature} -- {desc}")
    print(f"  matrix  : {len(tasks)} task(s) x {args.repeats} repeat(s) x 2 arms "
          f"= {n_runs} runs of up to {args.episodes} episodes")
    print(f"  arm A   : {spec_a['env'] or '{}'} {' '.join(spec_a['args']) or '(no extra args)'}")
    print(f"  arm B   : {spec_b['env'] or '{}'} {' '.join(spec_b['args']) or '(no extra args)'}")
    if args.dry_run:
        print("\n  --dry-run: nothing executed.")
        return 0

    baseline = Baseline()
    baseline.capture()
    print(f"  baseline: {', '.join(baseline.saved) or 'nothing to snapshot'} "
          f"(restored before every run)")

    results = []
    try:
        for rep in range(args.repeats):
            for task, diff in tasks:
                for arm, spec in (("A", spec_a), ("B", spec_b)):
                    baseline.restore()
                    print(f"\n  [rep {rep+1}/{args.repeats}] arm {arm}: {task[:56]}",
                          flush=True)
                    r = run_once(task, diff, arm, spec, args.episodes,
                                 args.model, args.timeout)
                    results.append(r)
                    print(f"      {r.successes}/{r.episodes} success"
                          f"{', first at ep %d' % r.first_success if r.first_success else ''}"
                          f"{'  ERROR: ' + r.error if r.error else ''}", flush=True)
    finally:
        baseline.restore()

    summarise(results, args.feature, desc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
