#!/usr/bin/env python3
"""Planning skills: reusable procedure injected into the planner's prompt.

A skill is durable know-how about *how* to act -- retract before traversing,
descend vertically, read the refusal. It is not knowledge about *where things
are*. That distinction is the whole design, and it is enforced rather than
trusted.

## Skills carry procedure, never geometry

This repository's recurring defect is one fact living in several places and
the copies drifting apart. In a single session we found four disagreeing
copies of the joint limits, three of the cup's dimensions, a failure analyser
still teaching a grasp rule the simulator had stopped accepting, and a world
model asserting an error string that does not exist.

A skill is a *more* durable hiding place for that rot, because it reads as
authority and nobody re-reads it. So `load_skills` REJECTS any skill
containing a measurement or a coordinate. Positions and dimensions come from
the registry at runtime, where there is exactly one copy and it is read from
the scene.

## Skills must earn their place

A skill that does not change behaviour is prose. The episode loop reports
episodes-to-first-success, which makes skills falsifiable: run a task with and
without one and compare. `--no-skills` on the entry points exists for that.

    python -m agent.skills --list
    python -m agent.skills --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

SKILLS_DIR = Path("agent/skills")

#: Text that looks like a measurement or a coordinate. A skill containing any
#: of these is rejected at load time rather than silently trusted.
GEOMETRY_PATTERNS = (
    re.compile(r"-?\d+(?:\.\d+)?\s*(?:mm|cm|millimet|centimet)", re.I),
    re.compile(r"\(\s*-?\d+\s*,\s*-?\d+"),          # (x, y ...
    re.compile(r"\b[xyz]\s*=\s*-?\d", re.I),        # z=810
    re.compile(r"\brail\s+(?:at\s+)?-?\d+", re.I),  # rail at 350
)


class SkillRejected(ValueError):
    pass


def _check_no_geometry(name: str, text: str) -> None:
    hits = []
    for pat in GEOMETRY_PATTERNS:
        hits += pat.findall(text)
    if hits:
        raise SkillRejected(
            f"skill '{name}' contains geometry ({hits[:3]}). Skills carry "
            f"procedure only; positions and sizes must come from the registry "
            f"at runtime, where there is one copy read from the scene.")


def load_skills(skills_dir: Path = SKILLS_DIR, strict: bool = False,
                audience: str = "planner") -> list:
    """[(name, body)] for skills addressed to `audience`. Rejects geometry.

    Audience matters because the two consumers need opposite things. The
    planner needs procedure -- how to move without hitting anything. The
    prompt refiner needs the opposite instruction: say what the task IS, and
    do not describe how to do it. Feeding the planner's movement procedure to
    the refiner would produce task statements with the plan already baked in,
    which pre-empts the planner and freezes one strategy across every episode
    -- exactly what the loop's exploration depends on not happening.

    `strict` re-raises rejections instead of skipping.
    """
    out = []
    if not Path(skills_dir).is_dir():
        return out
    for path in sorted(Path(skills_dir).glob("*.md")):
        text = path.read_text()
        name = path.stem.replace("_", "-")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        body = text[m.end():] if m else text
        skill_audience = "planner"
        if m:
            n = re.search(r"^name:\s*(.+)$", m.group(1), re.M)
            if n:
                name = n.group(1).strip()
            a = re.search(r"^audience:\s*(.+)$", m.group(1), re.M)
            if a:
                skill_audience = a.group(1).strip()
        if skill_audience != audience:
            continue
        try:
            _check_no_geometry(name, body)
        except SkillRejected as exc:
            if strict:
                raise
            print(f"[Skills] {exc}")
            continue
        out.append((name, body.strip()))
    return out


def render_skills_section(skills: list) -> str:
    if not skills:
        return "(no planning skills loaded)"
    parts = []
    for name, body in skills:
        parts.append(f"### Skill: {name}\n\n{body}")
    return "\n\n".join(parts)


def self_test() -> int:
    import tempfile
    fails = 0

    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "good.md"
        good.write_text("---\nname: good\n---\nLift before traversing.\n")
        bad = Path(d) / "bad.md"
        bad.write_text("---\nname: bad\n---\nGrasp the cube at z=810 mm.\n")

        loaded = {n for n, _ in load_skills(Path(d))}
        ok = loaded == {"good"}
        fails += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] a skill containing geometry is "
              f"rejected (loaded={sorted(loaded)})")

        try:
            load_skills(Path(d), strict=True)
            ok2 = False
        except SkillRejected:
            ok2 = True
        fails += 0 if ok2 else 1
        print(f"  [{'PASS' if ok2 else 'FAIL'}] strict mode raises instead of skipping")

    real = load_skills()
    ok3 = len(real) > 0
    fails += 0 if ok3 else 1
    print(f"  [{'PASS' if ok3 else 'FAIL'}] the shipped skills load clean "
          f"({[n for n, _ in real]})")

    print(f"\n  {3 - fails} PASS  {fails} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    for name, body in load_skills():
        print(f"  {name:20} {len(body):5} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
