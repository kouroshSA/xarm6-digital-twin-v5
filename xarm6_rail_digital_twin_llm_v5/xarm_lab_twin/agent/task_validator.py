#!/usr/bin/env python3
"""Layer 1 task validation: resolve a task against the scene before planning.

The planner's job is to choose actions. It should not also be guessing which
object was meant, or how tall it is, or whether the task is possible at all.
Watching the episode loop, it spent six episodes proposing grasp heights of
760, 795, 810, 830 and 845 mm for a cube whose centre the registry already
knew was at 780 -- and the task it was attempting was, at that moment,
impossible for any plan. Both are cheaper to settle before planning starts.

Two things happen here, and neither uses an LLM:

1. **Resolution.** Every object referred to in the task is bound to exactly
   one scene body, with its measured position and dimensions. Ambiguity is
   reported rather than silently resolved -- "the red cube" matches two.

2. **Feasibility.** Deterministic checks that can prove a task impossible:
   the object is wider than the jaws; it does not fit the destination
   container; no legal, collision-free grasp pose exists from any rail
   position. A task that fails these cannot be rescued by a better plan, and
   saying so in one second beats discovering it over twelve episodes.

Deliberately NOT here: advice, procedure, or strategy. This layer emits facts
that were read from the scene. Anything that rewrites the task into better
prose belongs in Layer 2, where the standing rule is that every number must
trace back to something this module measured.

    python -m agent.task_validator --self-test
    python -m agent.task_validator "put the red cube in the translucent cup"
"""
from __future__ import annotations

import argparse
import re
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.getcwd())


@dataclass
class Resolution:
    phrase: str
    matches: list                      # LabObject
    @property
    def ok(self) -> bool:
        return len(self.matches) == 1


@dataclass
class TaskVerdict:
    task: str
    resolutions: list = field(default_factory=list)
    facts: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.blockers

    def render(self) -> str:
        out = [f"Task: {self.task}", ""]
        if self.facts:
            out.append("Ground truth read from the scene:")
            out += [f"  - {f}" for f in self.facts]
        if self.warnings:
            out.append("")
            out.append("Warnings -- ambiguity, or a likely but unproven problem:")
            out += [f"  ! {w}" for w in self.warnings]
        if self.blockers:
            out.append("")
            out.append("BLOCKERS -- no plan can succeed while these hold:")
            out += [f"  x {b}" for b in self.blockers]
        return "\n".join(out)


#: Rail positions sampled when asking "is there anywhere the arm can stand to
#: reach this?". Coarse on purpose -- this is a feasibility screen, not a
#: planner, and 8 IK solves is already the expensive part of this module.
RAIL_SAMPLES_MM = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0)

#: IK attempts per sampled rail position. The solver is branch-unstable, so a
#: single failed attempt is not evidence the pose is unreachable.
IK_ATTEMPTS_PER_RAIL = 3


#: How much of an alias an n-gram must cover to count as naming it. Substring
#: matching alone is far too eager: "the rail" is a substring of the alias
#: "red cube behind the rail", so a prompt mentioning the rail resolved to a
#: red cube, invented a fact about it, and then made Layer 2's contract reject
#: a perfectly good rewrite for "dropping" a referent that was never real.
NGRAM_ALIAS_COVERAGE = 0.5


def _ngram_matches(registry, phrase: str) -> list:
    """Objects an n-gram plausibly names, by alias coverage rather than mere
    containment. Exact name or exact alias always wins."""
    out = []
    for obj in registry.objects.values():
        if phrase == obj.name.lower():
            return [obj]
        for alias in obj.aliases:
            a = alias.lower().strip()
            if phrase == a:
                out.append(obj)
                break
            if phrase in a and len(phrase) / max(len(a), 1) >= NGRAM_ALIAS_COVERAGE:
                out.append(obj)
                break
    return out


def resolve_referents(task: str, registry) -> list:
    """Bind phrases in `task` to scene objects via the registry's aliases.

    Matches aliases against the raw task text rather than trying to parse it.
    Parsing English is the part that would need an LLM, and getting it wrong
    silently is worse than matching a little too eagerly: an extra resolved
    object costs one line of context, a missed one costs an episode.
    """
    low = task.lower()
    seen, out = set(), []

    # Phrases FROM THE TASK, matched against the registry. This direction
    # matters and was missing: the loop below asks "does this alias appear in
    # the task", which needs the operator to use an alias verbatim. No red
    # cube carries the bare alias "red cube" -- they are registered as "front
    # red cube", "near red cube" and so on -- so "put the red cube in the cup"
    # matched NOTHING and Layer 1 reported no ambiguity at all. A silent
    # non-resolution is worse than an ambiguous one: the planner is left to
    # guess and nobody is told. registry.find_all does substring matching the
    # other way round, so feeding it n-grams from the task catches it.
    words = re.findall(r"[a-z0-9_]+", low)
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            matches = _ngram_matches(registry, phrase)
            if not matches:
                continue
            key = (phrase, tuple(sorted(o.name for o in matches)))
            if key in seen:
                continue
            # Skip a shorter phrase already covered by a longer one that
            # resolved to the same objects ("red cube" under "front red cube").
            if any(phrase in p_ and set(o.name for o in matches) ==
                   set(o.name for o in m_) for p_, m_ in
                   [(r.phrase, r.matches) for r in out]):
                continue
            seen.add(key)
            out.append(Resolution(phrase=phrase, matches=matches))

    for obj in registry.objects.values():
        # The body NAME counts too, and is tried first. Naming an object
        # explicitly is exactly how an operator disambiguates "the red cube",
        # so a resolver that only understood aliases would ignore the one
        # phrasing guaranteed to be unambiguous.
        candidates = [obj.name] + sorted(obj.aliases, key=len, reverse=True)
        for alias in candidates:
            a = alias.lower().strip()
            if len(a) < 3 or a not in low:
                continue
            matches = registry.find_all(a)
            key = (a, tuple(sorted(o.name for o in matches)))
            if key in seen:
                continue
            seen.add(key)
            out.append(Resolution(phrase=a, matches=matches))
            break
    return out


def _reachable_at_rail(arm, target_m, rail_mm: float) -> bool:
    """Can the arm reach `target_m` (metres, world) with the rail at `rail_mm`?

    The rail has to be moved for real before asking, because both the IK solver
    and the validator read the LIVE qpos. The first version of this called
    set_rail_position(speed_mm_s=0, wait=False), which writes a ctrl target but
    does not move qpos -- so all eight samples silently tested whatever position
    the rail already held, and the sweep reported red_cube_front unreachable
    from everywhere while a grasp at that exact height demonstrably worked.

    A false blocker is worse than no check: it refuses work that is possible.
    So the rail qpos is set directly, and restored in a finally.
    """
    import mujoco
    rail_m = float(rail_mm) / 1000.0
    with arm.lock:
        saved = float(arm.data.qpos[arm.rail_jid])
        try:
            arm.data.qpos[arm.rail_jid] = rail_m
            mujoco.mj_forward(arm.model, arm.data)
            # Several attempts, because the IK is branch-unstable: the same
            # pose yields different solutions on consecutive calls, and only
            # some branches validate. One attempt made this screen flap
            # between "reachable at rail 100" and "reachable nowhere" for an
            # identical scene.
            for _ in range(IK_ATTEMPTS_PER_RAIL):
                angles = arm.ik_solver.solve(target_m, target_rot=None)
                if angles is None:
                    continue
                if arm.validator.validate(angles, target_m,
                                          rail_pos_m=rail_m).is_valid:
                    return True
            return False
        except Exception:                                # noqa: BLE001
            return False
        finally:
            arm.data.qpos[arm.rail_jid] = saved
            mujoco.mj_forward(arm.model, arm.data)


def check_graspable(arm, obj) -> tuple:
    """(blockers, warnings, facts) for whether `obj` can be picked up at all."""
    from sim.mujoco_env import (GRASP_APERTURE_M, GRASP_APERTURE_BIO_M,
                                GRASP_MAX_AXIAL_M)
    blockers, warnings, facts = [], [], []
    try:
        width = arm.object_width_m(obj.name)
    except Exception:                                    # noqa: BLE001
        return blockers, warnings, facts

    rc, pose = arm.get_body_pose(obj.name)
    if rc == 0 and pose is not None:
        facts.append(f"{obj.name}: at ({pose[0]:.0f}, {pose[1]:.0f}, "
                     f"{pose[2]:.0f}) mm, {width*1000:.0f} mm wide")

    if obj.is_container or obj.object_type in ("bin", "rack", "instrument"):
        return blockers, warnings, facts        # containers are not picked up

    if width > GRASP_APERTURE_BIO_M:
        blockers.append(f"{obj.name} is {width*1000:.0f} mm wide; the widest "
                        f"effector spans {GRASP_APERTURE_BIO_M*1000:.0f} mm")
    elif width > GRASP_APERTURE_M:
        warnings.append(f"{obj.name} is {width*1000:.0f} mm wide -- too wide "
                        f"for the standard jaws ({GRASP_APERTURE_M*1000:.0f} "
                        f"mm); the bio gripper is required")

    # Is there anywhere the arm can stand and legally reach a grasp pose?
    if rc == 0 and pose is not None:
        import numpy as np
        grasp_z = pose[2] + GRASP_MAX_AXIAL_M * 1000.0 * 0.6
        tgt = np.array([pose[0], pose[1], grasp_z]) / 1000.0
        reachable_at = [r for r in RAIL_SAMPLES_MM
                        if _reachable_at_rail(arm, tgt, r)]
        if not reachable_at:
            # A WARNING, not a blocker, and the distinction is the point.
            # Width is provable geometry: an object wider than the jaws cannot
            # be grasped, full stop. Reachability is not provable this way --
            # this samples 8 rail positions and takes ONE IK branch at each,
            # and the controller's own IK is known to alternate branches for an
            # identical pose. Absence of a solution here is failure to find
            # one, not proof that none exists. Calling that a blocker would
            # refuse work the arm can do, which is the more expensive mistake.
            warnings.append(
                f"no grasp pose found for {obj.name} at z={grasp_z:.0f} mm "
                f"from any sampled rail position "
                f"({int(RAIL_SAMPLES_MM[0])}-{int(RAIL_SAMPLES_MM[-1])} mm) -- "
                f"likely infeasible, but this is a screen, not a proof")
        else:
            facts.append(f"{obj.name}: grasp at z={grasp_z:.0f} mm, reachable "
                         f"with rail at {', '.join(f'{r:.0f}' for r in reachable_at)} mm")
    return blockers, warnings, facts


def check_fits_container(arm, obj, container) -> list:
    """Blockers for putting `obj` into `container`."""
    try:
        w_obj = arm.object_width_m(obj.name)
        w_con = arm.object_width_m(container.name)
    except Exception:                                    # noqa: BLE001
        return []
    if w_obj >= w_con:
        return [f"{obj.name} ({w_obj*1000:.0f} mm) is not narrower than "
                f"{container.name} ({w_con*1000:.0f} mm) -- it will not go in"]
    return []


def validate_task(task: str, registry, arm) -> TaskVerdict:
    v = TaskVerdict(task=task)
    v.resolutions = resolve_referents(task, registry)

    for r in v.resolutions:
        if not r.matches:
            continue
        if len(r.matches) > 1:
            v.warnings.append(
                f"'{r.phrase}' matches {len(r.matches)}: "
                f"{', '.join(o.name for o in r.matches)}. Name one explicitly.")
            continue
        b, w, f = check_graspable(arm, r.matches[0])
        v.blockers += b; v.warnings += w; v.facts += f

    singles = [r.matches[0] for r in v.resolutions if r.ok]
    containers = [o for o in singles if o.is_container or o.object_type in ("bin", "rack")]
    movables = [o for o in singles
                if not (o.is_container or o.object_type in ("bin", "rack", "instrument"))]
    for c in containers:
        for m in movables:
            v.blockers += check_fits_container(arm, m, c)

    # The two matchers can both find the same object, so the same measured
    # fact appears twice. Dedupe while preserving order -- a report that
    # repeats itself reads as two separate observations.
    for attr in ("facts", "warnings", "blockers"):
        seen, uniq = set(), []
        for item in getattr(v, attr):
            if item not in seen:
                seen.add(item); uniq.append(item)
        setattr(v, attr, uniq)
    return v


# --------------------------------------------------------------------------
# Self-test -- no sim, no LLM, so it can gate a commit.
# --------------------------------------------------------------------------

class _FakeObj:
    def __init__(self, name, aliases, container=False, otype="cube"):
        self.name, self.aliases = name, aliases
        self.is_container, self.object_type = container, otype


class _FakeRegistry:
    def __init__(self, objs): self.objects = {o.name: o for o in objs}
    def find_all(self, q):
        q = q.lower().strip()
        for o in self.objects.values():
            if q == o.name.lower():
                return [o]
        return [o for o in self.objects.values()
                if any(q in a.lower() for a in o.aliases)]


def self_test() -> int:
    reg = _FakeRegistry([
        _FakeObj("red_cube_front", ["red cube", "red cube front"]),
        _FakeObj("red_cube_back", ["red cube", "red cube back"]),
        _FakeObj("translucent_cup", ["translucent cup", "cup"], container=True, otype="bin"),
    ])
    fails = 0

    r = resolve_referents("put the red cube in the translucent cup", reg)
    amb = [x for x in r if not x.ok]
    ok = any(x.phrase == "red cube" and len(x.matches) == 2 for x in amb)
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] ambiguous 'red cube' is reported, not guessed")

    r2 = resolve_referents("put red_cube_front in the translucent cup", reg)
    ok2 = any(x.phrase == "red_cube_front" and x.ok for x in r2)
    fails += 0 if ok2 else 1
    print(f"  [{'PASS' if ok2 else 'FAIL'}] an explicit name resolves to exactly one")

    v = TaskVerdict(task="t", blockers=["x"])
    ok3 = not v.feasible and TaskVerdict(task="t").feasible
    fails += 0 if ok3 else 1
    print(f"  [{'PASS' if ok3 else 'FAIL'}] a blocker makes the task infeasible")

    print(f"\n  {3 - fails} PASS  {fails} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scene", default="envs/lab_scene.xml")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.task:
        ap.error("need a task or --self-test")

    os.environ.setdefault("MUJOCO_GL", "egl")
    from sim.mujoco_env import SimXArmAPI
    from agent.object_registry import build_default_registry
    arm = SimXArmAPI(scene_xml=args.scene, render=False)
    try:
        reg = build_default_registry()
        try:
            reg.refresh_from_sim(arm)
        except Exception:                                # noqa: BLE001
            pass
        v = validate_task(args.task, reg, arm)
        print("\n" + v.render())
        print(f"\n  feasible: {v.feasible}")
        return 0 if v.feasible else 1
    finally:
        try:
            arm.disconnect()
        except Exception:                                # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
