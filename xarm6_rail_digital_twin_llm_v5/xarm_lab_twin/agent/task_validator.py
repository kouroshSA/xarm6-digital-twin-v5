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

#: Extra height for CARRYING over a target, above the height for releasing onto
#: it. Measured on the bench: carrying at 870 clipped a target whose release
#: height is 867 and shoved it 18 mm; 875 was the first height that cleared.
#: 25 mm keeps a margin instead of sitting on the measured edge.
CARRY_CLEARANCE_MM = 25.0

#: How far below the tool a held object's centre sits. Derived from the same
#: expression the grasp height uses (centre + GRASP_MAX_AXIAL_M*0.6), so the
#: grasp fact and the placement fact cannot drift apart: if you grasp an object
#: by putting the tool this far above its centre, you place it by putting the
#: tool this far above where its centre must end up.
def _held_centre_below_tool_mm() -> float:
    from sim.mujoco_env import GRASP_MAX_AXIAL_M
    return GRASP_MAX_AXIAL_M * 1000.0 * 0.6


def _underside_below_tool_mm(arm, obj) -> float:
    """How far below the tool a held object's UNDERSIDE sits.

    Measured, not assumed: with the tool at 862 a grasped red cube spanned
    806..868, so its underside was 56 mm down -- the tool-to-centre offset
    plus half the object's height. Modelling it as offset + half of a
    NOMINAL 30 mm cube gave 42 mm, and that 14 mm error is exactly why a
    carry at 870 grazed a bin top at 810 while the arithmetic said it cleared
    by 18 mm. Clearance is about the underside, so compute it from the
    object's real extent.
    """
    lo_m, hi_m = arm.object_z_extent_m(obj.name)
    half_mm = (hi_m - lo_m) * 1000.0 / 2.0
    return _held_centre_below_tool_mm() + half_mm


def check_container_placement(arm, mover, container, obstacles) -> list:
    """Measured facts for putting `mover` INTO `container`.

    Two numbers, and the binding one is not the obvious one. Dropping into a
    115 mm cup is forgiving -- releases from 845 mm upward all landed the cube
    inside. Getting there is not: the traverse crosses other bench furniture,
    and a carry height chosen by the planner (870) was refused for grazing a
    bin, halting the task before the cup was ever reached.
    """
    try:
        under = _underside_below_tool_mm(arm, mover)
        _, rim_m = arm.object_z_extent_m(container.name)
    except Exception:                                    # noqa: BLE001
        return []
    rim_mm = rim_m * 1000.0
    release_mm = rim_mm + under

    tallest, tallest_name = rim_mm, container.name
    for o in obstacles:
        if o.name in (mover.name, container.name):
            continue
        try:
            _, top_m = arm.object_z_extent_m(o.name)
        except Exception:                                # noqa: BLE001
            continue
        if top_m * 1000.0 > tallest:
            tallest, tallest_name = top_m * 1000.0, o.name
    carry_mm = tallest + under + CARRY_CLEARANCE_MM

    return [f"to put {mover.name} in {container.name}: carry it at "
            f"z>={carry_mm:.0f} mm across the bench, then release at "
            f"z>={release_mm:.0f} mm once over the container "
            f"({container.name}'s rim is at z={rim_mm:.0f} mm and the tallest "
            f"thing on the route is {tallest_name} at z={tallest:.0f} mm; a "
            f"held {mover.name}'s underside hangs {under:.0f} mm below the tool)"]


def check_placement(arm, mover, target) -> list:
    """Measured fact: where to release `mover` so it lands on top of `target`.

    Layer 1 reported grasp heights and nothing else, so a task like "put the
    blue cube on top of the red cube" left the planner to invent a release
    height. It chose one below the target's top surface, drove the carried
    cube into it and knocked it flat -- repeatedly.

    This is geometry, not a primitive: target's top surface, plus half the
    carried object's height, plus the offset between the tool and a held
    object's centre. Verified against the sim -- the computed 867 mm is
    exactly the lowest release that produces a stack; 855 knocks the target
    over.
    """
    try:
        _, target_top_m = arm.object_z_extent_m(target.name)
        lo_m, hi_m = arm.object_z_extent_m(mover.name)
    except Exception:                                    # noqa: BLE001
        return []
    half_mover_mm = (hi_m - lo_m) * 1000.0 / 2.0
    release_mm = target_top_m * 1000.0 + half_mover_mm + _held_centre_below_tool_mm()
    # Carrying needs MORE height than releasing. At the release height the
    # carried object's underside sits exactly on the target's top surface, so
    # travelling at that height drags it across the target instead of over it.
    # Measured: carrying at 870 clipped the target and shoved it 18 mm; 875 was
    # the first height that cleared. CARRY_CLEARANCE_MM keeps a real margin
    # rather than sitting on the measured edge.
    carry_mm = release_mm + CARRY_CLEARANCE_MM
    # The re-grasp height is the SAME number, and it has to be said, because
    # the grasp height Layer 1 reports for the mover is measured where the
    # mover is standing NOW -- on the bench. A task with a later leg that
    # picks it back off the target is asking about a position that will not
    # exist until the first leg runs, so the planner guesses. It guessed 885
    # against a correct 867, and 18 mm too high matters more than it looks:
    # gripper_close welds whatever tool-to-object pose exists at that instant,
    # so a high grasp is frozen as a LOW hold. Measured on this scene --
    # re-grasp at 867 holds the cube 25 mm below the tool and the next descent
    # is accepted; at 885 it holds it 43 mm below and the descent is refused
    # for driving the cargo into the benchtop.
    return [f"to place {mover.name} on top of {target.name}: carry it at "
            f"z>={carry_mm:.0f} mm until it is over {target.name}, then release "
            f"with the tool at z={release_mm:.0f} mm "
            f"({target.name}'s top surface is at z={target_top_m*1000:.0f} mm; "
            f"travelling at the release height drags the carried object "
            f"through the target instead of over it)",
            f"to pick {mover.name} back off {target.name} afterwards: grasp it "
            f"with the tool at z={release_mm:.0f} mm, the same height it was "
            f"released at -- the grasp height reported for {mover.name} above "
            f"is where it stands now, not where it will be once stacked. "
            f"Grasping higher still catches it, but freezes a lower hold, and "
            f"the carried object then collides on the next descent"]


def _bench_top_mm(arm) -> float:
    """World z of the benchtop surface, read from the scene.

    Not a constant here. The bench height already lives in the scene XML, and
    a second copy in this file is the repo's oldest failure mode -- it starts
    equal and drifts.
    """
    gid = arm.model.geom("bench_top").id
    half_z = float(arm.model.geom_size[gid][2])
    return (float(arm.data.geom_xpos[gid][2]) + half_z) * 1000.0


def check_bench_placement(arm, mover) -> list:
    """Measured fact: where to release `mover` so it lands on open benchtop.

    Every placement fact this module produced was "on top of object X". A task
    whose leg is "put it back where it started" is a placement onto the bench
    itself, and for that there was no fact at all -- so the planner invented
    the number, and invented a different one each run. Across five runs of one
    task it chose 830 three times (works) and 807 twice, and 807 is exactly
    the height at which a cargo held at the nominal offset grazes the bench.

    Verified against the sim: with a nominal grasp, releases from 800 to 840
    all leave the cube resting at z=780 -- so this is a wide target, and the
    planner was missing it only because nobody handed it the range.
    """
    try:
        lo_m, hi_m = arm.object_z_extent_m(mover.name)
        bench_mm = _bench_top_mm(arm)
    except Exception:                                    # noqa: BLE001
        return []
    half_mover_mm = (hi_m - lo_m) * 1000.0 / 2.0
    release_mm = bench_mm + half_mover_mm + _held_centre_below_tool_mm()
    carry_mm = release_mm + CARRY_CLEARANCE_MM
    return [f"to set {mover.name} back down on the open benchtop: carry it at "
            f"z>={carry_mm:.0f} mm, then release with the tool at "
            f"z={release_mm:.0f} mm or a little above; it falls the last short "
            f"distance (the benchtop surface is at z={bench_mm:.0f} mm). Do "
            f"not descend below that release height while still holding it -- "
            f"the carried object reaches lower than the tool does, and the "
            f"move is refused for driving it into the bench"]




#: How much of an alias an n-gram must cover to count as naming it. Substring
#: matching alone is far too eager: "the rail" is a substring of the alias
#: "red cube behind the rail", so a prompt mentioning the rail resolved to a
#: red cube, invented a fact about it, and then made Layer 2's contract reject
#: a perfectly good rewrite for "dropping" a referent that was never real.
NGRAM_ALIAS_COVERAGE = 0.5


#: The one place the Layer 1 fact block's opening marker is spelled. Both the
#: producer (run_task's single-shot and --loop task builders) and the consumer
#: (LLMBrain.prepare_for_task, which must strip it back off) import this rather
#: than repeating the literal -- two copies of a separator drift the same way
#: two copies of a joint limit do, and here a drifted copy silently disables
#: the stripping rather than raising.
SCENE_FACTS_MARKER = " Measured from the scene: "


def append_scene_facts(task: str, facts) -> str:
    """Ride Layer 1's measurements along with the task, for the planner."""
    if not facts:
        return task
    return f"{task}{SCENE_FACTS_MARKER}" + "; ".join(facts) + "."


def strip_scene_facts(task: str) -> str:
    """Recover the task as the operator wrote it.

    Layer 1's facts name every object it resolved, so anything that keyword-
    scans the task string reads those names as if the operator had typed
    them. That is how "put it back down at its original position on the
    bench" equipped the bio-gripper: 'on the bench' resolved to well_plate_B,
    whose measurements were appended to the task, and the gripper scan saw
    'plate'. The facts are for the planner; every other consumer wants the
    original.
    """
    return task.split(SCENE_FACTS_MARKER, 1)[0] if task else task



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

    # Objects the n-gram pass above already pinned down. Used to suppress a
    # short, ambiguous alias that adds nothing: "blue" is an alias of
    # blue_cube, but it also appears in blue_bin's and three blue-capped
    # tubes' aliases, so it resolved 5 ways and dragged all five into the
    # fact list. A two-object task emitted ~15 facts that way -- more text
    # with no more signal, which is exactly what measurably hurt Layer 0.
    already = {o.name for r in out for o in r.matches}

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
            # Drop a single-word alias that resolves several ways when a
            # longer phrase has already resolved one of them. "blue cube"
            # having pinned blue_cube, "blue" adds only ambiguity. A
            # single-word alias that resolves several ways and pins NOTHING is
            # kept -- there the ambiguity is the real answer and the operator
            # needs to see it.
            if (len(matches) > 1 and len(a.split()) == 1
                    and any(o.name in already for o in matches)):
                continue
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
        # The grasp height is GEOMETRY -- object centre plus jaw depth -- so it
        # is always computable and always worth telling the planner. It used to
        # be reported only when the reachability sweep below also succeeded,
        # and that sweep is flaky because the controller's IK alternates
        # branches. The single most useful number was therefore delivered
        # intermittently: watching a live run, the planner got no grasp height,
        # guessed 795 mm for a block whose top is at 810, and drove the gripper
        # into it. Report it unconditionally; reachability is reported apart.
        facts.append(f"{obj.name}: grasp at z={grasp_z:.0f} mm "
                     f"(jaws reach ~{GRASP_MAX_AXIAL_M*1000:.0f} mm below the "
                     f"tool, so descending lower drives the gripper into it)")

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
            facts.append(f"{obj.name}: reachable with the rail at "
                         f"{', '.join(f'{r:.0f}' for r in reachable_at)} mm")
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
            # Still measure EVERY candidate. Ambiguity about which object is
            # meant is not ambiguity about their geometry, and skipping this
            # left the planner with no grasp height at all for "the red cube"
            # -- so it guessed, exactly as it did before facts were wired in.
            # Measured: the two red-cube tasks in the 2026-08-22 A/B trailed
            # the unambiguous blue-block one on BOTH arms, and this is why.
            #
            # Facts only. A blocker found on one candidate must not refuse the
            # task, because the operator may well have meant the other one.
            for cand in r.matches:
                _b, w, f = check_graspable(arm, cand)
                v.facts += f
                # Keep the WARNINGS too. Dropping them meant Layer 1 reported
                # "tube_L2: grasp at z=844" while silently discarding its own
                # finding that no reachable grasp pose exists there -- the
                # descent is blocked at 870 by the gripper hitting the tube
                # cap. The planner was handed a height it could not achieve
                # and no hint that it could not, so it guessed 920 and failed.
                # Blockers stay suppressed: one candidate being unreachable
                # must not refuse a task that meant the other.
                v.warnings += w
            continue
        b, w, f = check_graspable(arm, r.matches[0])
        v.blockers += b; v.warnings += w; v.facts += f

    singles = [r.matches[0] for r in v.resolutions if r.ok]
    containers = [o for o in singles if o.is_container or o.object_type in ("bin", "rack")]

    # EVERY candidate, not just unambiguously-resolved ones. check_graspable
    # was already fixed to measure all candidates of an ambiguous referent;
    # this list was not, so placement heights were withheld exactly when the
    # referent was ambiguous. Measured consequence: on "place the blue cube on
    # top of the red cube" -- where "the red cube" matches two -- the planner
    # got the grasp height (807, which it used correctly) and no placement
    # height, guessed 820, and drove the carried cube into the target block.
    # Every attempt. The number it needed existed and was not shown to it.
    candidates = []
    for r in v.resolutions:
        for o in r.matches:
            if o not in candidates:
                candidates.append(o)

    # Rank by how explicitly the task names each object, because everything
    # downstream is CAPPED -- two objects get bench-placement facts, four
    # ordered pairs get stacking facts -- and a cap over an unranked list
    # spends its budget on whatever resolved first. On "put the blue cube on
    # red_cube_front, then put it back on the bench", an incidental phrase
    # resolved a well plate, and the plate and red cube consumed every slot:
    # six placement facts, not one of them about blue_cube, which is the only
    # object the task moves. The facts existed, were correct, and described
    # the wrong things.
    _low_task = task.lower()

    def _explicitness(o):
        if o.name.lower() in _low_task:
            return 0                      # named outright
        if any(a.lower() in _low_task for a in o.aliases):
            return 1                      # named by an alias
        return 2                          # pulled in by something incidental

    candidates.sort(key=_explicitness)    # stable: ties keep resolution order
    movables = [o for o in candidates
                if not (o.is_container or o.object_type in ("bin", "rack", "instrument"))]
    for c in containers:
        for m in movables:
            v.blockers += check_fits_container(arm, m, c)

    # Placement heights for stacking one named object on another. Capped:
    # every ordered pair would be O(n^2) facts, and the planner only needs the
    # ones it was actually asked about.
    # Facts must cover what the task NAMES, and not much else. Capping the
    # container facts at the first three movables meant that on "put the red
    # cube in the cup, then put the blue tube in the cup" the tube -- one of
    # the two objects actually named -- got no placement guidance, while four
    # stacking facts were emitted for a task that never mentions stacking. The
    # planner had 21 facts, none of them the one it needed, and guessed.
    for c in containers:
        for mvr in movables:
            v.facts += check_container_placement(arm, mvr, c, candidates)

    low = task.lower()

    # Bench placement, on the same terms as the stacking facts: only when the
    # instruction implies setting something down on the bench rather than into
    # or onto a named thing. "Put it back" is the common form and it names no
    # target at all, which is precisely why it went unmeasured.
    if any(k in low for k in ("back down", "back on", "put it back",
                              "original position", "where it started",
                              "back where", "on the bench", "on the benchtop",
                              "return it", "returned to its")):
        for mvr in movables[:2]:
            v.facts += check_bench_placement(arm, mvr)

    # Stacking facts only when the instruction actually implies stacking.
    # Otherwise they are noise, and noise measurably costs accuracy here.
    if any(k in low for k in ("on top", "stack", "onto", " atop")):
        placements = 0
        for mover in movables:
            for target in movables:
                if mover.name == target.name or placements >= 4:
                    continue
                v.facts += check_placement(arm, mover, target)
                placements += 1

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

    reg2 = _FakeRegistry([
        _FakeObj("blue_cube", ["blue cube", "blue"]),
        _FakeObj("blue_bin", ["blue bin", "blue"], container=True, otype="bin"),
        _FakeObj("tube_B1", ["blue capped tube", "blue"]),
    ])
    r4 = resolve_referents("put the blue cube in the blue bin", reg2)
    phrases = {x.phrase for x in r4}
    ok4 = "blue" not in phrases and "blue cube" in phrases
    fails += 0 if ok4 else 1
    print(f"  [{'PASS' if ok4 else 'FAIL'}] a short ambiguous alias is dropped once a "
          f"longer phrase resolved   {sorted(phrases)}")

    # ...but an ambiguous short alias that pins NOTHING must still be reported:
    # there the ambiguity IS the answer and the operator has to see it.
    r5 = resolve_referents("grab the blue one", reg2)
    ok5 = any(x.phrase == "blue" and len(x.matches) == 3 for x in r5)
    fails += 0 if ok5 else 1
    print(f"  [{'PASS' if ok5 else 'FAIL'}] an ambiguous alias that pins nothing is "
          f"still reported            {[(x.phrase, len(x.matches)) for x in r5]}")

    v = TaskVerdict(task="t", blockers=["x"])
    ok3 = not v.feasible and TaskVerdict(task="t").feasible
    fails += 0 if ok3 else 1
    print(f"  [{'PASS' if ok3 else 'FAIL'}] a blocker makes the task infeasible")

    print(f"\n  {5 - fails} PASS  {fails} FAIL")
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
