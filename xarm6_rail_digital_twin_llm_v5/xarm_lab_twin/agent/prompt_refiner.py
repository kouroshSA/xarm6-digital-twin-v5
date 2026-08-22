#!/usr/bin/env python3
"""Layer 2: rewrite an instruction into an unambiguous task statement.

Runs AFTER Layer 1, never before. Layer 1 is what produces ground truth --
which objects the words resolve to, where they are, how wide they are. A
refiner invoked before that has nothing true to write with and can only
paraphrase, and a paraphraser asked to "clarify the red cube" will pick one by
guessing. Guessing upstream of everything is the worst place to guess.

    user prompt
      -> Layer 1   resolve + measure          (deterministic; facts)
      -> Layer 2   rewrite using those facts  (this module; only when needed)
      -> Layer 1   re-validate the rewrite    (cheap; catches invention)
      -> planner

The re-validation is the point. "Every number must come from the registry" is
a guardrail that only means something if something checks it, and Layer 1
already knows how. If the rewrite fails the contract, the ORIGINAL prompt is
used. Fail-safe, not fail-open: a refiner that cannot be trusted is simply
skipped, and the run continues exactly as it would have without it.

## When it runs

Only when Layer 1 reports something unresolved or ambiguous. A prompt that
already resolves cleanly is left alone -- rewriting it adds risk and buys
nothing. That also keeps the fast path free of an API call.

    python -m agent.prompt_refiner --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.getcwd())

REFINER_MODEL_DEFAULT = "claude-haiku-4-5-20251001"

#: Words that turn a task statement into a plan. The refiner is told not to
#: write steps; this checks that it did not.
IMPERATIVE_MARKERS = (
    "step 1", "step 2", "first,", "then ", "next,", "finally,",
    "move_to", "set_rail", "gripper_close", "gripper_open",
    "approach height", "clearance height", "waypoint",
)


@dataclass
class RefineResult:
    original: str
    refined: str
    used: bool
    reason: str

    @property
    def task(self) -> str:
        return self.refined if self.used else self.original


def _numbers(text: str) -> set:
    return set(re.findall(r"-?\d+(?:\.\d+)?", text))


def check_contract(original: str, refined: str, verdict_before, verdict_after) -> tuple:
    """(ok, reason). Everything here is checkable without a model.

    1. Resolution must not get worse -- the rewrite may collapse an ambiguity
       but may not introduce one, and may not invent an object.
    2. Every number must already appear in Layer 1's measured facts or in the
       operator's own words. A number the refiner produced from nowhere reads
       as authoritative and will be acted on.
    3. It must still be a task statement, not a plan.
    """
    if not refined.strip():
        return False, "refiner returned nothing"
    if len(refined) > 8 * max(len(original), 40):
        return False, "rewrite is disproportionately long; likely a plan"

    before = {r.phrase: {o.name for o in r.matches} for r in verdict_before.resolutions}
    after = {r.phrase: {o.name for o in r.matches} for r in verdict_after.resolutions}
    names_before = set().union(*before.values()) if before else set()
    names_after = set().union(*after.values()) if after else set()
    invented = names_after - names_before
    if invented:
        return False, f"rewrite introduced objects Layer 1 never resolved: {sorted(invented)}"
    if names_before and not names_after:
        return False, "rewrite resolves to no objects at all"

    # Every referent the original had must still be represented. Comparing
    # name SETS is not enough: a rewrite naming something that resolves to
    # nothing at all (a hallucinated object) shrinks the after-set silently
    # and looks like a clean subset. Requiring each original group to survive
    # catches that, while still permitting the legitimate case -- collapsing
    # {red_cube_front, red_cube_back} down to one of them.
    for phrase, group in before.items():
        if group and not (group & names_after):
            return False, (f"rewrite dropped the referent for '{phrase}' "
                           f"(expected one of {sorted(group)}) -- it may name "
                           f"something that does not exist in the scene")

    amb_after = [p for p, s in after.items() if len(s) > 1]
    amb_before = [p for p, s in before.items() if len(s) > 1]
    if len(amb_after) > len(amb_before):
        return False, f"rewrite is MORE ambiguous than the original ({amb_after})"

    allowed = _numbers(original) | _numbers("\n".join(verdict_before.facts))
    stray = _numbers(refined) - allowed
    if stray:
        return False, (f"rewrite contains numbers Layer 1 never measured: "
                       f"{sorted(stray)}")

    low = refined.lower()
    hits = [m for m in IMPERATIVE_MARKERS if m in low]
    if hits:
        return False, f"rewrite reads as a plan, not a task statement ({hits})"

    return True, "contract satisfied"


def build_refiner_prompt(task: str, verdict) -> str:
    lines = [f"Operator instruction:\n{task}\n"]
    if verdict.facts:
        lines.append("Measured facts (use these; do not invent others):")
        lines += [f"  - {f}" for f in verdict.facts]
        lines.append("")
    amb = [r for r in verdict.resolutions if len(r.matches) > 1]
    if amb:
        lines.append("Ambiguous references you must resolve:")
        for r in amb:
            lines.append(f"  - '{r.phrase}' could be: "
                         f"{', '.join(o.name for o in r.matches)}")
        lines.append("")
    lines.append("Rewrite the instruction as one or two sentences naming exactly "
                 "what should end up where. Output only the rewritten "
                 "instruction, with no preamble.")
    return "\n".join(lines)


def refine(task: str, registry, arm, model: str = REFINER_MODEL_DEFAULT,
           call_model=None) -> RefineResult:
    """Refine `task` if Layer 1 says it needs it. Falls back to the original."""
    from agent.task_validator import validate_task
    from agent.skills import load_skills, render_skills_section

    before = validate_task(task, registry, arm)
    needs = [r for r in before.resolutions if len(r.matches) > 1]
    if not needs:
        return RefineResult(task, task, False,
                            "nothing ambiguous; left unchanged")

    system = ("You rewrite robot task instructions so they cannot be "
              "misread.\n\n" + render_skills_section(load_skills(audience="refiner")))
    user = build_refiner_prompt(task, before)

    if call_model is None:
        def call_model(system, user):                       # noqa: ANN001
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model, max_tokens=400,
                system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in resp.content if b.type == "text").strip()

    try:
        refined = call_model(system, user)
    except Exception as exc:                                # noqa: BLE001
        return RefineResult(task, task, False,
                            f"refiner call failed ({type(exc).__name__}); using original")

    after = validate_task(refined, registry, arm)
    ok, reason = check_contract(task, refined, before, after)
    return RefineResult(task, refined, ok, reason)


# --------------------------------------------------------------------------
# Self-test -- no model, no sim.
# --------------------------------------------------------------------------

class _Obj:
    def __init__(self, name, aliases, container=False, otype="cube"):
        self.name, self.aliases = name, aliases
        self.is_container, self.object_type = container, otype


class _Reg:
    def __init__(self, objs): self.objects = {o.name: o for o in objs}
    def find_all(self, q):
        q = q.lower().strip()
        for o in self.objects.values():
            if q == o.name.lower():
                return [o]
        return [o for o in self.objects.values()
                if any(q in a.lower() for a in o.aliases)]


def self_test() -> int:
    from agent.task_validator import validate_task

    class _NoArm:
        def object_width_m(self, n): raise RuntimeError("no sim")
        def get_body_pose(self, n): return 1, None

    reg = _Reg([_Obj("red_cube_front", ["red cube", "red cube front"]),
                _Obj("red_cube_back", ["red cube", "red cube back"]),
                _Obj("translucent_cup", ["translucent cup", "cup"],
                     container=True, otype="bin")])
    arm = _NoArm()
    orig = "put the red cube in the translucent cup"
    before = validate_task(orig, reg, arm)
    fails = 0

    cases = [
        ("a clean disambiguation is accepted",
         "put red_cube_front in the translucent cup", True),
        ("an invented object is rejected",
         "put the purple_widget in the translucent cup", False),
        ("a rewrite that is still ambiguous is not an improvement",
         "put the red cube in the translucent cup", True),
        ("a fabricated number is rejected",
         "put red_cube_front at 812 into the translucent cup", False),
        ("a rewrite that is really a plan is rejected",
         "First, move_to the red_cube_front then gripper_close and lift", False),
    ]
    for label, refined, expect in cases:
        after = validate_task(refined, reg, arm)
        ok, reason = check_contract(orig, refined, before, after)
        good = ok == expect
        fails += 0 if good else 1
        print(f"  [{'PASS' if good else 'FAIL'}] {label:52} ok={ok} ({reason})")

    r = refine(orig, reg, arm, call_model=lambda s, u: "put red_cube_front in the translucent cup")
    good = r.used and r.task == "put red_cube_front in the translucent cup"
    fails += 0 if good else 1
    print(f"  [{'PASS' if good else 'FAIL'}] {'refine() accepts a good rewrite':52} used={r.used}")

    r2 = refine(orig, reg, arm, call_model=lambda s, u: "put the purple_widget in the cup")
    good2 = (not r2.used) and r2.task == orig
    fails += 0 if good2 else 1
    print(f"  [{'PASS' if good2 else 'FAIL'}] {'refine() falls back to the original':52} used={r2.used}")

    r3 = refine("put red_cube_front in the translucent cup", reg, arm,
                call_model=lambda s, u: "SHOULD NOT BE CALLED")
    good3 = (not r3.used) and "nothing ambiguous" in r3.reason
    fails += 0 if good3 else 1
    print(f"  [{'PASS' if good3 else 'FAIL'}] {'an unambiguous prompt skips the model call':52} used={r3.used}")

    total = len(cases) + 3
    print(f"\n  {total - fails} PASS  {fails} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    ap.error("only --self-test is supported standalone; use run_task.py --refine-prompt")


if __name__ == "__main__":
    sys.exit(main())
