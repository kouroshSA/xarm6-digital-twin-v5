#!/usr/bin/env python3
"""Layer 0: turn a general instruction into a well-formed task for Layer 1.

Every prompt passes through here, which is why it is a layer rather than a
fallback: it needs no scene knowledge, so it costs nothing to run on the easy
cases. It works on the *structure* of the instruction and on *class-level*
knowledge -- what a Falcon tube requires, what a well plate requires -- and
hands Layer 1 something worth resolving.

    user prompt
      -> Layer 0  decompose, attach class constraints, or push back   <- here
      -> Layer 1  resolve instances, measure, screen feasibility
      -> Layer 2  instance-level rewrite using Layer 1's facts
      -> planner

The division from Layer 2 is deliberate and keeps them non-redundant. Layer 0
knows about *classes* and never sees the scene. Layer 2 knows about
*instances* and needs Layer 1's measurements. "A Falcon tube must be kept
upright" is Layer 0. "The red cube you mean is red_cube_front at (0, -250)"
is Layer 2.

## Constraints, never steps

A class fact like "approach a tube from directly above" is a **constraint on
any acceptable plan** -- it describes what the task IS, because a tube grasped
side-on tips over. That belongs here. A route -- waypoints, heights, rail
positions, gripper calls -- is the planner's to choose, and it must stay free
to choose differently when its first attempt fails. Baking a route into the
task text freezes one strategy across every episode of the learning loop,
which is precisely what the loop depends on not happening.

So the contract below rejects any output naming a motion primitive, and the
skill tells the model to emit requirements rather than sequences.

## Pushing back

Layer 0 may refuse. An instruction that is ambiguous, contradictory or
meaningless is cheaper to question than to attempt. But *asking* only works
when a human is present: under `--loop` the run is autonomous, so a question
becomes a refusal that records what it would have asked, rather than blocking
forever on input nobody will type.

Note what Layer 0 can and cannot judge. It has no scene, so it can catch
SEMANTIC impossibility ("pick up the building") and self-contradiction. It
cannot catch physical infeasibility -- whether the arm can actually reach the
thing. That stays Layer 1's job, and the two should not be confused.

    python -m agent.instruction_intake --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.getcwd())

INTAKE_MODEL_DEFAULT = "claude-haiku-4-5-20251001"

#: Naming one of these means a plan has been written, not a task described.
PRIMITIVE_NAMES = (
    "move_to", "set_rail", "gripper_close", "gripper_open", "set_servo_angle",
    "set_position", "push_object", "place_tube_in_rack", "go_home",
)

STATUS_OK = "ok"
STATUS_ASK = "needs_clarification"
STATUS_REJECT = "rejected"

SYSTEM_PROMPT = """\
You prepare instructions for a robot task planner. You do not plan the task.

Given an operator's instruction, return JSON with these fields:

  status      "ok", "needs_clarification", or "rejected"
  task        the instruction, reworded to be clear and complete
  components  a list of sub-tasks if it contains more than one; else []
  constraints a list of requirements any acceptable plan must satisfy,
              drawn from what the named class of object needs
  question    when status is "needs_clarification", the single question that
              would resolve it; otherwise ""

Rules:
- Constraints describe REQUIREMENTS ("the tube must stay upright", "approach
  it from directly above"). They must never be steps, waypoints, heights,
  rail positions, or gripper commands. The planner chooses those.
- Do not invent measurements. You have not seen the scene and do not know
  where anything is or how big it is.
- Use "needs_clarification" when the instruction could mean materially
  different things, and "rejected" when it is impossible or meaningless.
  Otherwise "ok".
- If the instruction is already clear, return it unchanged with status "ok".

Output only the JSON object.
"""


@dataclass
class IntakeResult:
    original: str
    task: str
    status: str = STATUS_OK
    components: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    question: str = ""
    used: bool = False
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.status in (STATUS_ASK, STATUS_REJECT)

    def render(self) -> str:
        out = []
        if self.used and self.task != self.original:
            out.append(f"[Layer0] task: {self.task}")
        if self.components:
            out.append("[Layer0] components:")
            out += [f"          {i}. {c}" for i, c in enumerate(self.components, 1)]
        if self.constraints:
            out.append("[Layer0] constraints any plan must satisfy:")
            out += [f"          - {c}" for c in self.constraints]
        if self.question:
            out.append(f"[Layer0] question for the operator: {self.question}")
        return "\n".join(out)

    def to_task_prompt(self) -> str:
        """The text handed onward. Constraints ride WITH the task, not as a plan."""
        if not self.used:
            return self.original
        parts = [self.task]
        if self.constraints:
            parts.append("Requirements: " + "; ".join(self.constraints) + ".")
        return " ".join(parts)


def _numbers(t: str) -> set:
    return set(re.findall(r"-?\d+(?:\.\d+)?", t))


def check_contract(original: str, data: dict) -> tuple:
    """(ok, reason). No scene and no model needed."""
    if not isinstance(data, dict):
        return False, "intake did not return an object"
    status = data.get("status", STATUS_OK)
    if status not in (STATUS_OK, STATUS_ASK, STATUS_REJECT):
        return False, f"unknown status {status!r}"
    task = (data.get("task") or "").strip()
    if status == STATUS_OK and not task:
        return False, "status ok but no task returned"
    if status == STATUS_ASK and not (data.get("question") or "").strip():
        return False, "needs_clarification without a question"

    blob = " ".join([task] + list(data.get("constraints") or [])
                    + list(data.get("components") or [])).lower()

    hits = [p for p in PRIMITIVE_NAMES if p in blob]
    if hits:
        return False, f"names motion primitives, so it is a plan not a task ({hits})"

    stray = _numbers(blob) - _numbers(original)
    if stray:
        return False, (f"invented measurements Layer 0 cannot know: "
                       f"{sorted(stray)} -- it has not seen the scene")

    if task and len(task) > 12 * max(len(original), 40):
        return False, "rewrite is disproportionately long"

    # The goal must survive. Without a scene the best available check is that
    # the instruction's distinctive words are still present somewhere.
    stop = {"the", "a", "an", "in", "into", "on", "to", "and", "it", "that",
            "is", "of", "put", "please", "then", "with", "up", "from", "for"}
    key = {w for w in re.findall(r"[a-z0-9_]+", original.lower())
           if len(w) > 2 and w not in stop}
    if key and status == STATUS_OK:
        kept = {w for w in key if w in blob}
        if len(kept) < max(1, len(key) // 3):
            return False, (f"rewrite dropped most of the instruction "
                           f"(kept {sorted(kept)} of {sorted(key)})")
    return True, "contract satisfied"


def intake(task: str, model: str = INTAKE_MODEL_DEFAULT,
           call_model=None, interactive: bool = False) -> IntakeResult:
    """Run Layer 0. Falls back to the original instruction on any doubt."""
    from agent.skills import load_skills, render_skills_section

    system = SYSTEM_PROMPT + "\n\n" + render_skills_section(
        load_skills(audience="intake"))

    if call_model is None:
        def call_model(system, user):                       # noqa: ANN001
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=model, max_tokens=700,
                system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in resp.content if b.type == "text").strip()

    try:
        raw = call_model(system, task)
    except Exception as exc:                                # noqa: BLE001
        return IntakeResult(task, task, reason=
                            f"intake call failed ({type(exc).__name__}); using original")

    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0) if m else raw)
    except Exception:                                       # noqa: BLE001
        return IntakeResult(task, task, reason="intake returned unparseable JSON; using original")

    ok, reason = check_contract(task, data)
    if not ok:
        return IntakeResult(task, task, reason=f"{reason}; using original")

    return IntakeResult(
        original=task,
        task=(data.get("task") or task).strip(),
        status=data.get("status", STATUS_OK),
        components=list(data.get("components") or []),
        constraints=list(data.get("constraints") or []),
        question=(data.get("question") or "").strip(),
        used=True,
        reason=reason,
    )


# --------------------------------------------------------------------------
# Self-test -- no model, no sim.
# --------------------------------------------------------------------------

def self_test() -> int:
    orig = "pick up a falcon tube and put it in the rack"
    fails = 0

    cases = [
        ("class constraints are accepted", {
            "status": "ok", "task": "Move a Falcon tube into the rack.",
            "components": [], "question": "",
            "constraints": ["approach the tube from directly above",
                            "keep the tube upright throughout"]}, True),
        ("a plan disguised as constraints is rejected", {
            "status": "ok", "task": "Move a Falcon tube into the rack.",
            "components": [], "question": "",
            "constraints": ["move_to above the tube", "gripper_close"]}, False),
        ("invented measurements are rejected", {
            "status": "ok", "task": "Grasp the Falcon tube at 95 mm.",
            "components": [], "constraints": [], "question": ""}, False),
        ("a question without a question is rejected", {
            "status": "needs_clarification", "task": orig,
            "components": [], "constraints": [], "question": ""}, False),
        ("a genuine clarification is accepted", {
            "status": "needs_clarification", "task": orig,
            "components": [], "constraints": [],
            "question": "Which rack should the tube go into?"}, True),
        ("a rewrite that loses the instruction is rejected", {
            "status": "ok", "task": "Do the thing.",
            "components": [], "constraints": [], "question": ""}, False),
    ]
    for label, data, expect in cases:
        ok, reason = check_contract(orig, data)
        good = ok == expect
        fails += 0 if good else 1
        print(f"  [{'PASS' if good else 'FAIL'}] {label:48} ok={ok} ({reason})")

    r = intake(orig, call_model=lambda s, u: json.dumps({
        "status": "ok", "task": "Move a Falcon tube into the rack.",
        "components": [], "question": "",
        "constraints": ["approach the tube from directly above",
                        "keep the tube upright throughout"]}))
    good = r.used and "upright" in r.to_task_prompt()
    fails += 0 if good else 1
    print(f"  [{'PASS' if good else 'FAIL'}] {'constraints ride with the task downstream':48} used={r.used}")

    r2 = intake(orig, call_model=lambda s, u: "not json at all")
    good2 = (not r2.used) and r2.to_task_prompt() == orig
    fails += 0 if good2 else 1
    print(f"  [{'PASS' if good2 else 'FAIL'}] {'unparseable output falls back to the original':48} used={r2.used}")

    r3 = intake(orig, call_model=lambda s, u: json.dumps({
        "status": "needs_clarification", "task": orig, "components": [],
        "constraints": [], "question": "Which rack?"}))
    good3 = r3.blocked and r3.question == "Which rack?"
    fails += 0 if good3 else 1
    print(f"  [{'PASS' if good3 else 'FAIL'}] {'a clarification request blocks the run':48} blocked={r3.blocked}")

    total = len(cases) + 3
    print(f"\n  {total - fails} PASS  {fails} FAIL")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.task:
        ap.error("need a task or --self-test")
    # Standalone use needs the key that run_task.py normally loads for us.
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except Exception:                                       # noqa: BLE001
        pass
    r = intake(args.task)
    print(r.render() or "[Layer0] no change")
    print(f"\n  status={r.status} used={r.used} ({r.reason})")
    print(f"  downstream task: {r.to_task_prompt()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
