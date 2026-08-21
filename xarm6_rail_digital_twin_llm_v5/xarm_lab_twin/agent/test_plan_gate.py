"""Hardware-free test for the pre-action plan gate (A8 step 5).

No arm, no controller container, no Anthropic call: the LLM response is stubbed
and the validator is a plain function. What is being pinned down is the one
property that matters on real hardware —

    when the gate rejects a plan, NOTHING is dispatched.

Checking per-command during execution would be worse than useless on a physical
arm: the prefix would already have run before the bad command was reached.

    python -m agent.test_plan_gate
"""
from __future__ import annotations

import json
import sys
import types


class _StubResponse:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(text=text)]
        self.usage = types.SimpleNamespace(input_tokens=0, output_tokens=0)


class _StubMessages:
    def __init__(self, text):
        self._text = text

    def create(self, **kwargs):
        return _StubResponse(self._text)


class _StubClient:
    def __init__(self, text):
        self.messages = _StubMessages(text)


class _StubArm:
    """Minimal arm: records every dispatch so we can assert none happened."""
    scene_xml = "envs/lab_scene.xml"

    def __init__(self):
        self.calls = []

    def _record(self, name):
        def f(*a, **k):
            self.calls.append(name)
            return 0
        return f

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._record(name)


PLAN = [
    {"action": "set_rail", "params": {"position_mm": 350, "speed_mm_s": 100}},
    {"action": "move_to", "params": {"x": 2000, "y": 0, "z": 400, "roll": 180,
                                     "pitch": 0, "yaw": 0, "speed_mm_s": 50}},
    {"action": "done", "params": {"message": "x"}},
]


def _make_brain(arm):
    from agent.llm_brain import LLMBrain
    from agent.object_registry import build_default_registry
    brain = LLMBrain(arm=arm, registry=build_default_registry(), recorder=None)
    brain.client = _StubClient(json.dumps(PLAN))
    return brain


def test_rejected_plan_dispatches_nothing():
    arm = _StubArm()
    brain = _make_brain(arm)
    brain.plan_validator = lambda cmds: (False, ["#1 move_to: pose unreachable"])
    out = brain.execute_task("put the front red cube in the green bin")
    if not out.get("rejected"):
        return "FAIL  rejected plan was not flagged as rejected"
    if out["results"]:
        return f"FAIL  rejected plan still produced results: {out['results']}"
    motion = [c for c in arm.calls if c not in ("get_body_pose", "get_position")]
    if motion:
        return f"FAIL  rejected plan still dispatched: {motion}"
    return "PASS  rejected plan dispatched nothing"


def test_accepted_plan_runs():
    arm = _StubArm()
    brain = _make_brain(arm)
    brain.plan_validator = lambda cmds: (True, [])
    out = brain.execute_task("put the front red cube in the green bin")
    if out.get("rejected"):
        return "FAIL  accepted plan was rejected"
    if not out["results"]:
        return "FAIL  accepted plan produced no results"
    return f"PASS  accepted plan ran ({len(out['results'])} commands dispatched)"


def test_no_validator_is_unchanged():
    """Sim runs attach no gate; behaviour must be exactly as before."""
    arm = _StubArm()
    brain = _make_brain(arm)
    out = brain.execute_task("put the front red cube in the green bin")
    if out.get("rejected"):
        return "FAIL  plan rejected with no validator attached"
    if not out["results"]:
        return "FAIL  no commands dispatched with no validator attached"
    return "PASS  no validator attached -> unchanged behaviour"


TESTS = [test_rejected_plan_dispatches_nothing,
         test_accepted_plan_runs,
         test_no_validator_is_unchanged]


def main() -> int:
    failures = 0
    for fn in TESTS:
        try:
            line = fn()
        except Exception as exc:  # noqa: BLE001
            line = f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}"
        if line.startswith("FAIL"):
            failures += 1
        print("  " + line)
    print(f"\n  {len(TESTS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
