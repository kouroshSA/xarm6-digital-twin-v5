"""The planner's reply -> command list.

Run: ``python -m agent.test_plan_parse``  (no sim, no model)

A live loop run lost two episodes to "Extra data": Sonnet reasoned before its
plan ("it spans [25, 77] ..."), the greedy first-"[" to last-"]" fallback began
in the prose, and a valid plan was discarded -- the episode then graded an
untouched scene as a planning failure. Replies that parsed before must parse
the same way; this only rescues ones that raised.
"""
from __future__ import annotations

import json
import sys

from agent.llm_brain import extract_command_list as parse

PLAN = [{"action": "gripper_open", "params": {}},
        {"action": "done", "params": {"message": "placed"}}]
TEXT = json.dumps(PLAN)

CASES = [
    ("bare plan", TEXT, PLAN),
    ("prose with brackets before the plan",
     "The cucumber spans [25, 77] and the cupcake [82, 157].\n\n" + TEXT, PLAN),
    ("prose after the plan", TEXT + "\n\nThat keeps clear of [the cupcake].", PLAN),
    ("a revised plan after a draft keeps the last",
     json.dumps(PLAN[:1]) + "\nActually:\n" + TEXT, PLAN),
    ("no JSON at all", "I cannot plan this.", None),
    ("old greedy behaviour kept where it parsed", "values [1, 2]", [1, 2]),
]

if __name__ == "__main__":
    failed = 0
    for label, raw, want in CASES:
        got = parse(raw)
        ok = got == want
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f": got {got!r}"))
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)
