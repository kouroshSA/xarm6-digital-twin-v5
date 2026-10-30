"""A requested displacement is graded with a tolerance, not as a literal.

Run: ``python -m agent.test_move_grading``  (no sim, no model)

The per-session grader wrote "cucumber moved (150" for "put it down 150 mm to
the right", which as a substring passes only an x displacement whose digits
begin 150 -- a correct placement 147 mm away fails, and 1500 mm passes. The
checks are on check_outcome's verdict for real physical_outcome() strings from
the live runs, i.e. the consumer of the criterion.
"""
from __future__ import annotations

import sys

from agent.outcome_checker import check_outcome

TASK = "pick up the cucumber and put it down 150 mm to the right"
FULL = ("any", ["cucumber moved (150, 0)mm"])
PARTIAL = ("any", ["cucumber moved (150"])           # what the grader actually wrote

CASES = [
    ("147 mm with a little drift passes", FULL,
     "cucumber moved (147, -6)mm; cucumber closer to cupcake", True),
    ("the partial form is read as numbers too", PARTIAL,
     "cucumber moved (147, -6)mm; cucumber closer to cupcake", True),
    ("the dropped cucumber slid 39 mm across: fails", FULL,
     "cucumber moved (153, -39)mm; cucumber closer to cupcake", False),
    ("moved toward the front instead: fails", FULL,
     "cucumber moved (-1, -161)mm; cucumber farther from cupcake", False),
    ("1500 mm no longer passes on its leading digits", PARTIAL,
     "cucumber moved (1500, 0)mm", False),
    ("another object moving 150 mm does not count", FULL,
     "cupcake moved (150, 0)mm", False),
    ("nothing moved", FULL, "no objects displaced", False),
    ("an exact literal still matches as before", ("any", ["cucumber moved"]),
     "cucumber moved (12, 3)mm", True),
]

if __name__ == "__main__":
    failed = 0
    for label, spec, physical, want in CASES:
        got, why = check_outcome(TASK, physical, fallback_spec=spec)
        ok = got is want
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f": {got} ({why})"))
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)
