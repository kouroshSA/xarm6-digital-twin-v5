# agent/outcome_checker.py
"""
Outcome checker: parse a task prompt to determine the expected physical
state, then compare to physical_outcome() to decide if the episode succeeded.

This is best-effort, not exhaustive. It handles the most common task patterns:
  - "put/place/move <color> cube in <color> bin"   -> expect "<color>_cube in <color>_bin"
  - "push <object> off the bench/edge/table"        -> expect "<object> fell to floor" OR "off bench"
  - "put all cubes in <color> bin"                  -> expect all three "*_cube in <color>_bin"
  - "sort the cubes [by color]"                     -> expect each cube in matching bin

For tasks the parser doesn't recognise, returns None (success unknown -- the
loop should fall back to command-level success, or ask the user).
"""
import re
from typing import Optional, List, Tuple

COLORS = ("red", "green", "blue")


def expected_outcome(task: str) -> Optional[List[str]]:
    """
    Return a list of substrings that ALL must appear in physical_outcome()
    for the task to count as a success. Returns None if the parser can't
    determine an expectation (caller should fall back to command-level check).
    """
    t = task.lower()

    # "push X off the bench/edge/table/floor" -> X should fall or be off bench
    if re.search(r"\b(push|knock|drop|shove|slide).*\b(off|edge|floor|away)\b", t):
        # Find the target object
        for color in COLORS:
            if f"{color} cube" in t or f"{color}_cube" in t:
                return [f"{color}_cube fell to floor", f"{color}_cube off bench"]  # either is OK
        # Generic "clear the table" / "everything off"
        if re.search(r"\b(all|everything|clear)\b", t):
            return ["fell to floor"]  # at least something fell
        return None

    # "put/place all cubes in <color> bin" -> all three cubes in that bin
    if re.search(r"\b(all|every|each).*\bcubes?\b", t):
        for color in COLORS:
            if f"{color} bin" in t:
                return [f"{c}_cube in {color}_bin" for c in COLORS]
        return None

    # "sort the cubes [by color]" -> each in its matching bin
    if re.search(r"\bsort.*cubes?\b", t):
        return [f"{c}_cube in {c}_bin" for c in COLORS]

    # "put/place/move <color> cube in <color> bin" (canonical pick-and-place)
    m = re.search(r"\b(red|green|blue)\s*(cube|block)\b.*\b(red|green|blue)\b.*\b(bin|container|box)\b", t)
    if m:
        src = m.group(1)
        dst = m.group(3)
        return [f"{src}_cube in {dst}_bin"]

    return None  # Unknown task pattern


def check_outcome(task: str, physical: str) -> Tuple[Optional[bool], str]:
    """
    Compare expected outcome (from task) to actual physical_outcome() string.

    Returns:
        (success, reason)
        success: True/False/None (None = couldn't determine)
        reason:  human-readable explanation
    """
    expected = expected_outcome(task)
    if expected is None:
        return (None, f"Task pattern not recognised; physical: {physical or 'no displacement'}")

    # For "push off" tasks, ANY of the substrings counts
    if any("fell to floor" in e or "off bench" in e for e in expected):
        if any(e in physical for e in expected):
            return (True, f"Met physical condition: {physical}")
        return (False, f"Expected one of {expected}; got: {physical or 'no displacement'}")

    # For placement tasks, ALL expected substrings must appear
    missing = [e for e in expected if e not in physical]
    if not missing:
        return (True, f"All expected placements present: {physical}")
    return (False, f"Missing: {missing}; got: {physical or 'no displacement'}")
