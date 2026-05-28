# agent/outcome_checker.py
"""
Outcome checker: parse a task prompt to determine the expected physical
state, then compare to physical_outcome() to decide if the episode succeeded.

Best-effort parser. Recognised task templates:
  - "put/place/move <color> cube in <color> bin"        -> "<src>_cube in <dst>_bin"
  - "put all cubes in <color> bin"                       -> all three "*_cube in <c>_bin"
  - "sort the cubes [by color]"                          -> each cube in matching bin
  - "push <object> off the bench/edge/table"             -> "<object> fell to floor" or "off bench"
  - "put/place/pick <tube-ref> in/into <rack-ref>"       -> "<tube> in <rack>"

Targets the parser recognises (matched against physical_outcome() strings):
  - cubes: red_cube, green_cube, blue_cube
  - bins:  red_bin, green_bin, blue_bin
  - racks: left_tube_rack, right_tube_rack
  - tubes: tube_L1..L3, tube_R1..R3 (by name, by cap color, or by source rack)

For tasks the parser doesn't recognise, returns None (success unknown -- the
loop should fall back to command-level success, or ask the user).

expected_outcome() returns (mode, substrings):
  - mode "all": every substring must appear in physical_outcome()
  - mode "any": at least one substring must appear (used when the user gave
    the LLM freedom to choose among several acceptable outcomes, or when the
    task can succeed via any of several end states -- e.g., "fell to floor"
    OR "off bench" both count as a successful push-off).
"""
import re
from typing import Optional, List, Tuple

COLORS = ("red", "green", "blue")

ALL_TUBES   = ("tube_L1", "tube_L2", "tube_L3",
               "tube_R1", "tube_R2", "tube_R3")
LEFT_TUBES  = ("tube_L1", "tube_L2", "tube_L3")
RIGHT_TUBES = ("tube_R1", "tube_R2", "tube_R3")

# Cap-color groupings (mirrors agent/object_registry.py).
BLUE_CAP_TUBES   = ("tube_L2", "tube_R1", "tube_R3")
ORANGE_CAP_TUBES = ("tube_L1", "tube_L3", "tube_R2")

# Home-rack mapping (mirrors RACK_TUBE_GROUPS in sim/mujoco_env.py).
TUBE_HOME_RACK = {
    "tube_L1": "left_tube_rack",  "tube_L2": "left_tube_rack",  "tube_L3": "left_tube_rack",
    "tube_R1": "right_tube_rack", "tube_R2": "right_tube_rack", "tube_R3": "right_tube_rack",
}

# Task-family classifiers. These mirror the patterns expected_outcome() uses
# below -- centralised here so other modules (lesson filtering, the loop's
# failure-mode hints) can ask "what kind of task is this?" without
# re-implementing the regexes. Add new families here when new task templates
# are introduced.
_PUSH_OFF_RE  = re.compile(r"\b(push|knock|drop|shove|slide)\b.*\b(off|edge|floor|away)\b")
_PLACEMENT_RE = re.compile(r"\b(put|place|move|drop|stash|pick)\b.*\b(in|into|inside)\b.*\b(bin|container|box|rack|slot)\b")
_SORT_RE      = re.compile(r"\bsort\b.*\bcubes?\b")


def classify_task(task: str) -> str:
    """Return one of: 'push_off', 'placement', 'sort', 'unknown'.

    Used to scope cross-run lessons (so push-task wins don't leak into
    placement-task prompts) and to pick task-appropriate failure hints.
    """
    t = task.lower()
    if _PUSH_OFF_RE.search(t):
        return "push_off"
    if _SORT_RE.search(t):
        return "sort"
    if _PLACEMENT_RE.search(t):
        return "placement"
    return "unknown"


def _off_bench_terms(names: List[str]) -> List[str]:
    """Return the union of '<name> fell to floor' and '<name> off bench' for each name."""
    out = []
    for n in names:
        out.append(f"{n} fell to floor")
        out.append(f"{n} off bench")
    return out


def _identify_tubes(t: str) -> List[str]:
    """Extract candidate tube names from task text. Empty if no tube reference."""
    # Specific tube ID (tube_L1, tube L1, L1 tube)
    m = re.search(r"\btube[_\s-]?([lr])[_\s-]?([123])\b", t)
    if m:
        return [f"tube_{m.group(1).upper()}{m.group(2)}"]
    # Cap-color descriptor: "blue cap tube", "blue-cap tube", "blue capped tube",
    # "tube with a blue cap", or just "blue tube"
    if (re.search(r"\bblue[\s-]?cap(ped)?\s*tube\b", t)
            or re.search(r"\btube\s+with\s+(a\s+)?blue\b", t)
            or re.search(r"\bblue\s+tube\b", t)):
        return list(BLUE_CAP_TUBES)
    if (re.search(r"\borange[\s-]?cap(ped)?\s*tube\b", t)
            or re.search(r"\btube\s+with\s+(an?\s+)?orange\b", t)
            or re.search(r"\borange\s+tube\b", t)):
        return list(ORANGE_CAP_TUBES)
    # Source-rack reference: "from the left rack" -> left tubes
    if re.search(r"\bfrom\b.*\bleft\b.*\brack\b", t):
        return list(LEFT_TUBES)
    if re.search(r"\bfrom\b.*\bright\b.*\brack\b", t):
        return list(RIGHT_TUBES)
    # Bare "tube" without qualifiers is too ambiguous to grade.
    return []


def _identify_dest_racks(t: str, sources: List[str]) -> List[str]:
    """
    Resolve the destination rack(s) from task text. The "other rack" idiom
    depends on each source tube's home rack, so this returns either:
      - a concrete list of rack names (specific destination), or
      - the marker ['__OPPOSITE_OF_HOME__'] when destination is per-tube relative.
    """
    if re.search(r"\bother\s+(tube[\s_-]?)?rack\b", t):
        return ["__OPPOSITE_OF_HOME__"]
    if re.search(r"\bright[\s_-]?(tube[\s_-]?)?rack\b", t):
        return ["right_tube_rack"]
    if re.search(r"\bleft[\s_-]?(tube[\s_-]?)?rack\b", t):
        return ["left_tube_rack"]
    return []


def expected_outcome(task: str) -> Optional[Tuple[str, List[str]]]:
    """
    Parse the task into a (mode, expected_substrings) pair, or None if the
    parser doesn't recognise the task.

    mode = "all" (all substrings required) or "any" (one substring suffices).
    """
    t = task.lower()

    # ---- Push-off family ----
    if re.search(r"\b(push|knock|drop|shove|slide).*\b(off|edge|floor|away)\b", t):
        # Cube targets
        for color in COLORS:
            if f"{color} cube" in t or f"{color}_cube" in t:
                return ("any", _off_bench_terms([f"{color}_cube"]))

        # Bin targets
        for color in COLORS:
            if f"{color} bin" in t or f"{color}_bin" in t:
                return ("any", _off_bench_terms([f"{color}_bin"]))

        # Specific tube target
        m = re.search(r"\btube[_\s-]?([lr])[_\s-]?([123])\b", t)
        if m:
            tname = f"tube_{m.group(1).upper()}{m.group(2)}"
            return ("any", _off_bench_terms([tname]))

        # Rack targets (a rack's tubes are welded during push -- any of them
        # appearing off-bench counts).
        has_left  = bool(re.search(r"\bleft\b", t))
        has_right = bool(re.search(r"\bright\b", t))
        has_rack  = bool(re.search(r"\bracks?\b", t)) or "tube_rack" in t

        if has_rack:
            if has_left and not has_right:
                return ("any", _off_bench_terms(["left_tube_rack", *LEFT_TUBES]))
            if has_right and not has_left:
                return ("any", _off_bench_terms(["right_tube_rack", *RIGHT_TUBES]))
            return ("any", _off_bench_terms(
                ["left_tube_rack", "right_tube_rack", *ALL_TUBES]))

        # "tubes" plural with no specific id
        if re.search(r"\btubes\b", t):
            if has_left and not has_right:
                return ("any", _off_bench_terms(list(LEFT_TUBES)))
            if has_right and not has_left:
                return ("any", _off_bench_terms(list(RIGHT_TUBES)))
            return ("any", _off_bench_terms(list(ALL_TUBES)))

        # Generic "clear the table" / "push everything off"
        if re.search(r"\b(all|everything|clear)\b", t):
            return ("any", ["fell to floor", "off bench"])
        return None

    # ---- Placement family ----
    #
    # Tube-into-rack ("put/place/move/pick the <tube-ref> in/into <rack-ref>").
    # Detected by presence of a tube reference + a rack/slot destination.
    has_tube_in_dest = bool(re.search(r"\b(rack|slot)s?\b", t))
    sources = _identify_tubes(t) if has_tube_in_dest else []
    if sources:
        dests = _identify_dest_racks(t, sources)
        if dests:
            expected: List[str] = []
            if dests == ["__OPPOSITE_OF_HOME__"]:
                for tube in sources:
                    home = TUBE_HOME_RACK.get(tube)
                    if home is None:
                        continue
                    other = ("right_tube_rack" if home == "left_tube_rack"
                             else "left_tube_rack")
                    expected.append(f"{tube} in {other}")
            else:
                for tube in sources:
                    home = TUBE_HOME_RACK.get(tube)
                    for rack in dests:
                        if rack == home:
                            continue  # placing in its own rack isn't a move
                        expected.append(f"{tube} in {rack}")
            if expected:
                # Any acceptable outcome counts (the user gave the LLM freedom
                # to choose which tube / which slot).
                return ("any", expected)

    # "put/place all cubes in <color> bin" -> all three cubes in that bin
    if re.search(r"\b(all|every|each).*\bcubes?\b", t):
        for color in COLORS:
            if f"{color} bin" in t:
                return ("all", [f"{c}_cube in {color}_bin" for c in COLORS])
        return None

    # "sort the cubes [by color]" -> each in its matching bin
    if re.search(r"\bsort.*cubes?\b", t):
        return ("all", [f"{c}_cube in {c}_bin" for c in COLORS])

    # "put/place/move <color> cube in <color> bin" (canonical pick-and-place)
    m = re.search(r"\b(red|green|blue)\s*(cube|block)\b.*\b(red|green|blue)\b.*\b(bin|container|box)\b", t)
    if m:
        src = m.group(1)
        dst = m.group(3)
        return ("all", [f"{src}_cube in {dst}_bin"])

    return None  # Unknown task pattern


def check_outcome(task: str, physical: str,
                  fallback_spec: Optional[Tuple[str, List[str]]] = None,
                  ) -> Tuple[Optional[bool], str]:
    """
    Compare expected outcome (from task) to actual physical_outcome() string.

    Returns:
        (success, reason)
        success: True/False/None (None = couldn't determine)
        reason:  human-readable explanation

    `fallback_spec`, when provided, is consulted ONLY if the regex grader
    returns None (i.e. the task pattern isn't recognised). It has the same
    (mode, expected_substrings) shape and is typically produced by
    agent.dynamic_grader.infer_criteria as a once-per-session Haiku call.
    The fast path stays free of LLM cost for known task patterns.
    """
    spec = expected_outcome(task)
    used_fallback = False
    if spec is None and fallback_spec is not None:
        spec = fallback_spec
        used_fallback = True
    if spec is None:
        return (None, f"Task pattern not recognised; physical: {physical or 'no displacement'}")

    mode, expected = spec
    phys = physical or ""
    src = " [LLM-grader]" if used_fallback else ""

    if mode == "any":
        if any(e in phys for e in expected):
            return (True, f"Met physical condition{src}: {phys}")
        return (False, f"Expected one of {expected}{src}; got: {phys or 'no displacement'}")

    # mode == "all"
    missing = [e for e in expected if e not in phys]
    if not missing:
        return (True, f"All expected placements present{src}: {phys}")
    return (False, f"Missing: {missing}{src}; got: {phys or 'no displacement'}")
