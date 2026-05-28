# agent/dynamic_grader.py
"""
Dynamic (LLM-based) fallback grader.

`outcome_checker.expected_outcome` knows a fixed set of task templates --
push-off, placement, sort. Anything outside those returns None, which the
loop reports as "ungraded" (no success/failure attribution). For novel
task phrasings we'd rather have the loop attempt to grade than silently
drop the episode from the totals.

This module makes one Haiku call per session at task start, asking the
model to inspect the task prompt and produce the *same* (mode, substrings)
spec shape that the regex grader produces -- which plugs directly into
`check_outcome` with no further changes.

Design choices:
  - Fallback-only: invoked only when the regex grader returns None, so the
    known-task fast path stays free.
  - One call per session, cached: criteria are deterministic w.r.t. the
    task text, so calling once and reusing across all N episodes is
    correct and cheap.
  - Non-fatal: any API error / parse error returns None and the loop
    falls back to "ungraded" as before.
  - Returns a vocabulary the simulator actually emits (see
    `physical_outcome` in sim/mujoco_env.py for the format).
"""
import json
import re
import time
from typing import List, Optional, Tuple

import anthropic


GRADER_MODEL_DEFAULT = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Speed-cap tiers (mm/s) for the motion primitives that take a speed param
# (move_to, set_rail, set_joints). The LLM picks a tier from the task
# prompt; the dispatch layer clamps any command-emitted speed to the tier's
# cap. None means "no cap" -- only used for the crazy_fast tier, which the
# user must opt into via explicit phrasing.
#
# These values are tuned for cautious real-arm transfer; adjust here if the
# real hardware can take more. Medium = the prior de-facto cap (matches the
# system prompt's "keep speed_mm_s <= 100" guidance, slightly conservative).
# ---------------------------------------------------------------------------

SPEED_TIERS = {
    "crazy_fast": None,
    "fast":       120,
    "medium":      80,   # default when the task gives no speed cue
    "slow":        40,
    "very_slow":   15,
}
DEFAULT_SPEED_TIER = "medium"


SPEED_SYSTEM_PROMPT = """\
You analyse a task prompt for a robot arm and pick a single SESSION-LEVEL
motion-speed tier that serves as the CEILING for the whole task. Per-
command tier downgrades happen later (the planning model can attach a
slower tier to individual delicate commands); your job is to set the
upper bound the planner is allowed to use.

## Tier list

  - "crazy_fast" : NO speed cap. Only pick this when the prompt EXPLICITLY
                   asks for uncapped / unlimited / no-safety-limit speed,
                   e.g. "go as fast as possible with no limits", "crazy
                   fast", "uncapped speed", "no safety cap". Otherwise do
                   NOT pick this -- removing the safety cap is dangerous
                   in real hardware.
  - "fast"       : explicit speed cues like "quickly", "fast", "rapidly",
                   "hurry", "asap", "as fast as you can".
  - "medium"     : DEFAULT when no speed cue is present, or generic task
                   descriptions. If you're unsure, pick this.
  - "slow"       : cues like "slowly", "carefully", "gently", "take your
                   time", "with care".
  - "very_slow"  : cues like "very slowly", "extremely carefully",
                   "delicately", "fragile", "in slow motion", "with
                   extreme care".

## Multiple cues in one prompt

When the prompt mentions DIFFERENT speeds for different phases -- e.g.
"pick up the tube quickly then carefully insert it" -- pick the MOST
PERMISSIVE (fastest) tier mentioned. That becomes the session ceiling.
The planning model will downgrade individual commands to slower tiers
for the careful phases. If you picked the slow tier as the session
ceiling, the quick phase couldn't happen at all (the ceiling would
hold the planner back). The most permissive tier preserves the
planner's freedom to honour every cue in the prompt.

For the example above ("pick up quickly then carefully insert"), the
correct session tier is `fast` -- the planner will then attach
`speed_tier: "slow"` to the insert commands and run the pickup at the
fast cap.

## Output

Return a single JSON object with exactly two fields:

{
  "tier": "<one of the five names above>",
  "reasoning": "<one short sentence citing the speed cue(s) you matched,
                or 'no speed cue, defaulting to medium'. If multiple
                cues were present, name them and explain that you picked
                the most permissive as the session ceiling.>"
}

Do not wrap the JSON in markdown fences. Do not add prose around it.
"""


SYSTEM_PROMPT = """\
You are a grading-criteria designer for a robot-arm digital twin. Given a
task prompt, produce the success criteria that will be matched against
the simulator's `physical_outcome()` string. Your output is a JSON object
in a strict schema; downstream code parses it deterministically.

## What `physical_outcome()` looks like

The simulator emits a semicolon-joined list of facts about objects whose
state has changed since the last scene reset. Vocabulary (these are the
ONLY shapes the simulator ever produces):

Categorical events (exclusive per object -- only one per object per call):
  - `<object> fell to floor`     -- the object's z is near floor level
  - `<object> in <bin>`          -- a cube is sitting in a bin's footprint
  - `<object> in <rack>`         -- a tube is seated in a non-home rack
                                    slot (tubes still in their home rack
                                    are NOT reported)
  - `<object> off bench`         -- xy is past the bench edge, still elevated

Displacement / proximity facts (added on top of the above):
  - `<object> moved (Δx, Δy)mm`         -- emitted for any movable body
                                           whose xy shifted >=20 mm from
                                           its position at the last scene
                                           reset (only for objects that
                                           did NOT trigger a categorical
                                           event above)
  - `<a> closer to <b>`                 -- inter-object xy distance shrank
                                           by >=20 mm
  - `<a> farther from <b>`              -- inter-object xy distance grew
                                           by >=20 mm

  - `no objects displaced`       -- no facts emitted (nothing happened)

Examples:
  - "red_cube in red_bin"
  - "tube_L2 in right_tube_rack"
  - "tube_R3 fell to floor; blue_bin off bench"
  - "green_bin moved (180, 5)mm; green_bin closer to blue_bin"
  - "red_cube moved (-50, 100)mm; red_cube closer to green_cube; red_cube farther from blue_cube"
  - "no objects displaced"

For a "push X closer to Y" task, the natural success criterion is
`<X> closer to <Y>`. Note that `<a>` and `<b>` are in a canonical order
(both pair members emit just one fact, alphabetical-first first); when
you don't know which order the simulator will use, list BOTH orderings
in expected_substrings with mode "any":
`["green_bin closer to blue_bin", "blue_bin closer to green_bin"]`.

## Valid object names (use these EXACTLY, including underscores)

Cubes: red_cube, green_cube, blue_cube
Bins:  red_bin, green_bin, blue_bin
Tube racks: left_tube_rack, right_tube_rack
Falcon tubes:
  - In left rack: tube_L1 (orange cap), tube_L2 (blue cap), tube_L3 (orange cap)
  - In right rack: tube_R1 (blue cap), tube_R2 (orange cap), tube_R3 (blue cap)

## Output schema

Return a single JSON object, nothing else:

{
  "mode": "any" | "all",
  "expected_substrings": ["<substring>", "<substring>", ...],
  "reasoning": "<one short sentence explaining your choice>"
}

- "any": at least one substring must appear in physical_outcome() for the
  task to count as success. Use this when the task gives the model freedom
  to choose between equivalent targets (e.g. "push any cube off" -- any
  cube falling counts).
- "all": EVERY substring must appear. Use this when the task names
  multiple specific outcomes that all must happen (e.g. "sort all three
  cubes into their matching bins").

Each substring should match the physical_outcome() vocabulary above (e.g.
"red_cube in red_bin", "tube_L2 fell to floor"). Do not invent shapes the
simulator wouldn't emit.

## When you cannot grade

If the task is too vague, ambiguous, or describes a state the simulator
cannot detect (e.g. "make the arm wave gracefully"), return:

{"mode": "any", "expected_substrings": [], "reasoning": "task is not graded physically (no detectable end state)"}

An empty expected_substrings array signals "ungraded" to the caller; it
is NOT treated as success.

Do not wrap your JSON in markdown fences. Do not add prose around it.
"""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user_message(task: str, registry_context: str = "") -> str:
    parts = [
        "Task to grade:",
        f"  \"{task}\"",
    ]
    if registry_context.strip():
        parts.extend([
            "",
            "Current scene registry (object positions, in case the task",
            "references them by alias rather than canonical name):",
            registry_context.strip(),
        ])
    parts.extend([
        "",
        "Produce the grading criteria as a single JSON object per the schema.",
    ])
    return "\n".join(parts)


def _parse_criteria(raw: str) -> Optional[Tuple[str, List[str]]]:
    """Extract (mode, expected_substrings) from Haiku's response. Returns
    None when the response is malformed -- the caller treats None as
    'fall back to ungraded'.
    """
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mode = obj.get("mode")
    subs = obj.get("expected_substrings")
    if mode not in ("any", "all") or not isinstance(subs, list):
        return None
    # Empty expected_substrings is intentional ("ungraded") -- propagate
    # as None so the loop doesn't try to grade with an empty spec (which
    # would degenerate to "any of nothing matches" = always False).
    if not subs:
        return None
    # Coerce all entries to strings and drop empties.
    subs = [str(s) for s in subs if str(s).strip()]
    if not subs:
        return None
    return (mode, subs)


def infer_criteria(task: str, registry=None,
                   model: str = GRADER_MODEL_DEFAULT,
                   client: Optional[anthropic.Anthropic] = None,
                   ) -> Optional[Tuple[str, List[str]]]:
    """Ask Haiku what physical-outcome substrings indicate this task succeeded.

    Returns the same (mode, substrings) tuple shape that
    outcome_checker.expected_outcome produces, so the caller can splice
    the result into check_outcome unchanged. Returns None on any error
    (API failure, malformed response, task ungradeable per Haiku's
    judgment) -- the caller is expected to treat None as "fall back to
    the existing ungraded behaviour".
    """
    if client is None:
        try:
            client = anthropic.Anthropic()
        except Exception as e:
            print(f"[DynamicGrader] No Anthropic client available: {e}")
            return None

    registry_ctx = ""
    if registry is not None and hasattr(registry, "to_llm_context"):
        try:
            registry_ctx = registry.to_llm_context()
        except Exception:
            registry_ctx = ""

    user_msg = _build_user_message(task, registry_ctx)

    t_start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        print(f"[DynamicGrader] API call failed: {type(e).__name__}: {e}")
        return None

    latency = time.time() - t_start
    raw = response.content[0].text
    in_tok  = getattr(response.usage, "input_tokens", 0)
    out_tok = getattr(response.usage, "output_tokens", 0)
    print(f"[DynamicGrader] Haiku graded task in {latency:.1f}s "
          f"({in_tok}->{out_tok} tokens)")

    parsed = _parse_criteria(raw)
    if parsed is None:
        # Try to extract reasoning for the log even when criteria parse failed.
        m = _JSON_OBJ_RE.search(raw)
        reason_snippet = ""
        if m:
            try:
                obj = json.loads(m.group(0))
                reason_snippet = f" (reason: {obj.get('reasoning', '?')})"
            except json.JSONDecodeError:
                pass
        print(f"[DynamicGrader] No usable criteria returned{reason_snippet}; "
              f"episodes will fall back to ungraded.")
        return None

    mode, subs = parsed
    preview = subs[:4] + (["..."] if len(subs) > 4 else [])
    print(f"[DynamicGrader] Criteria: mode={mode}, expects {preview}")
    return parsed


def infer_speed_cap(task: str,
                    model: str = GRADER_MODEL_DEFAULT,
                    client: Optional[anthropic.Anthropic] = None,
                    ) -> str:
    """Ask Haiku which speed tier this task implies. Returns one of
    SPEED_TIERS keys; defaults to DEFAULT_SPEED_TIER ('medium') on any
    error or unrecognised response. Never returns None -- there is always
    *some* cap (unless the model picks 'crazy_fast', which is uncapped).

    The caller looks up SPEED_TIERS[result] to get the numeric mm/s cap
    (or None for 'crazy_fast' = no clamp).
    """
    if client is None:
        try:
            client = anthropic.Anthropic()
        except Exception as e:
            print(f"[DynamicGrader] Speed inference unavailable: {e}; "
                  f"defaulting to '{DEFAULT_SPEED_TIER}'.")
            return DEFAULT_SPEED_TIER

    t_start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=128,
            system=SPEED_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Task: \"{task}\""}],
        )
    except Exception as e:
        print(f"[DynamicGrader] Speed inference API failure: "
              f"{type(e).__name__}: {e}; defaulting to '{DEFAULT_SPEED_TIER}'.")
        return DEFAULT_SPEED_TIER

    latency = time.time() - t_start
    raw = response.content[0].text
    m = _JSON_OBJ_RE.search(raw)
    tier = None
    reasoning = ""
    if m:
        try:
            obj = json.loads(m.group(0))
            tier = obj.get("tier")
            reasoning = obj.get("reasoning", "")
        except json.JSONDecodeError:
            pass
    if tier not in SPEED_TIERS:
        print(f"[DynamicGrader] Speed inference returned unrecognised tier "
              f"({tier!r}); defaulting to '{DEFAULT_SPEED_TIER}'.")
        return DEFAULT_SPEED_TIER

    cap = SPEED_TIERS[tier]
    cap_str = "UNCAPPED" if cap is None else f"{cap} mm/s"
    print(f"[DynamicGrader] Speed tier in {latency:.1f}s: {tier} "
          f"(cap: {cap_str}) -- {reasoning}")
    return tier
