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


SYSTEM_PROMPT = """\
You are a grading-criteria designer for a robot-arm digital twin. Given a
task prompt, produce the success criteria that will be matched against
the simulator's `physical_outcome()` string. Your output is a JSON object
in a strict schema; downstream code parses it deterministically.

## What `physical_outcome()` looks like

The simulator emits a semicolon-joined list of facts about objects whose
state has changed. Vocabulary (these are the ONLY shapes the simulator
ever produces):

  - `<object> fell to floor`     -- the object's z is near floor level
  - `<object> in <bin>`          -- a cube is sitting in a bin's footprint
  - `<object> in <rack>`         -- a tube is seated in a non-home rack
                                    slot (tubes still in their home rack
                                    are NOT reported)
  - `<object> off bench`         -- xy is past the bench edge, still elevated
  - `no objects displaced`       -- nothing moved

Examples:
  - "red_cube in red_bin"
  - "tube_L2 in right_tube_rack"
  - "tube_R3 fell to floor; blue_bin off bench"
  - "no objects displaced"

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
