# agent/review_session.py
"""
Phase 2 of the learning architecture: end-of-session review by a strong model.

After EpisodeRetry.run() finishes its N episodes, this module invokes Claude
Opus 4.7 on the *full* session and asks it to produce abstracted,
hypothesis-framed observations the in-loop per-episode analysers can't
produce.

What Opus is asked to do that the in-loop analysers cannot:
  - Generalize across multiple similar failures.
  - Spot false-positive successes (grader said yes but the trace looks lucky
    or destructive in a way the grader missed).
  - Diagnose *why* a deviation failed when the model tried something new
    that didn't work.
  - Compare what successes had in common.

What Opus is explicitly NOT asked to do:
  - Write prescriptive rules ("Always use rail=140mm"). These create lock-in
    in future sessions. Instead, observations are phrased as "episodes
    A,B,C succeeded with rail=140-160mm; episode D deviated to rail=180mm
    and IK failed." -- corroboration counts visible, hypothesis-framed.

The review output goes through agent.lessons.append_review(), where it is
clearly tagged so Phase 3 can prefer review entries over raw per-episode
ones when pre-seeding future prompts.

The whole pass is non-fatal: if Opus is unavailable, rate-limited, or
returns malformed JSON, the session completes normally and a short
diagnostic is printed. Future sessions just don't get a review for this run.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

import anthropic


REVIEW_MODEL_DEFAULT = "claude-opus-4-7"

# Sessions shorter than this don't get reviewed -- one or two episodes is
# too thin a base for Opus to abstract from, and the failure analysers in
# the loop already capture what's there.
MIN_EPISODES_FOR_REVIEW = 3


SYSTEM_PROMPT = """\
You are reviewing one training-session of a robot-arm digital twin in which
a smaller model (typically Haiku) attempted the SAME task across multiple
episodes. The scene was reset between episodes; each episode is an
independent attempt.

## Your job

Produce *abstracted observations* that will help future sessions on this
task -- not a prescriptive plan. The next session's smaller model will see
your writeup; if you write rules ("always do X"), it will copy them
verbatim and stop exploring. So phrase observations as:

  - "Episodes 2, 4, 6 succeeded with rail in 140-160mm; episode 5 deviated
    to rail=180mm and IK failed -- suggests this task is sensitive to rail
    position in that range."
  - "Both successful plans grasped the cube before lifting the rail;
    episode 3 set the rail first and the cube was nudged. Worth testing
    whether grasp-before-rail is required."

Never phrase as:
  - "Always use rail=140mm." (rule, not observation)
  - "Don't grasp after moving the rail." (rule, not observation)

## What to look for (per-task)

1. **Cross-episode patterns in successes**: what did the working plans have
   in common? What numeric ranges did the parameters cluster in?
2. **Cross-episode patterns in failures**: did the same kind of failure
   recur? Was it a coordinate range, a command ordering, a missing step?
3. **False-positive successes**: the grader is loose -- it can declare
   success when the physical outcome was destructive or lucky (objects
   knocked off the bench, gripper closed near but not on the target). If
   any success episode's physical outcome reads as suspect to you, flag it.
4. **Failed deviations**: if an episode tried something new and it didn't
   work, what changed between the deviation and the working plans?

## Cross-task observations (the world model)

In addition to the per-task writeup, look for observations that are
*general* -- true beyond this one task. These accumulate into a separate
`world_model.md` file across sessions. Four categories:

  - **geometric**: arm reachability, collision envelopes, validator
    quirks, IK behaviour at extremes. Examples: "validator rejects
    descents below z~895mm anywhere inside a tube-rack footprint",
    "IK fails for wrist pitches outside +/-90deg at the workspace edge".
  - **object_class**: regularities that apply to all members of a class
    (all tubes, all bins, all racks). Examples: "tubes seated in racks
    can be grasped from z=900mm regardless of which rack",
    "bin walls top out at z=810mm across all three colors".
  - **primitive**: behaviour of a command primitive. Examples:
    "`place_tube_in_rack` requires the rail to be at the rack's
    optimal_rail_mm; otherwise it returns IK failure",
    "`push_object` on a rack body carries all 3 tubes with it".
  - **grader**: regularities about how `physical_outcome` / `check_outcome`
    grade outcomes. Examples: "loose stringency accepts tubes within 30mm
    of the slot xy; strict requires <10mm", "the grader counts a tube as
    in-rack even if its lid is missing".

Only emit a cross-task observation when there's evidence in THIS session
that the pattern is real. A one-session observation will be recorded as
provisional; it will only become high-confidence if future sessions
corroborate it.

### Existing world-model entries (from prior sessions)

If existing entries are shown below, you must indicate for each new
observation whether it is a CORROBORATION of an existing entry (use its
`merge_with_index`) or a genuinely NEW one. Near-duplicates -- same fact
phrased differently -- should be merged, not duplicated. Only emit truly
new content as new entries.

## Output format

Return your analysis as Markdown, then a single fenced JSON block at the
end with structured fields. The Markdown is human-readable; the JSON is
machine-parsed for Phase 3 use.

The Markdown writeup should be 3-6 short paragraphs or a bulleted list of
observations -- not a wall of text. Use episode numbers as evidence when
making claims.

The trailing JSON block must have this exact shape:

```json
{
  "observations": [
    {
      "text": "<one-line observation phrased as an observation, not a rule>",
      "evidence_episodes": [<int>, ...],
      "confidence": "high" | "medium" | "low"
    }
  ],
  "false_positives": [
    {
      "episode": <int>,
      "reason": "<why this success looks suspect>"
    }
  ],
  "exploration_diagnoses": [
    {
      "episode": <int>,
      "deviation": "<what was tried differently>",
      "diagnosis": "<why it failed compared to the working plan>"
    }
  ],
  "cross_task_observations": [
    {
      "text": "<one-line observation general enough to apply beyond this task>",
      "category": "geometric" | "object_class" | "primitive" | "grader",
      "merge_with_index": <int or null>
    }
  ]
}
```

For `cross_task_observations`:
- `category` must be one of the four strings above.
- `merge_with_index` is the index (0-based) of an existing entry in that
  same category that this observation corroborates -- use `null` for a
  genuinely new entry. Indexes refer to the existing entries listed under
  "Existing world-model entries" in the user message; if no entries exist
  in that category, always use `null`.

If you have nothing to put in a list, use `[]`. Do not omit fields. Do not
add commentary after the JSON block.
"""


def _serialize_episode(idx: int, ctx) -> Dict[str, Any]:
    """Pull one episode's worth of data out of EpisodeContext.

    EpisodeContext stores per-episode outcomes and failure_history entries
    keyed by episode_num, but successful plans only carry the plan + episode
    number. So we reconstruct each episode by cross-referencing the lists.
    """
    ep_num = idx + 1  # episodes are 1-indexed in the loop
    outcome = ctx.episode_outcomes[idx] if idx < len(ctx.episode_outcomes) else None

    failures = [f for f in ctx.failure_history if f.get("episode") == ep_num]
    successes = [p for p in ctx.successful_plans if p.get("episode") == ep_num]

    entry: Dict[str, Any] = {
        "episode": ep_num,
        "outcome": ("success" if outcome is True
                    else "failure" if outcome is False
                    else "ungraded"),
    }
    if successes:
        s = successes[0]
        entry["plan_commands"] = s["commands"]
        entry["n_commands"] = s["n_commands"]
        entry["physical_outcome"] = s["physical"]
    if failures:
        # One episode can have at most one failure entry (command or physical).
        f = failures[0]
        entry["failure_kind"] = f.get("kind")
        if f.get("kind") == "command":
            entry["failed_step_index"] = f.get("step")
            entry["failed_action"] = f.get("action")
            entry["failure_code"] = f.get("code")
        elif f.get("kind") == "physical":
            entry["physical_outcome"] = f.get("physical")
            entry["grader_reason"] = f.get("reason")
        entry["learned_constraint"] = f.get("constraint")
    return entry


def _render_existing_world_model(world_model) -> str:
    """Render the existing WorldModel entries in a form Opus can reference
    by `merge_with_index`. Returns '' if there are no entries.

    The format intentionally numbers entries 0..N-1 within each category
    so Opus's `merge_with_index` can refer to them unambiguously.
    """
    if world_model is None:
        return ""
    # Lazy import to avoid forcing world_model.py at import time of
    # review_session.py.
    from agent.world_model import SECTIONS, SECTION_KEYS

    have_any = any(world_model.entries.get(k) for k in SECTION_KEYS)
    if not have_any:
        return ""

    lines = ["## Existing world-model entries"]
    if world_model.scene_changed:
        lines.append("**Note: scene XML has changed since these entries "
                     "were recorded -- treat as untested.**")
    lines.append("Reference these by category + 0-based index when "
                 "deciding `merge_with_index` for your cross-task "
                 "observations.")
    lines.append("")

    for title, key in SECTIONS:
        section = world_model.entries.get(key, [])
        if not section:
            continue
        lines.append(f"### {title} (category=`{key}`)")
        for i, entry in enumerate(section):
            lines.append(f"  [{i}] {entry.text} "
                         f"_(confidence={entry.confidence}, "
                         f"corroborations={len(entry.corroborations)})_")
        lines.append("")
    return "\n".join(lines)


def _build_user_message(task: str, ctx, prior_lessons: str,
                        world_model=None) -> str:
    """Serialize the full session into a clean prompt the reviewer can read."""
    n_total = len(ctx.episode_outcomes)
    episodes = [_serialize_episode(i, ctx) for i in range(n_total)]

    n_succ = sum(1 for o in ctx.episode_outcomes if o is True)
    n_fail = sum(1 for o in ctx.episode_outcomes if o is False)
    n_ung  = sum(1 for o in ctx.episode_outcomes if o is None)

    summary = {
        "task": task,
        "max_episodes": ctx.max_episodes,
        "episodes_run": n_total,
        "totals": {"success": n_succ, "failure": n_fail, "ungraded": n_ung},
        "successful_plans_pinned": len(ctx.successful_plans),
        "best_plan_n_commands": (
            ctx.successful_plans[ctx.best_plan_idx]["n_commands"]
            if ctx.best_plan_idx is not None else None
        ),
        "learned_failure_constraints": ctx.learned_constraints,
    }

    parts = [
        "## Session summary",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Episode-by-episode trace",
        "```json",
        json.dumps(episodes, indent=2),
        "```",
        "",
    ]

    if prior_lessons.strip():
        parts.extend([
            "## Prior lessons for this task family (from earlier sessions)",
            "Use these as background. If your observations corroborate or",
            "refine any of them, say so explicitly in your writeup.",
            "",
            prior_lessons.strip(),
            "",
        ])

    wm_block = _render_existing_world_model(world_model)
    if wm_block:
        parts.append(wm_block)
        parts.append("")

    parts.append(
        "Write your review now. Remember: observations with episode-level "
        "evidence, never prescriptive rules. Emit cross-task observations "
        "only when there's evidence in this session; mark them as "
        "corroborations of existing entries via merge_with_index when "
        "applicable. End with the JSON block as specified in the system "
        "prompt."
    )
    return "\n".join(parts)


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_review_response(raw: str) -> Dict[str, Any]:
    """Split Opus's response into (markdown_writeup, structured_dict).

    The JSON block is the LAST fenced ```json block in the response.
    Anything before it is the writeup; we strip the block out of the
    writeup so it isn't duplicated in lessons.md.
    """
    matches = list(_JSON_BLOCK_RE.finditer(raw))
    if not matches:
        # Reviewer didn't follow the format. Treat the whole thing as
        # writeup, leave structured fields empty.
        return {
            "task_writeup": raw.strip(),
            "observations": [],
            "false_positives": [],
            "exploration_diagnoses": [],
            "cross_task_observations": [],
        }

    last = matches[-1]
    writeup = (raw[:last.start()] + raw[last.end():]).strip()
    try:
        parsed = json.loads(last.group(1))
    except json.JSONDecodeError as e:
        print(f"[ReviewSession] JSON block failed to parse: {e}")
        return {
            "task_writeup": raw.strip(),
            "observations": [],
            "false_positives": [],
            "exploration_diagnoses": [],
            "cross_task_observations": [],
        }

    return {
        "task_writeup": writeup,
        "observations": parsed.get("observations", []),
        "false_positives": parsed.get("false_positives", []),
        "exploration_diagnoses": parsed.get("exploration_diagnoses", []),
        "cross_task_observations": parsed.get("cross_task_observations", []),
    }


def review_session(task: str,
                   ctx,
                   model: str = REVIEW_MODEL_DEFAULT,
                   prior_lessons: str = "",
                   world_model=None,
                   client: Optional[anthropic.Anthropic] = None,
                   ) -> Dict[str, Any]:
    """Invoke Opus on the full session; return parsed structured review.

    Args:
        task:           The task prompt as the user typed it.
        ctx:            EpisodeContext from EpisodeRetry.run (full history).
        model:          Reviewer model. Defaults to claude-opus-4-7.
        prior_lessons:  Existing relevant lessons.md content, for the
                        reviewer to corroborate/refine rather than duplicate.
        client:         Optional pre-built anthropic.Anthropic client
                        (mainly for tests).

    Returns:
        {
          "task_writeup":           str (Markdown for lessons.md),
          "observations":           List[Dict],
          "false_positives":        List[Dict],
          "exploration_diagnoses":  List[Dict],
          "model":                  str (resolved model id),
          "latency_s":              float,
          "input_tokens":           int,
          "output_tokens":          int,
        }

    Raises:
        anthropic.APIError if the call itself fails. The caller is
        expected to wrap this in try/except and treat the review as
        optional.
    """
    if client is None:
        client = anthropic.Anthropic()

    user_msg = _build_user_message(task, ctx, prior_lessons,
                                   world_model=world_model)

    t_start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    latency = time.time() - t_start

    raw = response.content[0].text
    in_tok  = getattr(response.usage, "input_tokens", 0)
    out_tok = getattr(response.usage, "output_tokens", 0)
    print(f"[ReviewSession] Opus reviewed session in {latency:.1f}s "
          f"({in_tok}->{out_tok} tokens)")

    parsed = _parse_review_response(raw)
    parsed.update({
        "model": model,
        "latency_s": latency,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    })
    return parsed
