# agent/lessons.py
"""
Cross-session reflection files.

Two artifacts live here:

  * `lessons.md` -- one-line entries appended after each task run. The
    next session's system prompt reads recent lessons so Claude can avoid
    repeating mistakes. Capped at MAX_LESSONS entries (soft cap to keep
    the prompt bounded).

  * `reviews.md` -- Phase-2 abstracted writeups produced by Opus after a
    full training session of N episodes. Each entry is multi-paragraph
    Markdown plus a structured-evidence block. Capped at MAX_REVIEWS
    entries. Kept in a SEPARATE file because `append_lesson` rewrites
    `lessons.md` wholesale from the bullet list, which would clobber
    any embedded multi-line review prose.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LESSONS_FILE = Path("lessons.md")
REVIEWS_FILE = Path("reviews.md")
MAX_LESSONS = 20
MAX_REVIEWS = 10

_HEADER = (
    "# Cross-session lessons\n"
    "\n"
    "Auto-appended after each task run. Most recent first. Capped at "
    f"{MAX_LESSONS} entries.\n"
    "\n"
)


def _summarize_outcome(planned_commands: list, results: list,
                       physical_outcome: str,
                       task_success: Optional[bool] = None) -> str:
    """Distill the dispatch results into a one-line outcome string.

    `task_success` carries the grader's verdict (from outcome_checker), if known:
      True  -> physical outcome matched the task -> "SUCCESS"
      False -> commands ran but physical outcome did NOT match -> "FAILED (wrong end state)"
      None  -> grader couldn't classify the task -> "EXECUTED (ungraded)"
    When None (e.g. legacy callers that don't grade), we fall back to the old
    behaviour of reporting "SUCCESS" once all commands returned 0 -- but that
    is misleading for tasks where command success != task success, so callers
    are strongly encouraged to pass the grader's verdict.
    """
    n_planned = len(planned_commands)
    n_executed = len(results)

    # Locate first hard failure (skip done/wait/get_pose which are advisory)
    for i, r in enumerate(results):
        if r["result"] != 0 and r["action"] not in ("done", "wait", "get_pose"):
            params = planned_commands[i].get("params", {}) if i < n_planned else {}
            reason = {
                1: "IK could not solve target",
                2: "validation failed (collision or FK error)",
                -1: "unknown action",
            }.get(r["result"], f"code {r['result']}")
            return (f"FAILED at step {i+1}/{n_planned} "
                    f"({r['action']} {params}) — {reason}")

    if n_executed < n_planned:
        return f"INCOMPLETE: executed {n_executed}/{n_planned} commands"

    # All commands returned 0. Now distinguish by the grader's verdict so we
    # don't poison future runs with "SUCCESS" lessons that were actually
    # failed task outcomes (e.g. knocking the rack off the bench while trying
    # to place a tube in it).
    suffix = f" ({physical_outcome})" if physical_outcome else ""
    if task_success is True:
        return "SUCCESS" + suffix
    if task_success is False:
        return "FAILED (wrong end state)" + suffix
    if task_success is None:
        # Legacy / ungraded caller: be honest about the ambiguity.
        return "EXECUTED (ungraded)" + suffix
    return "SUCCESS" + suffix  # unreachable, defensive


def append_lesson(task_prompt: str, model_short: str,
                  planned_commands: list, results: list,
                  physical_outcome: str = "",
                  task_success: Optional[bool] = None,
                  stringency: Optional[str] = None,
                  lessons_file: Path = None) -> Path:
    """Write a one-line lesson to lessons.md and trim to MAX_LESSONS.

    `stringency`, when set, is rendered as a `[stringency=X]` tag in the
    entry so users can tell at a glance which grading mode a past SUCCESS
    was claimed under. Lines without the tag (legacy entries) are treated
    as loose-mode for filtering purposes.
    """
    if lessons_file is None:
        lessons_file = LESSONS_FILE

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    outcome = _summarize_outcome(planned_commands, results, physical_outcome,
                                 task_success=task_success)
    stringency_tag = f"[stringency={stringency}] " if stringency else ""
    new_entry = (f'- {ts} [{model_short}] {stringency_tag}'
                 f'"{task_prompt}" -> {outcome}')

    existing_entries = []
    if lessons_file.exists():
        for line in lessons_file.read_text().splitlines():
            if line.startswith("- "):
                existing_entries.append(line)

    entries = [new_entry] + existing_entries
    entries = entries[:MAX_LESSONS]

    lessons_file.write_text(_HEADER + "\n".join(entries) + "\n")
    return lessons_file


# Regex to pull the task prompt out of a lesson line. Tolerates one or more
# bracket-tagged annotations (model + optional stringency etc.) before the
# quoted task. Examples:
#   - 2026-05-27 09:49 [haiku] "pick the blue-cap tube ..." -> SUCCESS (...)
#   - 2026-05-27 09:49 [haiku] [stringency=normal] "pick ..." -> FAILED ...
_LESSON_TASK_RE = re.compile(
    r'^\-\s+\S+\s+\S+\s+(?:\[[^\]]+\]\s+)+"(?P<task>.+?)"\s+->'
)


def _extract_task_from_lesson_line(line: str) -> Optional[str]:
    """Return the embedded task prompt from a lesson line, or None if it
    doesn't parse (header text, blank, malformed)."""
    m = _LESSON_TASK_RE.match(line)
    return m.group("task") if m else None


def read_lessons_section(lessons_file: Path = None,
                         current_task: Optional[str] = None) -> str:
    """Return the contents of lessons.md, or '' if absent/empty.

    If `current_task` is provided, filters entries to only those whose
    embedded task classifies into the same task family as `current_task`
    (push_off / placement / sort / unknown). This stops e.g. push-task
    "SUCCESS (rack off bench)" lessons from leaking into placement-task
    prompts and teaching the model that knocking objects off counts as a
    win for placement.

    Lessons from `unknown`-family tasks (parser couldn't classify them) are
    shown regardless of the current task -- they're already
    pattern-agnostic and tend to carry generally useful info (e.g.
    geometry quirks). The current task being unknown-family also disables
    filtering entirely (no signal to filter on).

    Used by LLMBrain to inject lessons into the system prompt.
    """
    if lessons_file is None:
        lessons_file = LESSONS_FILE
    if not lessons_file.exists():
        return ""

    text = lessons_file.read_text()
    if not text.strip():
        return ""

    if current_task is None:
        return text.strip()

    # Local import avoids a circular import (outcome_checker is a leaf).
    from agent.outcome_checker import classify_task
    current_family = classify_task(current_task)
    if current_family == "unknown":
        # No discriminator to filter on; show everything.
        return text.strip()

    lines = text.splitlines()
    kept: List[str] = []
    n_total = 0
    for line in lines:
        if not line.startswith("- "):
            kept.append(line)  # preserve header lines, blanks, etc.
            continue
        n_total += 1
        embedded = _extract_task_from_lesson_line(line)
        if embedded is None:
            kept.append(line)  # malformed -- be permissive, keep it
            continue
        fam = classify_task(embedded)
        # Keep same-family lessons + unknown-family (generally applicable).
        if fam == current_family or fam == "unknown":
            kept.append(line)

    filtered_text = "\n".join(kept).strip()
    n_kept = sum(1 for ln in kept if ln.startswith("- "))
    if n_total > 0 and n_kept < n_total:
        print(f"[Lessons] Filtered to {n_kept}/{n_total} relevant to "
              f"task family '{current_family}'")
    return filtered_text


# ---------------------------------------------------------------------------
# Phase 2: Opus session-review writeups (reviews.md)
# ---------------------------------------------------------------------------

_REVIEWS_HEADER = (
    "# Opus session reviews\n"
    "\n"
    "Auto-appended after each EpisodeRetry session of 3+ episodes. Each\n"
    "entry is an abstracted writeup produced by Claude Opus reading the\n"
    f"whole session. Most recent first. Capped at {MAX_REVIEWS} entries.\n"
    "\n"
    "Entries below are HYPOTHESES, not rules. They reference episodes by\n"
    "number so future sessions can corroborate or refute them.\n"
    "\n"
)

# Each review entry begins with a `### YYYY-MM-DD HH:MM` line so we can split
# the file back into entries when trimming.
_REVIEW_ENTRY_HEAD_RE = re.compile(r"^### \d{4}-\d{2}-\d{2} \d{2}:\d{2}\b",
                                   re.MULTILINE)


def _render_review_entry(task: str, review: Dict[str, Any],
                         model: str) -> str:
    """Render one review (as returned by review_session) to Markdown."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"### {ts} [{model}] \"{task}\"",
        "",
        (review.get("task_writeup") or "(no writeup returned)").strip(),
        "",
    ]

    obs = review.get("observations") or []
    if obs:
        lines.append("**Observations:**")
        for o in obs:
            text = o.get("text", "(no text)")
            ev = o.get("evidence_episodes") or []
            ev_str = f"ep {', '.join(str(e) for e in ev)}" if ev else "no episodes cited"
            conf = o.get("confidence", "?")
            lines.append(f"- {text}  _({conf}; {ev_str})_")
        lines.append("")

    fps = review.get("false_positives") or []
    if fps:
        lines.append("**False-positive successes flagged:**")
        for fp in fps:
            lines.append(f"- Episode {fp.get('episode', '?')}: "
                         f"{fp.get('reason', '(no reason)')}")
        lines.append("")

    diags = review.get("exploration_diagnoses") or []
    if diags:
        lines.append("**Exploration diagnoses:**")
        for d in diags:
            lines.append(f"- Episode {d.get('episode', '?')} -- "
                         f"deviation: {d.get('deviation', '?')}; "
                         f"diagnosis: {d.get('diagnosis', '?')}")
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def _split_review_entries(body: str) -> List[str]:
    """Split a `reviews.md` body (no header) into individual entries."""
    if not body.strip():
        return []
    # Find the start indices of every entry head; slice body into entries.
    starts = [m.start() for m in _REVIEW_ENTRY_HEAD_RE.finditer(body)]
    if not starts:
        return []
    starts.append(len(body))
    return [body[starts[i]:starts[i + 1]].strip()
            for i in range(len(starts) - 1)]


def append_review(task: str, review: Dict[str, Any], model: str,
                  reviews_file: Path = None) -> Path:
    """Write a rendered review to reviews.md, most-recent-first.

    `review` is the dict returned by agent.review_session.review_session
    (has `task_writeup`, `observations`, `false_positives`,
    `exploration_diagnoses`). `model` is the short reviewer-model label
    (e.g. "opus-4-7") for the entry header.
    """
    if reviews_file is None:
        reviews_file = REVIEWS_FILE

    new_entry = _render_review_entry(task, review, model)

    existing_entries: List[str] = []
    if reviews_file.exists():
        existing_body = reviews_file.read_text()
        # Strip the header by finding the first entry; everything before is
        # boilerplate we'll rewrite from _REVIEWS_HEADER.
        first = _REVIEW_ENTRY_HEAD_RE.search(existing_body)
        if first:
            existing_entries = _split_review_entries(existing_body[first.start():])

    entries = [new_entry] + existing_entries
    entries = entries[:MAX_REVIEWS]

    reviews_file.write_text(_REVIEWS_HEADER + "\n\n".join(entries) + "\n")
    return reviews_file


def read_reviews_section(reviews_file: Path = None,
                         current_task: Optional[str] = None) -> str:
    """Return reviews.md contents, optionally filtered to the task's family.

    Mirrors `read_lessons_section`'s family-filtering behaviour so the
    Opus reviewer (Phase 2) and any future Phase-3 system-prompt injector
    can pull only entries relevant to the current task family.
    """
    if reviews_file is None:
        reviews_file = REVIEWS_FILE
    if not reviews_file.exists():
        return ""

    text = reviews_file.read_text()
    if not text.strip():
        return ""

    if current_task is None:
        return text.strip()

    from agent.outcome_checker import classify_task
    current_family = classify_task(current_task)
    if current_family == "unknown":
        return text.strip()

    # Split into entries; keep header + same-family/unknown-family entries.
    first = _REVIEW_ENTRY_HEAD_RE.search(text)
    if not first:
        return text.strip()  # nothing to filter

    header = text[:first.start()].rstrip()
    entries = _split_review_entries(text[first.start():])

    # The entry header line embeds the task in quotes: ### TS [model] "task"
    task_re = re.compile(r'^### \d{4}-\d{2}-\d{2} \d{2}:\d{2}\s+\[[^\]]+\]\s+"(?P<task>.+?)"')
    kept: List[str] = []
    for entry in entries:
        head_line = entry.splitlines()[0] if entry else ""
        m = task_re.match(head_line)
        if not m:
            kept.append(entry)  # malformed -- keep permissively
            continue
        fam = classify_task(m.group("task"))
        if fam == current_family or fam == "unknown":
            kept.append(entry)

    if not kept:
        return ""
    return (header + "\n\n" + "\n\n".join(kept)).strip()
