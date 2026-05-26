# agent/lessons.py
"""
Cross-session reflection file. After each run, we append a one-line
lesson summarizing the task, the model, and the outcome. The next session's
system prompt reads recent lessons so Claude can avoid repeating mistakes.

File format: a markdown list, most-recent-first. We keep at most MAX_LESSONS
entries -- a soft cap to stop the prompt from inflating without bound.
"""
from datetime import datetime
from pathlib import Path
from typing import Iterable

LESSONS_FILE = Path("lessons.md")
MAX_LESSONS = 20

_HEADER = (
    "# Cross-session lessons\n"
    "\n"
    "Auto-appended after each task run. Most recent first. Capped at "
    f"{MAX_LESSONS} entries.\n"
    "\n"
)


def _summarize_outcome(planned_commands: list, results: list,
                       physical_outcome: str) -> str:
    """Distill the dispatch results into a one-line outcome string."""
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

    base = "SUCCESS"
    if physical_outcome:
        base += f" ({physical_outcome})"
    return base


def append_lesson(task_prompt: str, model_short: str,
                  planned_commands: list, results: list,
                  physical_outcome: str = "",
                  lessons_file: Path = None) -> Path:
    """Write a one-line lesson to lessons.md and trim to MAX_LESSONS."""
    if lessons_file is None:
        lessons_file = LESSONS_FILE

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    outcome = _summarize_outcome(planned_commands, results, physical_outcome)
    new_entry = f'- {ts} [{model_short}] "{task_prompt}" -> {outcome}'

    existing_entries = []
    if lessons_file.exists():
        for line in lessons_file.read_text().splitlines():
            if line.startswith("- "):
                existing_entries.append(line)

    entries = [new_entry] + existing_entries
    entries = entries[:MAX_LESSONS]

    lessons_file.write_text(_HEADER + "\n".join(entries) + "\n")
    return lessons_file


def read_lessons_section(lessons_file: Path = None) -> str:
    """Return the contents of lessons.md, or '' if absent/empty.
    Used by LLMBrain to inject lessons into the system prompt."""
    if lessons_file is None:
        lessons_file = LESSONS_FILE
    if not lessons_file.exists():
        return ""
    text = lessons_file.read_text().strip()
    return text if text else ""
