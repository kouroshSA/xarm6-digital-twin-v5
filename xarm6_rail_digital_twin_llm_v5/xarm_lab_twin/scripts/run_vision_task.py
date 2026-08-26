#!/usr/bin/env python3
"""Run a task with the wrist camera as the source of truth.

    python scripts/run_vision_task.py "put the blue cube in the cup"
    python scripts/run_vision_task.py "..." --mode real --ip 192.168.1.xxx

Identical to ``run_task.py`` in every respect -- same flags, same preflight, same
pre-action gate, same episode loop -- except that the planner is told to locate
objects with the camera instead of trusting registry coordinates. Every argument
``run_task.py`` accepts works here.

Which one to use
----------------
``run_task.py`` when the coordinates are right and precision matters: OT-2 deck
slots, rack slots, placing a plate on a fixed fixture. Those positions are
surveyed geometry, and a camera measurement is a worse number than the survey.

``run_vision_task.py`` when the coordinates are a guess: anything that has been
moved, anything the operator repositioned, and **everything on real hardware**.
There, ``RealXArmAPI.get_body_pose`` is a stub, so ``registry.refresh_from_sim``
silently skips every object and the registry keeps the static seeds it was built
with. They describe where things were placed once, not where they are.

Why a second entry point rather than a flag on the first
--------------------------------------------------------
The choice is a property of the *session*, not of one command, and putting it in
the name makes it visible in shell history, in a recording's metadata and in
whatever someone pastes into a bug report. A flag buried among twenty others
does not.

It is a wrapper, not a copy. ``run_task.main()`` does all the work; this file
sets one environment variable and calls it. Duplicating 500 lines of argument
parsing, preflight and loop handling to change one paragraph of prompt would be
this repo's first standing defect class, and the copies would drift the first
time anyone fixed a bug in one of them.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

from agent.llm_brain import VISION_FIRST_ENV  # noqa: E402


def main() -> int:
    # Set before run_task is imported: it builds the brain during main(), and
    # the flag is read when the system prompt is rendered.
    os.environ[VISION_FIRST_ENV] = "1"

    from scripts.run_task import main as run_task_main

    print("[vision-task] wrist camera is the source of truth for object "
          "positions; registry coordinates are a prior for aiming it.")
    return run_task_main()


if __name__ == "__main__":
    sys.exit(main())
