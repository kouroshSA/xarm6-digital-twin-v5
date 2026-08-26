"""Write the wrist-camera block into ``lab_scene_primitive.xml`` from the calibration.

    python -m perception.sync_scene            # rewrite the block, then regen
    python -m perception.sync_scene --check    # is the scene in sync? (exit 1 if not)

The camera's numbers live in ``perception/d435i_calib.py``. This script is how
they get into the scene, and it is idempotent -- it replaces any existing
``<body name="wrist_camera">`` block rather than appending a second one, so
re-running after a re-calibration is the whole update procedure.

Deliberately *not* hand-editing the XML: CLAUDE.md's first standing defect class
is one fact stored in several places that then drift. A camera pose typed into
the scene and also written in the calibration module is precisely that. Here the
XML is a build product, ``--check`` proves it is current, and
``scripts/sim_checks.py::check_wrist_camera_matches_calib`` runs that same check
inside the normal sweep so a stale scene fails the suite rather than surviving
to confuse someone later.

The block goes inside ``<body name="gripper">`` on purpose: ``build_mesh_scene.py``
copies that subtree verbatim into the generated ``lab_scene.xml``, so one edit
reaches both the primitive and the mesh scene with no second copy to maintain.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from . import d435i_calib as calib
from . import observer_calib

HERE = Path(__file__).resolve().parent
ENVS = HERE.parent / "envs"
PRIMITIVE = ENVS / "lab_scene_primitive.xml"
GENERATED = ENVS / "lab_scene.xml"

_OBSERVER_RE = re.compile(
    r"[ \t]*<!-- Fixed observer cameras\..*?"
    r"<body name=\"" + observer_calib.OBSERVER_BODY_NAME + r"\".*?\n[ \t]*</body>\n",
    re.DOTALL,
)

_BLOCK_RE = re.compile(
    r"[ \t]*<!-- Intel RealSense D435i wrist camera.*?"
    r"<body name=\"" + calib.CAMERA_BODY_NAME + r"\".*?\n[ \t]*</body>\n",
    re.DOTALL,
)


def _indent_of_gripper_children(text: str) -> str:
    """Indentation for direct children of ``<body name="gripper">``."""
    m = re.search(r"^([ \t]*)<body name=\"gripper\"", text, re.MULTILINE)
    if not m:
        raise SystemExit("no <body name=\"gripper\"> in lab_scene_primitive.xml")
    return m.group(1) + "  "


def _gripper_close_index(lines: list[str], open_idx: int, open_indent: str) -> int:
    """Line index of the ``</body>`` that closes the gripper body."""
    want = open_indent + "</body>"
    for i in range(open_idx + 1, len(lines)):
        if lines[i].rstrip() == want:
            return i
    raise SystemExit("could not find the </body> closing <body name=\"gripper\">")


def render_block() -> str:
    text = PRIMITIVE.read_text()
    return calib.emit_scene_xml(indent=_indent_of_gripper_children(text))


def _sync_observers(text: str, check_only: bool) -> tuple[str, int]:
    """Write (or check) the fixed observer block. Same contract as the wrist one.

    The observers live in the worldbody rather than in the gripper subtree --
    they do not move with the arm -- so they get their own anchored block
    instead of riding along with build_mesh_scene.py's verbatim copy.
    """
    block = observer_calib.emit_scene_xml(indent=" " * 4)
    existing = _OBSERVER_RE.search(text)

    if check_only:
        if existing is None:
            print("FAIL: lab_scene_primitive.xml has no observer block")
            return text, 1
        if existing.group(0).rstrip("\n") != block:
            print("FAIL: the observer block no longer matches "
                  "perception/observer_calib.py.\n"
                  "      Run: python -m perception.sync_scene")
            return text, 1
        print("OK: scene observer cameras match the calibration")
        return text, 0

    if existing is not None:
        return text[:existing.start()] + block + "\n" + text[existing.end():], 0

    # First insertion: immediately before the arm assembly, so the observers are
    # near the top of the worldbody and clearly not part of the robot.
    marker = text.index("    <!-- Rail assembly height")
    return text[:marker] + block + "\n\n" + text[marker:], 0


def sync(check_only: bool = False) -> int:
    text = PRIMITIVE.read_text()
    block = render_block()

    existing = _BLOCK_RE.search(text)
    if check_only:
        if existing is None:
            print("FAIL: lab_scene_primitive.xml has no wrist_camera block")
            return 1
        if existing.group(0).rstrip("\n") != block:
            print("FAIL: the wrist_camera block in lab_scene_primitive.xml no "
                  "longer matches perception/d435i_calib.py.\n"
                  "      Run: python -m perception.sync_scene")
            return 1
        print("OK: scene wrist camera matches the calibration")
        _, rc = _sync_observers(text, check_only=True)
        return rc

    if existing is not None:
        new_text = text[:existing.start()] + block + "\n" + text[existing.end():]
    else:
        lines = text.splitlines()
        open_idx = next(i for i, ln in enumerate(lines)
                        if '<body name="gripper"' in ln)
        open_indent = lines[open_idx][:len(lines[open_idx]) - len(lines[open_idx].lstrip())]
        close_idx = _gripper_close_index(lines, open_idx, open_indent)
        lines = lines[:close_idx] + block.splitlines() + lines[close_idx:]
        new_text = "\n".join(lines) + "\n"

    new_text, _ = _sync_observers(new_text, check_only=False)

    if new_text == text:
        print("lab_scene_primitive.xml already current")
    else:
        PRIMITIVE.write_text(new_text)
        print(f"updated {PRIMITIVE.relative_to(HERE.parent)}")

    # The generated scene is what every entry point loads, so regenerating is
    # part of the same operation -- leaving it stale would mean the twin runs a
    # camera the calibration no longer describes.
    result = subprocess.run(
        [sys.executable, str(ENVS / "build_mesh_scene.py")],
        cwd=HERE.parent, capture_output=True, text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify the scene matches the calibration; do not write")
    return sync(check_only=ap.parse_args().check)


if __name__ == "__main__":
    sys.exit(main())
