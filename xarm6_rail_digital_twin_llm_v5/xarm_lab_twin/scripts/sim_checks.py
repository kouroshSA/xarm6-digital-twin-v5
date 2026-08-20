"""Static sim invariants — fast, no physics stepping, no LLM, no hardware.

Each check returns one or more `CheckResult` records so `scripts/task_sweep.py` can
render them as text or JSON and set an exit code. Checks must be *fast* (the whole
static suite should run in a second or two) because they gate an edit-verify loop.

The central invariant: **the scene XML is the single source of truth for geometry.**
Anything holding a second copy of a coordinate is drift waiting to happen.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.getcwd())

from agent import scene_geometry
from sim.mujoco_env import GRIPPABLE_BODIES

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# Tolerance for "the prompt agrees with the scene". A millimetre is far tighter than
# the arm's real accuracy, but these are both nominal figures from the same source —
# any disagreement at all is a bug, not noise.
XY_TOL_MM = 1.0


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Coordinates hardcoded in agent/llm_brain.py's SYSTEM_PROMPT_TEMPLATE.
#
# THIS TABLE IS TEMPORARY. It is a hand-transcribed mirror of prose in the prompt,
# which is exactly the duplication that caused the drift it detects. A1 replaces the
# hardcoded prompt block with one generated from the scene; when that lands, this
# table and `check_prompt_geometry` are deleted and replaced by a generated-vs-
# committed diff check (see `check_objects_json_fresh` for the shape that takes).
#
# Transcribed 2026-08-20 from llm_brain.py lines 120-202 and the worked example at
# lines 209-214. Line numbers are recorded so a reviewer can confirm each claim.
# ---------------------------------------------------------------------------
PROMPT_XY_CLAIMS: list[tuple[str, tuple[float, float], str]] = [
    ("heater_shaker", (-300.0, -250.0), "llm_brain.py:141 'Heater-Shaker on the bench at (-300, -250)'"),
    ("vortex_genie", (-200.0, 250.0), "llm_brain.py:154 'Vortex-Genie 2 ... at world (-200, +250)'"),
    ("pcr_module", (200.0, -300.0), "llm_brain.py:168 'thermocycler at world (+200, -300)'"),
    ("well_plate_B", (200.0, -300.0), "llm_brain.py:131 'Plate B starts on the bench at (+200, -300, 762)'"),
    ("well_plate_A", (867.0, 132.0), "llm_brain.py:120,130 'Plate A starts in slot 1' slot1=(867,+132)"),
    ("tip_box", (1133.0, 132.0), "llm_brain.py:132 'Tip box starts in slot 10 at (1133, +132, 795)'"),
    ("green_cube", (0.0, 150.0), "llm_brain.py:209 worked example move_to y=150"),
    ("green_bin", (0.0, 350.0), "llm_brain.py:213 worked example move_to y=350"),
]

# push_object targets advertised to the LLM at llm_brain.py:63-71.
PROMPT_PUSH_TARGETS = [
    "green_cube", "blue_cube",
    "green_bin", "blue_bin",
    "left_tube_rack", "right_tube_rack",
    "tube_L1", "tube_L2", "tube_L3", "tube_R1", "tube_R2", "tube_R3",
    "well_plate_A", "well_plate_B", "tip_box",
    "heater_shaker", "vortex_genie", "pcr_module",
]

OBJECTS_JSON = "agent/objects.json"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_prompt_objects_exist(scene) -> list[CheckResult]:
    """Every body the prompt names must exist in the scene."""
    named = {n for n, _, _ in PROMPT_XY_CLAIMS} | set(PROMPT_PUSH_TARGETS)
    missing = scene.missing(sorted(named))
    if missing:
        return [CheckResult("prompt.objects_exist", FAIL,
                            f"named in prompt but absent from scene: {missing}")]
    return [CheckResult("prompt.objects_exist", PASS,
                        f"all {len(named)} prompt-named bodies exist")]


def check_prompt_geometry(scene) -> list[CheckResult]:
    """Every coordinate the prompt hardcodes must match the scene within XY_TOL_MM."""
    out = []
    for name, claimed, src in PROMPT_XY_CLAIMS:
        err = scene.xy_error_mm(name, claimed)
        if err is None:
            out.append(CheckResult(f"prompt.geometry[{name}]", FAIL, f"body absent; {src}"))
        elif err > XY_TOL_MM:
            actual = scene.xy_mm(name)
            out.append(CheckResult(
                f"prompt.geometry[{name}]", FAIL,
                f"prompt says ({claimed[0]:.0f},{claimed[1]:.0f}) but scene has "
                f"({actual[0]:.0f},{actual[1]:.0f}) — off by {err:.0f}mm; {src}"))
        else:
            out.append(CheckResult(f"prompt.geometry[{name}]", PASS, f"matches scene ({err:.2f}mm)"))
    return out


def check_push_targets_pushable(scene) -> list[CheckResult]:
    """Advertised push_object targets need a free joint, or the call fails at runtime."""
    missing, static = [], []
    for name in PROMPT_PUSH_TARGETS:
        info = scene.get(name)
        if info is None:
            missing.append(name)
        elif not info.pushable:
            static.append(name)
    if missing or static:
        bits = []
        if missing:
            bits.append(f"absent: {missing}")
        if static:
            bits.append(f"static (no free joint, push_object will fail): {static}")
        return [CheckResult("prompt.push_targets", FAIL, "; ".join(bits))]
    return [CheckResult("prompt.push_targets", PASS,
                        f"all {len(PROMPT_PUSH_TARGETS)} advertised targets are pushable")]


def check_grippable_bodies_exist(scene) -> list[CheckResult]:
    """Every GRIPPABLE_BODIES key must be a real body, or gripper_close cannot weld it."""
    missing = scene.missing(sorted(GRIPPABLE_BODIES))
    if missing:
        return [CheckResult("sim.grippable_bodies", FAIL,
                            f"in GRIPPABLE_BODIES but absent from scene: {missing}")]
    return [CheckResult("sim.grippable_bodies", PASS,
                        f"all {len(GRIPPABLE_BODIES)} entries exist in scene")]


def check_objects_json_exist(scene) -> list[CheckResult]:
    """Every registry key must be a real body."""
    if not os.path.exists(OBJECTS_JSON):
        return [CheckResult("registry.objects_exist", SKIP, f"{OBJECTS_JSON} not found")]
    raw = json.load(open(OBJECTS_JSON))
    keys = list(raw) if isinstance(raw, dict) else [o.get("name") for o in raw]
    missing = scene.missing([k for k in keys if k])
    if missing:
        return [CheckResult("registry.objects_exist", FAIL,
                            f"in {OBJECTS_JSON} but absent from scene: {missing}")]
    return [CheckResult("registry.objects_exist", PASS, f"all {len(keys)} registry keys exist")]


def check_recording_units(scene=None) -> list[CheckResult]:
    """Spatial datasets in trajectory.h5 should declare their units.

    `body_poses` is written in metres while `ee_pos_mm` / `rail_mm` are millimetres.
    Until A4 resolves that, an explicit `units` attribute is the minimum needed to
    stop a downstream VLA export silently mixing scales.
    """
    import glob
    sessions = sorted(glob.glob("recordings/*/trajectory.h5"))
    if not sessions:
        return [CheckResult("recording.units", SKIP, "no recordings/*/trajectory.h5 found")]
    try:
        import h5py
    except ImportError:
        return [CheckResult("recording.units", SKIP, "h5py not available")]

    latest = sessions[-1]
    spatial = ["rail_mm", "ee_pos_mm", "ee_rpy_deg", "body_poses", "joints_deg"]
    with h5py.File(latest, "r") as f:
        present = [d for d in spatial if d in f]
        undeclared = [d for d in present if "units" not in f[d].attrs]
    if undeclared:
        return [CheckResult("recording.units", FAIL,
                            f"{os.path.basename(os.path.dirname(latest))}: datasets without a "
                            f"'units' attribute: {undeclared}")]
    return [CheckResult("recording.units", PASS, f"all {len(present)} spatial datasets declare units")]


STATIC_CHECKS = [
    check_prompt_objects_exist,
    check_prompt_geometry,
    check_push_targets_pushable,
    check_grippable_bodies_exist,
    check_objects_json_exist,
    check_recording_units,
]


def run_static_checks(scene_xml: str = scene_geometry.DEFAULT_SCENE) -> list[CheckResult]:
    """Run every static check. Errors become FAIL records rather than propagating,
    so one broken check cannot hide the results of the others."""
    scene = scene_geometry.load(scene_xml)
    results: list[CheckResult] = []
    for fn in STATIC_CHECKS:
        try:
            results.extend(fn(scene))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            results.append(CheckResult(fn.__name__, FAIL, f"{type(exc).__name__}: {exc}"))
    return results
