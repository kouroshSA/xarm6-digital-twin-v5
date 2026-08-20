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


# Literal coordinates that were hardcoded into the system prompt before A1 and are
# now generated from the scene. If any reappears in the *rendered* prompt, someone
# has re-typed a coordinate by hand and the drift has started again.
#
# Each entry is (needle, what it used to assert). Matching is done against the
# rendered prompt text, so a value that legitimately reappears via the generated
# section (because the scene really does put an object there) is excluded below.
RETIRED_PROMPT_COORDS: list[tuple[str, str]] = [
    ("(-300, -250", "old heater_shaker xy"),
    ("(+200, -300", "old pcr_module / well_plate_B xy"),
    ("(867,", "old OT-2 deck row-1 x"),
    ("(956,", "old OT-2 deck row-2 x"),
    ("(1044,", "old OT-2 deck row-3 x"),
    ("(1133,", "old OT-2 deck row-4 x"),
    ('"y": 150', "old green_cube y in the worked example"),
    ('"y": 350', "old green_bin y in the worked example"),
]

OBJECTS_JSON = "agent/objects.json"
REGISTRY_SEEDS = "agent/object_registry.py"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _render_prompt(scene) -> str:
    """Render the system prompt exactly as LLMBrain would, minus the runtime-only
    sections (registry/world-model/lessons), which carry no hardcoded geometry."""
    from agent.llm_brain import SYSTEM_PROMPT_TEMPLATE
    from agent.scene_geometry import render_geometry_section, worked_example_coords
    return SYSTEM_PROMPT_TEMPLATE.format(
        registry_context="(omitted)",
        speed_cap_section="(omitted)",
        world_model_section="(omitted)",
        lessons_section="(omitted)",
        scene_geometry_section=render_geometry_section(scene),
        **worked_example_coords(scene),
    )


def check_prompt_renders(scene) -> list[CheckResult]:
    """The template must format cleanly — a missing placeholder is a hard failure
    at LLM-call time, which is the worst place to discover it."""
    try:
        text = _render_prompt(scene)
    except KeyError as exc:
        return [CheckResult("prompt.renders", FAIL, f"missing template placeholder: {exc}")]
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("prompt.renders", FAIL, f"{type(exc).__name__}: {exc}")]
    if "GENERATED from" not in text:
        return [CheckResult("prompt.renders", FAIL,
                            "rendered prompt does not contain the generated geometry section")]
    return [CheckResult("prompt.renders", PASS, f"renders, {len(text)} chars, geometry injected")]


def check_prompt_objects_exist(scene) -> list[CheckResult]:
    """Every body the generated geometry section names must exist in the scene."""
    from agent.scene_geometry import PROMPT_BODIES
    missing = scene.missing(PROMPT_BODIES)
    if missing:
        return [CheckResult("prompt.objects_exist", FAIL,
                            f"listed in PROMPT_BODIES but absent from scene: {missing}")]
    return [CheckResult("prompt.objects_exist", PASS,
                        f"all {len(PROMPT_BODIES)} prompt-named bodies exist")]


def check_prompt_no_stale_coords(scene) -> list[CheckResult]:
    """Guard against re-hardcoding: none of the retired literals may reappear."""
    try:
        text = _render_prompt(scene)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("prompt.no_stale_coords", SKIP, f"prompt did not render: {exc}")]
    hits = [f"{needle!r} ({why})" for needle, why in RETIRED_PROMPT_COORDS if needle in text]
    if hits:
        return [CheckResult("prompt.no_stale_coords", FAIL,
                            "retired coordinate(s) back in the prompt: " + "; ".join(hits))]
    return [CheckResult("prompt.no_stale_coords", PASS,
                        f"none of the {len(RETIRED_PROMPT_COORDS)} retired literals present")]


def check_push_targets_pushable(scene) -> list[CheckResult]:
    """The generated 'Movable objects' list must contain only genuinely pushable
    bodies — push_object fails at runtime on anything without a free joint."""
    from agent.scene_geometry import PROMPT_BODIES
    static = [n for n in PROMPT_BODIES
              if (info := scene.get(n)) is not None and not info.pushable]
    try:
        text = _render_prompt(scene)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("prompt.push_targets", SKIP, f"prompt did not render: {exc}")]

    movable_block = text.split("Movable objects")[-1].split("Static fixtures")[0]
    leaked = [n for n in static if n in movable_block]
    if leaked:
        return [CheckResult("prompt.push_targets", FAIL,
                            f"static bodies advertised as movable: {leaked}")]
    return [CheckResult("prompt.push_targets", PASS,
                        f"{len(static)} static fixture(s) correctly excluded from the movable list")]


def check_registry_positions(scene) -> list[CheckResult]:
    """agent/objects.json seeds must match the scene (regen with scripts/regen_registry.py)."""
    if not os.path.exists(OBJECTS_JSON):
        return [CheckResult("registry.positions", SKIP, f"{OBJECTS_JSON} not found")]
    raw = json.load(open(OBJECTS_JSON))
    drifted = []
    for name, obj in raw.items():
        info = scene.get(name)
        have = obj.get("position_xyz_m")
        if info is None or have is None:
            continue
        err = scene.xy_error_mm(name, (have[0] * 1000.0, have[1] * 1000.0))
        if err is not None and err > XY_TOL_MM:
            drifted.append(f"{name} ({err:.0f}mm)")
    if drifted:
        return [CheckResult("registry.positions", FAIL,
                            "stale vs scene, run scripts/regen_registry.py: " + ", ".join(drifted))]
    return [CheckResult("registry.positions", PASS, f"all {len(raw)} seeds match the scene")]


def check_registry_seed_literals(scene) -> list[CheckResult]:
    """The hardcoded position_xyz_m literals in object_registry.py must match too —
    they re-seed objects.json on a fresh clone, so a stale one resurrects the drift."""
    import re
    if not os.path.exists(REGISTRY_SEEDS):
        return [CheckResult("registry.seed_literals", SKIP, f"{REGISTRY_SEEDS} not found")]
    name_re = re.compile(r'^\s*name="([^"]+)",\s*$')
    pos_re = re.compile(r'^\s*position_xyz_m=\[([^\]]*)\],?\s*$')
    cur, drifted = None, []
    for line in open(REGISTRY_SEEDS):
        m = name_re.match(line)
        if m:
            cur = m.group(1)
            continue
        m = pos_re.match(line)
        if m and cur:
            try:
                have = [float(v) for v in m.group(1).split(",")]
            except ValueError:
                continue
            err = scene.xy_error_mm(cur, (have[0] * 1000.0, have[1] * 1000.0))
            if err is not None and err > XY_TOL_MM:
                drifted.append(f"{cur} ({err:.0f}mm)")
    if drifted:
        return [CheckResult("registry.seed_literals", FAIL,
                            "stale seeds vs scene: " + ", ".join(drifted))]
    return [CheckResult("registry.seed_literals", PASS, "seed literals match the scene")]


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
    check_prompt_renders,
    check_prompt_objects_exist,
    check_prompt_no_stale_coords,
    check_push_targets_pushable,
    check_grippable_bodies_exist,
    check_objects_json_exist,
    check_registry_positions,
    check_registry_seed_literals,
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
