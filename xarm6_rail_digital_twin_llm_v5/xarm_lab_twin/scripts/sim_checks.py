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
    """Trajectory datasets must declare their units.

    `body_poses` is metres while `ee_pos_mm` / `rail_mm` are millimetres, and that
    mismatch would silently corrupt a VLA export. An explicit per-dataset `units`
    attribute is the minimum guard.

    Only files written by the current format are checked. Recordings predating the
    change carry no `format_version` and are reported as legacy rather than FAIL —
    rewriting existing sessions would destroy captured data to satisfy a linter.
    """
    import glob
    sessions = sorted(glob.glob("recordings/*/trajectory.h5"))
    if not sessions:
        return [CheckResult("recording.units", SKIP, "no recordings/*/trajectory.h5 found")]
    try:
        import h5py
    except ImportError:
        return [CheckResult("recording.units", SKIP, "h5py not available")]

    from recording import TRAJECTORY_FORMAT_VERSION

    checked, legacy, bad = 0, 0, []
    for path in sessions:
        label = os.path.basename(os.path.dirname(path))
        try:
            with h5py.File(path, "r") as f:
                version = int(f.attrs.get("format_version", 1))
                if version < TRAJECTORY_FORMAT_VERSION:
                    legacy += 1
                    continue
                checked += 1
                undeclared = [d for d in f
                              if isinstance(f[d], h5py.Dataset) and "units" not in f[d].attrs]
                if undeclared:
                    bad.append(f"{label}: {undeclared}")
        except OSError as exc:
            bad.append(f"{label}: unreadable ({exc})")

    if bad:
        return [CheckResult("recording.units", FAIL,
                            f"v{TRAJECTORY_FORMAT_VERSION} recordings missing 'units': "
                            + "; ".join(bad))]
    detail = f"{checked} current-format recording(s) declare units"
    if legacy:
        detail += f"; {legacy} legacy (pre-v{TRAJECTORY_FORMAT_VERSION}) left untouched"
    return [CheckResult("recording.units", PASS, detail)]


def check_bin_bodies_exist(scene) -> list[CheckResult]:
    """Every container physical_outcome() can report a cube as being "in".

    These names drive the grader's `<cube> in <bin>` facts. A name here that is
    absent from the scene raises at snapshot time; a container in the scene that
    is missing here is worse -- it looks like a bin, the arm can drop a cube in
    it, and the grader silently never reports success.
    """
    from sim.mujoco_env import BIN_BODIES
    missing = scene.missing(list(BIN_BODIES))
    if missing:
        return [CheckResult("sim.bin_bodies", FAIL,
                            f"BIN_BODIES names bodies absent from the scene: {missing}")]
    return [CheckResult("sim.bin_bodies", PASS,
                        f"all {len(BIN_BODIES)} container(s) exist: {', '.join(BIN_BODIES)}")]


def check_perturbable_bodies_exist(scene) -> list[CheckResult]:
    """Bodies the scene randomizer jitters must exist.

    PERTURBABLE_BODIES named a bare "red_cube" long after that body was deleted,
    so a third of the intended domain randomisation was silently a no-op — the
    kind of failure that quietly weakens a dataset rather than breaking a run.
    """
    try:
        from envs.scene_randomizer import PERTURBABLE_BODIES
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("randomizer.bodies_exist", SKIP, f"import failed: {exc}")]
    missing = scene.missing(sorted(PERTURBABLE_BODIES))
    if missing:
        return [CheckResult("randomizer.bodies_exist", FAIL,
                            f"PERTURBABLE_BODIES names bodies absent from the scene "
                            f"(their jitter silently does nothing): {missing}")]
    return [CheckResult("randomizer.bodies_exist", PASS,
                        f"all {len(PERTURBABLE_BODIES)} perturbable bodies exist")]


def check_joint_limits_match_scene(scene) -> list[CheckResult]:
    """arm_backend's joint/rail limits must match the scene's jnt_range.

    They are duplicated on purpose — hardware/real_arm.py must run on a machine
    with no MuJoCo installed — so this is the guard that stops the copy drifting
    from the scene the way the prompt coordinates did.
    """
    import numpy as np
    from arm_backend import RAIL_LIMITS_MM, XARM6_JOINT_LIMITS_DEG
    m = scene._model
    bad = []
    for i, (lo, hi) in enumerate(XARM6_JOINT_LIMITS_DEG, start=1):
        try:
            s_lo, s_hi = np.degrees(m.jnt_range[m.joint(f"joint{i}").id])
        except Exception as exc:  # noqa: BLE001
            bad.append(f"joint{i} not in scene ({exc})")
            continue
        if abs(s_lo - lo) > 0.15 or abs(s_hi - hi) > 0.15:
            bad.append(f"joint{i}: arm_backend [{lo}, {hi}] vs scene "
                       f"[{s_lo:.1f}, {s_hi:.1f}]")
    r = m.jnt_range[m.joint("rail").id] * 1000.0
    if abs(r[0] - RAIL_LIMITS_MM[0]) > 1 or abs(r[1] - RAIL_LIMITS_MM[1]) > 1:
        bad.append(f"rail: arm_backend {RAIL_LIMITS_MM} vs scene "
                   f"({r[0]:.0f}, {r[1]:.0f})")
    if bad:
        return [CheckResult("arm.joint_limits_match_scene", FAIL, "; ".join(bad))]
    return [CheckResult("arm.joint_limits_match_scene", PASS,
                        "6 joint limits + rail travel match the scene")]


def check_base_offset_matches_scene(scene) -> list[CheckResult]:
    """arm_backend's base offset and bench height must match the scene.

    RealXArmAPI converts world<->base with these numbers. If they drift from the
    scene the twin was planned against, every Cartesian move on hardware lands
    somewhere else — the failure this constant was added to fix.
    """
    import mujoco as mj
    from arm_backend import (BENCH_TOP_Z_MM, MEASURED_BASE_Z_DELTA_MM,
                             SCENE_BASE_AT_RAIL_ZERO_MM)
    m, d = scene._model, mj.MjData(scene._model)
    d.qpos[m.jnt_qposadr[m.joint("rail").id]] = 0.0
    mj.mj_kinematics(m, d)
    base = d.xpos[m.body("xarm_base").id] * 1000.0
    bad = []
    for axis, got, want in zip("xyz", base, SCENE_BASE_AT_RAIL_ZERO_MM):
        if abs(got - want) > 1.0:
            bad.append(f"base {axis}: arm_backend {want:.0f} vs scene {got:.0f}")
    gid = m.geom("bench_top").id
    top = (d.geom_xpos[gid][2] + m.geom_size[gid][2]) * 1000.0
    if abs(top - BENCH_TOP_Z_MM) > 1.0:
        bad.append(f"bench top: arm_backend {BENCH_TOP_Z_MM:.0f} vs scene {top:.0f}")
    if bad:
        return [CheckResult("arm.base_offset_matches_scene", FAIL, "; ".join(bad))]
    detail = (f"scene base at rail=0 {SCENE_BASE_AT_RAIL_ZERO_MM} and bench top "
              f"{BENCH_TOP_Z_MM:.0f} match the scene")
    if MEASURED_BASE_Z_DELTA_MM:
        detail += (f"; measured cell base is {MEASURED_BASE_Z_DELTA_MM:+.0f} mm "
                   f"from the scene (known sim-to-real gap, scene not yet corrected)")
    return [CheckResult("arm.base_offset_matches_scene", PASS, detail)]


def check_arm_backend_parity(scene=None) -> list[CheckResult]:
    """Both backends must cover the ArmBackend contract (B2).

    A contract method that is merely absent on one backend is an AttributeError
    waiting to fire partway through a plan — after part of it has already run on
    a real arm. Hardware-free: the classes are inspected, never instantiated.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "test_arm_backend"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("PASS", "FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("arm.backend_parity", FAIL, "; ".join(failed))]
    return [CheckResult("arm.backend_parity", PASS,
                        f"{len(tail)} parity checks pass over "
                        f"{len(__import__('arm_backend').ARM_BACKEND_METHODS)} contract methods")]


def check_preflight_level1(scene=None) -> list[CheckResult]:
    """The Level 1 query gate must reject bad plans without a controller (A8.1).

    Hardware-free: the self-test drives the gate against a stub controller that
    deliberately misbehaves the way the real one does -- is_tcp_limit and
    is_joint_limit both lie -- and asserts the verdict does not depend on
    either. See docs/vendor_reference/README.md for the measurements.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "scripts/preflight_level1.py", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("preflight.level1_gate", FAIL, "; ".join(failed))]
    return [CheckResult("preflight.level1_gate", PASS,
                        f"{len(tail)} Level 1 gate tests pass (no controller needed)")]


def check_plan_gate(scene=None) -> list[CheckResult]:
    """The pre-action gate must dispatch nothing when it rejects a plan (A8).

    Hardware-free: the LLM response is stubbed and the validator is a plain
    function, so this runs with no arm, no controller container and no API call.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "agent.test_plan_gate"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("PASS", "FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("agent.plan_gate", FAIL, "; ".join(failed))]
    return [CheckResult("agent.plan_gate", PASS,
                        f"{len(tail)} pre-action gate tests pass")]


def check_motion_error_audit(scene=None) -> list[CheckResult]:
    """No motion command may fail and still report success (A7).

    Two records: the audit itself, and a self-test proving the detector still
    catches the bug. A clean audit from a detector that cannot detect anything
    is worse than no check, because it reads as evidence of safety.
    """
    import subprocess
    out = []

    proc = subprocess.run([sys.executable, "scripts/audit_motion_errors.py", "--self-test"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        out.append(CheckResult("motion.audit_self_test", FAIL,
                               "detector no longer catches known-bad patterns: "
                               + proc.stdout.strip().replace("\n", " ")[:200]))
    else:
        n = sum(1 for ln in proc.stdout.splitlines() if "PASS" in ln)
        out.append(CheckResult("motion.audit_self_test", PASS,
                               f"detector verified against {n} known-good/bad patterns"))

    proc = subprocess.run([sys.executable, "scripts/audit_motion_errors.py", "--check"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        findings = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().endswith("()")]
        out.append(CheckResult("motion.no_silent_failures", FAIL,
                               "motion command(s) return success on an exception path: "
                               + "; ".join(findings)))
    else:
        out.append(CheckResult("motion.no_silent_failures", PASS,
                               "no motion command returns success on an exception path"))
    return out


def check_real_arm_contract(scene=None) -> list[CheckResult]:
    """Run the hardware-free real_arm tests as part of the sweep.

    No arm, no network: the SDK is faked. These guard the silent-success defect,
    so they belong in the gate rather than in a file someone remembers to run.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "hardware.test_real_arm"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("PASS", "FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("hardware.real_arm_contract", FAIL, "; ".join(failed))]
    return [CheckResult("hardware.real_arm_contract", PASS,
                        f"{len(tail)} hardware-free contract tests pass")]


STATIC_CHECKS = [
    check_prompt_renders,
    check_prompt_objects_exist,
    check_prompt_no_stale_coords,
    check_push_targets_pushable,
    check_grippable_bodies_exist,
    check_objects_json_exist,
    check_registry_positions,
    check_registry_seed_literals,
    check_bin_bodies_exist,
    check_perturbable_bodies_exist,
    check_recording_units,
    check_real_arm_contract,
    check_joint_limits_match_scene,
    check_base_offset_matches_scene,
    check_arm_backend_parity,
    check_plan_gate,
    check_preflight_level1,
    check_motion_error_audit,
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
