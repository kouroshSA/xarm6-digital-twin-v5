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
import pathlib
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
        # Rendered for real, not stubbed: this is the switch run_vision_task.py
        # flips, and a check that stubbed it could not tell whether it still works.
        vision_policy_section=__import__(
            "agent.llm_brain", fromlist=["x"]).render_vision_policy(),
        registry_context="(omitted)",
        speed_cap_section="(omitted)",
        world_model_section="(omitted)",
        # Rendered for real, not stubbed: these checks scan the prompt for
        # stale coordinates, and a skill is exactly the kind of place a stale
        # coordinate could hide. Stubbing it would exempt skills from the one
        # check that would catch that.
        skills_section=__import__("agent.skills", fromlist=["x"]).render_skills_section(
            __import__("agent.skills", fromlist=["x"]).load_skills()),
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


#: Internal helpers in sim/ik_solver.py that touch the shared mjData without
#: taking the lock themselves. Every caller reaches them through solve() or
#: _pos_error_for(), both of which hold it, and self.lock is an RLock so the
#: nesting is safe. Listed explicitly so the check below stays strict about
#: everything else.
_LOCK_EXEMPT_FUNCS = {
    "_solve_pink", "_rot_error_for",
    "_solve_jacobian_position", "_solve_jacobian_6dof",
}


def check_layers_measure_after_reset(scene=None) -> list[CheckResult]:
    """The per-episode layer re-run must happen AFTER reset_scene(), not before.

    Layer 1 MEASURES the scene. Before the reset, the scene is whatever the
    previous episode left behind -- cube knocked aside, cup displaced -- and
    the reset then restores it out from under those numbers. So every episode
    after the first planned against coordinates for a world that no longer
    existed.

    Cost, measured: this task ran at 76% (n=12) before the feedback loop was
    wired, 40% after (n=8, deterministic 2/5 every run), and 60% once the
    re-run was moved after the reset.

    Structural rather than behavioural, for the same reason as
    sim.shared_state_locked: the defect is an ORDERING, and asserting the
    order directly cannot be flaky. A behavioural version would have to
    detect a degraded success rate, which needs many runs and still could not
    say why.
    """
    src = pathlib.Path("agent/episode_loop.py").read_text()
    i = src.find("while ctx.episode_num <= ctx.max_episodes:")
    if i < 0:
        return [CheckResult("agent.layers_measure_after_reset", FAIL,
                            "could not find the episode loop")]
    body = src[i:]
    reset_at = body.find("self.arm.reset_scene()")
    prep_at = body.find("self.prepare_task(")
    if reset_at < 0 or prep_at < 0:
        return [CheckResult("agent.layers_measure_after_reset", FAIL,
                            "reset_scene() or prepare_task() missing from the loop")]
    if prep_at < reset_at:
        return [CheckResult("agent.layers_measure_after_reset", FAIL,
                            "prepare_task() runs BEFORE reset_scene(): Layer 1 "
                            "would measure the previous episode's leftovers")]
    return [CheckResult("agent.layers_measure_after_reset", PASS,
                        "the per-episode layer re-run measures a freshly reset scene")]


#: Private helpers that are ALLOWED to write joint targets straight to ctrl:
#: the pacing layer itself, and the zero-duration fallback it delegates to.
_UNPACED_OK = {"_execute_paced_arm", "_execute_joint_angles", "reset_scene"}


def check_motion_primitives_paced(scene=None) -> list[CheckResult]:
    """No public motion primitive may drive joints straight to ctrl.

    MuJoCo position actuators have no velocity limit, so a primitive that
    calls _execute_joint_angles directly moves at whatever the PD gains allow
    and --speed-tier has no effect on it. That is not a cosmetic difference:
    the operator flagged the return-home as a motion nobody would command on
    hardware, and rehearsing it in the twin makes it look normal.

    Found by inspection three times now -- set_position/set_rail_position,
    then go_home, then wave_goodbye -- because each behavioural pacing check
    only looks at the primitive it names. This one looks at all of them, so
    the next unpaced primitive fails a check instead of waiting to be noticed.
    """
    import re
    src = pathlib.Path("sim/mujoco_env.py").read_text()
    lines = src.split("\n")

    # Per public method, does it ever PACE, and does it ever write ctrl直接?
    # A primitive that paces and falls back to a direct write for zero-length
    # moves is fine -- set_position does exactly that. One that ONLY ever
    # writes ctrl has no speed control at all, which is the defect.
    fn, seen = "?", {}
    for l in lines:
        m = re.match(r"    def (\w+)", l)
        if m:
            fn = m.group(1)
            seen.setdefault(fn, {"paced": False, "raw": False})
        if fn == "?":
            continue
        if "self._execute_paced_arm(" in l or "self._execute_paced_rail(" in l:
            seen[fn]["paced"] = True
        if "self._execute_joint_angles(" in l:
            seen[fn]["raw"] = True

    bad = [f"{fn}()" for fn, v in seen.items()
           if v["raw"] and not v["paced"]
           and not fn.startswith("_") and fn not in _UNPACED_OK]
    if bad:
        return [CheckResult("sim.motion_primitives_paced", FAIL,
                            f"never paced: {'; '.join(sorted(bad))} -- writes joint "
                            f"targets straight to ctrl, so speed caps do not apply")]
    return [CheckResult("sim.motion_primitives_paced", PASS,
                        "no public motion primitive writes joint targets unpaced")]


def check_shared_state_locked(scene=None) -> list[CheckResult]:
    """Every touch of the shared mjData in sim/ must hold the lock.

    The sim thread steps physics continuously while a viewer thread copies
    mjData to draw it. Any third party touching that data unlocked can land
    mid-step, and MuJoCo aborts the PROCESS with "attempting to copy mjData
    while stack is in use".

    That is not theoretical: it was losing whole runs. `launch_passive` copies
    mjData internally to build its scene and was called outside the lock, so
    construction itself could kill the run -- 2 of 6 rendered runs died before
    executing a single episode. Locking it took a 7-run sample to zero losses.

    A structural check rather than a behavioural one on purpose: the failure is
    a thread race, so a behavioural test would be flaky in exactly the way that
    makes a check untrustworthy. This cannot be flaky -- it either finds an
    unguarded call or it does not.
    """
    import re
    bad = []
    for f in sorted(pathlib.Path("sim").glob("*.py")):
        lines = f.read_text().split("\n")
        stack, fn = [], "?"
        for i, l in enumerate(lines):
            if not l.strip() or l.strip().startswith("#"):
                continue
            ind = len(l) - len(l.lstrip())
            stack = [d for d in stack if d < ind]
            m = re.match(r"\s*def (\w+)", l)
            if m:
                fn = m.group(1)
            if re.search(r"with\s+[\w\.]*lock\b", l):
                stack.append(ind)
                continue
            touches = re.search(r"\bmujoco\.mj_(step|forward|collision)\(", l) \
                or "viewer.launch_passive" in l
            if touches and not stack and fn not in _LOCK_EXEMPT_FUNCS:
                bad.append(f"{f.name}:{i+1} in {fn}()")
    if bad:
        return [CheckResult("sim.shared_state_locked", FAIL,
                            f"{len(bad)} unguarded: {'; '.join(bad[:4])}")]
    return [CheckResult("sim.shared_state_locked", PASS,
                        "every mjData touch in sim/ holds the lock")]


def check_home_pose_is_clear(scene) -> list[CheckResult]:
    """Home must be collision-free, and the twin must home where the cell does.

    Both halves failed silently for a long time. The sim had its own
    hand-picked home (joint1 = +90 deg, facing the wall) that rested the
    gripper inside the PCR module, while hardware/real_arm.py used the pose
    measured on the arm. Nothing compared them, and nothing checked that home
    was reachable without touching anything -- so it only surfaced once the
    rail gained swept-path validation and EVERY episode died on its first
    command, unable to move at all from a pose already in contact.
    """
    import numpy as np, mujoco
    from arm_backend import HOME_JOINTS_DEG, HOME_RAIL_MM
    out = []
    m, d = scene._model, mujoco.MjData(scene._model)

    for i, a in enumerate(HOME_JOINTS_DEG, start=1):
        d.qpos[m.joint(f"joint{i}").qposadr[0]] = np.deg2rad(a)
    d.qpos[m.joint("rail").qposadr[0]] = HOME_RAIL_MM / 1000.0
    mujoco.mj_forward(m, d)
    mujoco.mj_collision(m, d)

    ARM = {"base_link", "link1_geom", "link2_geom", "link3_geom", "link4_geom",
           "link5_geom", "link6_geom", "gripper_geom", "carriage_geom"}
    hits = []
    for k in range(d.ncon):
        c = d.contact[k]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if (ARM & {g1, g2}) and not ({g1, g2} <= ARM):
            hits.append((g1, g2))
    if hits:
        out.append(CheckResult("arm.home_pose_clear", FAIL,
                               f"home pose is in contact with {hits[:3]}"))
    else:
        out.append(CheckResult("arm.home_pose_clear", PASS,
                               "home pose is collision-free in this scene"))

    try:
        import importlib.util
        spec = importlib.util.find_spec("hardware.real_arm")
        shares = spec is not None
    except Exception:  # noqa: BLE001
        shares = False
    src = (pathlib.Path("hardware/real_arm.py").read_text()
           if pathlib.Path("hardware/real_arm.py").exists() else "")
    redefines = any(ln.startswith("HOME_JOINTS_DEG =") for ln in src.splitlines())
    if redefines:
        out.append(CheckResult("arm.home_pose_shared", FAIL,
                               "hardware/real_arm.py defines its own HOME_JOINTS_DEG "
                               "instead of importing arm_backend's -- two homes is "
                               "two robots"))
    else:
        out.append(CheckResult("arm.home_pose_shared", PASS,
                               "both backends take the home pose from arm_backend"))
    return out


def check_ab_harness(scene=None) -> list[CheckResult]:
    """The A/B harness must not confuse "no headroom" with "no benefit".

    It is the instrument every prompt-layer claim will be measured with, so a
    silent bug in its verdict logic would let unfalsifiable prose accumulate
    while appearing to be evidence-based.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "scripts/ab_test.py", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("harness.ab_test", FAIL, "; ".join(failed))]
    return [CheckResult("harness.ab_test", PASS, f"{len(tail)} A/B verdict tests pass")]


def check_instruction_intake(scene=None) -> list[CheckResult]:
    """Layer 0's contract must reject plans, invented measurements and drift.

    No model and no sim. Layer 0 runs on every prompt, so its guardrail is on
    the critical path for every task the system attempts.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "agent.instruction_intake", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("agent.instruction_intake", FAIL, "; ".join(failed))]
    return [CheckResult("agent.instruction_intake", PASS,
                        f"{len(tail)} Layer 0 contract tests pass")]


def check_prompt_refiner(scene=None) -> list[CheckResult]:
    """Layer 2's contract must reject invention, fabrication and plans.

    No model and no sim: the self-test drives check_contract directly and
    stubs the model call, so the guardrail is verified without spending a
    token. The guardrail is the only reason Layer 2 is safe to enable.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "agent.prompt_refiner", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("agent.prompt_refiner", FAIL, "; ".join(failed))]
    return [CheckResult("agent.prompt_refiner", PASS,
                        f"{len(tail)} Layer 2 contract tests pass")]


def check_skills(scene=None) -> list[CheckResult]:
    """Planning skills must load, and must not smuggle geometry into the prompt.

    Skills carry procedure; positions and dimensions come from the registry at
    runtime. A skill is a durable, authoritative-looking place for a stale
    coordinate to hide, so the loader rejects any that contain one and this
    check proves the rejection still works.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "agent.skills", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("agent.skills", FAIL, "; ".join(failed))]
    return [CheckResult("agent.skills", PASS, f"{len(tail)} skill-loader tests pass")]


def check_task_validator(scene=None) -> list[CheckResult]:
    """Layer 1 task validation must resolve referents and refuse the impossible.

    Hardware-free and sim-free: the self-test drives it against a stub registry
    containing the deliberate two-red-cube ambiguity.
    """
    import subprocess
    proc = subprocess.run([sys.executable, "-m", "agent.task_validator", "--self-test"],
                          capture_output=True, text=True)
    tail = [ln.strip() for ln in proc.stdout.splitlines()
            if ln.strip().startswith(("[PASS", "[FAIL"))]
    if proc.returncode != 0:
        failed = [ln for ln in tail if ln.startswith("[FAIL")] or [proc.stderr.strip()[-200:]]
        return [CheckResult("agent.task_validator", FAIL, "; ".join(failed))]
    return [CheckResult("agent.task_validator", PASS,
                        f"{len(tail)} Layer 1 task-validation tests pass")]


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


def check_wrist_camera_matches_calib(scene=None) -> list[CheckResult]:
    """The scene's wrist camera must still be the one perception/ describes.

    Two failure modes, both of which end with the twin quietly modelling a
    camera the code thinks it has calibrated:

    1. someone hand-edits the camera block in ``lab_scene_primitive.xml``
       instead of editing ``perception/d435i_calib.py`` and re-syncing;
    2. someone re-syncs the primitive scene but forgets
       ``envs/build_mesh_scene.py``, so the *generated* scene every entry point
       actually loads keeps the old pose.

    Neither would show up in a render -- a camera 30 mm off still produces a
    perfectly plausible picture. Cheap to check, invisible otherwise.
    """
    import mujoco as mj
    import numpy as np

    from perception import d435i_calib as calib
    from perception.sync_scene import PRIMITIVE, sync

    results: list[CheckResult] = []

    if sync(check_only=True) != 0:
        results.append(CheckResult(
            "perception.scene_in_sync", FAIL,
            f"{PRIMITIVE.name} does not match perception/d435i_calib.py; "
            "run `python -m perception.sync_scene`"))
    else:
        results.append(CheckResult("perception.scene_in_sync", PASS,
                                   "primitive scene matches the calibration"))

    # Check the model MuJoCo actually loads, not the XML text. The scene's
    # comments contain "--", which MuJoCo accepts and a strict XML parser does
    # not, and going through MuJoCo also resolves the camera's world pose through
    # the whole kinematic chain rather than trusting a local pos attribute.
    model = getattr(scene, "_model", None)
    if model is None:
        model = mj.MjModel.from_xml_path(scene_geometry.DEFAULT_SCENE)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    l6 = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link6")
    if l6 < 0:
        return results + [CheckResult("perception.generated_scene", FAIL,
                                      "scene has no link6 to measure against")]
    r_flange = data.xmat[l6].reshape(3, 3)

    # The invariant that matters: where does the camera sit relative to the
    # flange? That is what the hand-eye calibration states, and it survives any
    # change to the arm's pose or to the rail.
    for cam_name, (r_want, t_want) in (
        (calib.COLOR_CAM_NAME, calib.flange_to_color_optical()),
        (calib.DEPTH_CAM_NAME, calib.flange_to_depth_optical()),
    ):
        cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            results.append(CheckResult(
                "perception.generated_scene", FAIL,
                f"{scene_geometry.DEFAULT_SCENE} has no camera {cam_name!r}; "
                "run `python -m perception.sync_scene`"))
            continue

        t_got = r_flange.T @ (data.cam_xpos[cid] - data.xpos[l6])
        off_mm = float(np.max(np.abs(t_got - t_want))) * 1000.0

        r_got = (r_flange.T @ data.cam_xmat[cid].reshape(3, 3)) @ np.diag([1.0, -1.0, -1.0])
        ang_deg = float(np.degrees(np.arccos(np.clip(
            (np.trace(r_got.T @ r_want) - 1.0) / 2.0, -1.0, 1.0))))

        # Sub-micrometre and sub-millidegree: these are the same numbers passed
        # through a float formatter, so anything larger means a real edit.
        if off_mm > 1e-3 or ang_deg > 1e-3:
            results.append(CheckResult(
                "perception.generated_scene", FAIL,
                f"{cam_name} is {off_mm:.4f} mm / {ang_deg:.4f} deg off the "
                f"hand-eye calibration in {scene_geometry.DEFAULT_SCENE}; "
                f"rerun `python -m perception.sync_scene`"))
            continue

        # Intrinsics too -- a camera in the right place with the wrong focal
        # length is just as wrong, and just as invisible in a render.
        res, ss = model.cam_resolution[cid], model.cam_sensorsize[cid]
        intr = calib.COLOR_INTRINSICS if cam_name == calib.COLOR_CAM_NAME \
            else calib.DEPTH_INTRINSICS
        fx_len, fy_len, cx_len, cy_len = model.cam_intrinsic[cid]
        got = (fx_len / ss[0] * res[0], fy_len / ss[1] * res[1],
               (res[0] - 1) / 2.0 - cx_len / ss[0] * res[0],
               (res[1] - 1) / 2.0 - cy_len / ss[1] * res[1])
        want_intr = (intr.fx, intr.fy, intr.cx, intr.cy)
        worst = max(abs(a - b) for a, b in zip(got, want_intr))
        if worst > 0.01:
            results.append(CheckResult(
                "perception.generated_scene", FAIL,
                f"{cam_name} intrinsics differ from the device's by up to "
                f"{worst:.3f} px (scene fx={got[0]:.3f} cx={got[2]:.3f}, "
                f"device fx={intr.fx:.3f} cx={intr.cx:.3f})"))
        else:
            results.append(CheckResult(
                "perception.generated_scene", PASS,
                f"{cam_name} matches the hand-eye pose and the device's "
                f"{intr.width}x{intr.height} intrinsics"))

    return results


def check_ggcnn_weights_load(scene=None) -> list[CheckResult]:
    """The vendored GG-CNN weights must still fit the vendored architecture.

    Fast on purpose: constructs the detector and runs one synthetic frame. It
    does *not* pose the arm or check that grasps land on objects -- that is
    `python -m perception.grasp.test_ggcnn`, which needs physics and takes far
    too long for a gate meant to run in seconds.

    What this catches is drift between `weights/*.pt` and `_ggcnn*_net.py`:
    edit the network definition and `load_state_dict` fails loudly here rather
    than at the first grasp attempt. SKIPs rather than fails when torch is
    absent, since the sim itself does not require it.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return [CheckResult("perception.ggcnn_weights", SKIP,
                            "torch not installed; grasp detection unavailable")]

    import numpy as np

    from perception.d435i_calib import COLOR_INTRINSICS
    from perception.grasp import GGCNNDetector
    from perception.rgbd import RGBDFrame

    results = []
    for model in ("ggcnn", "ggcnn2"):
        try:
            det = GGCNNDetector(model=model)
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(
                f"perception.ggcnn_weights[{model}]", FAIL,
                f"{type(exc).__name__}: {exc}"))
            continue

        # A flat plane with a raised block: enough for the network to find
        # something, without depending on the scene or on physics.
        depth = np.full((480, 640), 0.40, dtype=np.float32)
        depth[200:280, 280:360] = 0.33
        frame = RGBDFrame(
            color=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=depth, intrinsics=COLOR_INTRINSICS, cam_to_world=None)
        try:
            grasps = det.detect(frame, top_k=1)
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(
                f"perception.ggcnn_weights[{model}]", FAIL,
                f"inference raised {type(exc).__name__}: {exc}"))
            continue

        if not grasps:
            results.append(CheckResult(
                f"perception.ggcnn_weights[{model}]", FAIL,
                "no grasp on a synthetic block; the weights load but the "
                "network is not producing usable output"))
        elif grasps[0].position_world is not None:
            # Guards the specific silent failure: a pose-less frame must not
            # yield world coordinates, or the arm gets camera coordinates.
            results.append(CheckResult(
                f"perception.ggcnn_weights[{model}]", FAIL,
                "a frame with no cam_to_world produced a world position"))
        else:
            results.append(CheckResult(
                f"perception.ggcnn_weights[{model}]", PASS,
                f"weights load into the vendored net; synthetic block gives "
                f"q={grasps[0].quality:.2f}, width={grasps[0].width_m * 1000:.0f} mm"))
    return results


def check_grounding_model_available(scene=None) -> list[CheckResult]:
    """Is the Grounding DINO checkpoint on disk, and does the package import?

    Checks the HuggingFace cache rather than loading the model: the weights are
    ~700 MB and 232 M parameters, so a real load has no place in a suite that
    gates an edit-verify loop. `python -m perception.language.test_targeting`
    does the actual work.

    The failure this catches is the annoying one -- everything imports, the code
    is correct, and the first `target()` call stalls trying to reach the network
    or dies offline because nobody noticed the checkpoint was never fetched.
    SKIPs rather than fails when the optional deps are absent.
    """
    try:
        import transformers  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return [CheckResult("perception.grounding_model", SKIP,
                            "transformers not installed; language targeting "
                            "unavailable")]

    from perception.language.grounding import MODEL_ID

    missing = [f for f in ("config.json", "preprocessor_config.json")
               if not isinstance(try_to_load_from_cache(MODEL_ID, f), str)]
    weights = any(isinstance(try_to_load_from_cache(MODEL_ID, f), str)
                  for f in ("model.safetensors", "pytorch_model.bin"))
    if missing or not weights:
        return [CheckResult(
            "perception.grounding_model", SKIP,
            f"{MODEL_ID} is not in the HuggingFace cache "
            f"(missing {missing or ['weights']}); the first target() call will "
            f"try to download ~700 MB")]
    return [CheckResult("perception.grounding_model", PASS,
                        f"{MODEL_ID} is cached locally")]


def check_vision_actions_wired(scene=None) -> list[CheckResult]:
    """The vision actions must be dispatchable, documented, and gate-aware.

    Three lists have to agree about the same set of actions, and nothing else
    compares them:

    * ``LLMBrain._dispatch`` -- can the action run at all;
    * the prompt's command vocabulary -- will the planner ever emit it;
    * ``validate_plan.DEFERRED`` -- does the pre-action gate know its pose is
      camera-resolved, or does it report an unrecognised action and reject the
      whole plan on real hardware.

    Dropping one is silent in every direction. An action missing from the prompt
    is simply never used; one missing from DEFERRED breaks only in ``--mode
    real``, which is the worst place to find out. Cheap to check here.

    No model loading and no arm: this asks whether the wiring exists, not
    whether vision works. ``agent/test_vision_dispatch.py`` does that.
    """
    from scripts.validate_plan import DEFERRED, IGNORED

    results: list[CheckResult] = []
    try:
        from agent.llm_brain import SYSTEM_PROMPT_TEMPLATE
        from agent.vision_targeting import VisionTargeting  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("agent.vision_actions", FAIL,
                            f"cannot import the vision layer: "
                            f"{type(exc).__name__}: {exc}")]

    # Probe _dispatch without an arm: an unknown action returns -1, and every
    # vision action refuses a missing description long before touching hardware.
    class _NullArm:
        last_refusal = ""
        scene_xml = scene_geometry.DEFAULT_SCENE

    brain = object.__new__(__import__("agent.llm_brain",
                                      fromlist=["x"]).LLMBrain)
    brain.arm = _NullArm()
    brain._vision = None
    brain.registry = None

    missing_dispatch = [a for a in sorted(DEFERRED)
                        if brain._dispatch(a, {}) == -1]
    if missing_dispatch:
        results.append(CheckResult(
            "agent.vision_actions", FAIL,
            f"in validate_plan.DEFERRED but not dispatchable: "
            f"{missing_dispatch}"))
    else:
        results.append(CheckResult(
            "agent.vision_actions", PASS,
            f"all {len(DEFERRED)} vision action(s) dispatch and are known to "
            f"the pre-action gate"))

    undocumented = [a for a in sorted(DEFERRED)
                    if f"- {a}" not in SYSTEM_PROMPT_TEMPLATE]
    if undocumented:
        results.append(CheckResult(
            "agent.vision_prompt", FAIL,
            f"dispatchable but absent from the prompt's command vocabulary, so "
            f"the planner will never emit them: {undocumented}"))
    else:
        results.append(CheckResult(
            "agent.vision_prompt", PASS,
            "every vision action appears in the prompt vocabulary"))

    overlap = DEFERRED & IGNORED
    if overlap:
        results.append(CheckResult(
            "agent.vision_actions", FAIL,
            f"{sorted(overlap)} are in both DEFERRED and IGNORED; IGNORED "
            f"claims they are not controller motion, which is false"))
    return results


def check_vision_first_entry_point(scene=None) -> list[CheckResult]:
    """run_vision_task.py must actually change the prompt, and only it.

    The whole entry point is one environment variable, which is exactly the kind
    of wiring that breaks silently: rename the variable, drop the placeholder, or
    stub it in one of the two format() call sites, and run_vision_task.py keeps
    running, keeps printing its banner, and quietly plans from registry
    coordinates like run_task.py. Nothing else would notice.

    So assert both directions -- the policy appears when the flag is set, and is
    absent when it is not, because a policy that is always on would silently
    change run_task.py too.
    """
    import os

    from agent.llm_brain import VISION_FIRST_ENV, render_vision_policy

    results: list[CheckResult] = []
    had = os.environ.get(VISION_FIRST_ENV)
    try:
        os.environ.pop(VISION_FIRST_ENV, None)
        off = _render_prompt(scene)
        os.environ[VISION_FIRST_ENV] = "1"
        on = _render_prompt(scene)
    except Exception as exc:  # noqa: BLE001
        return [CheckResult("agent.vision_first", FAIL,
                            f"prompt did not render: {type(exc).__name__}: {exc}")]
    finally:
        os.environ.pop(VISION_FIRST_ENV, None)
        if had is not None:
            os.environ[VISION_FIRST_ENV] = had

    marker = "THIS SESSION IS VISION-FIRST"
    if marker in off:
        results.append(CheckResult(
            "agent.vision_first", FAIL,
            "the vision-first policy renders with the flag UNSET, so plain "
            "run_task.py sessions are being told to use the camera too"))
    elif marker not in on:
        results.append(CheckResult(
            "agent.vision_first", FAIL,
            f"setting {VISION_FIRST_ENV} does not change the prompt; "
            f"run_vision_task.py is a no-op"))
    else:
        results.append(CheckResult(
            "agent.vision_first", PASS,
            f"{VISION_FIRST_ENV} adds {len(on) - len(off)} chars of policy; "
            f"unset leaves the prompt unchanged"))

    entry = pathlib.Path("scripts/run_vision_task.py")
    if not entry.exists():
        results.append(CheckResult("agent.vision_first", FAIL,
                                   "scripts/run_vision_task.py is missing"))
    elif "run_task" not in entry.read_text():
        results.append(CheckResult(
            "agent.vision_first", FAIL,
            "run_vision_task.py no longer delegates to run_task.main(); if it "
            "has been forked into its own copy, the two will drift"))
    return results


def check_camera_pose_backs_out_tcp_offset(scene=None) -> list[CheckResult]:
    """The wrist camera's pose must be built from the FLANGE, not the TCP.

    ``get_position()`` reports whatever the TCP points at. This cell runs a
    217 mm tool offset, so it reports the gripper FINGERTIP, while the hand-eye
    calibration is measured from the FLANGE. Composing one onto the other puts
    the camera a whole tool-length away.

    Measured on hardware 2026-08-26: uncorrected, a benchtop 296 mm from the
    lens deprojected to base z = -266 mm against a true -72, and the
    camera-to-surface range came out 200 mm short. Every log line looked
    healthy throughout -- the frames were fine, the intrinsics were fine, the
    detector would have been fine. Only the answer was wrong, which is defect
    class #2 exactly.

    This asks what ``cam_to_world`` ACTUALLY RETURNS for an arm that reports an
    offset, not whether some helper can compute one. It also pins the other
    half: an arm reporting no offset must be left alone, or the sim path that
    ``perception/test_projection.py`` validates would silently shift.
    """
    import numpy as np

    name = "perception.camera_pose_tcp"
    try:
        from perception.realsense_camera import RealSenseWristCamera
    except ImportError as exc:
        return [CheckResult(name, SKIP, f"pyrealsense2 unavailable: {exc}")]

    pose = [300.0, -20.0, 250.0, 180.0, 0.0, 0.0]

    class _Arm:
        def __init__(self, off=None):
            if off is not None:
                self.tcp_offset_z_mm = off
        def get_position(self):
            return (0, list(pose))

    cam = object.__new__(RealSenseWristCamera)      # no device required
    cam.arm = _Arm()                                # reports no offset
    bare = cam.cam_to_world()
    cam.arm = _Arm(0.0)
    zero = cam.cam_to_world()
    cam.arm = _Arm(217.0)
    offset = cam.cam_to_world()

    if bare is None or offset is None:
        return [CheckResult(name, FAIL, "cam_to_world() returned None for a posed arm")]

    if not np.allclose(bare, zero, atol=1e-9):
        return [CheckResult(name, FAIL,
                            "an arm reporting a ZERO offset moved the camera; the "
                            "sim path would shift under test_projection")]

    moved = float(np.linalg.norm(offset[:3, 3] - bare[:3, 3]) * 1000.0)
    if abs(moved - 217.0) > 1e-3:
        return [CheckResult(name, FAIL,
                            f"a 217 mm TCP offset moved the camera {moved:.3f} mm; "
                            f"the flange pose is not being recovered")]

    if not np.allclose(offset[:3, :3], bare[:3, :3], atol=1e-12):
        return [CheckResult(name, FAIL,
                            "backing out the TCP offset rotated the camera; it is a "
                            "pure translation along the tool axis")]

    # Direction matters as much as magnitude: with roll=180 the tool +z points
    # down, so the flange is 217 mm ABOVE the fingertip. A sign error would pass
    # a magnitude-only check and put every grasp 434 mm out.
    if offset[2, 3] - bare[2, 3] < 0:
        return [CheckResult(name, FAIL,
                            "the correction moved the camera DOWN from a tool "
                            "pointing down; the offset is being added, not removed")]

    return [CheckResult(name, PASS,
                        "cam_to_world backs a 217 mm TCP offset out to the flange "
                        "(217.000 mm, up, rotation untouched); zero offset is a no-op")]


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
    check_task_validator,
    check_skills,
    check_prompt_refiner,
    check_instruction_intake,
    check_ab_harness,
    check_home_pose_is_clear,
    check_shared_state_locked,
    check_motion_primitives_paced,
    check_layers_measure_after_reset,
    check_motion_error_audit,
    check_wrist_camera_matches_calib,
    check_camera_pose_backs_out_tcp_offset,
    check_ggcnn_weights_load,
    check_grounding_model_available,
    check_vision_actions_wired,
    check_vision_first_entry_point,
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
