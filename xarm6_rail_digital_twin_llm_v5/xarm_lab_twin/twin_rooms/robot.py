# twin_rooms/robot.py
"""Robot types: `robots/<type>/robot.json`, defined once, referenced by rooms.

    from twin_rooms.robot import load_robot_type
    rt = load_robot_type("xarm6_rail")
    rt.names["ee_site"]                   # 'end_effector'
    rt.value("home.joints_deg")           # resolved from arm_backend.HOME_JOINTS_DEG
    rt.model_path()                       # the scene the robot is extracted from

A robot.json holds NAMES and pointers, not copies. A number that already has
one source in code is written {"ref": "module:NAME"} and resolved by import, so
the definition cannot drift from that source. Joint and rail limits are not in
robot.json at all: the compiled model's jnt_range is their source.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco

from twin_rooms.schema import RoomError, validate_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_DIR = PROJECT_ROOT / "robots"
SCHEMA_PATH = Path(__file__).with_name("robot_schema.json")

#: Rooms may always say end_effector "none" (bare flange).
NO_EFFECTOR = "none"


def resolve_ref(ref: str) -> Any:
    """'module:NAME' -> the value of NAME in module (imported from the project root)."""
    mod, name = ref.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        raise RoomError(ref, [f"ref {ref!r} does not resolve: {e}"]) from None


def _resolve(v: Any) -> Any:
    if isinstance(v, dict) and set(v) == {"ref"}:
        v = resolve_ref(v["ref"])
    if isinstance(v, tuple):
        v = list(v)
    return v


@dataclass
class RobotType:
    type: str
    name: str
    description: str
    dir: Path
    model: dict
    names: Dict[str, Any]
    rail: dict
    home: dict
    ik: dict
    end_effectors: dict
    raw: dict = field(default_factory=dict, repr=False)

    # ---- model -----------------------------------------------------------
    def model_path(self, variant: Optional[str] = None) -> Path:
        v = variant or self.model["default"]
        if v not in self.model["variants"]:
            raise RoomError(self.type, [f"no model variant {v!r}; have "
                                        f"{sorted(self.model['variants'])}"])
        return (self.dir / self.model["variants"][v]).resolve()

    @property
    def mount_body(self) -> str:
        return self.model["bodies"][0]

    @property
    def has_rail(self) -> bool:
        return "rail_joint" in self.names

    # ---- values ----------------------------------------------------------
    def value(self, dotted: str) -> Any:
        """A field by dotted path with refs resolved, e.g. 'home.rail_mm' or
        'end_effectors.parallel_gripper.reach_m'."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"robot {self.type!r} has no field {dotted!r}")
            node = node[part]
        return _resolve(node)

    def end_effector_names(self) -> List[str]:
        return [NO_EFFECTOR] + [k for k in self.end_effectors if k != "default"]

    def end_effector(self, name: Optional[str]) -> dict:
        """The effector's entry, refs resolved; None means the type's default."""
        name = name or self.end_effectors["default"]
        if name == NO_EFFECTOR:
            return {}
        if name not in self.end_effectors or name == "default":
            raise RoomError(self.type, [f"no end effector {name!r}; have "
                                        f"{self.end_effector_names()}"])
        return {k: _resolve(v) for k, v in self.end_effectors[name].items()}

    def refs(self) -> List[str]:
        """Every {"ref": ...} in the definition (for tests and validation)."""
        out: List[str] = []

        def walk(n):
            if isinstance(n, dict):
                if set(n) == {"ref"}:
                    out.append(n["ref"])
                else:
                    for v in n.values():
                        walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)
        walk(self.raw)
        return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _subtree(m: mujoco.MjModel, root: int) -> List[int]:
    out = []
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = int(m.body_parentid[x])
        if x == root:
            out.append(b)
    return out


def check_against_model(rt: RobotType, m: mujoco.MjModel, where: str) -> List[str]:
    """Every name in robot.json must exist in `m`, inside the robot's own bodies,
    and the robot's bodies must carry no actuator that robot.json does not name
    (an unnamed actuator would be built into every room and driven by nobody)."""
    errs: List[str] = []

    def nid(kind, name):
        return mujoco.mj_name2id(m, kind, name)

    body_ids = set()
    for b in rt.model["bodies"] + rt.model.get("operator_aids", []):
        i = nid(mujoco.mjtObj.mjOBJ_BODY, b)
        if i < 0:
            errs.append(f"{where}: model.bodies: no body {b!r}")
        elif int(m.body_parentid[i]) != 0:
            errs.append(f"{where}: model.bodies: {b!r} is not a top-level body "
                        f"(its parent is {m.body(int(m.body_parentid[i])).name!r})")
        else:
            body_ids.update(_subtree(m, i))
    if errs:
        return errs

    n = rt.names
    eff = {k: v for k, v in rt.end_effectors.items() if k != "default"}
    want = {   # (field, object kind, name)
        mujoco.mjtObj.mjOBJ_JOINT: [("arm_joints", x) for x in n["arm_joints"]]
            + ([("rail_joint", n["rail_joint"])] if "rail_joint" in n else [])
            + [(f"end_effectors.{k}.finger_joints", x) for k, e in eff.items()
               for x in e.get("finger_joints", [])],
        mujoco.mjtObj.mjOBJ_ACTUATOR: [("arm_actuators", x) for x in n["arm_actuators"]]
            + ([("rail_actuator", n["rail_actuator"])] if "rail_actuator" in n else [])
            + [(f"end_effectors.{k}.finger_actuators", x) for k, e in eff.items()
               for x in e.get("finger_actuators", [])],
        mujoco.mjtObj.mjOBJ_SITE: [("ee_site", n["ee_site"])],
        mujoco.mjtObj.mjOBJ_BODY: [(f, n[f]) for f in ("base_body", "flange_body",
                                                          "gripper_body", "wrist_camera_body")
                                   if f in n] + [("led_strips", x) for x in n.get("led_strips", [])],
        mujoco.mjtObj.mjOBJ_GEOM: [("arm_collision_geoms", x) for x in n.get("arm_collision_geoms", [])]
            + [(f"end_effectors.{k}.geoms", x) for k, e in eff.items() for x in e.get("geoms", [])],
        mujoco.mjtObj.mjOBJ_CAMERA: [(f"cameras.{k}", x) for k, x in n.get("cameras", {}).items()],
        mujoco.mjtObj.mjOBJ_MATERIAL: [(f"end_effectors.{k}.materials", x) for k, e in eff.items()
                                       for x in e.get("materials", [])],
    }
    owner = {
        mujoco.mjtObj.mjOBJ_JOINT: lambda i: int(m.jnt_bodyid[i]),
        mujoco.mjtObj.mjOBJ_SITE: lambda i: int(m.site_bodyid[i]),
        mujoco.mjtObj.mjOBJ_BODY: lambda i: i,
        mujoco.mjtObj.mjOBJ_GEOM: lambda i: int(m.geom_bodyid[i]),
        mujoco.mjtObj.mjOBJ_CAMERA: lambda i: int(m.cam_bodyid[i]),
        mujoco.mjtObj.mjOBJ_ACTUATOR: lambda i: int(m.jnt_bodyid[int(m.actuator_trnid[i, 0])]),
    }
    for kind, items in want.items():
        label = kind.name.replace("mjOBJ_", "").lower()
        for fld, name in items:
            i = nid(kind, name)
            if i < 0:
                errs.append(f"{where}: names.{fld}: no {label} {name!r} in the model")
            elif kind in owner and owner[kind](i) not in body_ids:
                errs.append(f"{where}: names.{fld}: {label} {name!r} is outside the "
                            f"robot's bodies {rt.model['bodies']}")

    named_acts = {x for _, x in want[mujoco.mjtObj.mjOBJ_ACTUATOR]}
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if owner[mujoco.mjtObj.mjOBJ_ACTUATOR](a) in body_ids and m.actuator(a).name not in named_acts:
            errs.append(f"{where}: actuator {m.actuator(a).name!r} drives a joint of the "
                        "robot but robot.json does not name it")

    if len(n["arm_joints"]) != len(n["arm_actuators"]):
        errs.append(f"{where}: names: {len(n['arm_joints'])} arm_joints but "
                    f"{len(n['arm_actuators'])} arm_actuators")
    for a_name, j_name in zip(n["arm_actuators"], n["arm_joints"]):
        a, j = nid(mujoco.mjtObj.mjOBJ_ACTUATOR, a_name), nid(mujoco.mjtObj.mjOBJ_JOINT, j_name)
        if a >= 0 and j >= 0 and int(m.actuator_trnid[a, 0]) != j:
            errs.append(f"{where}: names: actuator {a_name!r} drives "
                        f"{m.joint(int(m.actuator_trnid[a, 0])).name!r}, not {j_name!r}")
    return errs


def validate_robot_dict(d: dict, robot_dir: Path, check_model: bool = True) -> List[str]:
    errs = validate_schema(d, json.loads(SCHEMA_PATH.read_text()))
    if errs:
        return errs
    if d["type"] != robot_dir.name:
        errs.append(f"type: {d['type']!r} must equal the folder name {robot_dir.name!r}")
    mdl = d["model"]
    if mdl["default"] not in mdl["variants"]:
        errs.append(f"model.default: {mdl['default']!r} is not one of the variants "
                    f"{sorted(mdl['variants'])}")
    eff = d["end_effectors"]
    if eff["default"] != NO_EFFECTOR and eff["default"] not in eff:
        errs.append(f"end_effectors.default: {eff['default']!r} is not defined here "
                    f"(have {[k for k in eff if k != 'default']} and {NO_EFFECTOR!r})")
    if ("rail_joint" in d["names"]) != ("rail_actuator" in d["names"]):
        errs.append("names: give both rail_joint and rail_actuator, or neither")
    for k, rel in mdl["variants"].items():
        if not (robot_dir / rel).is_file():
            errs.append(f"model.variants.{k}: file not found: {rel} (relative to {robot_dir})")
    if errs or not check_model:
        return errs
    rt = parse_robot(d, robot_dir)
    for ref in rt.refs():
        try:
            resolve_ref(ref)
        except RoomError as e:
            errs += e.problems
    for k in mdl["variants"]:
        m = mujoco.MjModel.from_xml_path(str(rt.model_path(k)))
        errs += check_against_model(rt, m, f"model.variants.{k}")
    return errs


def parse_robot(d: dict, robot_dir: Path) -> RobotType:
    return RobotType(type=d["type"], name=d["name"], description=d["description"],
                     dir=robot_dir, model=d["model"], names=d["names"],
                     rail=d.get("rail", {}), home=d["home"], ik=d.get("ik", {}),
                     end_effectors=d["end_effectors"], raw=d)


def load_robot_type(type_name: str, robots_dir: Path = None,
                    check_model: bool = True) -> RobotType:
    """Load and validate robots/<type_name>/robot.json. With check_model (the
    default) every name is looked up in every model variant and every ref is
    resolved. Raises RoomError listing every problem."""
    robots_dir = Path(robots_dir or ROBOTS_DIR)
    rd = robots_dir / type_name
    f = rd / "robot.json"
    if not f.is_file():
        raise RoomError(type_name, [f"no robot type {type_name!r}: {f} not found; "
                                    f"types: {', '.join(list_robot_types(robots_dir)) or '(none)'}"])
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise RoomError(str(f), [f"not valid JSON: {e}"]) from None
    errs = validate_robot_dict(d, rd, check_model)
    if errs:
        raise RoomError(str(f), errs)
    return parse_robot(d, rd)


def list_robot_types(robots_dir: Path = None) -> List[str]:
    robots_dir = Path(robots_dir or ROBOTS_DIR)
    if not robots_dir.is_dir():
        return []
    return sorted(p.name for p in robots_dir.iterdir() if (p / "robot.json").is_file())
