# twin_rooms/schema.py
"""Validate room.json: the JSON Schema in room_schema.json, then the rules a
schema cannot express (folder name, file existence, one controllable robot...).

The schema is standard JSON Schema so editors can use it; the validator below
implements only the subset that file uses, which keeps `jsonschema` out of the
dependency list. Every problem is collected, not just the first, and each is
reported with its JSON path, e.g. ``objects[2].pose.pos: expected 3 items, got 2``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

SCHEMA_PATH = Path(__file__).with_name("room_schema.json")

#: Parameters each builtin object kind accepts, with their types.
BUILTIN_PARAMS = {
    "cube": {"size": (int, float), "rgba": list, "mass": (int, float)},
    "bin": {"size": list, "wall": (int, float), "rgba": list, "mass": (int, float)},
    "tube_rack": {"rows": int, "cols": int, "pitch": (int, float), "rgba": list},
    "tube": {"radius": (int, float), "length": (int, float), "cap_rgba": list,
             "rgba": list},
}


class RoomError(ValueError):
    """A room definition that does not validate. `problems` lists every one."""

    def __init__(self, where: str, problems: List[str]):
        self.where = where
        self.problems = problems
        super().__init__(f"{where}: {len(problems)} problem(s)\n  - "
                         + "\n  - ".join(problems))


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


# ---------------------------------------------------------------------------
# JSON Schema subset
# ---------------------------------------------------------------------------
_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _is_type(v, t: str) -> bool:
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def _resolve(schema: dict, root: dict) -> dict:
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"unsupported $ref {ref}")
        node = root
        for part in ref[2:].split("/"):
            node = node[part]
        schema = {**node, **{k: v for k, v in schema.items() if k != "$ref"}}
    return schema


def validate_schema(value: Any, schema: dict, root: dict = None, path: str = "") -> List[str]:
    root = root or schema
    s = _resolve(schema, root)
    here = path or "(root)"
    errs: List[str] = []

    if "anyOf" in s:
        branches = [validate_schema(value, b, root, path) for b in s["anyOf"]]
        if all(branches):
            best = min(branches, key=len)
            errs.append(f"{here}: matches none of the allowed forms; closest: "
                        + "; ".join(best))
        return errs

    if "type" in s:
        ts = s["type"] if isinstance(s["type"], list) else [s["type"]]
        if not any(_is_type(value, t) for t in ts):
            return [f"{here}: expected {' or '.join(ts)}, got {type(value).__name__} "
                    f"{json.dumps(value)[:40]}"]
    if "enum" in s and value not in s["enum"]:
        errs.append(f"{here}: {value!r} is not one of {s['enum']}")
    if isinstance(value, str):
        if "minLength" in s and len(value) < s["minLength"]:
            errs.append(f"{here}: must not be empty")
        if "pattern" in s and not re.search(s["pattern"], value):
            want = s.get("description") or f"a string matching {s['pattern']}"
            errs.append(f"{here}: {value!r} is not valid; expected {want}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in s and value < s["minimum"]:
            errs.append(f"{here}: {value} < minimum {s['minimum']}")
        if "maximum" in s and value > s["maximum"]:
            errs.append(f"{here}: {value} > maximum {s['maximum']}")
    if isinstance(value, list):
        if "minItems" in s and len(value) < s["minItems"]:
            errs.append(f"{here}: expected at least {s['minItems']} items, got {len(value)}")
        if "maxItems" in s and len(value) > s["maxItems"]:
            errs.append(f"{here}: expected at most {s['maxItems']} items, got {len(value)}")
        if "items" in s:
            for i, item in enumerate(value):
                errs += validate_schema(item, s["items"], root, f"{path}[{i}]")
    if isinstance(value, dict):
        props = s.get("properties", {})
        for k in s.get("required", []):
            if k not in value:
                errs.append(f"{here}: missing required field '{k}'")
        if "minProperties" in s and len(value) < s["minProperties"]:
            errs.append(f"{here}: must have at least {s['minProperties']} entries")
        extra = s.get("additionalProperties", True)
        for k, v in value.items():
            sub = f"{path}.{k}" if path else k
            if k in props:
                errs += validate_schema(v, props[k], root, sub)
            elif extra is False:
                close = [p for p in props if p.lower() == k.lower() or p.startswith(k[:3])]
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                errs.append(f"{here}: unknown field '{k}'{hint}")
            elif isinstance(extra, dict):
                errs += validate_schema(v, extra, root, sub)
    return errs


# ---------------------------------------------------------------------------
# Rules beyond the schema
# ---------------------------------------------------------------------------
def validate_semantics(d: dict, room_dir: Path, robots_dir: Path = None) -> List[str]:
    errs: List[str] = []
    if d.get("id") != room_dir.name:
        errs.append(f"id: {d.get('id')!r} must equal the folder name {room_dir.name!r}")

    passthrough = "scene" in d
    if passthrough == ("environment" in d):
        errs.append("give exactly one of 'scene' (passthrough room) or 'environment' "
                    "(composed room)")

    def must_exist(rel: str, where: str):
        if not (room_dir / rel).is_file():
            errs.append(f"{where}: file not found: {rel} (relative to {room_dir})")

    if passthrough:
        sc = d["scene"]
        if sc.get("default") not in sc.get("variants", {}):
            errs.append(f"scene.default: {sc.get('default')!r} is not one of the "
                        f"variants {sorted(sc.get('variants', {}))}")
        for k, rel in sc.get("variants", {}).items():
            must_exist(rel, f"scene.variants.{k}")
    env = d.get("environment")
    if isinstance(env, str):
        must_exist(env, "environment")
    elif isinstance(env, dict):
        must_exist(env.get("extract_from", ""), "environment.extract_from")

    robots = d.get("robots", [])
    inst = [r.get("instance") for r in robots]
    for n in {n for n in inst if inst.count(n) > 1}:
        errs.append(f"robots: instance {n!r} is used more than once")
    n_ctrl = sum(1 for r in robots if r.get("controllable", True))
    if robots and n_ctrl != 1:
        errs.append(f"robots: exactly one robot must be controllable in v1, found {n_ctrl} "
                    "(set \"controllable\": false on the others)")
    if robots_dir is not None and robots_dir.is_dir():
        known = sorted(p.name for p in robots_dir.iterdir()
                       if (p / "robot.json").is_file())
        for i, r in enumerate(robots):
            if r.get("type") and r["type"] not in known:
                errs.append(f"robots[{i}].type: unknown robot type {r['type']!r}; "
                            f"known: {known}")
            elif r.get("end_effector"):
                # Read only the effector list: the full robot check (every name
                # looked up in the model) belongs to load_robot_type, not to
                # every room load.
                eff = json.loads((robots_dir / r["type"] / "robot.json").read_text()
                                 ).get("end_effectors", {})
                have = ["none"] + [k for k in eff if k != "default"]
                if r["end_effector"] not in have:
                    errs.append(f"robots[{i}].end_effector: {r['end_effector']!r} is not "
                                f"an end effector of {r['type']}; have {have}")

    objs = d.get("objects", [])
    names = [o.get("name") for o in objs]
    for n in {n for n in names if names.count(n) > 1}:
        errs.append(f"objects: name {n!r} is used more than once")
    for n in set(names) & set(inst):
        errs.append(f"objects: {n!r} is also a robot instance name")
    for i, o in enumerate(objs):
        where = f"objects[{i}] ({o.get('name')})"
        src = o.get("source", "")
        if src == "scene":
            if not passthrough:
                errs.append(f"{where}.source: 'scene' is only valid in a passthrough "
                            "room (one with 'scene'); use builtin:/mjcf:/simready:")
            if "pose" in o:
                errs.append(f"{where}.pose: a 'scene' object is where the scene XML "
                            "puts it; remove the pose")
            continue
        if passthrough:
            errs.append(f"{where}.source: passthrough rooms can only describe bodies "
                        f"already in the scene ('scene'), not add {src!r}")
        if "pose" not in o:
            errs.append(f"{where}: missing 'pose' (required for {src.split(':')[0]} objects)")
        if src.startswith("builtin:"):
            kind = src.split(":", 1)[1]
            allowed = BUILTIN_PARAMS.get(kind, {})
            for k, v in (o.get("params") or {}).items():
                if k not in allowed:
                    errs.append(f"{where}.params: unknown parameter {k!r} for {kind}; "
                                f"allowed: {sorted(allowed)}")
                elif not isinstance(v, allowed[k]) or isinstance(v, bool):
                    errs.append(f"{where}.params.{k}: wrong type {type(v).__name__}")
        elif src.startswith("mjcf:"):
            must_exist(src.split(":", 1)[1], f"{where}.source")

    wb = d.get("workspace_bounds")
    if wb and len(wb.get("min", [])) == 3 and len(wb.get("max", [])) == 3:
        for ax, lo, hi in zip("xyz", wb["min"], wb["max"]):
            if lo is not None and hi is not None and lo >= hi:
                errs.append(f"workspace_bounds: {ax} min {lo} is not below max {hi}")
    return errs


def validate_room_dict(d: dict, room_dir: Path, robots_dir: Path = None) -> List[str]:
    errs = validate_schema(d, load_schema())
    # Semantic rules assume the shapes the schema guarantees.
    if not errs:
        errs = validate_semantics(d, room_dir, robots_dir)
    return errs
