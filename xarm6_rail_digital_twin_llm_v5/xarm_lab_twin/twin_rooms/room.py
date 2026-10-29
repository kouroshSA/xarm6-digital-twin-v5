# twin_rooms/room.py
"""Room definitions: find, load and validate `rooms/<id>/room.json`.

    from twin_rooms import load_room, list_rooms
    room = load_room("lab_a2s2")          # validated; raises RoomError
    room.scene_path()                     # passthrough rooms: the XML to load

Two kinds of room (see room_schema.json):
  PASSTHROUGH  `scene` names an existing scene XML, loaded unchanged. The lab
               (lab_a2s2) and the basic project's scene (basic_cubes_bins) are
               these, so a run with no --room loads exactly the file it always did.
  COMPOSED     environment + robots + objects, assembled by twin_rooms.builder.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from twin_rooms.schema import RoomError, validate_room_dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent      # .../xarm_lab_twin
ROOMS_DIR = PROJECT_ROOT / "rooms"
ROBOTS_DIR = PROJECT_ROOT / "robots"
TEMPLATES = "_templates"

#: The room every entry point uses without --room.
DEFAULT_ROOM = "lab_a2s2"


@dataclass(frozen=True)
class Pose:
    pos: tuple
    euler: tuple = (0.0, 0.0, 0.0)          # degrees

    @classmethod
    def parse(cls, d: Optional[dict]) -> Optional["Pose"]:
        if d is None:
            return None
        return cls(tuple(d["pos"]), tuple(d.get("euler", (0.0, 0.0, 0.0))))


@dataclass(frozen=True)
class RobotSpec:
    instance: str
    type: str
    base_pose: Pose
    end_effector: Optional[str]     # None = the robot type's default
    controllable: bool


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    source: str
    params: dict
    pose: Optional[Pose]
    dynamic: bool
    type: str
    label: str
    aliases: tuple
    grippable: bool
    driver: Optional[str]

    @property
    def source_kind(self) -> str:
        return self.source.split(":", 1)[0]

    @property
    def source_arg(self) -> str:
        return self.source.split(":", 1)[1] if ":" in self.source else ""


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pos: Optional[tuple]
    lookat: Optional[tuple]
    fovy: Optional[float]
    role: Optional[str]


@dataclass
class RoomDef:
    id: str
    name: str
    description: str
    dir: Path
    scene: Optional[dict]
    environment: object
    robots: List[RobotSpec]
    objects: List[ObjectSpec]
    cameras: List[CameraSpec]
    workspace_bounds: Optional[dict]
    agent_context: str = ""
    task_ideas: List[str] = field(default_factory=list)
    agent_prompt_file: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def is_passthrough(self) -> bool:
        return self.scene is not None

    @property
    def is_template(self) -> bool:
        return self.dir.parent.name == TEMPLATES

    def scene_path(self, variant: Optional[str] = None) -> Path:
        """Passthrough rooms: the scene XML for `variant` (default variant if None)."""
        if not self.is_passthrough:
            raise RoomError(self.id, ["scene_path(): not a passthrough room; "
                                      "build it with twin_rooms.builder"])
        v = variant or self.scene["default"]
        if v not in self.scene["variants"]:
            raise RoomError(self.id, [f"no scene variant {v!r}; have "
                                      f"{sorted(self.scene['variants'])}"])
        return (self.dir / self.scene["variants"][v]).resolve()

    def controllable_robot(self) -> RobotSpec:
        return next(r for r in self.robots if r.controllable)

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"room {self.id!r} has no object {name!r}")

    def referenced_files(self) -> List[Path]:
        files = [self.dir / "room.json"]
        if self.is_passthrough:
            files += [(self.dir / p) for p in self.scene["variants"].values()]
        if isinstance(self.environment, str):
            files.append(self.dir / self.environment)
        elif isinstance(self.environment, dict):
            files.append(self.dir / self.environment["extract_from"])
        files += [self.dir / o.source_arg for o in self.objects if o.source_kind == "mjcf"]
        if self.agent_prompt_file:
            files.append(self.dir / self.agent_prompt_file)
        return files

    def definition_hash(self) -> str:
        """sha256[:12] over room.json and every file it references, so editing
        the scene XML behind a passthrough room changes the hash too."""
        h = hashlib.sha256()
        for p in self.referenced_files():
            h.update(p.name.encode())
            h.update(p.read_bytes() if p.is_file() else b"<missing>")
        return h.hexdigest()[:12]


def _room_dir(room_id: str, rooms_dir: Path) -> Path:
    for d in (rooms_dir / room_id, rooms_dir / TEMPLATES / room_id):
        if (d / "room.json").is_file():
            return d
    raise RoomError(room_id, [f"no room {room_id!r}: looked for room.json in "
                              f"{rooms_dir / room_id} and {rooms_dir / TEMPLATES / room_id}; "
                              f"rooms: {', '.join(list_rooms(rooms_dir)) or '(none)'}"])


def parse_room(d: dict, room_dir: Path) -> RoomDef:
    ag = d.get("agent", {})
    return RoomDef(
        id=d["id"], name=d["name"], description=d.get("description", ""),
        dir=room_dir, scene=d.get("scene"), environment=d.get("environment"),
        robots=[RobotSpec(r["instance"], r["type"],
                          Pose.parse(r.get("base_pose")) or Pose((0.0, 0.0, 0.0)),
                          r.get("end_effector"),
                          r.get("controllable", True)) for r in d["robots"]],
        objects=[ObjectSpec(o["name"], o["source"], dict(o.get("params", {})),
                            Pose.parse(o.get("pose")), o.get("dynamic", True),
                            o.get("type", "other"), o.get("label", o["name"].replace("_", " ")),
                            tuple(o.get("aliases", ())), o.get("grippable", o.get("dynamic", True)),
                            o.get("driver")) for o in d["objects"]],
        cameras=[CameraSpec(c["name"], tuple(c["pos"]) if "pos" in c else None,
                            tuple(c["lookat"]) if "lookat" in c else None,
                            c.get("fovy"), c.get("role")) for c in d.get("cameras", [])],
        workspace_bounds=d.get("workspace_bounds"),
        agent_context=ag.get("context", ""), task_ideas=list(ag.get("task_ideas", [])),
        agent_prompt_file=ag.get("prompt_file"),
        raw=d)


def load_room(room_id: str, rooms_dir: Path = None, robots_dir: Path = None) -> RoomDef:
    """Load and validate a room by id (or a template by name). Raises RoomError
    listing every problem."""
    rooms_dir = Path(rooms_dir or ROOMS_DIR)
    room_dir = _room_dir(room_id, rooms_dir)
    return load_room_dir(room_dir, robots_dir)


def load_room_dir(room_dir: Path, robots_dir: Path = None) -> RoomDef:
    room_dir = Path(room_dir)
    f = room_dir / "room.json"
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise RoomError(str(f), [f"not valid JSON: {e}"]) from None
    errs = validate_room_dict(d, room_dir, Path(robots_dir or ROBOTS_DIR))
    if errs:
        raise RoomError(str(f), errs)
    return parse_room(d, room_dir)


def list_rooms(rooms_dir: Path = None, include_templates: bool = False) -> List[str]:
    rooms_dir = Path(rooms_dir or ROOMS_DIR)
    if not rooms_dir.is_dir():
        return []
    out = sorted(p.name for p in rooms_dir.iterdir()
                 if (p / "room.json").is_file() and not p.name.startswith("_"))
    if include_templates and (rooms_dir / TEMPLATES).is_dir():
        out += sorted(f"{TEMPLATES}/{p.name}" for p in (rooms_dir / TEMPLATES).iterdir()
                      if (p / "room.json").is_file())
    return out
