# twin_rooms/context.py
"""RoomContext: what a built room is, in the names code uses.

Code and recordings use LOGICAL names -- the lab's own names (joint1,
end_effector, gripper, red_cube, grip_red_cube). In a passthrough room those
ARE the MJCF names. In a composed room the builder prefixes them:

    robot elements   "<instance>/"   arm0/joint1, arm0/act_rail, arm0/gripper
    objects          "obj/"          obj/red_cube, obj/red_cube_geom
    grasp welds      "<instance>/grip_<object>"   (logical: grip_<object>)
    environment      none            bench, cam_overhead, floor

    ctx.mjcf("joint1", "joint")       -> 'arm0/joint1'   (the controllable robot)
    ctx.robot("arm1").mjcf("joint1")  -> 'arm1/joint1'   (any robot, by instance)
    ctx.logical("obj/red_cube")       -> 'red_cube'

The flat map (`names`) holds the environment, the objects and the
CONTROLLABLE robot only, which is what lets the lab's names keep working
unchanged. Other robot instances are reached through ctx.robot(instance). If
two elements would get the same logical name, building fails and names both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import mujoco

from twin_rooms.robot import RobotType
from twin_rooms.room import ObjectSpec, RobotSpec, RoomDef
from twin_rooms.schema import RoomError

OBJ_PREFIX = "obj/"

#: Element kinds indexed, by the short names used in ctx.mjcf(name, kind).
KINDS = {
    "body": (mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    "joint": (mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    "geom": (mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
    "site": (mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    "camera": (mujoco.mjtObj.mjOBJ_CAMERA, "ncam"),
    "light": (mujoco.mjtObj.mjOBJ_LIGHT, "nlight"),
    "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
    "equality": (mujoco.mjtObj.mjOBJ_EQUALITY, "neq"),
    "sensor": (mujoco.mjtObj.mjOBJ_SENSOR, "nsensor"),
    "tendon": (mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
    "material": (mujoco.mjtObj.mjOBJ_MATERIAL, "nmat"),
    "mesh": (mujoco.mjtObj.mjOBJ_MESH, "nmesh"),
}


@dataclass
class RobotInstance:
    spec: RobotSpec
    type: RobotType
    prefix: str
    end_effector: dict            # resolved entry; {} for "none"

    @property
    def instance(self) -> str:
        return self.spec.instance

    def mjcf(self, logical: str) -> str:
        return self.prefix + logical

    @property
    def joint_names(self):
        return [self.mjcf(n) for n in self.type.names["arm_joints"]]

    @property
    def actuator_names(self):
        return [self.mjcf(n) for n in self.type.names["arm_actuators"]]


@dataclass
class HandleInstance:
    """A grip segment on one moving link of an articulated object (a fridge
    door, a drawer). The gripper holds it with a CONNECT constraint -- a
    point, not a weld -- so the link can turn under a gripper that keeps its
    own orientation."""
    link: str                     # link name in the asset (door_01)
    body: str                     # MJCF body of the link
    site: str                     # MJCF capsule site: the segment
    joint: Optional[str]          # MJCF joint that moves the link
    connects: Dict[str, str]      # robot instance -> MJCF name of its connect


@dataclass
class ObjectInstance:
    spec: ObjectSpec
    body: str                     # MJCF body name
    welds: Dict[str, str]         # robot instance -> MJCF name of its grasp weld
    handles: Dict[str, HandleInstance] = field(default_factory=dict)   # link -> handle

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class RoomContext:
    room: RoomDef
    model_path: Optional[Path]    # the XML that loads this model on its own
    hash: str
    robots: Dict[str, RobotInstance]
    objects: Dict[str, ObjectInstance]
    variant: Optional[str] = None
    names: Dict[str, Dict[str, str]] = field(default_factory=dict)   # kind -> logical -> mjcf
    _rev: Dict[str, str] = field(default_factory=dict, repr=False)

    # ---- lookup ----------------------------------------------------------
    @property
    def room_id(self) -> str:
        return self.room.id

    @property
    def controllable(self) -> RobotInstance:
        return next(r for r in self.robots.values() if r.spec.controllable)

    def robot(self, instance: Optional[str] = None) -> RobotInstance:
        if instance is None:
            return self.controllable
        if instance not in self.robots:
            raise KeyError(f"room {self.room_id!r} has no robot {instance!r}; "
                           f"have {sorted(self.robots)}")
        return self.robots[instance]

    def mjcf(self, logical: str, kind: str = "body") -> str:
        try:
            return self.names[kind][logical]
        except KeyError:
            raise KeyError(f"room {self.room_id!r} has no {kind} {logical!r}") from None

    def has(self, logical: str, kind: str = "body") -> bool:
        return logical in self.names.get(kind, {})

    def logical(self, mjcf_name: str) -> str:
        """MJCF name -> logical name. Elements of a non-controllable robot keep
        their prefixed name."""
        return self._rev.get(mjcf_name, mjcf_name)

    @property
    def workspace_bounds(self):
        return self.room.workspace_bounds

    # ---- built by the builder -------------------------------------------
    def index(self, model: mujoco.MjModel) -> None:
        ctrl_prefix = self.controllable.prefix
        other = [r.prefix for r in self.robots.values() if r.prefix and r.prefix != ctrl_prefix]
        clashes = []
        for kind, (obj, count) in KINDS.items():
            table: Dict[str, str] = {}
            for i in range(getattr(model, count)):
                name = mujoco.mj_id2name(model, obj, i)
                if not name:
                    continue
                if any(name.startswith(p) for p in other):
                    continue
                if name.startswith(OBJ_PREFIX):
                    logical = name[len(OBJ_PREFIX):]
                elif ctrl_prefix and name.startswith(ctrl_prefix):
                    logical = name[len(ctrl_prefix):]
                else:
                    logical = name
                if logical in table:
                    clashes.append(f"{kind} {logical!r} would mean both {table[logical]!r} "
                                   f"and {name!r}")
                    continue
                table[logical] = name
                self._rev[name] = logical
            self.names[kind] = table
        if clashes:
            raise RoomError(self.room_id, ["logical name clash: " + c for c in clashes]
                            + ["rename the object (names must not repeat environment "
                               "or robot names)"])


def room_of(arm) -> Optional[RoomContext]:
    """The arm's RoomContext, or None for an arm that is not a twin (real
    hardware, test doubles). Checks the TYPE, not hasattr: a mock that answers
    every attribute would otherwise pass for a twin."""
    room = getattr(arm, "room", None) if arm is not None else None
    return room if isinstance(room, RoomContext) else None
