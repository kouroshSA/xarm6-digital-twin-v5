# twin_rooms/builder.py
"""Build a room into a MuJoCo model.

    from twin_rooms.builder import build_room
    model, ctx = build_room("empty_bench")
    model.joint(ctx.mjcf("joint1", "joint"))          # 'arm0/joint1'

PASSTHROUGH rooms load their scene XML unchanged; the context maps every name
to itself, so nothing about the lab changes.

COMPOSED rooms are assembled with MjSpec:
  1. the environment: an MJCF file, or an existing scene with every top-level
     body deleted except the ones listed (MjSpec deletes the actuators, welds
     and contact excludes that pointed into what was removed);
  2. each robot: its type's scene, pruned to the type's bodies, attached at
     base_pose with the prefix "<instance>/";
  3. each object: a body named "obj/<name>" (builtin generators, or an MJCF
     file attached with the prefix "obj/<name>/");
  4. a grasp weld "<instance>/grip_<object>" from each gripper-carrying robot
     to each grippable object -- the magnetic gripper needs one per pair, and
     without it a placed object can be seen but never picked up;
  5. room cameras that give a pose.
The result is written to <room>/build/scene.xml (gitignored), which loads on
its own in the MuJoCo viewer.

Code uses LOGICAL names (the lab's names: joint1, gripper, red_cube); the
context maps them to MJCF names. See RoomContext.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

from twin_rooms.builtins import GENERATORS
from twin_rooms.context import OBJ_PREFIX, ObjectInstance, RobotInstance, RoomContext
from twin_rooms.robot import RobotType, load_robot_type
from twin_rooms.room import (ROBOTS_DIR, ROOMS_DIR, RoomDef, list_rooms, load_room,
                             load_room_dir, parse_room)
from twin_rooms.schema import RoomError

SIMREADY_DIR = Path(__file__).resolve().parent.parent / "assets" / "simready"


# ---------------------------------------------------------------------------
# spec helpers
# ---------------------------------------------------------------------------
def load_spec(path: Path) -> mujoco.MjSpec:
    """MjSpec.from_file with asset paths made absolute, so the spec can be
    attached into one from another directory and its to_xml() loads anywhere."""
    path = Path(path).resolve()
    s = mujoco.MjSpec.from_file(str(path))
    for items, d in ((s.meshes, s.meshdir), (s.textures, s.texturedir),
                     (s.hfields, s.meshdir), (s.skins, s.meshdir)):
        base = path.parent / d if d else path.parent
        for it in items:
            if getattr(it, "file", ""):
                it.file = str((base / it.file).resolve())
    s.meshdir = ""
    s.texturedir = ""
    return s


def prune(spec: mujoco.MjSpec, keep, where: str) -> mujoco.MjSpec:
    """Delete every top-level body not in `keep`. Raises if one of `keep` is
    missing, rather than building a room without it."""
    have = {b.name for b in spec.worldbody.bodies}
    missing = [b for b in keep if b not in have]
    if missing:
        raise RoomError(where, [f"no top-level body {b!r} to extract; top-level bodies: "
                                f"{sorted(have)}" for b in missing])
    for b in list(spec.worldbody.bodies):
        if b.name not in keep:
            spec.delete(b)
    for k in list(spec.keys):        # qpos layouts no longer match
        spec.delete(k)
    _drop_unused_assets(spec)
    return spec


def _drop_unused_assets(spec: mujoco.MjSpec) -> None:
    """Delete meshes, materials and textures nothing left in `spec` uses. The
    environment and the robot are often cut from the same scene; without this
    both copies carry every asset, and the robot's prefixed copy of e.g.
    bio_gripper_body_mat would clash with the environment's unused one."""
    users = list(spec.geoms) + list(spec.sites) + list(spec.skins) + list(spec.tendons)
    meshes = {g.meshname for g in spec.geoms if g.meshname}
    mats = {u.material for u in users if getattr(u, "material", "")}
    for m in list(spec.meshes):
        if m.name not in meshes:
            spec.delete(m)
    for m in list(spec.materials):
        if m.name not in mats:
            spec.delete(m)
    texs = {t for m in spec.materials for t in m.textures if t}
    for t in list(spec.textures):
        # skybox / builtin textures with no material are kept (they are used
        # by the renderer directly, not through a material)
        if t.name and t.name not in texs and t.type != mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(t)


def _strip_world(spec: mujoco.MjSpec, geoms=True, lights=True, cameras=True, sites=True):
    for flag, items in ((geoms, spec.worldbody.geoms), (lights, spec.worldbody.lights),
                        (cameras, spec.worldbody.cameras), (sites, spec.worldbody.sites)):
        if flag:
            for it in list(items):
                spec.delete(it)


def quat_from_euler_deg(euler) -> list:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad(np.asarray(euler, float)), "xyz")
    return list(q)


def _xyaxes(pos, lookat) -> list:
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return list(x) + list(y)


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def room_hash(room: RoomDef, robot_types: Dict[str, RobotType], variant: Optional[str]) -> str:
    """sha256[:12] over the room definition (room.json + the files it names)
    and, for each robot type, robot.json and the model file it extracts from."""
    h = hashlib.sha256(room.definition_hash().encode())
    for t in sorted(robot_types):
        rt = robot_types[t]
        for p in (rt.dir / "robot.json", rt.model_path(variant if variant in rt.model["variants"] else None)):
            h.update(p.name.encode())
            h.update(p.read_bytes() if p.is_file() else b"<missing>")
    for o in room.objects:
        if o.source_kind == "simready":
            p = SIMREADY_DIR / o.source_arg / f"{o.source_arg}.xml"
            h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_room(room_id: str, variant: Optional[str] = None, rooms_dir: Path = None,
               robots_dir: Path = None, write: bool = True
               ) -> Tuple[mujoco.MjModel, RoomContext]:
    """Build room `room_id`. `variant` picks the scene variant of a passthrough
    room or the robot model variant of a composed one (e.g. "primitive")."""
    if (Path(room_id) / "room.json").is_file():          # a room folder, e.g. a test's
        room = load_room_dir(Path(room_id), robots_dir)
    else:
        room = load_room(room_id, rooms_dir, robots_dir)
    return build_room_def(room, variant, robots_dir, write)


def build_room_def(room: RoomDef, variant: Optional[str] = None, robots_dir: Path = None,
                   write: bool = True) -> Tuple[mujoco.MjModel, RoomContext]:
    robots_dir = Path(robots_dir or ROBOTS_DIR)
    types = {r.type: load_robot_type(r.type, robots_dir, check_model=False) for r in room.robots}
    if room.is_passthrough:
        return _build_passthrough(room, types, variant)
    model, spec, ctx = _build_composed(room, types, variant)
    if write:
        out = room.dir / "build" / "scene.xml"
        out.parent.mkdir(exist_ok=True)
        out.write_text(spec.to_xml())
        ctx.model_path = out
    return model, ctx


def _robot_instances(room: RoomDef, types, prefixes: Dict[str, str]) -> Dict[str, RobotInstance]:
    return {r.instance: RobotInstance(r, types[r.type], prefixes[r.instance],
                                      types[r.type].end_effector(r.end_effector))
            for r in room.robots}


def _build_passthrough(room: RoomDef, types, variant, model=None):
    path = room.scene_path(variant)
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(path))
    robots = _robot_instances(room, types, {r.instance: "" for r in room.robots})
    if len(robots) != 1:
        raise RoomError(room.id, ["a passthrough room describes exactly one robot (its scene's)"])
    inst = next(iter(robots.values()))
    # Grasp welds already in the scene: gripper -> object.
    welds: Dict[str, str] = {}
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, inst.type.names["gripper_body"])
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and model.eq_obj1id[e] == gb:
            welds[model.body(int(model.eq_obj2id[e])).name] = model.equality(e).name
    objects = {o.name: ObjectInstance(o, o.name, {inst.spec.instance: welds[o.name]}
                                      if o.name in welds else {})
               for o in room.objects}
    ctx = RoomContext(room=room, model_path=path, hash=room_hash(room, types, variant),
                      robots=robots, objects=objects, variant=variant)
    ctx.index(model)
    return model, ctx


def context_for_scene(scene_xml, model: mujoco.MjModel, rooms_dir: Path = None,
                      robots_dir: Path = None) -> RoomContext:
    """The context for a model loaded straight from a scene file (the pre-rooms
    way every tool loads the lab). If a passthrough room wraps that file, its
    context; otherwise an anonymous passthrough room ("scene:<file>") whose
    objects are the scene's free and welded bodies, typed "other", with one
    xarm6_rail robot and identity names."""
    path = Path(scene_xml).resolve()
    rooms_dir = Path(rooms_dir or ROOMS_DIR)
    for rid in list_rooms(rooms_dir):
        room = load_room(rid, rooms_dir, robots_dir)
        if not room.is_passthrough:
            continue
        for v in room.scene["variants"]:
            if room.scene_path(v) == path:
                types = {r.type: load_robot_type(r.type, Path(robots_dir or ROBOTS_DIR),
                                                 check_model=False) for r in room.robots}
                return _build_passthrough(room, types, v, model)[1]
    return _anonymous_context(path, model, robots_dir)


def _anonymous_context(path: Path, model: mujoco.MjModel, robots_dir) -> RoomContext:
    names = set()
    for b in range(1, model.nbody):
        j = model.body_jntadr[b]
        if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            names.add(model.body(b).name)
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD:
            names.add(model.body(int(model.eq_obj2id[e])).name)
    d = {"id": "scene_" + path.stem.lower().replace("-", "_"), "name": path.name,
         "description": f"anonymous passthrough room for {path}",
         "scene": {"default": "file", "variants": {"file": str(path)}},
         "robots": [{"instance": "arm0", "type": "xarm6_rail"}],
         "objects": [{"name": n, "source": "scene", "type": "other"} for n in sorted(names)]}
    room = parse_room(d, path.parent)
    types = {"xarm6_rail": load_robot_type("xarm6_rail", Path(robots_dir or ROBOTS_DIR),
                                           check_model=False)}
    return _build_passthrough(room, types, "file", model)[1]


def _extract_bodies(spec: mujoco.MjSpec, room: RoomDef) -> None:
    """Attach every `mjcf:<scene>#<body>` object. All bodies taken from one
    scene go in ONE attach: two attaches of the same scene would each bring its
    materials and default classes, and the names would clash (both well
    plates use well_plate_mat). Elements keep their scene names under "obj/",
    so code that knows the scene's names (the vortex, PCR and heater-shaker
    drivers) knows these."""
    by_file: Dict[str, list] = {}
    for o in room.objects:
        if o.source_kind == "mjcf" and "#" in o.source_arg:
            file, _, body = o.source_arg.partition("#")
            by_file.setdefault(file, []).append((body, o))
    for file, items in by_file.items():
        where = f"{room.id}: objects from {file}"
        child = prune(load_spec(room.dir / file), [b for b, _ in items], where)
        _strip_world(child)
        for body, o in items:
            free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in child.body(body).joints)
            if free != o.dynamic:
                raise RoomError(where, [f"{body!r} {'has' if free else 'has no'} free joint "
                                        f"in {file} but the object is \"dynamic\": "
                                        f"{str(o.dynamic).lower()}"])
        spec.attach(child, prefix=OBJ_PREFIX, frame=spec.worldbody.add_frame())


def _object_body(spec: mujoco.MjSpec, room: RoomDef, o) -> mujoco.MjsBody:
    """Add object `o` to `spec`; returns its root body, named obj/<name>."""
    where = f"{room.id}: objects ({o.name})"
    if o.source_kind == "builtin":
        body = GENERATORS[o.source_arg](spec.worldbody, o.name, OBJ_PREFIX, o.params)
        if o.dynamic:
            body.add_freejoint(name=f"{OBJ_PREFIX}{o.name}_joint")
        return body
    if o.source_kind == "mjcf" and "#" in o.source_arg:
        return spec.body(OBJ_PREFIX + o.name)        # attached by _extract_bodies
    if o.source_kind == "mjcf":
        path = room.dir / o.source_arg
    elif o.source_kind == "simready":
        path = SIMREADY_DIR / o.source_arg / f"{o.source_arg}.xml"
        if not path.is_file():
            raise RoomError(where, [f"SimReady asset {o.source_arg!r} is not imported "
                                    f"({path} not found); run "
                                    "`python -m simready_import fetch ...` first"])
    else:
        raise RoomError(where, [f"cannot build source {o.source!r} in a composed room"])
    child = load_spec(path)
    _strip_world(child)
    tops = list(child.worldbody.bodies)
    if len(tops) != 1:
        raise RoomError(where, [f"{path.name} must have exactly one top-level body, "
                                f"has {[b.name for b in tops]}"])
    root = tops[0].name                  # read before attach renames it
    frame = spec.worldbody.add_frame()
    spec.attach(child, prefix=f"{OBJ_PREFIX}{o.name}/", frame=frame)
    body = spec.body(f"{OBJ_PREFIX}{o.name}/{root}")
    body.name = f"{OBJ_PREFIX}{o.name}"
    free = [j for j in body.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if o.dynamic and not free:
        body.add_freejoint(name=f"{OBJ_PREFIX}{o.name}_joint")
    elif not o.dynamic and free:
        raise RoomError(where, [f"{path.name}'s body has a free joint but the object is "
                                "\"dynamic\": false"])
    return body


def _build_composed(room: RoomDef, types, variant):
    env = room.environment
    if isinstance(env, str):
        spec = load_spec(room.dir / env)
    else:
        spec = prune(load_spec(room.dir / env["extract_from"]), env["bodies"],
                     f"{room.id}: environment")
        _strip_world(spec, geoms=not env.get("floor", True),
                     lights=not env.get("lights", True), cameras=False, sites=False)

    prefixes = {r.instance: f"{r.instance}/" for r in room.robots}
    robots = _robot_instances(room, types, prefixes)
    for inst in robots.values():
        rt = inst.type
        v = variant if variant in rt.model["variants"] else None
        child = prune(load_spec(rt.model_path(v)),
                      rt.model["bodies"] + rt.model.get("operator_aids", []),
                      f"robot type {rt.type}")
        _strip_world(child)
        bp = inst.spec.base_pose
        frame = spec.worldbody.add_frame(pos=list(bp.pos), quat=quat_from_euler_deg(bp.euler))
        spec.attach(child, prefix=inst.prefix, frame=frame)

    _extract_bodies(spec, room)
    objects: Dict[str, ObjectInstance] = {}
    for o in room.objects:
        body = _object_body(spec, room, o)
        if o.pose is not None:        # an extracted body without one stays put
            body.pos = list(o.pose.pos)
            body.quat = quat_from_euler_deg(o.pose.euler)
        welds = {}
        if o.grippable and o.dynamic:
            for inst in robots.values():
                if not inst.end_effector:          # bare flange: nothing to carry with
                    continue
                name = f"{inst.prefix}grip_{o.name}"
                spec.add_equality(name=name, type=mujoco.mjtEq.mjEQ_WELD,
                                  objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                                  name1=inst.mjcf(inst.type.names["gripper_body"]),
                                  name2=body.name)
                welds[inst.spec.instance] = name
        objects[o.name] = ObjectInstance(o, body.name, welds)

    for c in room.cameras:
        if c.pos is None:
            continue
        if c.lookat is None:
            raise RoomError(room.id, [f"cameras ({c.name}): a camera with pos needs lookat"])
        cam = spec.worldbody.add_camera(name=c.name, pos=list(c.pos), fovy=c.fovy or 45.0)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = _xyaxes(c.pos, c.lookat)

    try:
        model = spec.compile()
    except ValueError as e:
        raise RoomError(room.id, [f"MuJoCo could not compile the room: {e}"]) from None
    ctx = RoomContext(room=room, model_path=None, hash=room_hash(room, types, variant),
                      robots=robots, objects=objects, variant=variant)
    ctx.index(model)
    missing = [c.name for c in room.cameras
               if c.pos is None and mujoco.mj_name2id(
                   model, mujoco.mjtObj.mjOBJ_CAMERA, ctx.names["camera"].get(c.name, "")) < 0]
    if missing:
        raise RoomError(room.id, [f"cameras: {n!r} is not in the environment or robot; "
                                  "give pos and lookat to add it" for n in missing])
    return model, spec, ctx
