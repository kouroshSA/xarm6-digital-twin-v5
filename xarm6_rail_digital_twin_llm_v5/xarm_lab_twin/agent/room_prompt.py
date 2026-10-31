# agent/room_prompt.py
"""The room-specific parts of the planner's system prompt, and its registry.

SYSTEM_PROMPT_TEMPLATE (agent/llm_brain.py) is the same for every room; a few
seams in it are filled from the room the arm is in:

    rail_length / rail_range   the robot's rail, from the model's jnt_range
    bench_extents              SimXArmAPI.BENCH_X_MM / BENCH_Y_MM
    room_commands              extra actions the room's objects support
    push_notes                 notes on the movable set
    room_knowledge             what the planner must know about this room
    worked_example             the pick-and-place example, from the room's objects
    scene_geometry_section     object positions

For the lab the commands / push_notes / knowledge text comes from
rooms/lab_a2s2/agent_prompt.md, which is what the template held before rooms,
moved verbatim -- the lab's prompt is byte-identical (twin_rooms.test_prompt).
Any other room gets them from its objects and agent.context, so a prompt for
an empty bench says nothing about tube racks or a PCR it does not have.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import mujoco
import numpy as np

from twin_rooms.context import room_of
from twin_rooms.memory import memory_file, uses_lab_memory

SECTIONS = ("commands", "push_notes", "knowledge")
_MARK = re.compile(r"^<!-- room-prompt: (\w+) -->\n", re.M)

# Room object type -> registry object_type (the registry's prompt groups).
_REGISTRY_TYPE = {"cube": "cube", "bin": "bin", "container": "bin", "rack": "rack",
                  "tube": "tube", "plate": "plate", "instrument": "instrument",
                  "other": "other"}


def read_prompt_file(path: Path) -> Dict[str, str]:
    """{section: text} from a file of `<!-- room-prompt: name -->` blocks.
    Text is kept exactly, except the newlines next to the markers."""
    text = Path(path).read_text()
    marks = list(_MARK.finditer(text))
    out = {}
    for m, nxt in zip(marks, marks[1:] + [None]):
        name = m.group(1)
        if name == "end":
            break
        body = text[m.end(): nxt.start() if nxt else len(text)]
        out[name] = body.rstrip("\n")
    unknown = set(out) - set(SECTIONS)
    if unknown:
        raise ValueError(f"{path}: unknown room-prompt section(s) {sorted(unknown)}; "
                         f"known: {list(SECTIONS)}")
    return out


def initial_positions_mm(arm) -> Dict[str, np.ndarray]:
    """World position (mm) of every room object at the model's initial state,
    by logical name -- what the scene file says, not where the sim has since
    moved things (the live positions are the registry's job)."""
    m = arm.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = {}
    for name, obj in arm.room.objects.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj.body)
        if bid >= 0:
            out[name] = d.xpos[bid] * 1000.0
    return out


class SceneView:
    """What this module reads from an arm, for an arm with no twin model
    (RealXArmAPI). It describes the scene file the prompt has always described
    for real hardware -- the lab -- using SimXArmAPI's own derivation of the
    object lists rather than a second copy of it."""

    def __init__(self, scene_xml: Optional[str] = None):
        from sim.mujoco_env import SimXArmAPI
        from twin_rooms.builder import context_for_scene
        from agent.scene_geometry import DEFAULT_SCENE
        # A SceneGeometry (what the sweep passes around) or a path.
        self.scene_xml = getattr(scene_xml, "scene_xml", scene_xml) or DEFAULT_SCENE
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.room = context_for_scene(self.scene_xml, self.model)
        self.robot = self.room.controllable
        self.BENCH_X_MM, self.BENCH_Y_MM = SimXArmAPI.BENCH_X_MM, SimXArmAPI.BENCH_Y_MM
        SimXArmAPI._init_room_objects(self)

    def _mj(self, name, kind="body"):
        from sim.mujoco_env import SimXArmAPI
        return SimXArmAPI._mj(self, name, kind)


def prompt_view(arm):
    """The arm itself when it is a twin; a SceneView of its scene otherwise."""
    if room_of(arm) is not None:
        return arm
    scene = getattr(arm, "scene_xml", None)
    return SceneView(scene if isinstance(scene, (str, Path)) else None)


# ---------------------------------------------------------------------------
# prompt parts
# ---------------------------------------------------------------------------
def _example_pair(arm):
    objs = arm.room.objects
    cubes = [n for n in arm.cube_bodies]
    bins = [n for n in arm.bin_bodies if objs[n].spec.type == "bin"] or list(arm.bin_bodies)
    cube = "green_cube" if "green_cube" in cubes else (cubes[0] if cubes else None)
    binb = "green_bin" if "green_bin" in bins else (bins[0] if bins else None)
    return cube, binb


def render_worked_example(arm) -> str:
    cube, binb = _example_pair(arm)
    pos = initial_positions_mm(arm)
    rail_lo, rail_hi = arm._rail_limits_mm

    def rail_for(x_mm):
        return round(float(np.clip(x_mm - arm._rail_x0_mm, rail_lo, rail_hi)))

    def mv(x, y, z, speed):
        return ('  {"action": "move_to",       "params": {"x": %d, "y": %d, "z": %d, '
                '"roll": 180, "pitch": 0, "yaw": 0, "speed_mm_s": %d}},' % (x, y, z, speed))

    if cube is None or binb is None:
        # No cube-and-bin pair in this room: a plain motion example.
        x, y = 0, round(-150)
        return "\n".join([
            'Example for "move the gripper 100 mm above the bench"',
            "(coordinates below are an illustration; take real ones from the registry):",
            "[",
            '  {"action": "set_rail",      "params": {"position_mm": %d, "speed_mm_s": 100}},'
            % rail_for(x),
            mv(x, y, 850, 80),
            '  {"action": "done",          "params": {"message": "Gripper is above the bench"}}',
            "]"])
    clabel = arm.room.objects[cube].spec.label
    blabel = arm.room.objects[binb].spec.label
    cx, cy = (round(v) for v in pos[cube][:2])
    bx, by = (round(v) for v in pos[binb][:2])
    return "\n".join([
        f'Example for "put {clabel} in {blabel}"',
        "(coordinates below are generated from the live scene, so they are correct as written):",
        "[",
        '  {"action": "set_rail",      "params": {"position_mm": %d, "speed_mm_s": 100}},'
        % rail_for(cx),
        mv(cx, cy, 830, 80),
        mv(cx, cy, 795, 50),
        '  {"action": "gripper_close", "params": {}},',
        mv(cx, cy, 870, 80),
        mv(bx, by, 870, 80),
        mv(bx, by, 830, 50),
        '  {"action": "gripper_open",  "params": {}},',
        '  {"action": "done",          "params": {"message": "%s placed in %s"}}'
        % (clabel[:1].upper() + clabel[1:], blabel),
        "]"])


def render_room_geometry(arm) -> str:
    """Object positions for the SCENE GEOMETRY section. The lab keeps the
    scene-file renderer it has always used; other rooms list their own objects
    by logical name."""
    if uses_lab_memory(arm.room.room_id):
        from agent.scene_geometry import render_geometry_section
        return render_geometry_section(arm.scene_xml)
    pos = initial_positions_mm(arm)
    movable, static = [], []
    for name, obj in arm.room.objects.items():
        if obj.spec.type == "fixture" or name not in pos:
            continue
        x, y, z = pos[name]
        row = f"  {name:<17} ({x:>6.0f}, {y:>6.0f}, {z:>5.0f})  {obj.spec.label}"
        (movable if obj.spec.dynamic else static).append(row)
    lines = [f"GENERATED from room {arm.room.room_id} -- do not hand-edit these numbers.",
             "All coordinates are world-frame millimetres; xy is the body centre.",
             "", "Movable objects (valid `push_object` targets, and graspable):",
             *(movable or ["  (none)"])]
    if static:
        lines += ["", "Static fixtures — `push_object` on these FAILS (no free joint).",
                  "Plan around them; they are obstacles, not cargo:", *static]
    return "\n".join(lines)


def prompt_parts(arm) -> Dict[str, str]:
    arm = prompt_view(arm)
    room = arm.room.room
    sections = {k: "" for k in SECTIONS}
    if room.agent_prompt_file:
        sections.update(read_prompt_file(room.dir / room.agent_prompt_file))
    if not room.agent_prompt_file:
        sections["commands"] = _generated_commands(arm)
    knowledge = sections["knowledge"]
    if room.agent_context:
        about = f"   **This room ({room.name}):** {room.agent_context}"
        knowledge = f"{knowledge}\n\n{about}" if knowledge else about
    lo, hi = arm._rail_limits_mm
    bx, by = arm.BENCH_X_MM, arm.BENCH_Y_MM
    return {
        "rail_length": f"{hi - lo:.0f}",
        "rail_range": f"{lo:.0f}-{hi:.0f}",
        "bench_extents": f"x ∈ [{bx[0]:+.0f}, {bx[1]:+.0f}] mm, y ∈ [{by[0]:+.0f}, {by[1]:+.0f}] mm",
        "room_commands": sections["commands"] + "\n" if sections["commands"] else "",
        "push_notes": " " + sections["push_notes"] if sections["push_notes"] else "",
        "room_knowledge": "\n\n" + knowledge if knowledge else "",
        "worked_example": render_worked_example(arm),
        "scene_geometry_section": render_room_geometry(arm),
    }


def _generated_commands(arm) -> str:
    """Instrument and rack actions, offered only when the room has what they
    act on (the lab's wording, from its prompt file, is the model)."""
    lines = []
    racks = [r for r, tubes in arm.rack_tube_groups.items()]
    if racks and arm.tube_bodies:
        names = " | ".join(f'"{r}"' for r in racks)
        lines.append(f'- place_tube_in_rack  params: {{"rack_name": {names}}}  — while holding '
                     "a tube, this finds the first open slot in the named rack, flies the arm "
                     "there, and seats the tube.")
    if any(o.spec.driver == "pcr" for o in arm.room.objects.values()):
        lines += ["- pcr_open       params: {}  — open the PCR Thermocycler lid. MUST be called "
                  "before loading or removing a plate from the PCR.",
                  "- pcr_close      params: {}  — close the PCR Thermocycler lid."]
    parts = _operable_parts(arm)
    if parts:
        names = "; ".join(parts)
        lines.append(
            '- operate_part   params: {"part": "<object>.<part>", "opening": <0.0-1.0>}  — '
            "grasp the named door/drawer/lever by its handle and swing or slide it to "
            "`opening` (fraction of its travel: 0 = closed, 1 = fully open), then let go "
            "and back off. ONE action: do not move_to the handle or gripper_close yourself. "
            "A part behind a closed door (a shelf) needs that door opened first. Afterwards "
            "the scene reports '<object> <part> open N deg|mm'. Parts in this room: "
            f"{names}.")
    boxes = _containers(arm)
    if boxes:
        lines.append(
            '- put_in         params: {"object": "<name>", "container": "<object>[.<part>]"}  — '
            "pick the object off the bench with a sideways grasp and set it down inside the "
            "container (on a shelf, through the side facing the robot), then back out and go "
            "home. ONE action, called with the gripper EMPTY and the object still on the "
            "bench: do NOT pick the object up first, and do not set_rail, move_to, "
            "gripper_close or gripper_open around it -- put_in chooses its own rail "
            "positions, grasp and route. The container's door must be OPEN first "
            "(operate_part); if it is closed the action refuses and names the door. Shelves "
            "inside are too low for a gripper pointing down, which is why this exists. "
            "Afterwards the scene reports '<name> in <container> [part]'. Containers in this "
            f"room: {'; '.join(boxes)}.")
    return "\n".join(lines)


def _containers(arm) -> list:
    """Room objects put_in can fill (articulated, with a door), and their
    shelf parts, as the names put_in takes."""
    spec_of = getattr(arm, "_container_spec", None)
    if spec_of is None:
        return []
    out = []
    for o in sorted(getattr(arm.room, "objects", {}).values(), key=lambda o: o.name):
        if spec_of(o.name) is None:
            continue
        shelves = sorted(lbl.split(" ", 1)[1] for lbl, owner, *_ in getattr(arm, "_part_boxes", [])
                         if owner == o.name and " " in lbl and spec_of(lbl.replace(" ", ".")) is not None)
        out.append(o.name + (f" (shelves: {', '.join(o.name + '.' + s for s in shelves)})"
                             if shelves else ""))
    return out


def _operable_parts(arm) -> list:
    """`object.part (hinged door, 0-105 deg)` for every handle the arm can
    grasp, from the arm's own handle table -- the names operate_part takes."""
    handles = getattr(arm, "handles", None) or {}
    out = []
    for name, h in sorted(handles.items()):
        if h.get("jid") is None:
            continue
        m = arm.model
        if m.jnt_stiffness[h["jid"]] > 0:
            # Spring-return (the toaster lever): it cannot be LEFT open, so
            # offering it would hand the planner an action that always fails.
            continue
        lo, hi = m.jnt_range[h["jid"]]
        if m.jnt_type[h["jid"]] == mujoco.mjtJoint.mjJNT_HINGE:
            span = f"hinged, {abs(np.degrees(hi - lo)):.0f} deg"
        else:
            span = f"sliding, {abs(hi - lo) * 1000:.0f} mm"
        out.append(f"{name} ({span})")
    return out


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def registry_for(arm):
    """The object registry for the arm's room. The lab keeps its hand-curated
    registry (aliases, grasp configs and notes refined over many sessions);
    any other room's registry is generated from its room.json objects."""
    from agent.object_registry import (GraspConfig, LabObject, ObjectRegistry,
                                       build_default_registry)
    if room_of(arm) is None or uses_lab_memory(arm.room.room_id):
        return build_default_registry()
    reg = ObjectRegistry(registry_path=str(memory_file(arm.room.room_id, "objects.json")))
    pos = initial_positions_mm(arm)
    lo, hi = arm._rail_limits_mm
    for name, obj in arm.room.objects.items():
        s = obj.spec
        if s.type == "fixture" or name not in pos:
            continue
        container = s.type in ("bin", "container")
        grasp = GraspConfig(approach_direction=[0.0, 0.0, -1.0],
                            grip_orientation_rpy=[180.0, 0.0, 0.0],
                            grip_depth=0.0 if container else 0.7,
                            approach_standoff_mm=60.0 if container else 40.0)
        notes = "" if (s.grippable and s.dynamic) else "Not graspable."
        x = pos[name][0]
        reg.register(LabObject(
            name=name, aliases=[s.label, *s.aliases],
            position_xyz_m=list(pos[name] / 1000.0), grasp=grasp, safety_notes=notes,
            optimal_rail_mm=float(np.clip(x - arm._rail_x0_mm, lo, hi)),
            is_container=container, object_type=_REGISTRY_TYPE.get(s.type, "other")))
    return reg


# ---------------------------------------------------------------------------
# grader vocabulary
# ---------------------------------------------------------------------------
def object_names_text(arm=None) -> str:
    """Movable-object vocabulary for the dynamic grader's prompt: the lab scene
    file's (as before rooms) when there is no room or the room is the lab,
    otherwise the room's own movable objects, grouped by type."""
    from agent.scene_geometry import render_object_names
    if room_of(arm) is None or uses_lab_memory(arm.room.room_id):
        return render_object_names(getattr(arm, "scene_xml", None))
    groups = {"Cubes": ("cube",), "Bins": ("bin", "container"), "Tube racks": ("rack",),
              "Falcon tubes": ("tube",), "Plates / tip racks": ("plate",)}
    lines, used = [], set()
    for label, types in groups.items():
        names = [n for n, o in arm.room.objects.items()
                 if o.spec.dynamic and o.spec.type in types]
        used.update(names)
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    rest = [n for n, o in arm.room.objects.items()
            if o.spec.dynamic and n not in used and o.spec.type != "fixture"]
    if rest:
        lines.append(f"Other movable: {', '.join(rest)}")
    return "\n".join(lines) or "(no movable objects)"
