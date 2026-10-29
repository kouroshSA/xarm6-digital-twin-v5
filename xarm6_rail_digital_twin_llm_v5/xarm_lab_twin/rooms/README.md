# Rooms

A room is one folder holding a `room.json`: the environment, the robots and the
objects a tool runs in. Every entry point will take `--room <id>`; without it
they run in `lab_a2s2`, which is the original lab loaded unchanged.

| room | kind | what |
|---|---|---|
| `lab_a2s2` | passthrough | the original lab, `envs/lab_scene.xml` (default) |
| `basic_cubes_bins` | passthrough | the manual-control project's scene (its tools' default) |
| `_templates/empty_bench` | composed | the bench + xArm6 on its rail, nothing else |

## Two kinds of room

- **Passthrough** — `scene` names an existing scene XML, loaded as it is. The
  objects list only *describes* bodies already in it (`"source": "scene"`):
  labels, types, aliases, which instrument driver animates them. Nothing in
  the scene changes, so behaviour cannot change either.
- **Composed** — `environment` + `robots` + `objects`, assembled by
  `twin_rooms.builder`. The environment can be an MJCF fragment or
  bodies extracted from an existing scene at build time (`extract_from`), so a
  room built on the lab bench never holds a copy of it that could drift.

## room.json

Schema: [`twin_rooms/room_schema.json`](../twin_rooms/room_schema.json). Units
are metres and degrees.

| field | |
|---|---|
| `id` | must equal the folder name |
| `scene` | passthrough: `{"default": "meshes", "variants": {"meshes": "../../envs/lab_scene.xml"}}` |
| `environment` | composed: `"environment.xml"` or `{"extract_from": ..., "bodies": [...], "floor": true, "lights": true}` |
| `robots[]` | `instance`, `type` (a folder under [`robots/`](../robots/README.md)), `base_pose`, `end_effector` (one the type offers, or `none`; omit for its default), `controllable` (exactly one) |
| `objects[]` | `name`, `source` (`scene`, `builtin:cube\|bin\|tube_rack\|tube`, `mjcf:<path>`, `simready:<slug>`), `pose`, `dynamic`, `type`, `label`, `aliases`, `grippable`, `driver`, `params` |
| `cameras[]` | `name`, optional `pos`/`lookat`/`fovy`, `role` |
| `workspace_bounds` | `{"min": [x,y,z], "max": [x,y,z]}`; `null` = the robot's default for that axis |
| `agent` | `context` (notes for the LLM), `task_ideas` |

Loading validates against the schema and then the rules a schema cannot state
(the folder name, that referenced files exist, one controllable robot, unique
names, builtin parameters). Every problem is reported with its field path:

```
rooms/my_room/room.json: 2 problem(s)
  - robots[0].end_effector: 'suction' is not an end effector of xarm6_rail; have ['none', 'parallel_gripper', 'bio_gripper']
  - objects[0].pose.pos: expected at least 3 items, got 2
```

```python
from twin_rooms import load_room, list_rooms
room = load_room("lab_a2s2")
room.scene_path()            # passthrough: the XML to load
room.definition_hash()       # covers room.json AND the files it references
```

Tests: `python -m twin_rooms.test_schema`.

## Building a room

```python
from twin_rooms.builder import build_room
model, ctx = build_room("empty_bench")           # variant="primitive" for the box arm
ctx.mjcf("joint1", "joint")                      # 'arm0/joint1'
ctx.objects["red_cube"].welds                    # {'arm0': 'arm0/grip_red_cube'}
```

A passthrough room loads its scene XML unchanged, and every name maps to
itself. A composed room is assembled with MjSpec and written to
`<room>/build/scene.xml` (gitignored), which opens in the MuJoCo viewer on its
own.

| element | MJCF name | logical name (what code uses) |
|---|---|---|
| robot parts | `arm0/joint1`, `arm0/gripper`, `arm0/act_rail` | `joint1`, `gripper`, `act_rail` |
| objects | `obj/red_cube`, `obj/red_cube_geom`, `obj/red_cube_joint` | `red_cube`, ... |
| grasp welds | `arm0/grip_red_cube` | `grip_red_cube` |
| environment | `bench`, `cam_overhead` | the same |

The logical map covers the environment, the objects and the **controllable**
robot. Reach other robots through `ctx.robot("arm1").mjcf(...)`. If two
elements would share a logical name (for example an object called `gripper`),
the build fails and names both.

- **Grasp welds are generated.** Every object that is `dynamic` and
  `grippable` (both default to true) gets a weld from each robot that has an
  end effector. The magnetic gripper needs one weld per pair. Without it an
  object can be seen but never picked up.
- **Builtin objects reproduce the lab's** with default params: `cube` is
  `green_cube`, `bin` is `green_bin`, `tube_rack` is `left_tube_rack`, and
  `tube` is `tube_L1`. `test_builder` compares them geom by geom. `pose.pos`
  places, by kind:

  | kind | `pose.pos` is |
  |---|---|
  | cube | its centre (on the bench, z = 0.750 + half the size) |
  | bin | the underside of its floor (0.750 on the bench) |
  | tube_rack | the centre of its 10 mm base (0.755 on the bench) |
  | tube | its centre; tube slots are at ±20 / ±60 mm in x and ±20 mm in y from the rack centre |
- **`mjcf:` objects** must have exactly one top-level body. They are attached
  with the prefix `obj/<name>/` and the body is renamed `obj/<name>`, so one
  file can be placed twice.
- **Room cameras** that give `pos` + `lookat` are added. Cameras without a
  pose must already exist in the environment or the robot.
- The room hash covers room.json, the files it names, and each robot type's
  robot.json and model file.

Tests: `MUJOCO_GL=egl python -m twin_rooms.test_builder`.
