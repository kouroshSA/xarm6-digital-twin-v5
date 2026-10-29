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
  `twin_rooms.builder` (step 3). The environment can be an MJCF fragment or
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
| `robots[]` | `instance`, `type` (a folder under `robots/`), `base_pose`, `end_effector`, `controllable` (exactly one) |
| `objects[]` | `name`, `source` (`scene`, `builtin:cube\|bin\|tube_rack\|tube`, `mjcf:<path>`, `simready:<slug>`), `pose`, `dynamic`, `type`, `label`, `aliases`, `grippable`, `driver`, `params` |
| `cameras[]` | `name`, optional `pos`/`lookat`/`fovy`, `role` |
| `workspace_bounds` | `{"min": [x,y,z], "max": [x,y,z]}`; `null` = the robot's default for that axis |
| `agent` | `context` (notes for the LLM), `task_ideas` |

Loading validates against the schema and then the rules a schema cannot state
(the folder name, that referenced files exist, one controllable robot, unique
names, builtin parameters). Every problem is reported with its field path:

```
rooms/my_room/room.json: 2 problem(s)
  - robots[0].end_effector: 'suction' is not one of ['parallel_gripper', 'bio_gripper', 'none']
  - objects[0].pose.pos: expected at least 3 items, got 2
```

```python
from twin_rooms import load_room, list_rooms
room = load_room("lab_a2s2")
room.scene_path()            # passthrough: the XML to load
room.definition_hash()       # covers room.json AND the files it references
```

Tests: `python -m twin_rooms.test_schema`.
