# Robot types

A robot type is defined once, in `robots/<type>/robot.json`, and rooms refer to
it by name (`"type": "xarm6_rail"`). It maps the **logical names** control code
uses to the model's MJCF names, says where the model comes from, and lists the
end effectors the type can carry.

| type | model | end effectors |
|---|---|---|
| `xarm6_rail` | extracted from `envs/lab_scene.xml` (or `lab_scene_primitive.xml`) | `parallel_gripper` (default), `bio_gripper`, `none` |
| `xarm6_rail_basic` | the manual-control project's `basic_scene.xml` | `none` |

Schema: [`twin_rooms/robot_schema.json`](../twin_rooms/robot_schema.json).
Tests: `python -m twin_rooms.test_robot`.

```python
from twin_rooms import load_robot_type
rt = load_robot_type("xarm6_rail")     # every name checked in every model variant
rt.names["arm_joints"]                 # ['joint1', ..., 'joint6']
rt.value("home.joints_deg")            # resolved from arm_backend.HOME_JOINTS_DEG
rt.end_effector("bio_gripper")         # geoms, materials, reach, aperture
```

## No copies

- **The model is not copied.** `model.bodies` lists the top-level bodies of an
  existing scene that make up the robot: `rail_track` (the mount, carrying
  carriage, arm, gripper and wrist camera) and the rail's two LED strips.
  The room builder extracts them at build time, so a change to the lab scene
  reaches every room. `operator_aids` (the teach handle and the VR head) come
  along so the teleop tools work in any room. The spec suggested moving the
  MJCF into `robots/xarm6_rail/model/`; that was not done, because the lab
  scene stays where it is and a moved copy would drift from it.
- **Numbers that live in code are refs**, `{"ref": "module:NAME"}`, resolved by
  import: the home pose and homing speeds (`arm_backend`, shared with the real
  arm), the IK tolerance and the gripper constants (`sim.mujoco_env`).
- **Joint and rail limits are not stored.** The compiled model's `jnt_range` is
  their source; the lab scene's ranges are generated from
  `arm_backend.XARM6_JOINT_LIMITS_DEG`. A `joint_limits_deg` field is refused.

Loading checks every name against every model variant and requires it to lie
inside the robot's own bodies. It also refuses an actuator on the robot that
robot.json does not name, since every room would build it and nothing would
drive it.

`xarm6_rail_basic` exists because the basic scene uses the lab's names for a
different arm (simplified joint ranges, base 57 mm lower, no fingers, no welds).
A name map alone could not tell the two apart.

## Future types

- **UFACTORY Lite 6.** See [`lite6/README.md`](lite6/README.md). MuJoCo
  Menagerie has a model; it has not been added yet.
- **Another arm.** Add a folder with a `robot.json`. A model that does not come
  from an existing scene will need a `model.file` form (an MJCF path attached
  whole) next to the scene-extraction form; add it to the schema when the
  first such type arrives.
