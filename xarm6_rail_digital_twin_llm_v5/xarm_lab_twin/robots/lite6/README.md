# UFACTORY Lite 6 — not yet a robot type

A placeholder. There is no `robot.json` here, so `list_robot_types()` does not
report it and no room can use it.

## What MuJoCo Menagerie has (checked 2026-10-29)

`google-deepmind/mujoco_menagerie/ufactory_lite6`, at commit
`f3b6013629c7b94f8ee93fd977b161fa8350f9bc`. BSD-3-Clause (UFACTORY). Released
2023-11-03; requires MuJoCo ≥ 3.1.0. Derived from UFACTORY's `xarm_ros2` URDF.
The folder is 6.8 MB of STL meshes.

| file | bodies | joints | actuators | sites |
|---|---|---|---|---|
| `lite6.xml` | 8 | 6 | 6 | `attachment_site` |
| `lite6_gripper_narrow.xml` | 11 | 8 | 7 | `attachment_site`, `end_effector` |
| `lite6_gripper_wide.xml` | 11 | 8 | 7 | `attachment_site`, `end_effector` |

All three compile on this repo's MuJoCo 3.8.1.

- Joints are `joint1..6`, with ranges ±360°, ±150°, -3.5°..300°, ±360°, ±124°
  and ±360°. Note that joint3's range is nothing like the xArm6's.
- The actuators are `<position>` (kp 2000, kv 200) but **unnamed**. robot.json
  maps names, so the builder would have to name them, or the map would have to
  allow "actuator on joint X".
- The gripper is a single `motor` (the real one is pneumatic), with a joint
  equality keeping the fingers symmetric. It has no position control.
- `keyframe "home"` is `0 0 1.57 0 1.57 0`.
- The arm has **no rail**. A `lite6_rail` type would mount it on the
  `xarm6_rail` rail assembly. That needs the second model form mentioned in
  `../README.md`: attach an MJCF file whole, rather than extract bodies from a
  scene.

## To add it later

1. Vendor `ufactory_lite6/` at the pinned commit into `robots/lite6/model/`,
   with LICENSE and provenance.
2. Write `robot.json`, naming the actuators by joint.
3. Run `python -m twin_rooms.test_robot`.

The repo already supports the real Lite 6 elsewhere
(`run_kinesthetic_real.py --robot lite6`). Its home pose and limits should come
from the same place those do.
