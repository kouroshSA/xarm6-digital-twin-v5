# xArm Gripper visual meshes — provenance & attribution

The seven STL meshes here (`base_link.stl` and the left/right `outer_knuckle`,
`inner_knuckle` and `finger`) are UFACTORY's xArm Gripper, taken from MuJoCo
Menagerie's xArm7 model, which derives them from UFACTORY's public URDF:

- **Source:** https://github.com/google-deepmind/mujoco_menagerie
  `ufactory_xarm7/assets/`, commit `f3b6013` (the commit the Lite6 entry in
  `robots/lite6/README.md` pins).
- **License:** BSD-3-Clause, Copyright (c) 2018 UFACTORY Inc. — see
  `UFACTORY_LICENSE.txt` (vendored verbatim).
- **Units:** metres, no scaling.

## How they are used

`envs/lab_scene_primitive.xml` builds the gripper subtree from them (and
`envs/build_mesh_scene.py` copies that subtree into `lab_scene.xml`). Body and
joint positions are Menagerie's, turned 180 deg about the tool axis as
Menagerie mounts it. Differences from Menagerie:

- **Cosmetic linkage.** Menagerie closes the four-bar with two CONNECT
  constraints and a tendon. Here each follower and inner knuckle is tied to its
  driver by a joint equality (measured in Menagerie: follower = inner = driver
  to within 0.004 rad over the whole stroke), the finger geoms do not collide,
  and holding is the twin's magnetic weld, as before.
- **Mounted on the cell's tool stack.** The real arm has an F/T sensor and the
  camera adapter between the flange and the gripper. Their combined thickness
  is `sim/tool.py::TOOL_STACK_M`, derived from the measured flange-to-tip (see
  that file); the stack is drawn as one cylinder carrying the measured mass.

## Measured from Menagerie's model (gripper mount frame)

| | open | closed |
|---|---|---|
| fingertip | 162.2 mm | 174.5 mm |
| TCP (`link_tcp`) | 172.0 mm | 172.0 mm |
| jaw gap (pad to pad) | 88.7 mm | 2.4 mm |
| mass, base + linkage | 0.751 kg | |
