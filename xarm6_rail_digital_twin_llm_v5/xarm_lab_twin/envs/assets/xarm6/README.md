# xArm6 visual meshes — provenance & attribution

These STL meshes (`base.stl`, `link1.stl` … `link6.stl`) and the joint
kinematics used to place them come from UFACTORY's official ROS 2 description
package:

- **Source:** https://github.com/xArm-Developer/xarm_ros2
  (`xarm_description/meshes/xarm6/visual/` and
  `xarm_description/urdf/xarm6/xarm6.urdf.xacro`), `master` branch.
- **License:** BSD-3-Clause, Copyright (c) 2018 UFACTORY Inc. — see
  `UFACTORY_LICENSE.txt` in this folder (vendored verbatim).
- **Units:** metres (no scaling applied in the MJCF).

## How they're used here

These meshes are the **default** digital-twin arm. `envs/build_mesh_scene.py`
generates `envs/lab_scene.xml` (the file every entry point loads) from the
hand-edited source `envs/lab_scene_primitive.xml`, swapping the six primitive
box/cylinder arm links for these meshes at the real xArm6 URDF joint
origins/axes. The primitive arm remains available via `--scene primitive`.

Nothing else in the scene changes, and the joint names (`joint1..joint6`),
body names (`xarm_base`, `link1..link6`, `gripper`), the gripper subtree, the
`end_effector` site, the rail, actuators, and weld `<equality>` targets are all
preserved — so the mesh arm drops into the existing `SimXArmAPI` machinery.

Originally landed on branch `spike/xarm6-realistic-meshes`.

## What we measured (scripts/ik_sanity.py)

The real xArm6 kinematics use all-Z joint axes with rpy-rotated link frames,
differing from the primitive scene's simplified mixed-axis stack. The natural
worry was that the IK solver (`sim/ik_solver.py`) is tuned to the primitive
frames. The sanity harness (`scripts/ik_sanity.py`, 60 FK-sampled reachable
targets/scene) shows that worry is mostly unfounded:

- **Position IK transfers cleanly.** The damped-least-squares solver reads the
  Jacobian from MuJoCo's *actual* kinematics, so box-vs-mesh geometry is
  irrelevant. Median residual ~0.06 mm, 98% within 5 mm on the mesh arm —
  equal-or-better than the primitive (95%). No re-validation needed for the
  position path (`set_position`, "move to X", pick/place).
- **Orientation is the real limitation, and it is PRE-EXISTING.** 6-DOF solves
  leave a large orientation tail (p90 > 100°) on *both* arms — this is the
  known iterative-IK orientation dead-end, not something the meshes introduce.
  If anything the mesh arm is better (median 0° / 80% within 10° vs the
  primitive's 19° / 47%), because the real wrist has a fuller orientation
  workspace.

## Remaining risk before this becomes the default scene

Not the IK solver — the **geometry-dependent primitives** that hardcode the
primitive arm's proportions: the gripper approach offset (`end_effector` site
at link6 + 0.09), the per-object snap-z heights, and the fly-over+weld push
pattern (`sim/mujoco_env.py::push_object`). Those need a visual pass on the
mesh arm. Collision also uses the convex hull of each visual mesh (fine for the
spike; a real integration would add simplified collision meshes).

Run the harness yourself:

    python scripts/ik_sanity.py            # position-only, both scenes
    python scripts/ik_sanity.py --orient   # add 6-DOF orientation error

## Geometry pass (scripts/pickplace_check.py)

Drove the two geometry-dependent primitives CLAUDE.md flags -- grasp+lift
pick/place and the fly-over `push_object` -- through a scripted sequence on the
mesh arm (no LLM). **Both pass unchanged, no gripper-mount edits needed:**

- Grasp z-stack (mesh): fingertips land at ~777mm, inside the cube's 750-780mm
  span; the gripper stack sits within ~6mm of where it sits on the primitive
  arm. `close_lite6_gripper` grasps at ~26mm (< 70mm reach), the cube tracks
  the gripper through the lift, and `physical_outcome()` reports "green_cube in
  green_bin".
- `push_object` places blue_cube on target (rc=0).
- The 60mm flange->palm offset is a plausible real-gripper length, not a bug;
  the EE-site->fingertip offset (~15mm) matches the primitive.
- Bonus: the mesh arm is actually *cleaner* here -- the primitive scene throws
  `link2_geom`/`bench_top` self-collision validation warnings on some
  set_position moves that the mesh arm does not.

## Broad task sweep (scripts/task_sweep.py)

Extended the cube-only pass to every graspable object CLASS, each on a fresh
scene. **All pass on `--scene meshes`, identical-or-better vs the primitive:**

| Task | primitive | meshes |
| --- | --- | --- |
| cube pick/place       | PASS (24mm to bin) | PASS (7mm to bin) |
| tube -> rack          | PASS | PASS |
| well-plate (bio-grip) | PASS (lifted 762->850) | PASS (lifted 762->850) |
| bin push              | PASS (moved 120mm) | PASS (moved 120mm) |

(The tube descend returns a validation-collision code on BOTH scenes and grasps
the cap from above -- a pre-existing quirk, not mesh-specific.)

Net: IK, gripper geometry, and the full task suite are validated on the mesh
arm — which is why it's now the default (`--scene meshes`). The one optional
follow-up is adding simplified collision meshes (collision is convex-hull
today), only if the convex hulls prove too coarse for a tight-clearance task.
