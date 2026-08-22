#!/usr/bin/env python3
"""Generate the default ``lab_scene.xml`` from ``lab_scene_primitive.xml``.

This produces the production lab scene: the six primitive box/cylinder arm links
of the hand-edited source are replaced by the real UFACTORY xArm6 visual meshes,
positioned with the true URDF kinematics. ``lab_scene.xml`` is the file every
entry point loads; ``lab_scene_primitive.xml`` is the editable source and the
opt-in ``--scene primitive`` fallback.

Why a generator (rather than a hand-edited copy): the lab scene is ~1000 lines
and evolves. Regenerating keeps the realistic scene in lock-step with everything
downstream of the arm (bench, rail, cubes/bins/tubes, OT-2, instruments) so the
only delta is the arm subtree + the mesh <asset> declarations. Edit scene
furniture in lab_scene_primitive.xml, then rerun this script.

Provenance of the meshes + kinematics: xArm-Developer/xarm_ros2,
xarm_description (BSD-3, UFACTORY Inc. 2018). See assets/xarm6/README.md.

Invariants preserved so the rest of the stack is untouched:
  * joint NAMES: joint1..joint6 (+ rail, finger_*_joint) -> existing <actuator>
    entries bind by name, unchanged.
  * body NAMES: xarm_base, link1..link6, gripper (+ fingers) -> weld targets in
    <equality> (body1="gripper") and FK-validator ARM_GEOM_NAMES still resolve.
  * the gripper subtree (palm geom, cosmetic fingers, bio-gripper geoms) and the
    end_effector site are copied VERBATIM from the primitive scene.

What DOES change (expected; this is why it's a spike, not the default):
  * joint FRAMES/AXES now match the real xArm6 (all axis z + rpy-rotated link
    frames) instead of the simplified mixed-axis straight-up stack. The IK
    solver + FK validator are tuned to the old frames, so task execution needs
    re-validation before this becomes the default scene.
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# lab_scene_primitive.xml is the hand-edited SOURCE (primitive box/cylinder arm
# + all scene furniture). lab_scene.xml is the GENERATED default that every
# entry point loads (realistic xArm6 meshes swapped in). Edit the scene in
# lab_scene_primitive.xml, then rerun this script to regenerate lab_scene.xml.
SRC = HERE / "lab_scene_primitive.xml"
DST = HERE / "lab_scene.xml"

# --- Real xArm6 kinematics (xarm_description/urdf/xarm6/xarm6.urdf.xacro) -----
# Each joint's URDF <origin> becomes the child body's pos/euler; the joint sits
# at the body origin with axis z. rpy values are all single-axis (roll only),
# so MuJoCo's default eulerseq="xyz" is unambiguous. Limits use the xacro
# "limited" variant. Inertials copied verbatim (fullinertia order in MuJoCo is
# ixx iyy izz ixy ixz iyz).
# Joint ranges are DERIVED from arm_backend.XARM6_JOINT_LIMITS_DEG, never
# repeated here. This file used to carry PI99 = pi*0.99, the xacro "limited"
# bound, which silently disagreed with both the controller and the hand-edited
# primitive scene -- three copies, three different answers. One source now.
import sys as _sys
_sys.path.insert(0, str(HERE.parent))
from arm_backend import XARM6_JOINT_LIMITS_DEG as _LIM
import math as _math

def _rng(j: int) -> str:
    """MuJoCo range="lo hi" in radians for 1-indexed joint `j`."""
    lo, hi = _LIM[j - 1]
    return f"{_math.radians(lo):.5f} {_math.radians(hi):.5f}"

MESH_ASSETS = """    <!-- xArm6 visual meshes (UFACTORY / xarm_ros2 xarm_description, BSD-3).
         Meshes are in metres; no scaling. Used for both visual + (convex-hull)
         collision in this spike variant. Paths are relative to the compiler's
         meshdir="assets/", so they read as xarm6/<file>.stl. -->
    <mesh name="xarm6_base"  file="xarm6/base.stl"/>
    <mesh name="xarm6_link1" file="xarm6/link1.stl"/>
    <mesh name="xarm6_link2" file="xarm6/link2.stl"/>
    <mesh name="xarm6_link3" file="xarm6/link3.stl"/>
    <mesh name="xarm6_link4" file="xarm6/link4.stl"/>
    <mesh name="xarm6_link5" file="xarm6/link5.stl"/>
    <mesh name="xarm6_link6" file="xarm6/link6.stl"/>
    <material name="xarm6_link_mat" rgba="0.9 0.9 0.92 1"/>
"""


def _new_arm_subtree(gripper_block: str) -> str:
    """Realistic mesh arm. `gripper_block` is the verbatim <body name="gripper">
    ... </body> lifted from the primitive scene, re-indented under link6."""
    g = "\n".join(
        ("                  " + ln[8:] if ln.strip() else ln)
        for ln in gripper_block.splitlines()
    )
    return f'''        <!-- ================= xArm6 (realistic meshes) ==================
             SPIKE variant. Kinematics + meshes from UFACTORY xarm_ros2
             (BSD-3). Joint/body names preserved so actuators, welds, rail,
             and the FK validator bind unchanged. See build_mesh_scene.py. -->
        <body name="xarm_base" pos="0 0 0.02">
          <geom name="base_link" type="mesh" mesh="xarm6_base" mass="2.7"
                material="xarm6_link_mat" class="arm_link"/>

          <body name="link1" pos="0 0 0.267">
            <joint name="joint1" axis="0 0 1" range="{_rng(1)}" damping="10"/>
            <geom name="link1_geom" type="mesh" mesh="xarm6_link1" mass="2.16"
                  material="xarm6_link_mat" class="arm_link"/>

            <body name="link2" pos="0 0 0" euler="-1.5708 0 0">
              <joint name="joint2" axis="0 0 1" range="{_rng(2)}" damping="10"/>
              <geom name="link2_geom" type="mesh" mesh="xarm6_link2" mass="1.71"
                    material="xarm6_link_mat" class="arm_link"/>

              <body name="link3" pos="0.0535 -0.2845 0">
                <joint name="joint3" axis="0 0 1" range="{_rng(3)}" damping="8"/>
                <geom name="link3_geom" type="mesh" mesh="xarm6_link3" mass="1.384"
                      material="xarm6_link_mat" class="arm_link"/>

                <body name="link4" pos="0.0775 0.3425 0" euler="-1.5708 0 0">
                  <joint name="joint4" axis="0 0 1" range="{_rng(4)}" damping="6"/>
                  <geom name="link4_geom" type="mesh" mesh="xarm6_link4" mass="1.115"
                        material="xarm6_link_mat" class="arm_link"/>

                  <body name="link5" pos="0 0 0" euler="1.5708 0 0">
                    <joint name="joint5" axis="0 0 1" range="{_rng(5)}" damping="4"/>
                    <geom name="link5_geom" type="mesh" mesh="xarm6_link5" mass="1.275"
                          material="xarm6_link_mat" class="arm_link"/>

                    <body name="link6" pos="0.076 0.097 0" euler="-1.5708 0 0">
                      <joint name="joint6" axis="0 0 1" range="{_rng(6)}" damping="3"/>
                      <geom name="link6_geom" type="mesh" mesh="xarm6_link6" mass="0.1096"
                            material="xarm6_link_mat" class="arm_link"/>
                      <!-- IK target marker, at the tool flange (+z out of link6). -->
                      <site name="end_effector" pos="0 0 0.09"
                            size="0.005" rgba="1 0.3 0.3 0.5" type="sphere"/>
{g}
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>'''


def main() -> None:
    src = SRC.read_text()

    # 1) inject mesh <asset> declarations just before </asset>
    if "xarm6_base" not in src:
        src = src.replace("  </asset>", MESH_ASSETS + "  </asset>", 1)

    # 2) locate the primitive arm subtree: <body name="xarm_base" ...> ... </body>
    open_re = re.compile(r'[ \t]*<body name="xarm_base"')
    m = open_re.search(src)
    if not m:
        raise SystemExit("could not find <body name=\"xarm_base\"> in lab_scene.xml")
    start = src.rfind("\n", 0, m.start()) + 1
    # brace-match <body>/</body> from the opening tag
    depth, i = 0, m.start()
    tag_re = re.compile(r"<body\b|</body>")
    end = None
    for t in tag_re.finditer(src, m.start()):
        depth += 1 if t.group() == "<body" else -1
        if depth == 0:
            end = t.end()
            break
    if end is None:
        raise SystemExit("could not brace-match the xarm_base subtree")
    arm_block = src[start:end]

    # 3) lift the gripper subtree verbatim from the primitive arm
    gm = re.search(r'[ \t]*<body name="gripper"', arm_block)
    depth, gend = 0, None
    for t in tag_re.finditer(arm_block, gm.start()):
        depth += 1 if t.group() == "<body" else -1
        if depth == 0:
            gend = t.end()
            break
    gstart = arm_block.rfind("\n", 0, gm.start()) + 1
    gripper_block = arm_block[gstart:gend]

    # 4) splice
    out = src[:start] + _new_arm_subtree(gripper_block) + src[end:]

    # 5) stamp a GENERATED banner so nobody hand-edits the default by mistake.
    banner = ("<!-- GENERATED FILE - DO NOT EDIT BY HAND.\n"
              "     Produced by envs/build_mesh_scene.py from "
              "envs/lab_scene_primitive.xml\n"
              "     (primitive arm -> realistic UFACTORY xArm6 meshes). Edit the\n"
              "     source scene there and rerun the generator. -->")
    out = re.sub(r"^<!-- envs/lab_scene[^>]*-->", banner, out, count=1)

    DST.write_text(out)
    print(f"wrote {DST.relative_to(HERE.parent)}  ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
