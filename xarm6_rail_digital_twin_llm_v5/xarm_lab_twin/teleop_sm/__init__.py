"""SpaceMouse teleoperation of the MuJoCo digital twin.

Native to this repo rather than routed through the UFACTORY LeRobot plugin:
that plugin's SpaceMouse teleop is a 2-DOF planar jog (rotation commented out,
Z forced to zero, gripper hardcoded open) built for pushing a T-block, and it
requires Python >= 3.12 while this env is 3.11. See README.md.

Mirrors the structure of `vr/` and reuses its IK, workspace clamp, smoother
and recorder rather than duplicating them.
"""
