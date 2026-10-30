# sim/tool.py
"""The twin's tool: the cell's F/T sensor + camera adapter stack and the UFACTORY
xArm Gripper on it. ONE copy of these numbers -- build_mesh_scene.py writes the
end_effector site from them, and scripts/sim_checks.py ties the hand-edited
lab_scene_primitive.xml back to them.

Where the numbers come from:

  GRIPPER_TCP_M     UFACTORY's TCP for the xArm Gripper, from its mounting
                    face: 172 mm (Menagerie's `link_tcp` site agrees).
  TOOL_STACK_M      flange -> gripper mounting face: the F/T sensor plus the
                    camera adapter. DERIVED, not measured directly:
                      flange -> fingertip, measured on the cell 2026-08-21
                      (arm_backend.TOOL_LENGTH_MM)                     217.0 mm
                      gripper mount -> fingertip, jaws closed, in
                      Menagerie's model (as for a tip probe)          174.5 mm
                      stack                                           = 42.5 mm
                    With the jaws OPEN at measurement it would be 54.8 mm.
                    Measure the stack with a ruler and replace this.
  TOOL_STACK_MASS_KG  the measured payload past the flange
                    (arm_backend.CELL_TCP_LOAD, 1.247 kg) less the gripper
                    (0.751 kg, Menagerie) and the D435i housing in the scene
                    (0.072 kg) = 0.424 kg: the F/T sensor (~0.27 kg, the gap
                    between CELL_TCP_LOAD and CELL_FT_LOAD) plus the adapter.

GRIPPER_GAP_TABLE maps the linkage's driver angle to the pad-to-pad gap,
sampled from Menagerie's closed four-bar (follower = inner = driver to
0.004 rad, which is why the twin can couple them with joint equalities).
"""
from __future__ import annotations

import numpy as np

GRIPPER_TCP_M = 0.172
TOOL_STACK_M = 0.0425
TOOL_STACK_MASS_KG = 0.424

#: flange -> TCP, the end_effector site's offset along link6 +z.
TCP_FROM_FLANGE_M = TOOL_STACK_M + GRIPPER_TCP_M

#: (driver angle rad, jaw gap m), sampled in Menagerie's model.
GRIPPER_GAP_TABLE = np.array([
    (0.002, 0.0887), (0.097, 0.0803), (0.203, 0.0704), (0.309, 0.0598),
    (0.415, 0.0488), (0.521, 0.0375), (0.628, 0.0259), (0.734, 0.0142),
    (0.841, 0.0024)])

GRIPPER_DRIVER_OPEN = 0.0
GRIPPER_DRIVER_CLOSED = 0.85            # the driver joint's upper limit


def driver_for_gap(gap_m: float) -> float:
    """Driver angle that leaves the jaws `gap_m` apart (clamped to the stroke)."""
    gaps = GRIPPER_GAP_TABLE[::-1, 1]
    angles = GRIPPER_GAP_TABLE[::-1, 0]
    return float(np.interp(gap_m, gaps, angles))
