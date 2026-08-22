# World Model

scene_version: 0ef767749087
last_updated: 2026-08-22T12:55:52

Cross-task invariants learned from past sessions. Each entry's
confidence is derived from its corroboration count (1=provisional,
2=medium, 3+=high). If the scene XML changes, all entries are
flagged as untested until re-corroborated in the new scene.

## Geometric and kinematic invariants

- Top-down approach (roll=180,pitch=0,yaw=0) to points near y=-250 in front of the rail failed FK/collision validation at z=795, suggesting the rail body constrains descent corridors on its front face.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional

## Object-class regularities

- Red cubes on the bench appear to sit near z~795-810 based on the model's chosen descent targets, consistent with prior cube-manipulation traces.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional

## Primitive-behavior knowledge

- gripper_close was reported as failing with 'IK could not solve target' rather than a grasp-miss code, suggesting the primitive either performs an internal IK step or the simulator mislabels grasp-pose failures as IK errors.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional

## Grader and stringency regularities

_(no entries yet)_
