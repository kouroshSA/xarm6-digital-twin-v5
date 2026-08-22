# World Model

scene_version: 0ef767749087
last_updated: 2026-08-22T13:02:57

Cross-task invariants learned from past sessions. Each entry's
confidence is derived from its corroboration count (1=provisional,
2=medium, 3+=high). If the scene XML changes, all entries are
flagged as untested until re-corroborated in the new scene.

## Geometric and kinematic invariants

- The rail body's top-down collision envelope at y=-250 extends above z=830mm; approach corridors 'in front of the rail' at this y are more restricted than the cube's own height would suggest.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional

## Object-class regularities

_(no entries yet)_

## Primitive-behavior knowledge

- gripper_close can return failure_code 1 'IK could not solve target' -- unusual for a non-IK primitive, suggesting the sim internally re-solves EE pose at grasp time and reports upstream pose-reachability issues under the IK label.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional

## Grader and stringency regularities

_(no entries yet)_
