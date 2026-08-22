# World Model

scene_version: 9ad0bf8da29c
last_updated: 2026-08-22T17:08:46

Cross-task invariants learned from past sessions. Each entry's
confidence is derived from its corroboration count (1=provisional,
2=medium, 3+=high). If the scene XML changes, all entries are
flagged as untested until re-corroborated in the new scene.

## Geometric and kinematic invariants

- The rail body's top-down collision envelope at y=-250 extends above z=830mm; approach corridors 'in front of the rail' at this y are more restricted than the cube's own height would suggest.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22), put red_cube_front in the translucent cup (2026-08-22)
  Confidence: medium
- set_rail to 300 mm failed (code 2). Choose a rail position closer to optimal_rail_mm for the target object.
  Corroborated by: Place red_cube_front into the translucent cup Requirements: Release the cube above the rim of the cup, not inside it; Approach the cup from a direction that avoids contact with its walls during release. (2026-08-22)
  Confidence: provisional
- move_to (100, -150, 870) mm fails collision/FK validation. Lift z to >= 920 mm, or change rail position so the arm approaches from a less obstructed side.
  Corroborated by: Place red_cube_front into the translucent cup Requirements: Release the cube above the rim of the cup, not inside it; Approach the cup from a direction that avoids contact with its walls during release. (2026-08-22)
  Confidence: provisional
- set_rail to 550 mm failed (code 2). Choose a rail position closer to optimal_rail_mm for the target object.
  Corroborated by: Move the blue block into the translucent cup Requirements: Release the block above the rim of the cup, not inside it. (2026-08-22)
  Confidence: provisional
- set_rail to 400 mm failed (code 2). Choose a rail position closer to optimal_rail_mm for the target object.
  Corroborated by: Move the blue block into the translucent cup Requirements: Release the block above the rim of the cup, not inside it. (2026-08-22)
  Confidence: provisional

## Object-class regularities

_(no entries yet)_

## Primitive-behavior knowledge

- gripper_close can return failure_code 1 'IK could not solve target' -- unusual for a non-IK primitive, suggesting the sim internally re-solves EE pose at grasp time and reports upstream pose-reachability issues under the IK label.
  Corroborated by: pick up the red cube that is in front of the rail and put it in the translucent cup (2026-08-22)
  Confidence: provisional
- set_rail can return failure_code 2 for target positions that are within nominal travel range -- rail=150 and rail=200 were both rejected in this task family while rail=350 was accepted -- suggesting the primitive validates against per-object reachability envelopes rather than just mechanical limits.
  Corroborated by: put red_cube_front in the translucent cup (2026-08-22)
  Confidence: provisional

## Grader and stringency regularities

- The grader accepted a place-in-cup outcome with quality=0.47, 236mm target-object disturbance, and 213 direction reversals; loose stringency appears to key on the final containment relation rather than trajectory cleanliness.
  Corroborated by: put red_cube_front in the translucent cup (2026-08-22)
  Confidence: provisional
