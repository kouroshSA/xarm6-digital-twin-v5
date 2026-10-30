# World Model

scene_version: e1d5cc36637a
last_updated: 2026-10-30T23:15:45

Cross-task invariants learned from past sessions. Each entry's
confidence is derived from its corroboration count (1=provisional,
2=medium, 3+=high). If the scene XML changes, all entries are
flagged as untested until re-corroborated in the new scene.

## Geometric and kinematic invariants

- Carried-object collision envelope extends beyond the tool TCP: destinations within ~half-sum of object widths from an obstacle's center can trigger validator refusal even when the tool itself has clearance.
  Corroborated by: Pick up the cucumber and place it 150 mm to the right of its current position, avoiding the cupcake. Requirements: The cucumber must not collide with the cupcake during the move.; The cucumber must end 150 mm to the right of where it started.. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; cupcake: at (120, -250, 789) mm, 75 mm wide; cupcake: grasp at z=816 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cupcake: reachable with the rail at 0, 100, 200, 300, 400, 500, 600, 700 mm; to set cucumber back down on the open benchtop: carry it at z>=826 mm, then release with the tool at z=801 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench; to set cupcake back down on the open benchtop: carry it at z>=841 mm, then release with the tool at z=816 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench. (2026-10-30)
  Confidence: provisional

## Object-class regularities

_(no entries yet)_

## Primitive-behavior knowledge

- When a +X destination overlaps an obstacle's lateral footprint, releasing higher does not avoid the collision -- a dropped object isn't scored as placed, so the release pose itself must be collision-free at bench height.
  Corroborated by: Pick up the cucumber and place it 150 mm to the right of its current position, avoiding the cupcake. Requirements: The cucumber must not collide with the cupcake during the move.; The cucumber must end 150 mm to the right of where it started.. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; cupcake: at (120, -250, 789) mm, 75 mm wide; cupcake: grasp at z=816 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cupcake: reachable with the rail at 0, 100, 200, 300, 400, 500, 600, 700 mm; to set cucumber back down on the open benchtop: carry it at z>=826 mm, then release with the tool at z=801 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench; to set cupcake back down on the open benchtop: carry it at z>=841 mm, then release with the tool at z=816 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench. (2026-10-30)
  Confidence: provisional

## Grader and stringency regularities

_(no entries yet)_
