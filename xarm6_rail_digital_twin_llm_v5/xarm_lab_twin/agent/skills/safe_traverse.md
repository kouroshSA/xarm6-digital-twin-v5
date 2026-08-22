---
name: safe-traverse
applies_to: any task that moves the arm between two places in the cell
scene_contract: swept-path validation; jaw-depth grasp gate
---

# Moving without knocking things over

The arm sweeps a large volume. A move is validated along its whole path, so a
trajectory that clips something is refused outright -- the command returns a
failure and nothing moves. Plans fail far more often from *how* they travel
than from where they are going.

## Retract before you traverse

Never move the rail, and never make a long lateral move, while the tool is at
working height. Lift to a clearance height first, traverse, then descend. A
rail move at grasp height drags the whole arm sideways through whatever is on
the bench.

Order for any pick-and-place:

1. Lift straight up to clearance.
2. Traverse (rail and/or lateral) at clearance.
3. Descend straight down onto the target.
4. Act.
5. Lift straight up again before going anywhere else.

## Descend and ascend vertically

Keep x and y fixed while changing z. Diagonal moves toward a target sweep the
tool through the space beside it, which is exactly where neighbouring objects
sit.

## The path is not a straight line

Interpolation happens in joint space, so the tool bows away from the straight
line between two points. Two poses that are both clear can still be joined by
a path that is not. If a move is refused but both ends look fine, this is why:
insert an intermediate waypoint at clearance rather than raising the target.

## Read the refusal

A refused move names where it was blocked -- "Path blocked at 92% along the
path" plus the two geoms involved. Use it. Retrying the same move, or nudging
the target a few mm, will not clear an obstacle sitting mid-path.

## When a descent is refused

The obstacle is beside the target, not the target itself. Traverse to a
position that approaches from a clearer side, or lift and come down closer to
vertical. Do not close the gripper early to compensate: the jaws must actually
be around the object, and closing short is reported as a failed grasp.
