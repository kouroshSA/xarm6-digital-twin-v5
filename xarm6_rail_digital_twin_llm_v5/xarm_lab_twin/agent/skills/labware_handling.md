---
name: labware-handling
audience: intake
applies_to: instructions naming a class of laboratory object
---

# What each class of object requires

These are constraints on any acceptable plan, not a plan. State them as
requirements the task must satisfy; leave the route, the heights and the
order of moves to the planner.

## Tubes (Falcon, centrifuge, cryovial)

Tall, narrow, and top-heavy when full. The cap and the upper body are the
only sensible grip. Approach from directly above and lift vertically before
moving anywhere: a lateral nudge at body height tips them, and a tipped tube
in a rack takes the neighbouring tubes with it.

Keep upright throughout. If contents matter, say so as a constraint -- an
inverted tube is a spill even when the gripper never lost it.

## Well plates and tip boxes

Rigid, wide, shallow. Grip the skirt or the short sides; the wells themselves
are not a grip surface and lids are not attached. Keep level: tilting decants
the wells. These are wider than the standard jaws, so the task must state
that the bio gripper is required rather than leaving the planner to discover
it mid-plan.

## Open containers (cups, beakers, bins)

Usually the destination rather than the object. Release above the rim, not
inside it -- descending into a container risks the jaws touching its wall,
and the object only has to fall the last short distance. Never grasp an open
container by its rim while it holds anything.

## Blocks and cubes

The forgiving case. Any face is grippable and orientation rarely matters.
Where a task is ambiguous between a block and something fragile, prefer
naming the block explicitly.

## What to write

Convert the class into constraints the plan must meet: which surface may be
gripped, which orientation must be preserved, which effector is required.
Never convert it into steps, waypoints, or gripper commands -- those are the
planner's to choose, and it must stay free to choose differently when its
first attempt fails.
