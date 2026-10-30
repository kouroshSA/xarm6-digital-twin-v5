# Opus session reviews

Auto-appended after each EpisodeRetry session of 3+ episodes. Each
entry is an abstracted writeup produced by Claude Opus reading the
whole session. Most recent first. Capped at 10 entries.

Entries below are HYPOTHESES, not rules. They reference episodes by
number so future sessions can corroborate or refute them.

### 2026-10-30 23:15 [opus-4-7] "Pick up the cucumber and place it 150 mm to the right of its current position, avoiding the cupcake. Requirements: The cucumber must not collide with the cupcake during the move.; The cucumber must end 150 mm to the right of where it started.. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; cupcake: at (120, -250, 789) mm, 75 mm wide; cupcake: grasp at z=816 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cupcake: reachable with the rail at 0, 100, 200, 300, 400, 500, 600, 700 mm; to set cucumber back down on the open benchtop: carry it at z>=826 mm, then release with the tool at z=801 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench; to set cupcake back down on the open benchtop: carry it at z>=841 mm, then release with the tool at z=816 mm or a little above; it falls the last short distance (the benchtop surface is at z=750 mm). Do not descend below that release height while still holding it -- the carried object reaches lower than the tool does, and the move is refused for driving it into the bench."

# Review: cucumber +150mm with cupcake obstacle

## Summary
All 5 episodes failed. The task is: cucumber starts at (-99, -264, 771); needs to end at roughly (+51, -264, 771). Cupcake sits at (120, -250, 789) -- directly in or very near the destination zone. No episode produced a valid placement.

## Observations

- **Three of five episodes (2, 3, 5) produced "no objects displaced" with 0mm ee path** -- the planner emitted plans that never physically engaged. Episode 3 even scored quality=1.00 while doing literally nothing. Suggests the planner may have been bailing out early (empty/no-op plans) when it couldn't resolve the obstacle conflict, rather than attempting and failing.

- **Episode 1 attempted a release at (50, -250)** which is right next to the cupcake at (120, -250) -- only 70mm separation with cucumber 54mm wide and cupcake 75mm wide. The move was refused by validation (collision/FK). This is consistent with the destination zone being physically occluded by the cupcake footprint.

- **Episode 4 actually grasped and moved the cucumber, but ended up at (-100, -425)** -- it moved the cucumber 161mm in the -y direction (away from the cupcake) rather than +150mm in x. The planner appears to have interpreted "avoid cupcake" as "go around via -y" and then forgot to re-establish the +x target, or confused axis conventions.

- **The destination (+51, -264) sits ~70mm left-edge of the cupcake's bounding box** (cupcake left edge ~120-37=83, so destination at x=51 is ~32mm clear of cupcake's left edge, but with cucumber half-width 27mm, that's only ~5mm clearance). This is extremely tight; episode 1's validation refusal is consistent with this. The task as specified may require either wrist yaw rotation or lofting the cucumber over the cupcake with a very precise lateral offset.

- **"150mm to the right" direction ambiguity**: episode 4's -y motion suggests the planner isn't confident which axis is "right". None of the attempts placed the cucumber at the +x location shown by the registry delta.

## Cross-session note
This matches the prior-session trace exactly (same 5 failures replayed). The task appears structurally hard: the "right" destination overlaps the cupcake footprint, and simple pick-move-release plans either validate-fail or get bypassed.

**Observations:**
- Episodes 2, 3, 5 all produced 0mm ee path with 'no objects displaced', suggesting the planner emitted no-op plans rather than attempting a grasp when it couldn't resolve the cupcake conflict.  _(high; ep 2, 3, 5)_
- Episode 1 attempted release at (50, -250), ~70mm from cupcake center at (120, -250); with cucumber width 54 and cupcake width 75, that destination has only ~5mm edge clearance and validation refused the move.  _(high; ep 1)_
- Episode 4 was the only episode to actually move the cucumber, but transported it -161mm in y instead of +150mm in x -- suggests axis-direction confusion about which axis constitutes 'to the right'.  _(medium; ep 4)_
- The nominal destination (-99+150, -264) = (+51, -264) sits within the cupcake's lateral footprint envelope; no episode found a release pose that cleared cupcake without changing the destination.  _(medium; ep 1, 4)_

**Exploration diagnoses:**
- Episode 4 -- deviation: Routed cucumber in -y direction (away from cupcake) and released there instead of translating +x; diagnosis: Final cucumber xy = (-100, -425) is 161mm south of start, 0mm east -- the planner avoided cupcake geometrically but lost the +x displacement requirement, failing the grader's 'moved (150' pattern.
- Episode 1 -- deviation: Tried straight-line release at (50, -250) adjacent to cupcake; diagnosis: Validator rejected move_to at step 8 -- cucumber body would have collided with cupcake at that pose; releasing higher doesn't help because dropped objects aren't registered as 'placed'.

---
