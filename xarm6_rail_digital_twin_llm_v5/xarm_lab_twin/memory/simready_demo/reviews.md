# Opus session reviews

Auto-appended after each EpisodeRetry session of 3+ episodes. Each
entry is an abstracted writeup produced by Claude Opus reading the
whole session. Most recent first. Capped at 10 entries.

Entries below are HYPOTHESES, not rules. They reference episodes by
number so future sessions can corroborate or refute them.

### 2026-10-30 23:41 [opus-4-7] "Pick up the cucumber and put it down 150 mm to the right, avoiding collision with the cupcake Requirements: the cucumber must be kept upright throughout; the path must not pass through or contact the cupcake. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; cupcake: at (120, -250, 789) mm, 75 mm wide; cupcake: grasp at z=816 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cupcake: reachable with the rail at 0, 100, 200, 300, 400, 500, 600, 700 mm."

# Review: cucumber +150mm with cupcake obstacle (third run)

## Summary
Big improvement over the prior two sessions: 4/5 success, with episodes 2, 3, 4, 5 all placing the cucumber at approximately (+147, +2) from start. The breakthrough was wrist yaw rotation to 90° during transit and release, which was exactly what the lesson from the first ep 1 failure suggested. Episode 1 failed at the release move_to (yaw=0, direct) -- same failure signature as prior sessions.

## Observations

- **Episodes 2, 3, 5 all share an identical plan structure**: rail=100, grasp at (-99, -264, 798) with yaw=0, lift to z=870, rotate to yaw=90 at clearance height, translate to (51, -264, 870), descend to z=803, open gripper, lift. All three produced near-identical physical outcomes: cucumber moved (147, 2)mm. The one deviation (ep 2) rotated during the lateral move rather than at the start pose, and still succeeded.

- **Episode 1 failed at step 8 (move_to with yaw=0 descending to z=805 at x=51)** -- same validator refusal pattern that has recurred across every prior session when the direct release pose is attempted with cucumber long-axis along y. This is the fourth independent corroboration that the yaw=0 release pose at (51, -264) is inside the cupcake collision envelope.

- **The yaw=90 rotation was performed at clearance height (z=870) in all successes**, with the rotation either happening in place (eps 3, 5) or during the lateral translation (ep 2). The cucumber remained upright throughout (roll=180, pitch=0 held across all moves) -- so yaw rotation did not violate the upright requirement.

- **Rail set to 100mm in all 4 successes**, which is the smallest cucumber-reachable value that's also in the cupcake-reachable set. No episode tested other rail positions, so this isn't evidence of a required value -- just an unexplored dimension.

- **Release z was 803 across all successes** (slightly above the 798 grasp z). Episode 1's failure attempted z=805 with yaw=0 and was refused; same height class, so the yaw change (not z) appears to be what resolved the collision envelope issue.

- **Physical outcome "cucumber closer to cupcake"** appears in all success traces -- the grader accepted a destination that is near but not overlapping the cupcake. Quality scores clustered around 0.35-0.36, suggesting the grader passed on positional correctness even with moderate path cost.

## Cross-session note
Prior Opus writeups flagged that the planner was not exploring wrist yaw as a clearance strategy. This session the yaw=90 strategy was adopted (presumably via the pinned learned_constraint from ep 1) and it worked immediately in episode 2 and remained the template for eps 3-5. This corroborates existing primitive entry [0] (release pose must be collision-free) and refines it: *rotating the carried object's long axis parallel to the travel direction can clear the envelope at the same xy destination*.

**Observations:**
- Episodes 2, 3, 5 used an identical pick-rotate-translate-release structure with wrist yaw rotated 0->90 at clearance height z=870 and release at (51, -264, 803); all produced cucumber displacement of (147, 2)mm.  _(high; ep 2, 3, 5)_
- Episode 1 failed at the descent to z=805 at (51, -266) with yaw=0 -- same validator refusal signature as repeatedly seen in prior sessions when the straight-line release is attempted with cucumber long-axis along y.  _(high; ep 1)_
- In all four successes the cucumber remained roll=180, pitch=0 across every move with only yaw varying between 0 and 90, so the yaw rotation did not conflict with the 'keep upright' requirement graders apply.  _(high; ep 2, 3, 4, 5)_
- Rail was set to 100mm in all 4 successful plans and no other rail position was tested, so other reachable rail values (0, 200, 300, 400) remain unexplored for this task.  _(medium; ep 2, 3, 5)_
- Episode 2 rotated to yaw=90 during the lateral move (combined rotation+translation) while episodes 3, 5 rotated in place at the pick location first; both orderings succeeded, suggesting the rotation timing is not critical as long as the release pose has yaw=90.  _(medium; ep 2, 3, 5)_
- Release z=803 (just above grasp z=798) was accepted in all 4 successes; episode 1 attempted z=805 with yaw=0 and was refused -- indicating yaw orientation rather than z-height was the discriminating factor.  _(medium; ep 1, 2, 3, 5)_

**Exploration diagnoses:**
- Episode 1 -- deviation: Attempted direct release at (51, -266) with yaw=0 -- cucumber long-axis along y, pointing toward cupcake.; diagnosis: Validator refused the descent move_to; this is the same collision-envelope refusal that successes 2-5 avoided by rotating to yaw=90 so the cucumber's long axis ran along x instead of toward the cupcake.

---

### 2026-10-30 23:30 [opus-4-7] "pick up the cucumber and put it down 150 mm to the right, avoiding collision with the cupcake Requirements: the cucumber must be placed 150 mm to the right of its starting position; the path must not pass through or collide with the cupcake. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; cupcake: at (120, -250, 789) mm, 75 mm wide; cupcake: grasp at z=816 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cupcake: reachable with the rail at 0, 100, 200, 300, 400, 500, 600, 700 mm."

# Review: cucumber +150mm with cupcake obstacle (second run)

## Summary
All 5 episodes failed again. Same task structure as prior session: cucumber at (-99, -264, 771) needs to reach ~(+51, -264); cupcake at (120, -250, 789) sits very close to that destination. Two failure modes recur, matching last session's taxonomy.

## Observations

- **Episodes 1 and 5 both failed with command failure code 2 at step 7-8 (`move_to`) attempting release at x=51, y=-264/-267.** Both targeted the nominal "150mm right" destination and both were refused by the validator. This replicates the prior session's episode-1 pattern -- destination xy sits inside the cupcake collision envelope. Releasing higher (z=801) didn't help; the carried cucumber hangs below the tool.

- **Episodes 2, 3, 4 all produced a physical cucumber displacement in roughly (+146 to +147, -83 to -113)** -- i.e. the planner achieved the +x translation correctly (~146mm) but also dragged the cucumber ~83-113mm in -y. This is different from the prior session, where ep 4 moved only in -y. Here the planner seems to be routing *diagonally* around the cupcake, which gets the x-component right but still fails because the grader expects y-displacement ≈ 0.

- **The y-drift (-83 to -113) is suspiciously consistent across eps 2, 3, 4** -- suggests a repeated detour template (go around cupcake via -y) rather than random noise. The planner may be using cupcake-avoidance waypoints that it then releases from, instead of returning to y=-264 before releasing.

- **No episode produced a success.** The task appears genuinely hard: the ideal destination (+51, -264) is ~70mm from cupcake center at (120, -250), which with cucumber half-width 27 + cupcake half-width 37 ≈ 64mm leaves only ~5mm clearance. Any detour that doesn't return to y=-264 before release will fail the grader's tight y=0 tolerance.

- **Comparing to prior session**: previously 3/5 episodes emitted no-op plans (0mm ee path); this session all 5 episodes actually executed motion. That's progress in willingness to attempt, but the diagonal-release pattern is a new failure mode.

## Cross-session note
The prior Opus writeup already noted the destination sits within cupcake's footprint envelope. This session corroborates: when the planner respects that and detours, it releases at the detour waypoint (y≈-350) rather than returning to the correct y. The two paths through the task are (a) straight-line → validator rejection, or (b) detour → wrong release y. Neither has been resolved in 10 episodes across 2 sessions.

**Observations:**
- Episodes 1 and 5 both failed with move_to code 2 at step 7-8 when releasing at x≈51, y≈-264/-267 -- the nominal '150mm right' destination -- suggesting that pose sits inside the cupcake's collision envelope.  _(high; ep 1, 5)_
- Episodes 2, 3, 4 all achieved +146-147mm x-displacement (close to the +150 target) but also drifted -83 to -113mm in y, consistent with a detour-around-cupcake route that releases at the detour waypoint rather than returning to y=-264.  _(high; ep 2, 3, 4)_
- The y-drift magnitude clusters tightly (-83, -113, -113) across three independent episodes, suggesting a repeated planning template rather than random variance.  _(medium; ep 2, 3, 4)_
- Compared to the prior session where 3/5 episodes emitted no-op plans, all 5 episodes here executed motion -- the failure mode shifted from bail-out to diagonal-release; attempting at all is progress but the release y is wrong.  _(medium; ep 1, 2, 3, 4, 5)_
- No episode tried wrist-yaw rotation to reduce the carried cucumber's lateral extent toward cupcake, despite that being suggested in the learned_constraint text -- the planner is not exploring orientation as a clearance strategy.  _(medium; ep 1, 2, 3, 4, 5)_

**Exploration diagnoses:**
- Episode 2 -- deviation: Detoured in -y to avoid cupcake, then released at the detour waypoint (y=-347) instead of returning to y=-264.; diagnosis: Grader expects cucumber moved (150, 0); got (146, -83) -- the x-translation was correct but the release pose was never corrected back to the starting y.
- Episode 3 -- deviation: Same detour pattern as ep 2 but with larger y-offset.; diagnosis: cucumber ended at (146, -113) -- diagonal release again; the planner apparently treats the avoidance waypoint as the destination.
- Episode 4 -- deviation: Repeated the diagonal-release pattern despite 'keep upright' phrasing in the task.; diagnosis: Same (146, -113) outcome as ep 3; the task-text variations ('keep upright', 'avoid cupcake') didn't change plan structure.
- Episode 5 -- deviation: Returned to straight-line release at (51, -264) like ep 1.; diagnosis: Validator refused move_to -- same collision envelope issue; swapping back to the direct approach reverted to the ep-1 failure mode.

---

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
