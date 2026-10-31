# Opus session reviews

Auto-appended after each EpisodeRetry session of 3+ episodes. Each
entry is an abstracted writeup produced by Claude Opus reading the
whole session. Most recent first. Capped at 10 entries.

Entries below are HYPOTHESES, not rules. They reference episodes by
number so future sessions can corroborate or refute them.

### 2026-10-31 12:27 [opus-4-7] "Open the incubator door, grasp the cucumber and place it in the incubator, grasp the pastry and place it in the incubator, close the incubator door. Then open the cabinet door, grasp the clock and place it in the cabinet, close the cabinet door, and return to the home position. Requirements: When placing the cucumber in the incubator, avoid collision with the cabinet door frame or structure during approach and insertion; When placing the pastry in the incubator, avoid collision with the cabinet door frame or structure during approach and insertion; When placing the clock in the cabinet, approach with sufficient clearance from the cabinet door structure to avoid collisions; The arm must clear all static obstacles during all moves between the incubator and cabinet stations. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; incubator: at (0, 782, 1106) mm, 490 mm wide; incubator: not picked up; it has parts operate_part can open or close by their handles: incubator.Door, incubator.LowerShelf, incubator.UpperShelf; clock: at (580, -624, 831) mm, 63 mm wide; clock: grasp at z=858 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); clock: reachable with the rail at 300, 400, 500, 600, 700 mm; cabinet: at (-948, -300, 1220) mm, 372 mm wide; cabinet: not picked up; it has parts operate_part can open or close by their handles: cabinet.door_01, cabinet.door_02."

# Review: Multi-stage incubator + cabinet task (session 4, 3 failures)

## Summary
Fourth consecutive session on this task family with 0/3 success. This session's failure modes shifted again -- away from command-level failures (IK/validator refusals that previous sessions produced) toward **grader-level failures with high-quality plan execution**. Episode 1 actually executed the full task physically (cucumber in incubator, cupcake in incubator, clock in cabinet) but the grader rejected it for missing door-open states. Episodes 2 and 3 both produced no-op plans (0mm ee path, "no objects displaced").

## Observations

- **Episode 1 is the first episode across four sessions (12 episodes total) to achieve the three placement outcomes**: cucumber in incubator, cupcake in incubator, clock in cabinet. Quality was still low (0.16) with 29744mm ee path and 24 direction reversals, and the clock was disturbed 1532mm during planning. The grader rejected only on the door-state checks: `['incubator Door open', 'incubator Door open', 'cabinet door_01 open', 'cabinet door_01 open']` -- notably listed *twice each*, suggesting the grader checks door-open as a pre-condition for each put_in independently.

- **Episodes 2 and 3 produced empty/no-op plans** (0mm ee path, "no objects displaced"). This matches the bail-out pattern previously seen on the cucumber+cupcake task when the planner couldn't resolve conflicts. The learned_constraint from ep 1 ("Re-check the task target and use registry coordinates rather than estimating positions") appears to have been interpreted so conservatively that no plan was emitted.

- **The grader's door-check phrasing is informative**: it reports *missing* `'incubator Door open'` even though the physical outcome in ep 1 shows the cucumber and cupcake landed in the incubator. This suggests either (a) the planner used `put_in` which teleports/magics placement without requiring the door open, or (b) the grader checks the final door state and the plan closed the doors before the grader snapshot. Either way, the LLM-grader scores the door state independently from object-in-container state.

- **The learned_constraint text pinned from ep 1 ("Re-check the task target") appears to have been counter-productive**: eps 2 and 3 emitted no-op plans, worse than ep 1's actual-execution failure. The constraint text didn't convey the specific missing requirement (door-open states) that ep 1 actually missed.

- **This session breaks the pattern of prior Opus writeups** (sessions 1-3) which observed command-level failures (move_to collisions, set_rail refusals, put_in IK/precondition). Here, ep 1 successfully executed all the manipulations the prior sessions couldn't even reach -- suggesting the planner may have used higher-level primitives that bypass the approach-collision problems. But in doing so, it skipped door operations entirely.

- **Clock disturbed 1532mm during execution in ep 1** suggests the clock was knocked or dragged significantly during motion, possibly by a collision envelope during traversal. The physical outcome still accepted "clock in cabinet", so the grader tolerated the disturbance.

## Cross-session note
Eps 2 and 3's no-op bail-out matches prior-session patterns on simpler tasks (cucumber+cupcake session 1), where overly-general learned_constraints suppressed attempts. This corroborates the pattern that pinned constraint text can regress planner behavior rather than refine it. The door-state grader finding is new for this task family -- prior sessions never got far enough physically to observe it.

**Observations:**
- Episode 1 is the first episode across 4 sessions (12 episodes) to achieve all three placements (cucumber in incubator, cupcake in incubator, clock in cabinet); it still failed on grader missing door-open states, not on object placement.  _(high; ep 1)_
- Episodes 2 and 3 emitted no-op plans (0mm ee path, 'no objects displaced'), suggesting the ep-1-pinned learned_constraint 'Re-check the task target' was interpreted as a reason to bail out rather than refine the plan.  _(high; ep 2, 3)_
- Episode 1's grader_reason lists each door-open requirement twice (incubator Door open x2, cabinet door_01 open x2), suggesting the grader checks door-open as a separate requirement per put_in rather than as a single global state.  _(medium; ep 1)_
- Episode 1 placed all three objects without observable use of a door-open step -- the grader's 'Door open' misses together with successful object-in-container outcomes suggest the planner's put_in primitive may complete placement without first opening the container door, or the plan closed doors before the grader snapshot.  _(medium; ep 1)_
- Episode 1's quality was 0.16 with 29744mm ee path, 24 direction reversals, and clock disturbed 1532mm -- the plan succeeded physically but was highly inefficient and perturbed the clock substantially before placing it.  _(high; ep 1)_
- Four consecutive sessions on this task family have produced zero successes but exhibited distinct dominant failure modes each session (move_to collisions, set_rail refusals, put_in preconditions, now grader door-state misses), suggesting pinned constraints redirect between failure categories rather than converging on completion.  _(high; ep 1, 2, 3)_

**Exploration diagnoses:**
- Episode 1 -- deviation: Used primitives that completed all three placements but did not include open-door steps before each container access (or closed them before plan end).; diagnosis: Grader-level failure on missing 'incubator Door open' and 'cabinet door_01 open' (each listed twice). Physical placements succeeded, so the task primitive clearly can bypass door state, but the grader scores door-open independently.
- Episode 2 -- deviation: Emitted a plan with 0mm ee path -- effectively a no-op -- after the ep-1 constraint 'Re-check the task target' was pinned.; diagnosis: Constraint text was too general to point at the specific missing requirement (door state); planner regressed from executing-and-failing to not attempting.
- Episode 3 -- deviation: Same no-op bail-out as episode 2 despite task text explicitly listing door-open/close requirements.; diagnosis: Both pinned constraints from eps 1 and 2 say 'Re-check the task target' / 'no objects displaced' -- neither points at how to fix the door-open gap, so the planner remained in bail-out mode.

---

### 2026-10-31 12:01 [opus-4-7] "Open the incubator door, grasp the cucumber and place it in the incubator, grasp the pastry and place it in the incubator, close the incubator door. Then open the cabinet door, grasp the clock and place it in the cabinet, close the cabinet door, and return to the home position. Requirements: Avoid collisions with the arm structure when opening and closing doors; Plan door opening positions carefully to prevent arm interference; Ensure clear paths between the arm's current position and target objects. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; incubator: at (0, 782, 1106) mm, 490 mm wide; incubator: not picked up; it has parts operate_part can open or close by their handles: incubator.Door, incubator.LowerShelf, incubator.UpperShelf; clock: at (580, -624, 831) mm, 63 mm wide; clock: grasp at z=858 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); clock: reachable with the rail at 300, 400, 500, 600, 700 mm; cabinet: at (-948, -300, 1220) mm, 372 mm wide; cabinet: not picked up; it has parts operate_part can open or close by their handles: cabinet.door_01, cabinet.door_02."

# Review: Multi-stage incubator + cabinet task (3 failures, no completed stages)

## Summary
Third consecutive session of 3 episodes with zero successes on this task family. Nine consecutive episodes across three sessions have now failed in the first half of the plan, none reaching a release, door-close, or home step. This session's failures: ep 1 and ep 2 refused at `move_to` above the cucumber (z=950, z=1050; collision/FK), ep 3 refused at `put_in cucumber -> incubator` with "gripper is already holding something" at step 10.

## Observations

- **Episodes 1 and 2 both failed at a `move_to` directly above the cucumber at (x=-99, y=-264)** -- ep 1 at z=950 (step 3), ep 2 at z=1050 (step 4). Each learned constraint lifted the next attempt by 100mm but the validator refused the new height too. This matches the prior session's pattern where incremental z-lifts near the cucumber footprint kept hitting new envelopes.

- **Episode 3 progressed further (step 10) but failed on `put_in` with code 1 and message "gripper is already holding something"** -- this is a semantic/state error, not IK or collision. The planner apparently called `put_in cucumber -> incubator` while something was already grasped (likely the cucumber itself), suggesting the primitive expects the object to NOT yet be grasped (it does its own grasp internally), OR the planner double-sequenced a grasp.

- **The learned-constraint text for ep 3's failure explicitly contradicts the pattern used in eps 1-2**: it says "Keep using put_in for anything going inside a container -- a gripper pointing down cannot reach its shelves". This suggests prior sessions' explicit move_to-based placements may have been structurally wrong, and the primitive is the intended path -- but the primitive has its own precondition (empty gripper) that the planner didn't satisfy.

- **The failure mode has now shifted across three sessions**: session 1 was move_to collisions during traversal; session 2 was set_rail refusals; this session mixes move_to-above-cucumber collisions (eps 1,2) with a put_in precondition violation (ep 3). The planner oscillates between "explicit motions" and "high-level primitive" strategies without settling.

- **No episode in 9 consecutive attempts has observed what happens past the first put_in** -- so whether the door-open before put_in is actually required, what rail position put_in needs, and whether release/close-door primitives work remain entirely untested in this evidence base.

- **The z=950 and z=1050 refusals above the cucumber at rail unspecified are a recurring motif** (seen in session 1 at z=950, this session at both heights). The constraint-text suggestion "change rail position so the arm approaches from a less obstructed side" has not been exercised across any session, consistent with existing primitive entry [3].

## Cross-session notes
This corroborates existing primitive entry [2] (3-episode cap vs multi-container envelopes), [3] (rail-change mitigation untested), and [5] (put_in has preconditions beyond IK). The put_in "gripper already holding" failure is new evidence refining entry [5]: the primitive takes an un-grasped object and performs the whole pick-and-place, so pre-grasping the object breaks it.

**Observations:**
- Episodes 1 and 2 failed at move_to directly above the cucumber (x=-99, y=-264) at z=950 and z=1050 respectively; each learned constraint lifted z by 100mm but validator still refused -- the obstacle envelope above the cucumber zone extends higher than incremental 100mm lifts have probed.  _(high; ep 1, 2)_
- Episode 3 reached step 10 and failed on put_in with code 1 and message 'gripper is already holding something' -- a precondition violation, not IK/collision, suggesting the planner grasped the cucumber manually before invoking put_in which expects to do its own grasp.  _(high; ep 3)_
- Across three sessions (9 consecutive episodes) no plan has reached a release, door-close, or home step; the task is bottlenecked entirely in the cucumber-approach / first-put_in region.  _(high; ep 1, 2, 3)_
- The learned_constraint text for ep 3 explicitly instructs continued use of put_in rather than move_to+gripper_open for container placement, which contradicts the explicit-motion strategy that eps 1-2 (and prior sessions) attempted; the planner has oscillated between strategies without a converged approach.  _(medium; ep 1, 2, 3)_
- No episode in this session attempted a rail-position change as mitigation for the cucumber-above collision, despite the learned_constraint text suggesting it; only the z-lift mitigation has been exercised across all three sessions.  _(medium; ep 1, 2)_
- The three sessions on this task have each exhibited a different dominant early-failure mode (move_to collisions, set_rail refusals, put_in precondition), suggesting pinned constraints are redirecting the planner between failure categories rather than converging toward completion.  _(medium; ep 1, 2, 3)_

**Exploration diagnoses:**
- Episode 1 -- deviation: Attempted move_to at (-99, -264, 950) early in the plan, likely as part of an explicit approach to the cucumber.; diagnosis: Validator refused -- z=950 at the cucumber xy is inside some arm/environment collision envelope; the specific obstacle (likely bench shelving above cucumber or an incubator-side structure) was not identified before trying.
- Episode 2 -- deviation: Lifted the above-cucumber traversal to z=1050 per ep-1 constraint, but kept the same xy approach.; diagnosis: Validator still refused; the obstacle envelope above the cucumber extends past z=1050 at that xy, so the +100mm step was insufficient. Rail-change path remained unexplored.
- Episode 3 -- deviation: Switched to the high-level put_in primitive but pre-grasped the cucumber (reaching step 10 before failing).; diagnosis: put_in returned code 1 with 'gripper is already holding something' -- the primitive expects to perform the grasp itself; manually grasping first breaks its precondition.

---

### 2026-10-31 11:50 [opus-4-7] "Open the incubator door, grasp the cucumber and place it in the incubator, grasp the pastry and place it in the incubator, close the incubator door. Open the cabinet door, grasp the clock and place it in the cabinet, close the cabinet door, and return to the home position. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; incubator: at (0, 782, 1106) mm, 490 mm wide; incubator: not picked up; it has parts operate_part can open or close by their handles: incubator.Door, incubator.LowerShelf, incubator.UpperShelf; clock: at (580, -624, 831) mm, 63 mm wide; clock: grasp at z=858 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); clock: reachable with the rail at 300, 400, 500, 600, 700 mm; cabinet: at (-948, -300, 1220) mm, 372 mm wide; cabinet: not picked up; it has parts operate_part can open or close by their handles: cabinet.door_01, cabinet.door_02."

# Review: Multi-stage incubator + cabinet task (3 failures, zero progress on put_in)

## Summary
All 3 episodes failed before completing even the first placement. Failures were all command-level (not grader-level): one `put_in` IK failure at step 2, and two `set_rail` validator refusals at steps 7 and 17. Notably, the depth of progress *decreased* across the session (steps 2 → 17 → 7), opposite the pattern Opus observed in the previous 3-episode run on this same task where depth increased.

## Observations

- **Episode 1 failed on the very first `put_in`** (step 2, code 1 = IK failure). This is a different failure mode than the previous session, where the planner used explicit `move_to` sequences. Suggests the planner tried a high-level primitive this time and the primitive's internal pose solver couldn't reach the incubator interior.

- **Episodes 2 and 3 both failed on `set_rail`** (code 2 = collision/FK) -- ep 2 at rail=350, ep 3 at rail=0. Both of these rail values are *outside* the stated reachable sets for the object being approached at that step: rail=350 is not in cucumber's reachable set {0,100,200,300,400} nor clock's {300,400,500,600,700}; rail=0 is only in cucumber's set, so if ep 3 was transitioning toward clock/cabinet, rail=0 would be unreachable for clock.

- **The three distinct failures (put_in IK, set_rail 350, set_rail 0) exhaust three different primitive-level error modes** without the plan ever reaching a release, door-close, or home step. Together with the previous session's writeup, 6 consecutive episodes on this task have failed in the first half.

- **Rail selection appears to be a recurring friction point**: ep 2's rail=350 is almost exactly between the cucumber and clock reachable bands but in neither; ep 3's rail=0 is cucumber-reachable but likely attempted mid-task when clock was the current target. The learned_constraint hint "choose a rail closer to optimal_rail_mm" was emitted twice but doesn't appear to have changed subsequent rail-selection strategy within the session.

- **No episode in this session or the prior one has reached a `release` or door-close step**, so whether `put_in` on incubator/cabinet eventually works (given a valid approach rail) remains untested here.

- **Episode 1's single-step failure is especially cheap feedback**: it indicates the high-level `put_in` primitive does not auto-solve the full approach for a tall container (incubator at z=1106) from the home/default rail; the planner probably needs to pre-position the rail before invoking it.

## Cross-session note
Prior Opus writeup flagged that pinned constraints were pushing progress depth forward. This session, with only `set_rail` constraints getting pinned (not `move_to` z-lifts), the planner regressed to earlier failure points -- suggesting the two constraint categories don't compose cleanly, or that rail-constraint pinning is less informative than z-constraint pinning for forward progress.

**Observations:**
- Episode 1 failed at step 2 on put_in (code 1, IK), while eps 2 and 3 failed on set_rail (code 2) at steps 17 and 7 -- three distinct early-stage failure modes with no episode reaching a release or door-close.  _(high; ep 1, 2, 3)_
- Ep 2's rail=350 and ep 3's rail=0 both fall outside the reachable rail set for at least one of the two target objects (cucumber {0,100,200,300,400}, clock {300,400,500,600,700}); rail=350 is in neither set.  _(high; ep 2, 3)_
- Progress depth decreased across the session (step 2 -> 17 -> 7), opposite the previous session's monotonic-forward pattern -- suggesting rail-level learned constraints did not compose into forward progress the way z-level move_to constraints did previously.  _(medium; ep 1, 2, 3)_
- Episode 1's put_in failing at step 2 (immediately after presumed door-open) suggests the put_in primitive on a tall container like incubator (z=1106) does not auto-solve approach from the default rail position without an explicit prior set_rail.  _(medium; ep 1)_
- Across 6 consecutive episodes on this task (this session + the prior Opus-reviewed session), no episode has reached a release, door-close, or return-home step; the task is bottlenecked entirely in the first-half approach phase.  _(high; ep 1, 2, 3)_
- The learned_constraint text about rail selection was emitted twice (eps 2 and 3) but the second rail choice (0) was still outside the clock's reachable set, suggesting the planner did not use the constraint text to intersect reachable-rail sets across pending targets.  _(medium; ep 2, 3)_

**Exploration diagnoses:**
- Episode 1 -- deviation: Used high-level put_in primitive directly as step 2 rather than an explicit move_to approach sequence.; diagnosis: put_in returned code 1 (IK) -- the primitive could not solve a full approach-and-release into the incubator from whatever pose/rail followed door-opening; this is a different failure mode than the move_to-sequenced prior session which reached step 7-33.
- Episode 2 -- deviation: Chose rail=350 at step 17, likely as a compromise position mid-plan between cucumber (<=400) and clock (>=300) workspaces.; diagnosis: 350 is not in either object's listed reachable set; FK/collision validator refused -- rail positions between reachable bands are not themselves valid for transit.
- Episode 3 -- deviation: Chose rail=0 at step 7, early in the plan.; diagnosis: If the active target at that step was anything other than cucumber (e.g. transitioning toward clock), rail=0 is unreachable for it; failure at a much earlier step than ep 2 suggests the planner may have reshuffled task order rather than fixing the specific set_rail that failed in ep 2.

---

### 2026-10-31 10:32 [opus-4-7] "Open the incubator door, grasp the cucumber and place it in the incubator, grasp the pastry and place it in the incubator, close the incubator door. Then open the cabinet door, grasp the clock and place it in the cabinet, close the cabinet door, and return to the home position. Requirements: Avoid collisions with the arm structure and the environment throughout all movements; Plan paths that do not cause carried objects to strike obstacles or the incubator and cabinet structures; When placing objects in the incubator and cabinet, release them above the rim or opening rather than inserting the gripper deep inside. Measured from the scene: cucumber: at (-99, -264, 771) mm, 54 mm wide; cucumber: grasp at z=798 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); cucumber: reachable with the rail at 0, 100, 200, 300, 400 mm; incubator: at (0, 782, 1106) mm, 490 mm wide; incubator: not picked up; it has parts operate_part can open or close by their handles: incubator.Door, incubator.LowerShelf, incubator.UpperShelf; clock: at (580, -624, 831) mm, 63 mm wide; clock: grasp at z=858 mm (jaws reach ~45 mm below the tool, so descending lower drives the gripper into it); clock: reachable with the rail at 300, 400, 500, 600, 700 mm; cabinet: at (-948, -300, 1220) mm, 372 mm wide; cabinet: not picked up; it has parts operate_part can open or close by their handles: cabinet.door_01, cabinet.door_02."

# Review: Multi-stage incubator + cabinet task (3 failures)

## Summary
All 3 episodes failed, each at a different `move_to` step with validator refusal (code 2). The failures occurred progressively later in the plan sequence: step 7 (ep 1), step 11 (ep 2), step 33 (ep 3). This suggests the planner is incrementally learning from each pinned constraint and making it further through the multi-stage task before hitting the next collision envelope.

## Observations

- **Each episode failed at a `move_to` targeting a position above/near a container structure at z ≈ 50mm above the container's base reference**: ep 1 at (-100, -264, 950) approaching cucumber zone, ep 2 at (0, 500, 900) approaching incubator, ep 3 at (-900, -300, 1050) approaching cabinet. Each learned constraint pushes z up by 50mm but the planner then hits a different structure next time.

- **Progress through the task extended with each episode**: ep 1 failed at step 7/38 (never got past approach), ep 2 reached step 11/41 (into the incubator-side motion), ep 3 reached step 33/39 (near the end, approaching the cabinet). The cumulative pinned constraints appear to be redirecting the planner successfully past the earlier obstacles.

- **The failing moves are all at z-heights 50-100mm above the stated container/object top**: incubator at z=1106 was approached at z=900 (below the opening), cabinet at z=1220 was approached at z=1050 (below rim). This suggests the planner is routing the TCP *below* the container top during lateral traverse, which pulls the arm into the container's wall/frame.

- **Rail-position variation unknown from the trace**, but the recurring collision theme suggests the arm structure (not just the gripper) is intersecting container frames when approaching containers from a rail position that puts the shoulder/elbow near the obstacle.

- **No episode reached a release or close-door step**, so there's no evidence yet whether the "release above the rim" requirement or door-closing primitives work as expected. The task is bottlenecked on reaching the containers, not on manipulating them.

- **Cumulative learned constraints across the 3 episodes** form a corridor hint: z ≥ 1000 near cucumber zone, z ≥ 950 near incubator, z ≥ 1100 near cabinet. Future runs that lift traverse height uniformly to ~1150mm might clear all three in one plan, but this is untested here.

## Cross-session note
Prior Opus writeups on the cucumber-only task repeatedly identified that *release/approach poses near obstacle footprints get refused by the validator even at reasonable z-clearance*. This session extends that observation to large static structures (incubator and cabinet) at much higher z, consistent with the carried-envelope principle but now applied to the arm itself near container frames.

**Observations:**
- All 3 episodes failed at move_to with validator code 2, each at a different approach pose near a container (incubator at step 11, cabinet at step 33) or above the cucumber (step 7).  _(high; ep 1, 2, 3)_
- Progress depth increased each episode (step 7, 11, 33 of ~40), suggesting pinned learned_constraints redirected the planner past earlier failure points but new collision envelopes appeared further in.  _(high; ep 1, 2, 3)_
- Failing z-heights (950 near cucumber, 900 near incubator at z=1106, 1050 near cabinet at z=1220) are each roughly 150-200mm below the container's reference z; the planner appears to traverse laterally below the container top and hit its frame.  _(medium; ep 1, 2, 3)_
- No episode in this session reached a release or door-close step, so the back half of the plan (release-above-rim, close-door, return-home) remains entirely unexercised.  _(high; ep 1, 2, 3)_
- Cumulative pinned constraints across the 3 episodes form an approximate safe-corridor hint (z >= 1000 near cucumber, >= 950 near incubator, >= 1100 near cabinet) but no single plan has tried a uniformly high traverse height.  _(medium; ep 1, 2, 3)_
- The learned_constraint text suggests two mitigations (lift z OR change rail position); all 3 episodes appear to have attempted only the z-lift path, leaving rail-position variation as an unexplored dimension.  _(low; ep 1, 2, 3)_

**Exploration diagnoses:**
- Episode 2 -- deviation: Planner lifted z to 1000+ near cucumber per ep-1 constraint, successfully reached step 11, then approached the incubator at (0, 500, 900).; diagnosis: The incubator at (0, 782, 1106) has structure extending down; z=900 at y=500 pulls the arm into the incubator's front frame before clearing it.
- Episode 3 -- deviation: Planner cleared both cucumber and incubator approach constraints and reached step 33 before failing at the cabinet approach (-900, -300, 1050).; diagnosis: Cabinet at z=1220 with 372mm width means the arm needed higher clearance (>=1100) to approach from outside the cabinet's footprint; z=1050 was still inside its vertical envelope.

---

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
