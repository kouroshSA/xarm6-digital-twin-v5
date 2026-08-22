# Opus session reviews

Auto-appended after each EpisodeRetry session of 3+ episodes. Each
entry is an abstracted writeup produced by Claude Opus reading the
whole session. Most recent first. Capped at 10 entries.

Entries below are HYPOTHESES, not rules. They reference episodes by
number so future sessions can corroborate or refute them.

### 2026-08-22 13:50 [opus-4-7] "put red_cube_front in the translucent cup"

## Review

This session is a near-exact replay of the prior Opus-reviewed session for the same task: episode 4 succeeded again with essentially the same plan (rail=350, top-down at (0,-250,970), descend to 810, close, lift, transit to (-200,-350), descend to 830, open). Same success, same 9-command structure, same physical_outcome text. The other five episodes all failed at the same generic failure modes as before.

The recurring pattern across eps 1, 2, 3, 5, 6 is that the smaller model kept probing the top-down corridor at (0, -250, z) by varying z (870→920→970→1020) or the rail alone (ep 1: rail=150 rejected; ep 3: implied a set_rail probe), without combining rail=350 with the descent -- which was the winning combination in ep 4. Ep 5 in particular retried (0,-250,970) — the exact successful xyz of ep 4 — but without ep 4's preceding rail=350, and it failed. That's a direct A/B: rail is the operative variable at this xy, not z.

Ep 1's failure at `set_rail=150` (code 2) and the prior session's ep 3 failure at `set_rail=200` both indicate the rail primitive has validator-rejected positions in the low-mid range for this pick target; rail=350 is the value that has now worked twice.

The escalating-z pattern (870, 920, 970, 1020) is the same monotonic z-lift behavior flagged in every prior Opus review of this task. The learned_failure_constraints keep telling the model to raise z, and the model keeps raising z, and it keeps failing -- the constraint text itself may be misleading the search.

Ep 4's physical_outcome shows "disturbed red_cube_front 236mm; ee path 1451mm, 213 direction reversals" -- the cube is in the cup but the trajectory quality is low (quality=0.47 per the log). Not a false positive in the end-state sense, but the grader is accepting a messy execution. Consistent with prior session's ep 4.

Ep 6 tried yaw=90 instead of yaw=0 at (0,-250,970) -- a new degree of freedom this session -- and still failed. Suggests wrist-yaw variation alone doesn't clear the corridor when rail is at default; corroborates that rail is dominant.

## Cross-task observations

Both the "rail-envelope-at-y=-250-extends-high" and "gripper_close reports IK error" entries have supporting evidence here (the first directly; the second indirectly via non-appearance -- ep 4 succeeded once rail moved, meaning the "corridor" story holds). I'll corroborate geometric [0] with the new upper-bound data. New: `set_rail` itself has validator-rejected positions.

**Observations:**
- Episode 4 succeeded with the same rail=350 + top-down-at-(0,-250,970) template as the prior session's success; two independent successes now share the same rail value and approach xy, while every failure kept rail at default.  _(high; ep 4)_
- Episode 5 failed at move_to (0,-250,970) -- the exact xyz+orientation that episode 4 succeeded at -- differing only in that ep 4 preceded it with set_rail=350; this A/B pair points to rail state, not z, as the operative variable at this xy.  _(high; ep 4, 5)_
- Failing episodes 2, 3, 5, 6 escalated z monotonically (870→920→970→1020) at (0,-250) without ever adopting rail=350; the learned_failure_constraint text explicitly recommends raising z, which appears to steer the search away from the rail lever.  _(high; ep 2, 3, 5, 6)_
- Episode 1's set_rail=150 was rejected (code 2), matching prior session's ep 3 rejection of set_rail=200; rail values in the 150-200 range appear validator-rejected for this pick target while rail=350 is accepted.  _(medium; ep 1)_
- Episode 6 varied wrist yaw to 90 (a new DOF explored this session) at (0,-250,970) without changing rail, and still failed validation; wrist orientation alone did not clear the corridor.  _(medium; ep 6)_
- Episode 4's success reports quality=0.47 with 236mm cube disturbance and 213 direction reversals over a 1451mm EE path; end-state grader accepts the outcome but the trajectory is objectively messy, matching the prior session's success profile.  _(medium; ep 4)_

**Exploration diagnoses:**
- Episode 5 -- deviation: retried z=970 (ep 4's successful z) without ep 4's preceding set_rail=350; diagnosis: rail state was the differentiator; validator still saw the rail obstructing (0,-250) with rail at default
- Episode 6 -- deviation: escalated z to 1020 and varied yaw to 90 while keeping rail at default; diagnosis: neither higher z nor wrist yaw clears the obstruction when rail carriage occupies the approach column
- Episode 1 -- deviation: tried set_rail=150 as opening move; diagnosis: rail=150 is in the validator-rejected band for this target; rail=350 (which worked in ep 4 here and in prior session) was not attempted first
- Episode 2 -- deviation: kept default rail, tried z=870 at (0,-250); diagnosis: z=870 well within rail collision envelope at this xy; failure predicted by prior session's constraints
- Episode 3 -- deviation: kept default rail, tried z=920 at (0,-250); diagnosis: same corridor obstruction; z-escalation without rail change

---

### 2026-08-22 13:46 [opus-4-7] "put red_cube_front in the translucent cup"

## Review

This session finally produced a success (ep 4), giving us the first positive template for this task. The critical differentiator: ep 4 used `set_rail=350` before its top-down approach at (0, -250, z), while the failing episodes either kept the default rail or tried rail=200 (ep 3, which itself failed at `set_rail`). This suggests the rail's collision envelope at y=-250 is not a fixed obstacle -- moving the rail carriage away from that x column opens the top-down corridor.

The failing top-down `move_to (0, -250, z)` attempts spanned z=870, 920, 970, 1020 (eps 1, 2, 5, 6) -- all rejected. Notably ep 4 succeeded at z=970 *after* setting rail to 350, whereas ep 5 failed at z=970 *without* the rail move. Same xyz, opposite outcomes, differing only in rail state -- strong evidence that rail position, not z-lift, is the operative variable.

Ep 4's plan structure is worth noting: rail=350 → approach z=970 → descend to z=810 → close → lift to z=970 → rail=150 → move to cup xy (-200, -350) at z=970 → descend to z=830 → open. Two rail changes: one to clear the pick corridor, one to reach the cup placement.

The physical_outcome shows quality=0.48 with a 269mm cube disturbance and 201 direction reversals during the 1564mm EE path -- the cube did end up in the cup, but the grasp/carry was messy. Not obviously a false positive (end state is correct), but the quality score and reversal count suggest the arm was thrashing.

Ep 3's failure at `set_rail=200` (code 2) is an interesting distinct failure mode -- the rail itself has forbidden positions, not just move_to targets. The learned constraint attributes this to "optimal_rail_mm for the target object," suggesting the rail primitive validates against task-object reachability.

Eps 1, 2, 5, 6 kept trying to solve the problem by raising z (870→920→970→1020) without ever changing the rail, corroborating the prior Opus observation that the smaller model treats z-lift as the sole knob. Ep 4 broke the pattern with rail=350 and succeeded.

## Cross-task observations

The rail-position/approach-corridor coupling seems to be the key geometric fact here, extending prior observation [0]. And ep 3 shows `set_rail` is not free -- certain positions are validator-rejected outright.

**Observations:**
- Episode 4 (the only success) set rail to 350mm before the top-down approach; failing episodes 1, 2, 5, 6 all kept default rail and repeatedly raised z, so rail position appears to be the operative variable rather than z-lift.  _(high; ep 1, 2, 4, 5, 6)_
- Episode 5 failed at move_to (0, -250, 970) while episode 4 succeeded at the exact same xyz+orientation; the only difference was ep 4's preceding set_rail=350, suggesting the rail carriage itself contributes to the top-down obstruction at (0, -250).  _(high; ep 4, 5)_
- Failing episodes tried z=870, 920, 970, 1020 (eps 1, 2, 5, 6) without ever varying rail; the monotonic z-escalation without rail exploration mirrors the pattern flagged in prior Opus reviews.  _(high; ep 1, 2, 5, 6)_
- Episode 3 failed at set_rail=200 (code 2), showing that set_rail itself has validator-rejected positions rather than being a free primitive.  _(medium; ep 3)_
- Episode 4's physical_outcome reports quality=0.48 with 269mm cube disturbance and 201 direction reversals over a 1564mm path -- the end-state grader passed the task, but the trajectory was not clean.  _(medium; ep 4)_

**Exploration diagnoses:**
- Episode 5 -- deviation: retried z=970 (same z as ep 4's success) but without ep 4's preceding set_rail=350; diagnosis: rail state, not z, was the differentiator; validator still saw the rail obstructing at (0, -250)
- Episode 6 -- deviation: escalated z to 1020; diagnosis: still failed, suggesting no amount of z-lift clears the obstruction while rail is at its default position
- Episode 3 -- deviation: tried set_rail=200 (a novel value); diagnosis: rejected outright, likely outside the object's optimal_rail range; other episodes avoided the rail entirely rather than trying safe values like 350

**Observations:**
- Episode 4 (the only success) set rail to 350mm before the top-down approach; failing episodes 1, 2, 5, 6 all kept default rail and repeatedly raised z, so rail position appears to be the operative variable rather than z-lift.  _(high; ep 1, 2, 4, 5, 6)_
- Episode 5 failed at move_to (0, -250, 970) while episode 4 succeeded at the exact same xyz+orientation; the only difference was ep 4's preceding set_rail=350.  _(high; ep 4, 5)_
- Failing episodes escalated z monotonically (870, 920, 970, 1020) across eps 1, 2, 5, 6 without ever varying rail position -- rail was never explored as a lever.  _(high; ep 1, 2, 5, 6)_
- Episode 3 failed at set_rail=200 (code 2), indicating set_rail has validator-rejected positions rather than being a free primitive.  _(medium; ep 3)_
- Episode 4's success had quality=0.48 with 269mm cube disturbance and 201 direction reversals; end-state graded as success but trajectory was notably unclean.  _(medium; ep 4)_
- Ep 4 used two rail changes: rail=350 for pick at (0,-250) and rail=150 for place at (-200,-350); different xy targets appear to have different optimal_rail values.  _(medium; ep 4)_

**Exploration diagnoses:**
- Episode 5 -- deviation: retried z=970 (same as ep 4's successful z) but without ep 4's preceding set_rail=350; diagnosis: rail state, not z, was the differentiator; validator still saw rail obstructing (0, -250)
- Episode 6 -- deviation: escalated z to 1020; diagnosis: still failed; z-lift alone cannot clear the obstruction while rail is at default position
- Episode 3 -- deviation: tried set_rail=200; diagnosis: rejected outright, likely outside optimal_rail range for the cube; other episodes avoided rail changes entirely rather than trying values like 350

---

### 2026-08-22 13:02 [opus-4-7] "pick up the red cube that is in front of the rail and put it in the translucent cup"

## Review

All 6 episodes failed again, and this session shows an even more scattered exploration pattern than the prior Opus review flagged. The failures split between two modes: validator rejection of top-down `move_to` at y=-250 (eps 1, 2, 5, 6) and `gripper_close` returning "IK could not solve" (eps 3, 4). No successes, so no positive template to compare against.

The `move_to (0, -250, z)` target was tried at z=760, 795, 795, 830 across episodes -- all rejected. The learned_failure_constraints repeatedly say "lift z to >=845" or ">=880", yet subsequent episodes went back to lower z values (ep 6 tried z=795 again after already-learned constraints said >=845). This directly corroborates the prior Opus finding that the smaller model does not systematically incorporate its own learned constraints across episodes.

The recurring failure at (x=0, y=-250) with roll=180/pitch=0/yaw=0 across every episode suggests the model never varied xy or approach orientation, only z. Given the task says "in front of the rail," a top-down descent at y=-250 apparently intrudes into the rail's collision envelope up to fairly high z (ep 5 failed even at z=830). The z lower bound for a clear top-down approach at that xy seems to be somewhere above 830mm, but no episode tested this.

Eps 3 and 4 reached `gripper_close` (so their preceding `move_to` succeeded -- likely at a higher z) but then the close-time solve failed, consistent with the EE being too high above the cube when the close was attempted. This matches the prior Opus hypothesis that once z is raised to pass validation, no episode re-descends near the cube before closing.

## Cross-task observations

The pattern where a validator-rejected `move_to` produces a "learned constraint" telling the agent a minimum safe z, but the agent then ignores that constraint in later episodes, is a recurring meta-issue rather than a task fact. Not a world-model entry.

The `gripper_close` primitive returning "IK could not solve target" (failure_code 1) is worth noting -- gripper_close should not require IK unless the sim internally reposes the EE. This corroborates the prior Opus cross-task observation.

**Observations:**
- Every episode targeted (x=0, y=-250) with top-down orientation (roll=180, pitch=0, yaw=0); only z varied across attempts, so xy/orientation alternatives were never tested.  _(high; ep 1, 2, 3, 4, 5, 6)_
- move_to at (0, -250, z) with top-down orientation was rejected by the validator at z=760, 795, and 830 (eps 1, 2, 5, 6), suggesting the rail's collision envelope extends up to at least z~830 at that xy.  _(medium; ep 1, 2, 5, 6)_
- Episode 6 retried z=795 after prior episodes had already learned constraints requiring z>=845; the smaller model does not appear to apply its own accumulated failure constraints across later episodes in the same session.  _(high; ep 2, 6)_
- Episodes 3 and 4 got past the approach move_to (implying a higher z was used) but then failed at gripper_close with 'IK could not solve'; consistent with the prior Opus hypothesis that raising z to clear the validator leaves the EE too high above the cube with no intermediate descent.  _(medium; ep 3, 4)_
- No episode this session attempted a rail-position change or an alternate approach direction, even though the learned constraints explicitly suggested 'change rail position so the arm approaches from a less obstructed side'.  _(high; ep 1, 2, 3, 4, 5, 6)_

**Exploration diagnoses:**
- Episode 5 -- deviation: raised z to 830 (higher than prior attempts); diagnosis: still rejected by validator, indicating the rail collision envelope at (0, -250) extends higher than 830mm for top-down approach
- Episode 6 -- deviation: reverted to z=795 despite constraints from eps 1-2 requiring z>=845; diagnosis: regression rather than exploration; failed validation for the same reason as ep 2
- Episode 3 -- deviation: used a higher z that passed the approach move_to, then called gripper_close directly; diagnosis: no descent between approach and close; EE was above the cube's grasp band when close was invoked
- Episode 4 -- deviation: identical replay of ep 3; diagnosis: no variation from previous failure

---

### 2026-08-22 12:55 [opus-4-7] "pick up the red cube that is in front of the rail and put it in the translucent cup"

## Review

All 6 episodes failed. Episode 1 failed at a `move_to (0, -250, 795)` with a collision/FK validation error. Episodes 2-6 all failed at step 3/4, `gripper_close`, with IK-could-not-solve-target -- notably the same failure mode repeated five times in a row, suggesting the smaller model latched onto a single approach template and kept re-running it after the ep-1 z-lift adjustment without re-examining xy alignment or z-descent depth.

The learned_failure_constraints hint that after ep 1 the model raised z to >=845, but that likely left the EE too high above the cube (cubes typically sit around z~795-810 on the bench). Raising to clear the collision envelope but never re-descending before `gripper_close` would explain the persistent "IK could not solve" on the close -- the solver may be reporting a bad configuration when the pre-close pose is unreachable rather than a true "too far from object" issue. The constraint text mixing "IK" with "within 70mm of object" is itself suspect: an IK-solve error at `gripper_close` is a kinematics failure, not a grasp-distance failure.

No successes to compare against, so no cross-episode success pattern. The recurring nature of ep 2-6 is itself the strongest signal: the model did not meaningfully vary its plan between episodes despite five identical failures.

The red cube "in front of the rail" sits at roughly y=-250 based on ep 1's target xy. The rail's presence at that y likely constrains approach angles -- ep 1's collision at z=795 with roll=180,pitch=0,yaw=0 (top-down) suggests the rail body intrudes into the top-down descent corridor at that y.

## Cross-task observations

- The failure code reported for `gripper_close` was "IK could not solve target," which is unusual for a gripper primitive -- gripper_close is not typically an IK-solved motion. This may indicate that in this simulator `gripper_close` internally re-solves for a grasp pose, or the error is being mislabeled.
- Ep 1 shows z=795 with top-down orientation collides near y=-250 (in front of rail), consistent with a rail body extending into that footprint.

**Observations:**
- Episodes 2-6 all failed identically at gripper_close with 'IK could not solve target' after ep 1's z-lift adjustment; the model appears to have kept z high enough to clear the collision but never re-descended near the cube before closing.  _(high; ep 2, 3, 4, 5, 6)_
- Episode 1's move_to (0, -250, 795) with top-down orientation was rejected by validation, suggesting the rail body occupies part of the descent corridor at y~-250.  _(medium; ep 1)_
- Across 6 episodes the smaller model did not meaningfully vary its plan after finding one working z-lift value; five identical repeat failures suggest little exploration between episodes.  _(high; ep 2, 3, 4, 5, 6)_

**Exploration diagnoses:**
- Episode 2 -- deviation: raised z to clear ep-1 collision, then called gripper_close; diagnosis: raising z avoided the validator collision but left EE above the cube; close-time pose could not be solved / grasp missed
- Episode 3 -- deviation: same as ep 2; diagnosis: identical replay; no variation in xy or descent
- Episode 4 -- deviation: same as ep 2; diagnosis: identical replay
- Episode 5 -- deviation: same as ep 2; diagnosis: identical replay
- Episode 6 -- deviation: same as ep 2; diagnosis: identical replay

---
