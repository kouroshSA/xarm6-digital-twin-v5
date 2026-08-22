# A/B results

Evidence for whether a prompt layer changes behaviour. Raw harness output is
kept verbatim so a later reader can re-derive the conclusion rather than
trust the summary.

## 2026-08-23 (fourth attempt) — Layer 0, grounding wired AND ambiguity fixed

`--feature intake --repeats 5 --episodes 5 --render`, 30 runs, 1 lost.

| task | A (no intake) | B (intake) | ep-to-first A | B | |
|---|---|---|---|---|---|
| move the blue block into the cup | 70% | **92%** | 2.2 | **1.4** | B better |
| carefully place the red cube, no tipping | 88% | 88% | 1.2 | **1.0** | B better |
| red cube then blue cube into the cup | 4% | **16%** | 3.0 | **1.5** | B better |

**B better on 3, A better on 0.**

### The arc across four attempts is the real result

Same feature, same tasks, four measurements as the stack underneath got less
broken:

| attempt | state of the stack | verdict |
|---|---|---|
| 1 | home pose in contact; nothing could move | VOID (0% both arms) |
| 2 | Layer 1's facts never reached the planner | B better on 3 |
| 3 | facts wired; ambiguous referents got no geometry | B better on 1, A better on 2 |
| 4 | facts wired; every candidate measured | B better on 3 |

Attempt 2's win was Layer 0 compensating for missing grounding. Attempt 3
showed the cost of that same constraint text once the planner had numbers for
SOME objects but not the ambiguous ones -- more to read, nothing extra to use.
Attempt 4 is the first where both arms had complete geometry, and Layer 0
helps on all three.

The single largest movement in the whole sequence was not a prompt layer. The
"careful red cube" task went 67% -> 88% on the CONTROL arm from one fix:
reporting geometry for ambiguous referents.

### How much to believe attempt 4

Less than the clean sweep suggests. Two of the three margins are thin:

- The careful-cube task is a TIE on success rate (88% both); B wins only on
  episodes-to-first, 1.0 vs 1.2.
- The two-component task is 4% vs 16% -- that is 1/25 episodes against 4/25.
  Both arms mostly fail it.

Only the blue-block result (70% -> 92%, and 2.2 -> 1.4 episodes to first
success) is comfortably clear of the noise at n=5 per arm.

Error rate dropped from 5 lost runs of 30 to 1, with zero mjData races,
suggesting the earlier losses were tied to the missing-geometry path rather
than to rendering.

## 2026-08-22 (third attempt, rendered) — Layer 0, on a fully wired stack

`--feature intake --repeats 5 --episodes 5 --render`, 30 runs. The first run
of this experiment was void; the second ran while Layer 1's facts were not
reaching the planner. This is the first version measuring the stack as
described.

| task | A (no intake) | B (intake) | |
|---|---|---|---|
| move the blue block into the cup | 72% | **95%** | B better |
| carefully place the red cube, without tipping the cup | **67%** | 32% | A better |
| red cube then blue cube into the cup | **30%** | 25% | A better |

**B better on 1, A better on 2.**

### The headline is not Layer 0

Compare arm A across the two runs. Same tasks, same control condition:

| task | A before grounding | A after |
|---|---|---|
| blue block | 20% | **72%** |
| careful red cube | 0% | **67%** |
| two-component | 0% | **30%** |

Wiring Layer 1's measured facts into the prompt -- one commit, 60125e3 --
moved the baseline further than any prompt layer has. That was a bug fix, not
a feature.

### Layer 0's apparent benefit was compensation for that bug

The second run showed Layer 0 ahead on all three tasks. It was measured on a
stack where the planner got no grasp heights, and Layer 0's rewording helped
it cope. With the facts actually delivered, the benefit disappears and Layer 0
is neutral-to-harmful here.

That is worth stating plainly: **an A/B run against a broken baseline can make
a layer look useful when it is only compensating.** Both of this experiment's
first two attempts were invalid for different reasons, and only the third
measured what the summary claimed.

### A gap this exposed, not yet fixed

On both red-cube tasks the planner received NO geometry for the cube --
Layer 1 skips `check_graspable` for ambiguous referents, and "the red cube"
matches two. Both candidates are identical 30x30x60 blocks at z=780, so the
grasp height could be reported regardless. That is likely why those two tasks
are harder than the blue-block one, and why extra constraints hurt there:
the planner is guessing heights while carrying more text.

### Caveats

5 of 30 runs died and produced no summary (one `mj_copyDataVisual` race under
the viewer, four other early exits), so per-task n is 4-5 per arm rather than
5. Rendering is implicated in at least one of those.

## 2026-08-22 (second attempt) — Layer 0 (instruction intake)

`scripts/ab_test.py --feature intake --repeats 3 --episodes 5`, 18 runs, on
three tasks chosen to exercise what Layer 0 does: class constraints,
constraint-laden phrasing, and a two-component instruction.

| task | A (no intake) | B (intake) |
|---|---|---|
| move the blue block into the cup | 20% | **33%** |
| carefully place the red cube, without tipping the cup | 0% | **20%** |
| red cube then blue cube into the cup | 0% | **13%** |

Episode level: **A 3/45 successful episodes in 1 of 9 runs; B 10/45 in 4 of
9.** B was ahead or level in every repeat.

**Read this as directional, not settled.** Nine runs per arm is small, and
the variance is large -- repeat 1 on the blue-block task favoured A, and
repeat 3 produced no successes for either arm. Because the harness
interleaves A and B within each repeat, whatever caused that decline hit
both arms, which is why the per-repeat ordering is more informative than the
totals.

The honest claim: on these tasks Layer 0 did not hurt, and it produced
successes on two tasks where the control never succeeded at all. That is
worth having. It is not proof of an effect size.

### An earlier run of the same experiment was void

The first attempt returned 0/5 on both arms across all 18 runs. That was not
a null result -- the system was broken. Every episode died on its first
command because the twin's home pose rested the gripper inside the PCR
module, and the rail had just gained swept-path validation, so the arm could
not move at all from a pose already in contact. Fixed in 1a3edd7.

Worth remembering as a harness lesson: an A/B that returns zero on both arms
is measuring the apparatus, not the feature.
