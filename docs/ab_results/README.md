# A/B results

Evidence for whether a prompt layer changes behaviour. Raw harness output is
kept verbatim so a later reader can re-derive the conclusion rather than
trust the summary.

## 2026-08-23 (sixth) — a regression I introduced, found by the re-check

`--feature intake --repeats 8 --episodes 5`, 16 runs, **0 lost** (the harness
fix holding). A 40%, B 22%.

The number that mattered was not A vs B but **A across sessions**: 76% (n=12)
before the prompt layers were wired per-episode, 40% after. And arm A scored
**exactly 2/5, first success at episode 3, in all eight repeats** -- identical,
not scattered. Deterministic outcomes rule out variance and make isolation
cheap: one run per condition decides.

### Three hypotheses, all killed by measurement

- *the carried-object check refuses legitimate moves into the cup* -- no:
  "the carried" appears zero times in 16 runs
- *the 0/2 seen earlier proves a regression* -- no: that was two-episode runs
  on a task whose first success lands around episode 3
- *the placement fact bloats the prompt* -- no: **2/5 with it, 2/5 without it**

### The actual cause

The per-episode layer re-run (added with the Layer 0 feedback loop, ec1be96)
ran BEFORE `reset_scene()`. Layer 1 measures the scene, so from episode 2
onward it reported coordinates for the scene the PREVIOUS episode left behind
-- cube knocked aside, cup displaced -- and the reset then restored the world
out from under those numbers.

The signature fits exactly: episode 1 unaffected, later episodes degraded,
first success pushed from episode 1 to episode 3.

Moving the re-run after the reset: **2/5 -> 3/5** on the same deterministic
task. Pinned by `agent.layers_measure_after_reset`, which fails if the order
is ever swapped back.

Not fully back to 76%, and that is left honest rather than tuned: the
historical figure came from a session whose per-run rates ranged 0.4-1.0,
whereas this build is deterministic, so the two are not cleanly comparable.

## 2026-08-23 (fifth) — 12 repeats on one task, and a retraction

24 runs, 5 lost, 3 mjData races. One task, the only one whose margin looked
clear at n=5.

    success rate      A 76%   B 26%
    episodes to first A 1.8   B 1.3
    -> A better

Within this run the difference is solid: per-run rates A 0.76 (sd 0.21, n=9)
vs B 0.26 (sd 0.22, n=10), Welch t = 4.8, barely-overlapping ranges.

### This retracts the previous entry

The fourth attempt reported **B 92%** on this same task. Same code -- only a
docs commit between them -- and the prompts are BYTE-IDENTICAL: Layer 0
produced the same rewrite and the same single constraint in every run of both
sessions. Nothing about the experiment changed.

B measured 0.92 on 5 runs, then 0.26 on 10. **The between-session shift is
larger than the effect being measured.** At n=5 the harness cannot distinguish
Layer 0 from noise, and the "B better on 3" headline should not have been
written.

Best current reading: on this task Layer 0 does not help and may hurt. Held
loosely -- the same caution that makes this a retraction applies to reading
too much into 10 runs either.

### A bug in the harness, found by this run

It printed "B better" for 76% vs 26%. The verdict rule was
`b_rate > a_rate OR b_ttf < a_ttf`, so a faster time-to-first overrode a
50-point collapse in success rate. Worse, episodes-to-first is conditioned on
succeeding: an arm that fails most runs and gets lucky early in the rest looks
FAST. Success rate now decides; time-to-first only breaks a tie within 5
points, and says so when it does.

Every verdict in the entries below was produced by the buggy rule. Re-checked
by hand: attempt 4's three verdicts are unchanged under the corrected rule,
because its rate differences all pointed the same way as its tiebreakers.

### The harness is too lossy to settle anything yet

5 of 24 runs lost (21%), 3 of them `mj_copyDataVisual` races under the viewer
-- after a run that lost 1 of 30 with none. Losses are not evenly spread
either: 3 of arm A's 12 runs died. A harness that discards a fifth of its
runs, unevenly, cannot support a conclusion this fine.

**Fix the losses before running more comparisons.**

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
