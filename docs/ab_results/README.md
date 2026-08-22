# A/B results

Evidence for whether a prompt layer changes behaviour. Raw harness output is
kept verbatim so a later reader can re-derive the conclusion rather than
trust the summary.

## 2026-08-22 — Layer 0 (instruction intake)

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
