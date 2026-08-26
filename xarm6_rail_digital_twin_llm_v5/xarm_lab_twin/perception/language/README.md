# Language-conditioned targeting

Say what you want; get the grasp for it.

```python
from perception import SimWristCamera            # or RealSenseWristCamera
from perception.language import LanguageTargeter

t     = LanguageTargeter()
frame = SimWristCamera(arm).capture()

grasp = t.grasp_for("the blue cube", frame, max_size_m=0.08)
if grasp:
    arm.set_position(*grasp.to_arm_pose())
```

Three stages, each doing only what it is good at:

| stage | does | why not the others |
|---|---|---|
| **Ground** — Grounding DINO | free text → image regions | open vocabulary; no training needed for "blue cube" |
| **Physicalise** — depth | region → measured object | an image has no scale; depth does |
| **Select** — GG-CNN | object → grasp pose | grounding says *where*, not *how to hold it* |

## Why the depth stage is not optional

Ask Grounding DINO for **"a green cube"** over this bench and it returns the
green cube *and* the green bin, both confidently — the bin often scoring
**higher**. It is not wrong: a bin is a green box, and a single image carries no
scale, so nothing in the picture separates a 40 mm cube from a 150 mm bin. No
amount of prompt tuning fixes this reliably, because the distinction is
physical, not linguistic.

Depth carries the scale the image does not:

```
$ python scripts/target_check.py "a green cube" --max-size 0.08 --compare

  max_size=0.08: score 0.42, measured 30 mm across
      centroid  x=4.9  y=-150.7 z=779.1 mm     <- green_cube
  no size bound: score 0.47, measured 83 mm across
      centroid  x=-5.6 y=-332.2 z=790.6 mm     <- green_bin
```

`max_size_m` lets a caller say "a cube is small" without hard-coding this
scene's objects into the perception stack.

The same depth step also turns the box into a **mask** for free — pixels inside
the box at roughly the object's own depth are the object; pixels at the bench's
depth are not. That matters more than it sounds: a grounding box around a 40 mm
cube seen from 300 mm is mostly bench, so a centroid over the whole box lands
*beside* the cube. Measured here, the mask is ~65% of the box.

One subtlety worth knowing if you touch `_physicalise`: the depth band is
centred on the region's **lower quartile** depth, not its median. For a small
object the median *is* the bench, and a band around it would mask the bench
instead of the object.

## Verified, not just demonstrated

```bash
MUJOCO_GL=egl python -m perception.language.test_targeting
```

Ground truth is the scene: MuJoCo knows where every body is, so a phrase can be
checked against the object it names. A targeter that confidently returns the
wrong object produces exactly as convincing a picture as one that works.

| phrase | resolves to | error |
|---|---|---|
| `"a blue cube"` | `blue_cube` | 2.7 mm |
| `"a red cube"` | `red_cube_front` | 4.0 mm |
| `"a green cube"` | `green_cube` | 4.9 mm |
| `"a rubber duck"` | `None` | — |

The suite includes `check_size_filter_is_what_fixes_it`, which asserts **both**
halves of the claim above: that unfiltered targeting picks the bin, and that the
size bound picks the cube. If someone later drops the depth stage believing
grounding alone suffices, the first half starts passing for the wrong reason and
the second fails outright.

## API

`LanguageTargeter.target(phrase, frame, ...) -> Target | None`

`Target` carries `phrase`, `score`, `box`, `mask`, `centroid_px`,
`position_cam`, `position_world`, `size_m` / `max_size_m`, `n_pixels`, and
`grasps` (the GG-CNN grasps landing on the mask).

`LanguageTargeter.grasp_for(phrase, frame, ...) -> Grasp | None` — the
one-liner.

Useful arguments:

- **`max_size_m` / `min_size_m`** — physical extent bounds. The cube-vs-bin
  discriminator.
- **`distractors`** — extra phrases to put in the prompt. Grounding DINO
  calibrates better when the alternatives are named, so
  `distractors=("a bin", "a cup")` sharpens the score for what you *do* want,
  even though those detections are discarded.

`grasp_for` returns `None` when GG-CNN proposes no grasp on the object it found,
rather than falling back to the region centroid. A centroid is a *position*, not
a grasp — it has no jaw angle and no evidence the gripper can close there.
Callers who want the position anyway should use `target()` and read
`position_world`, which says what it is.

## Speed

`xarm6sim` runs `torch 2.13.0+cu126`, so `device="auto"` selects the GPU.

| device | model load | per call |
|---|---|---|
| **CUDA (RTX 3080)** — the default here | 3.2 s | **0.37 s** |
| CPU (`torch+cpu`) | 0.5 s | 4.90 s |

13x. The two devices return the same detections — same count, same phrases,
boxes agreeing to 3e-4 px and scores to 8e-4 — but not bit-for-bit, as ordinary
cross-device float variation. Immaterial at these thresholds; worth knowing
before writing an exact-equality test across devices.

Note the load/inference trade: CUDA takes
~3 s longer to initialise, so a script that grounds exactly once and exits sees
little benefit. Build the targeter once and reuse it.

If a CPU-only environment is ever needed:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
```

`device="cpu"` also forces it without reinstalling.

## In the planner

`LLMBrain` exposes this as three dispatch actions — `locate_object`,
`move_to_object`, `grasp_object` — so a plan can say
`{"action": "grasp_object", "params": {"description": "the blue cube"}}`.
The adapter is `agent/vision_targeting.py`, which also supplies the size prior
for a phrase (`"cube"` -> at most 80 mm) so callers get the cube-vs-bin
discrimination without knowing it exists. See CLAUDE.md for why those actions
resolve-check-move internally instead of binding late-bound references.

## Limits

- **Grounding DINO is a proposal stage, not an oracle.** It is doing zero-shot
  detection on flat-shaded synthetic renders, well outside its training
  distribution. Scores of 0.4–0.7 here are normal; treat ranking as a hint and
  let depth arbitrate.
- **Spatial language is not handled.** "the cube *behind* the bin", "the
  *leftmost* tube" — Grounding DINO has weak spatial grounding, and nothing here
  adds any. `Target.position_world` is in base coordinates, so relational
  filtering is straightforward to add on top, and belongs above this layer.
- **One object per query.** `target()` returns the best match. The grounder
  returns all of them, so multi-instance selection is a small extension.
- **Real-cell depth is much sparser** (~50–70% valid vs 100% in sim). The mask
  and size estimate degrade with it; `MIN_REGION_PIXELS` guards the worst case
  by returning `None` rather than a number built from noise.

## Model

`IDEA-Research/grounding-dino-base` (Apache-2.0), 232 M parameters, ~700 MB, from
the HuggingFace cache. `check_grounding_model_available` in the sweep confirms
it is on disk without loading it, so a missing checkpoint surfaces as a sweep
note rather than as a stalled demo.

Adapted from `~/vision_stack/scripts/live_grounding_dino_realsense.py`.
