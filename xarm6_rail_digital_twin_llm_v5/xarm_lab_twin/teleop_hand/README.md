# `teleop_hand/` — whole-arm hand-tracking teleop (dex-retargeting / AnyTeleop)

Teleoperate the rail, the arm and the gripper with a bare hand in front of a
RealSense D435. Spec: `~/Dropbox/00002026_Q4/DEX_RETARGETING_TELEOP_IMPLEMENTATION.md`
(v2). **Status: twin side working and tested live on a D435i (steps 1–5 below);
real-arm runner and operator docs NOT written yet — see "To do".**

```
tracker process (env xarm6sim_hand)              teleop process (env xarm6sim)
D435 -> MediaPipe -> palm pose (depth) ----UDP---> ArmTeleopMapper -> guarded IK -> twin
        dex-retargeting -> gripper opening  :47100  clutch, filters, rail, gripper, Recorder
```

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
# terminal 1
conda activate xarm6sim_hand
python scripts/run_hand_tracker.py --show            # --list, --webcam N, --demo (no camera)
# terminal 2
conda activate xarm6sim
python scripts/run_handteleop_sim.py --operator-view # --orient yaw|full, --rail edge, --map absolute
```

Hold **Right Ctrl** (clutch) to move; pinch thumb+index to grip; Insert (in the
MuJoCo viewer) records a take with `interface == "handtrack_sim"`.

## Environments

* `xarm6sim` — teleop side. Added `dex_retargeting==0.5.0` (pure Python + nlopt;
  numpy stays 2.4.6).
* `xarm6sim_hand` — tracker. A clone of `xarm6sim` plus `mediapipe==1.1.1`
  (`requirements-hand.txt`). Kept separate because mediapipe pulls
  `opencv-contrib-python`, which would sit on top of `opencv-python` in the same
  `cv2/` directory.
* The HandLandmarker model (7.8 MB) downloads on first use to
  `assets/mediapipe/hand_landmarker.task` (gitignored), checked against a
  pinned SHA-256 in `config.py`.

## Findings (Phase 0 and live D435 sessions, 2026-10-25)

Each of these changed the code; most are guarded by a test.

1. **`mp.solutions` is gone** in mediapipe >= 0.10.30, and releases that still have it
   need numpy < 2. `detector.py` is upstream's `SingleHandDetector` ported to the
   Tasks `HandLandmarker` (MIT header kept, changes listed in the file).
2. **MediaPipe world axes == camera axes** (x right, y down, z away): measured on
   MediaPipe's own hand images, bone directions cos ≈ 1.0. `MEDIAPIPE_TO_CAMERA = I`.
3. **Handedness is NOT inverted on a D435.** Upstream assumes a mirrored image and
   swaps MediaPipe's label; live, that put the skeleton on the operator's LEFT
   hand. Default now trusts the label; `--swap-hands` exists for a camera that
   needs it.
4. **Fixed-base 1-DoF retargeting reads open hands as closed.** The thumb→index
   direction in the MANO frame swings so much between poses that its projection
   on a fixed jaw axis changes sign. `xarm_gripper.yml` adds a dummy free joint
   so only the pinch LENGTH drives `finger_joint` (`test_1b` is the control).
5. **Five palm points cannot grade a plane fit** (2 residual DoF: at 20 mm noise
   quality spread 0.0–0.9). The plane now also uses 21 depth samples inside the
   palm polygon, and the noise estimate is de-biased (`rms*sqrt(n/(n-3))`).
6. **One-Euro must run in millimetres.** With metres, beta=0.02 adds nothing and
   the filter is a fixed 1 Hz low-pass ~160 ms behind the hand.
7. **Jump rejection (50 mm / 30°) means scripted steps are outliers** — test
   helpers move the hand at human speed. Re-anchoring resets the filters onto
   the newest raw palm, or their lag drifts the target after the anchor.
8. **Joint-limit guard must let the arm back OUT.** Live, an arm parked at the
   3° margin refused even the return move (IK jitter read it as "inside").
   In-margin solutions are refused only if deeper than the current setpoint by
   more than 0.25°, with a hard stop 2° from the limit. Same rule for the wrist band.
9. **IKSolver returns orientation as best-effort** (position-only fallback reported
   as success). `guards.solve_guarded` now checks FK of every answer and refuses
   a miss > 2 mm / 2° (`pose_missed`).
10. **The twin's direct servo path has no collision checking** (same as SpaceMouse/VR
    direct). A test reaching toward x = 620 mm drove link 5 into the OT-2 and the
    rail stalled against the contact. Real arm: the controller's collision
    detection applies; twin: the operator can push into instruments.
11. **The shared workspace box spans the whole rail**, so rail edge-push uses a reach
    band of ±(400+30) mm around the arm base instead (`REACH_HALF_SPAN_MM`). Home
    already has the tool ~290 mm along the rail from the base.
12. **Front-row cubes (y = −250) are outside the shared box (y ≥ −200)** — same open
    decision as kinesthetic teaching. `run_handteleop_sim.py --workspace-y-min -400`
    widens it for one run only.

## Live results on the D435i (serial 346122072631, USB 3.2)

| Step | What | Result |
|---|---|---|
| 1 | still palm at 0.65 m | 30 FPS; palm std 0.4/0.8/0.5 mm, peak-to-peak ≤ 4.2 mm (spec ±10); rot_quality 1.00 |
| 2 | 3 slow pinches | opening 0.04–1.00, r = 0.977 vs direct pinch; exactly one close/open per pinch. Saved as `fixtures/pinch_real_d435.json` (test 1 runs on it) |
| 3 | axes, `facing` preset | right → +x (operator's right), up → +z, toward camera → +y (away from operator). All correct |
| 4 | yaw mode | 71° turn reproduced, same direction, 0.0° tilt leak; 13 s held still → 0.0° (deadband) |
| 5 | full mode | correct rotation axes; steady tracking 0.7° / 1.0 mm median; branch-flip and low-quality holds fired correctly |
| — | latency | capture→command p50 26 ms, p95 36 ms (8284 samples; excludes camera exposure) |

**Parameters settled in steps 1–5** (all in `config.py`): preset `facing`;
`PINCH_CLOSED_M=0.02`, `PINCH_OPEN_M=0.10`, yml `scaling_factor=1.5`, URDF pad gap
30 mm; grip thresholds 0.30 / 0.60; `ROT_DEADBAND_DEG=5` (exit 2.5);
`ROT_QUALITY_GOOD/BAD_MM=4/15`; One-Euro 1.0 Hz / beta 0.02 per mm/s; jump
50 mm / 30°; dropout 150 ms, silence 500 ms; 150 mm/s, 90 °/s; guards: branch
flip 20°, joint margin 3° (jitter 0.25°, hard stop 2°), wrist band 5°, FK
tolerance 2 mm / 2°.

## Tests (no hardware)

```bash
python -m teleop_hand.test_retarget                 # 6: tests 1 (synthetic + real D435), 1b control, 2, effectors
python -m teleop_hand.test_palm                     # 6: test 3 + frame convention, depth, fist
python -m teleop_hand.test_mapping                  # 9: tests 4-9 (mapper half), precision
MUJOCO_GL=egl python -m teleop_hand.test_twin       # 4: tests 9, 10, 11 + no-clutch-no-motion
```

## To do

1. **Live pick in the twin (step 6)**: `run_handteleop_sim.py --start-j1-deg -70
   --workspace-y-min -400 --orient locked`, pick `red_cube_front`, record, grade
   the take (gripper channel, export). Ready to run; not yet done.
2. **Per-joint speed cap.** In full mode the wrist joints lagged their commands by
   up to 20–27° during fast rotations (90 °/s EE rotation demands more joint speed
   than the twin's PD delivers near some poses). Slow the command when the IK
   step implies joint speed above a cap. Important before the real arm.
3. **`scripts/run_handteleop_real.py`** — not written. `teleop_hand/real.py` has the
   pieces (`RealHandStreamer`: mode 7 streaming via `RealXArmAPI.set_position(wait=False)`,
   twin-IK veto seeded from mirrored joints, measured twin→real tool offset,
   rail ≤ 10 Hz, latched faults + explicit recovery; `HandFakeController/Backend`).
   Needs: the runner (mirror via `teleop_kin.real_teach.RealMirror`, clutch,
   continuous standard gripper via `RealXArmAPI.set_gripper_opening`, binary
   BIO/vacuum, every exit through `return_to_position_mode`), a fake-controller
   SIGTERM test like `teleop_kin.test_real`, and Phase 0 on hardware (firmware
   supports mode 7?).
4. **Docs**: `docs/hand_teleop.md` (camera placement, presets, modes, clutch, effector
   table, guards, safety checklist) and a section in the top-level README.
5. **Not yet exercised live**: `--map absolute` (+ `--calibrate-absolute`), `--rail edge`,
   precision key, pedal and gesture clutch (test 13), webcam fallback with
   `--calibrate-depth`, operator view on the real arm.
6. **Regression**: run the task sweep and the SpaceMouse/VR suites once more after
   the above.
7. **Hardware tests 12–17** (operator present): real axes, clutch reliability,
   6-DoF usability, latency incl. camera, holds/errors, SIGTERM.
8. Port the joint-limit "allow moving out" fix and the FK verification to
   kinesthetic teaching's `servo_direct` path, which trusts IKSolver the same way.
