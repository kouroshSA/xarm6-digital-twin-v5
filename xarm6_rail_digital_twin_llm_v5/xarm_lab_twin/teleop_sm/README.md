# `teleop_sm/` — SpaceMouse teleoperation of the twin

Jog the digital twin's TCP in six axes with a 3Dconnexion SpaceMouse, drive the
rail, toggle the gripper, and record replayable sessions.

```bash
cd xarm6_rail_digital_twin_llm_v5/xarm_lab_twin
python scripts/run_spacemouse.py --device pro
```

Setup (spacenavd + libspnav + the Python binding) is in
[`docs/spacemouse_setup.md`](../docs/spacemouse_setup.md). Tests need none of it:

```bash
python -m teleop_sm.test_device      # 19 tests
python -m teleop_sm.test_receiver    # 19 tests
```

## Why this is not the LeRobot SpaceMouse teleop

The UFACTORY LeRobot plugin ships one, and it is a **2-DOF planar jog**. In its
`get_action()` the rotation branch is commented out, `dpos[2] = 0` zeroes the Z
axis, and `gripper_action = 1.0` is a hardcoded constant; its only shipped
config is `xarm7_pushT` with `gripper_type: 0`. It was built for pushing a
T-block around a table, and it does that well.

It is also Python ≥ 3.12, and `xarm6sim` is 3.11.

So this package reads that implementation for the spnav event-loop pattern and
writes the rest against the twin's own interfaces — reusing `vr/`'s IK,
workspace clamp, smoother and `Recorder` rather than routing through LeRobot.
There is **no LeRobot dependency here**.

## Layout

| File | Does |
|---|---|
| `device.py` | spnav event loop → normalised six-axis state, button edges |
| `buttons.py` | semantic actions and the per-model button maps |
| `receiver.py` | the control-loop consumer: integrates, clamps, servos |
| `config.py` | every tunable, including the speed caps |

`scripts/run_spacemouse.py` wires them to `SimXArmAPI` + `Recorder`.
`scripts/spacemouse_probe.py` dumps raw events for mapping and fixtures.

## The three things that actually matter here

**A SpaceMouse is a velocity jog, not a pose tracker.** There is no absolute
hand pose, so there is nothing to clutch against — the VR path's "freeze the
controller-to-EE offset on engage" has no counterpart. The receiver integrates
deflection into a target pose it *owns*.

**The target must not be re-read from the arm each tick.** Re-reading the EE
and adding a delta feeds IK tracking error back into the command: the arm lags,
the lag is read as position, the next delta lands on the lagged value, and the
target creeps or runs away. `test_target_is_not_reread_from_the_arm` pins this.

**Smoothing is an output filter, never part of the integrator state.** Writing
the smoother's output back into the target makes the smoother's state *be* the
target, and each tick then advances by only `alpha * delta` — the jog silently
runs at 30% of the requested speed and just feels sluggish. This was a real bug
during development, caught by `test_target_integrates_over_time`, which is why
that test asserts distance over a full second rather than a single tick.

## Modes

`MODE_CYCLE` switches between:

- **`arm`** — all six axes jog the TCP (3 translation + 3 rotation)
- **`rail`** — Y translation drives the 700 mm rail; every other axis is ignored

The arm is 6 joints + rail = 7 DOF and the mouse has 6 axes, so something has to
give. Explicit mode switching beats auto-allocating the 7th DOF: an operator can
predict a mode and cannot predict a heuristic. `RAIL_MODE` is also available as
a momentary hold.

On the **Compact** (2 buttons) button 1 is `MODE_CYCLE` when tapped and
`RAIL_MODE` when held — forced by the button count, and the receiver suppresses
the edge action while the key is held. On the **Pro** the rail gets its own key.

## Two things that are PROVISIONAL until measured on hardware

**The axis permutation.** `config.AXIS_PERMUTATION` starts as the matrix vendor
code ships, which encodes *their* bench orientation. There is a second reason to
distrust it: spacenavd reports `device flags: swap y-z invert y-z` for the
SpaceMouse Pro, so the daemon has already transformed the axes before we see
them, and a matrix derived from raw HID reports is one transform too many.

Confirm it by pushing the cap forward and checking the TCP moves **+X**. If it
does not, fix `AXIS_PERMUTATION` / `AXIS_SIGNS` — the signs array exists so a
wrong direction can be corrected without re-deriving the matrix.

**The Pro's button indices.** The daemon's own log says it "reports 15 buttons
before disjointed button remapping", and the spnav protocol carries no model
string — `SpnavButtonEvent` gives a bare `bnum`. Run the probe, press each key,
read the index, fix `BUTTON_MAPS["pro"]`:

```bash
python scripts/spacemouse_probe.py --quiet-motion --seconds 60
```

The Compact's two buttons are unambiguous.

## Recording

`Recorder` is reused unchanged — the session schema already carries `rail_mm`,
`joints_deg`, `ee_pos_mm`, `ee_rpy_deg`, `ctrl`, `body_poses`, `weld_active`
and `gripper`. Sessions are tagged `interface="spacemouse_teleop"`.

Mode changes are written via `log_command()` so a replayed session is
interpretable: which mode was active changes how the trace reads.

`scripts/export_lerobot.py` needs **no changes** — it reads `task_label`
generically and never hardcodes `vr_teleop`.
