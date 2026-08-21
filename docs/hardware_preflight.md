# Hardware pre-flight checklist (B0)

**Run this before every powered session.** Not once at setup — every session.
Most of it takes two minutes; the parts that don't are the parts that matter.

Operating model for all of Stream B: **attended operation only.** A human stands
outside the swept envelope with a hand on or beside the e-stop, for the whole
session. No unattended running.

> Every SDK call below was verified against the pinned `xarm-python-sdk==1.18.4`
> (see `requirements.txt`). Signatures differ in older builds — see
> [`a8_controller_validator.md`](a8_controller_validator.md) for the version split.

---

## Part 1 — One-time cell setup

Done once, then re-checked whenever the arm, rail or bench is moved.

### 1.1 Mark the envelope on the floor  ☐

The swept volume is where anything can be struck. For a rail-mounted arm it is
roughly **double** what people picture, because the rail drags the whole reach
along 700 mm of travel.

Recompute any time the mount or limits change:

```bash
python scripts/compute_envelope.py
```

Current figures (20,000 sampled configurations, all links tracked, full rail
travel, 100 mm margin):

| | swept | tape line (with margin) |
|---|---|---|
| x | −1163 … 1173 mm | **2.54 m wide** |
| y | −904 … 802 mm | **1.91 m deep** |
| z | 312 … 1921 mm | — |

Three consequences worth stating:

- **Top of reach is 1.92 m — head height.** Ducking under is not a safe strategy.
- **Bottom of reach is 312 mm**, well below the 750 mm benchtop, so the arm can
  swing past the bench edge into the space a person stands in.
- Self- and bench-collision are ignored in the sampling, so the figure is
  conservative — the correct direction for a safety line.

These are arm-relative coordinates from the sim's kinematics. Transfer them using
the arm base's real position. **The operator station goes outside the tape.**

### 1.2 Set the payload and TCP for the fitted effector  ☐

**Do this before anything else that depends on force.** Collision detection and
every F/T reading are derived from the load model: if the payload is wrong, the
readings are *wrong but plausible*, which is more dangerous than having none.

```python
arm.set_tcp_load(weight_kg, [cx_mm, cy_mm, cz_mm])   # centre of gravity, mm
arm.set_tcp_offset([x, y, z, roll, pitch, yaw])      # flange -> tool tip, mm/deg
arm.save_conf()                                      # else lost on reboot
```

Re-do this **after every effector change** — you swap by hand, there is no tool
changer. Then re-identify the load so F/T zeroes against the new tool:

```python
arm.iden_ft_sensor_load_offset()
```

While here, record the flange-to-grip-point distance. `ufactory_vision` ships
`GRIPPER_Z_MM = 150` with a comment saying it must be tuned per setup, and it
differs substantially between your standard parallel gripper and the BIO gripper.

### 1.3 Collision detection  ☐

```python
arm.set_collision_sensitivity(3)        # >= 3, per ufactory_vision's README
arm.set_self_collision_detection(True)
arm.save_conf()
```

Record the value used in the session log. Sensitivity is meaningless if 1.2 was
skipped.

### 1.4 Hard limits — in firmware, not just Python  ☐

Python-level limits die with the Python process. Firmware limits survive a crash,
which is the case that matters.

```python
arm.set_reduced_joint_range([j1min, j1max, ..., j6min, j6max])   # degrees
arm.set_reduced_tcp_boundary([x_max, x_min, y_max, y_min, z_max, z_min])  # mm
arm.set_reduced_max_tcp_speed(speed_mm_s)
arm.set_reduced_max_joint_speed(speed_deg_s)
arm.set_reduced_mode(True)      # boundary only takes effect in reduced mode
arm.save_conf()
```

Then **verify persistence** rather than assuming it:

```python
arm.get_reduced_states()        # read back
# power-cycle the control box, reconnect, read again
```

`RealXArmAPI` also enforces a workspace box in Python. Keep both: the Python
layer gives a clear error before motion starts, the firmware layer catches what
Python never gets to.

If you set reduced joint range here, **re-run `compute_envelope.py` with those
limits** — the tape line from 1.1 assumes full travel and will be too large,
which wastes floor space, or too small if you later relax the limits.

### 1.5 Write the session log location  ☐

One file per session: date, operator, effector fitted, payload/TCP values,
collision sensitivity, speed tier, firmware version (`arm.get_version()`), and
what was run. When something goes wrong later, this is what makes it
reconstructable.

---

## Part 2 — Every session, before power

- ☐ **Bench cleared.** Nothing precious, biological or breakable inside the
  envelope. Not for the first sessions, and not "just this once".
- ☐ **Envelope clear.** No people, chairs, cables or bags inside the tape.
  Operator station outside it.
- ☐ **Cameras mounted and streaming.** Wrist and observer. See Part 4.
- ☐ **You know what the arm is about to do.** Read the plan, or the task, first.

## Part 3 — Every session, at power-on

- ☐ **Test the e-stop.** Enable the arm, press it, confirm the arm drops out of
  ready state. Then reset and re-enable.

  Not "is the e-stop installed" — **press it**. An untested e-stop is a
  decoration, and this is the one item on the list that is never worth skipping.

- ☐ **Clear errors and confirm state.**
  ```python
  arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
  print(arm.get_err_warn_code(), arm.get_state())   # expect ([0, 0], 0)
  ```
  `RealXArmAPI.ready()` does this, and asserts it *after* effector and F/T setup —
  that setup leaves the controller in state 5, so asserting readiness earlier
  produces a first motion that fails with code −2 and no obvious cause.

- ☐ **Confirm the fitted effector matches the plan.**
  ```python
  arm.set_gripper("standard")   # raises if a different effector is fitted
  ```

- ☐ **Confirm F/T is live.** Stream `get_ft_sensor_data()` for a few seconds and
  push the flange by hand. The numbers must respond. A dead sensor reading zero
  looks exactly like no contact.

- ☐ **Verify the speed cap by measuring it.** Command a known distance at the
  session tier, time it, print the implied mm/s.

  Worth actually doing: this repo's `CLAUDE.md` records that the *sim* silently
  ignored the `speed` argument for two weeks after `--speed-tier` shipped. The
  real SDK does honour it — but that is an assumption until you time it.

- ☐ **Session speed tier is `slow` (40 mm/s) or `very_slow` (15 mm/s).**
  Every Stream B session. No exceptions while a human is in the room.

## Part 4 — Recording

- ☐ **Frames on.** `--save-frames`, with the cameras streaming.

  None of the 15 existing recordings contain images, and `run_vr.py` hardcodes
  `enable_frames=False`. Every session recorded without images is training data
  permanently lost, and teleop demonstrations are the only realistic route to the
  50–100 episodes a VLA fine-tune needs.

- ☐ **Session metadata filled in** — task label, effector, speed tier.

## Part 5 — Before an autonomous (agent-driven) run

- ☐ **The controller validator is running.**
  ```bash
  docker start uf-sim && docker exec uf-sim /xarm_scripts/xarm_start.sh 6 6
  ```
  `--mode real` refuses to start without it rather than running an unchecked
  plan. `--no-preflight` is the explicit opt-out and warns.

- ☐ **Dry-run the task first.**
  ```bash
  python scripts/run_task.py "<task>" --mode dryrun   # real backend, no arm
  ```

- ☐ **Hand on the e-stop for the whole run.** Not nearby. On it.

---

## Abort criteria — stop immediately if

- Any unexpected motion, however small.
- A grasp that `verify_grasp()` reports as `held` but visibly is not, or the
  reverse. That means the grasp-detection threshold is wrong and every
  subsequent decision built on it is unreliable.
- F/T readings that do not respond to a hand push.
- A collision-detection trip. Do not clear and retry until you know what it hit;
  a trip that gets cleared reflexively is a trip that teaches nothing.
- Any command reporting success while nothing moved. This is the failure mode
  `hardware/real_arm.py` was rewritten to make impossible — if it reappears,
  something regressed and the arm should not be running.

## Known limitations of the pre-flight itself

- **The controller validator does not simulate the rail.** Rail moves are
  range-checked locally only. A rail move that passes validation has not been
  checked by firmware.
- **The MuJoCo twin validates world geometry; the container validates
  kinematics.** Neither knows about the other's domain, and neither knows about
  a person standing in the envelope. That is what the tape and the e-stop are for.
- **Sim-to-real pose error is not yet measured** (B4). Until it is, treat every
  taught pose as approximate and approach slowly.
