# Kinesthetic teaching — operator guide

Record demonstrations by moving the arm by hand: on the real xArm 6 / Lite 6
in manual (teach) mode, or in the digital twin by dragging the end effector
with the mouse. Both write ordinary sessions to `recordings/` that
`scripts/export_lerobot.py` exports unchanged.

All commands run from `xarm6_rail_digital_twin_llm_v5/xarm_lab_twin/` in the
`xarm6sim` conda env.

---

## 1. Real arm

```bash
python scripts/run_kinesthetic_real.py --ip 192.168.1.xxx              # xArm 6, standard gripper
python scripts/run_kinesthetic_real.py --ip 192.168.1.xxx --robot lite6
python scripts/run_kinesthetic_real.py --fake                          # rehearse with no robot
```

Needs `xarm-python-sdk` (in `requirements.txt`) and `pynput`.

### Before you start

1. **Set the payload and TCP for the attached gripper** (xArm Studio, or the
   SDK's `set_tcp_load` / `set_tcp_offset`). Manual mode compensates for
   gravity with these; if they are wrong the arm sags or drifts when you let
   go. The script prints the controller's current load at start-up — check it.
2. **Keep the e-stop within reach.** One person teaches; nobody else stands in
   the arm's reach.

### Teaching

| Key | Action |
|---|---|
| **hold Space** | Manual mode: the arm releases and you can move it. **Release** and it locks in place. |
| Up / Down | Open / close the gripper (same as the twin) |
| Insert | Start / stop a take |
| Delete | Discard the take in progress (or the last saved one) — recoverable from `recordings/trash/` |
| Ctrl-C | Quit. Manual mode is always left first. |

3. Move smoothly. **Hold the link you want to move** rather than levering at
   the wrist, and don't push into joint limits.
4. Keys are read **globally** (whatever window has focus), because holding a
   key needs key-release events, which the viewer cannot report. If `pynput`
   or the display is unavailable, run with `--teach-toggle`: Space *in the
   viewer window* then toggles manual mode on and off instead.
5. **Never command motion while in manual mode.** This script never does. It
   returns the controller to `set_mode(0)` + `set_state(0)` on key release,
   Ctrl-C, `kill` (SIGTERM), any exception, and any controller error.

The twin window shows the arm mirrored live. That is also what is recorded —
the twin's joints are set from the controller every tick. Frames are not
recorded on real takes (the twin's cameras would show the twin, not the
bench).

### Exit tests (operator present, before trusting real takes)

These need the hardware; the automated tests (`python -m teleop_kin.test_real`)
cover everything that can be checked without it.

1. **Release and lock.** Hold Space, move joint 1 about 20° by hand, release.
   The arm holds position and the console prints `locked (mode 0)`.
2. **Mirror fidelity.** Move the TCP by hand to a tape-measured point; compare
   `get_position()` with the twin's EE. Expect joint-space agreement, but note
   **the twin's tool is ~90 mm and the real one 217 mm**, so the two TCPs
   differ by the tool length until the twin's tool matches the fitted one —
   compare flange positions, or correct for the offset, before applying a
   5 mm tolerance.
3. **Crash safety.** In manual mode, `kill -TERM <pid>`; the arm ends in mode 0.
   (Automated against the fake controller in `test_sigterm_in_manual_mode_ends_in_mode_0`.)
4. **Export.** Record a 30-second take with a grasp, then
   `python scripts/export_lerobot.py --dry-run` and
   `python scripts/export_lerobot.py --out /tmp/ds --allow-state-only`. The
   gripper column changes where Down and Up were pressed.

**Lite 6:** the twin is an xArm 6, so the twin's EE pose is not the Lite 6's.
Joint data is recorded faithfully. If plain `set_mode(2)` does not release the
joints, the script retries with `set_mode(2, 1)` automatically.

---

## 2. Digital twin

```bash
python scripts/run_kinesthetic_sim.py
python scripts/run_kinesthetic_sim.py --task-label "red cube to green bin"
```

The arm moves to home, and an **orange ball with an RGB axis tripod** appears
on the end effector.

| Input | Action |
|---|---|
| double-click the ball | select it |
| **Ctrl + right-drag** | move it — the arm follows |
| **Ctrl + left-drag** | rotate it |
| Up / Down | open / close the gripper |
| Left / Right | rail −/+ 25 mm |
| Insert | start / stop a take |
| Delete | discard the take in progress (or the last saved one) |
| Home | arm to home pose |
| End | reset the scene |

- The arm follows at up to 150 mm/s and 90°/s (`--pos-speed`, `--rot-speed`),
  so a fast drag is smoothed into a steady motion.
- **A red ball means the arm can't go there** (out of reach, outside the
  workspace box, or an IK branch flip). The arm holds; let go and the ball
  snaps back onto the arm.
- To grasp: bring the fingertips around the object (tool pointing down), press
  Down. The gripper channel records 1 only if the grasp actually took — a
  close on empty air records 0.
- The ball never appears in recorded frames.
- The front-row cubes (`red_cube_front`, `blue_cube`, y = −250 mm) are outside
  the shared teleop workspace box (y ≥ −200 mm) and turn the ball red.
