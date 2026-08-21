# B1a — Ground-truth SDK call patterns from UFACTORY's own generator

**What this is.** Reference artifacts, not code to run. `b1a_probe.xml` is a
Blockly project exercising the three open B1 questions; `b1a_probe_generated.py`
is what UFACTORY's own converter emitted for it. The point is to settle method
names by reading the vendor's generator rather than by consulting a table.

**Versions at time of capture (2026-08-22).**

| | Version |
|---|---|
| `xarm-python-sdk` | **1.18.4** |
| Converter | `xarm.tools.blockly_tool.BlocklyTool`, scripted (not the GUI button) |
| Container controller firmware | **v2.4.0** (`6,6, ,XX0000,v2.4.0`) |
| Real control box firmware | **v2.7.1** (`6,6,XI1305,MC1305,v2.7.1`), recorded 2026-08-21 |
| Studio version | **not captured** — the container's controller was down when this ran |

Firmware differs between the validator and the cell. A8.1 is explicit that a
validator running different firmware than the arm is a validator that lies. Any
result here that depends on firmware, rather than on the SDK, needs re-checking
against v2.7.1.

## Resolved method names

| Question | Appendix A of the recommendation says (for 1.18.4) | What the generator emits | Verdict |
|---|---|---|---|
| Move rail | `set_linear_motor_pos(pos, speed, wait, timeout)` | `set_linear_track_pos(pos, speed=, wait=, auto_enable=True)` | **DISAGREE** |
| Home rail | `set_linear_motor_back_origin(wait)` | `set_linear_track_back_origin(wait=True, auto_enable=True)` | **DISAGREE** |
| Enable rail | `set_linear_motor_enable(enable)` | *no separate call* — `auto_enable=True` on the move | **DISAGREE** |
| Standard gripper | `set_gripper_position` | `set_gripper_position(pos, wait=, speed=, auto_enable=True)` | agree |
| BIO gripper | `set_bio_gripper_enable`, `open/close_bio_gripper` | `set_bio_gripper_enable(True)`, then `open_bio_gripper(speed=, wait=)` / `close_bio_gripper(speed=, wait=)` | agree |
| Cartesian move | `set_position(x, y, z, roll, pitch, yaw, speed, wait)` | `set_position(*[...], speed=, mvacc=, radius=, wait=)` | agree |
| Mode/state boilerplate | not stated | `clean_warn()`, `clean_error()`, `motion_enable(True)`, `set_mode(0)`, `set_state(0)`, `sleep(1)` | new |

Per B1a, **the generated code wins.** The disagreements are reported here rather
than quietly folded into the appendix.

## The disagreement, stated precisely

Appendix A frames the rail API as *renamed* between 1.14.x and 1.18.x, implying
the `set_linear_track_*` family is gone in 1.18.4. It is not. Measured on a
1.18.4 instance, **both families are present**:

```
set_linear_track_pos          present      set_linear_motor_pos          present
set_linear_track_enable       present      set_linear_motor_enable       present
set_linear_track_back_origin  present      set_linear_motor_back_origin  present
get_linear_track_pos          present      get_linear_motor_pos          present
```

They resolve through an instance-level alias map (`xarm_api.py:110-134`), which
is why checking `dir(XArmAPI)` on the *class* misses them. So this is not a
question of availability but of which name the vendor actually drives, and the
vendor drives `set_linear_track_pos`.

Two consequences worth carrying forward:

1. `hardware/real_arm.py` probes `set_linear_motor_*` **first** and only falls
   back to `set_linear_track_*`. Since both always resolve, it will always pick
   the `_motor_` family — the one the vendor's generator does not use.
2. `scripts/validate_plan.py` records that `set_linear_motor_pos` returns
   success while the reported position stays at 0. That was attributed to the
   container not simulating the rail. It is *also* consistent with the `_motor_`
   alias behaving differently. **These two explanations have not been
   distinguished**, and cannot be without the rail on real hardware.

## Read the generated code, do not adopt it

`b1a_probe_generated.py` is a flat script with a hardcoded IP, hardcoded speeds,
and its own `_check_code` error wrapper. Extract the call pattern; keep our own
wrapper. Pasting the vendor's error handling into `hardware/` would reintroduce
exactly the silent-success defect B1 removed.

One thing it does *not* do, contrary to expectation: it does not emit
`check_joint_limit=False`. It emits `XArmAPI('<ip>', baud_checkset=False)`. The
`check_joint_limit=False` warning in the instructions applies to code copied out
of Studio's UI into an external IDE, not to converter output — at least for this
SDK version. Nothing in `xarm/tools/` references it.

## Regenerating

```bash
python - <<'PY'
from xarm.tools.blockly_tool import BlocklyTool
BlocklyTool("docs/vendor_reference/b1a_probe.xml").to_python(
    "docs/vendor_reference/b1a_probe_generated.py", arm="<ip>")
PY
```

Re-run after any SDK or firmware update and re-check the table. The converter
itself has had bugs fixed in recent SDK releases, so treat its output as
evidence to verify, not as scripture.

## Not done here

- **Trajectory record/replay is exposed** by the pinned SDK:
  `start_record_trajectory`, `stop_record_trajectory`, `save_record_trajectory`,
  `load_trajectory`, `playback_trajectory`, `get_trajectories`,
  `delete_trajectory`, `get_trajectory_rw_status`, `get_traj_speeding`.
  Reported as B1a asks; **not built on** — Phase 4, not Stream B.
- **Studio's own GUI was not used.** These blocks were authored against the
  converter's handler contract rather than exported from a live Studio session.
  The emitted call patterns are the vendor's; the block XML is ours. Round-
  tripping through the real Studio UI would strengthen this and should be done
  when a controller is up.
