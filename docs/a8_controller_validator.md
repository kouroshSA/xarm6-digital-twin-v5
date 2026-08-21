# Controller-in-the-loop plan validation (A8)

Validate a planned command sequence against **real xArm controller firmware**,
with no robot on the network and nothing able to move.

UFACTORY publish a Docker image that runs the actual controller plus Studio with
no hardware attached. Point `XArmAPI` at `127.0.0.1` and the same code path that
drives the bench runs against a simulated controller.

## Running it

```bash
docker pull danielwang123321/uf-ubuntu-docker

docker run -dit --name uf-sim \
  -p 18333:18333 -p 502:502 -p 503:503 -p 504:504 \
  -p 30000:30000 -p 30001:30001 -p 30002:30002 -p 30003:30003 \
  danielwang123321/uf-ubuntu-docker /bin/bash

docker exec uf-sim /xarm_scripts/xarm_start.sh 6 6    # axis=6, type=6 -> xArm 6
```

The container needs `-dit`: `xarm_start.sh` backgrounds the controller and Studio
under `screen` and then execs `/bin/bash`, so without a TTY the container exits
immediately with status 0 and nothing is running.

Give it ~20 s, then:

```bash
python scripts/validate_plan.py --self-test     # prove the validator works
python scripts/validate_plan.py plan.json       # validate a real plan
```

Studio's web UI is on <http://127.0.0.1:18333>. Dismiss the "unable to get robot
SN" prompt if it appears — see limitation 2 below.

**Which machine.** x86-64 only. Verified on the laptop (Docker 29.1.3). Do not
assume it runs on the ARM64 DGX Sparks.

## What it does and does not check

The container simulates the **controller**, not the world. There is no bench, no
OT-2, no cube, no camera. It will tell you a pose is unreachable, a joint is out
of range, or the arm is in the wrong mode. It will **not** tell you the arm would
hit the bench.

That is the clean division with the MuJoCo twin:

| | validates |
|---|---|
| Docker controller | kinematics, joint limits, singularities, mode/state machine, error codes |
| MuJoCo twin | world geometry, collisions, reachability among actual objects |

Neither replaces the other. A plan should pass both.

Measured behaviour (firmware `v2.4.0`, reported as `6,6, ,XX0000,v2.4.0`):

| command | result |
|---|---|
| `set_position(300, 0, 300)` | `rc=0`, no error — accepted |
| `set_position(2000, 0, 400)` | **`rc=-9`, controller error 21** — rejected |
| `set_position(0, 0, -500)` | `rc=1` — rejected |

## Two measured limitations

### 1. The rail is not simulated

The container has no linear motor. Every rail call returns success and the
position never changes:

```
set_linear_track_enable(True) -> (0, [])
set_linear_motor_pos(200, wait=True) -> (0, [])
get_linear_motor_pos() -> (0, 0)     # still zero
get_linear_motor_is_enabled() -> (0, 0)
```

So a rail move "passing" here means nothing. `validate_plan.py` therefore
**range-checks rail targets locally and does not send them to the controller** —
sending them would manufacture a meaningless pass, which is worse than not
checking at all.

### 2. The serial number is blank, which breaks SDK joint-limit checking

`arm.sn` is `' '`. The SDK's joint-limit path does
`int(self.sn[2:6])` (`xarm/x3/xarm.py:71`), and `int('')` raises `ValueError`, so
`set_servo_angle` crashes before reaching the controller.

UFACTORY's documented workaround is `check_joint_limit=False`. **We do not use
it**: it disables precisely the check this tool exists to perform. Instead joint
targets are validated locally against the scene's `jnt_range` — which came from
the xArm6 URDF, so it is the same source of truth the simulator uses — and the
motion is still offered to the controller, with the `ValueError` reported rather
than swallowed.

Joint limits in use (degrees), read from the scene at runtime:

```
joint1  -178.2 ..  178.2      joint4  -178.2 .. 178.2
joint2  -118.0 ..  120.0      joint5   -97.0 .. 178.2
joint3  -178.2 ..   11.0      joint6  -178.2 .. 178.2
rail       0.0 ..  700.0 mm
```

## A correction to the A5 / B1 analysis

A5 and B1 claimed that because UFACTORY renamed `set_linear_track_pos` to
`set_linear_motor_pos`, an unpinned install landing on SDK 1.18.4 would raise
`AttributeError` in `hardware/real_arm.py`, get swallowed, and return success
while the rail never moved.

**That specific scenario was wrong, and this container is what surfaced it.**
SDK 1.18.4 builds a backwards-compatibility alias map per instance in
`XArmAPI.__init__` (`xarm/wrapper/xarm_api.py:120-134`) covering exactly those
renames:

```python
'set_linear_track_pos':  self.set_linear_motor_pos,
'set_linear_track_enable': self.set_linear_motor_enable,
'ft_sensor_enable':      self.set_ft_sensor_enable,
'ft_sensor_set_zero':    self.set_ft_sensor_zero,
'set_impedance_mbk':     self.set_ft_sensor_admittance_parameters,
'config_force_control':  self.set_ft_sensor_force_parameters,
```

Verified against firmware: `set_linear_track_pos(100, speed=50)` returns
`(0, [])` on 1.18.4. The old names work.

The earlier check was done with `dir(XArmAPI)` and a `grep` for `def` in the
source — both of which see only the **class**, while the aliases are installed on
the **instance**. `hasattr(XArmAPI, 'set_linear_track_pos')` is `False`;
`hasattr(arm, 'set_linear_track_pos')` is `True`. A made-up name is still
`False`, so this is a curated alias table, not a blanket proxy.

What still stands from B1:

- **The `except AttributeError: ... return 0` swallow was a real defect** and
  removing it was right. A command that does not execute must never report
  success. That is independent of whether this particular exception ever fired.
- The Lite6 gripper calls were genuinely wrong for an xArm 6.
- Pinning the SDK is still correct — versions differ in real ways (`set_bio_gripper_force`
  and `get_gripper_status` exist only in newer builds).
- The capability probe is harmless and still fails loudly on a build with
  neither naming, but its stated premise ("only one spelling resolves") was
  overstated: on 1.18.4 both do.

The severity claim in the A5 and B1 commit messages should be read with this
correction alongside.

## Wiring it into the real-robot path

Two entry points now use it.

### `--mode dryrun`

```bash
python scripts/run_task.py "put the front red cube in the green bin" --mode dryrun
```

Runs the **real** backend (`hardware/real_arm.py`) against the container instead
of the bench. Nothing can move. This is the only way to exercise that file short
of powering the arm — sim mode never touches it — so it is where B1's changes get
tested before B3's first powered motion.

### Pre-action gate on `--mode real`

```bash
python scripts/run_task.py "<task>" --mode real --ip <arm-ip>
```

Before any command reaches the arm, the whole plan is replayed against the
container and execution aborts if the controller rejects any of it. Whole plan,
not per-command: checking as you go would let the arm execute the prefix before
reaching the bad command.

If the container is unreachable the run **aborts** rather than proceeding
unchecked — a gate that silently passes when its backend is down is worse than no
gate, because the operator believes a check happened. `--no-preflight` is the
explicit opt-out and says so in the log.

Guarded by `agent/test_plan_gate.py` (in the sweep as `agent.plan_gate`), which
pins the property that matters: **a rejected plan dispatches nothing.**

## Where Studio itself fits

Two different pre-flights, worth not confusing:

- **The containerised controller** (this document) is the *software* pre-flight.
  It answers "is this plan executable?" and needs no hardware.
- **Studio connected to the real arm** is the *operator* pre-flight, and belongs
  in B0's checklist. It is where you clear errors, confirm payload and TCP, set
  collision sensitivity, check the joint-range and safety-boundary settings, and
  jog the arm by hand before handing control to the agent. The SDK can set most
  of that, but Studio is the honest place to *verify* it, and firmware-level
  limits set there survive a crashed Python process.

Neither substitutes for the other, and neither substitutes for a human on the
e-stop.
