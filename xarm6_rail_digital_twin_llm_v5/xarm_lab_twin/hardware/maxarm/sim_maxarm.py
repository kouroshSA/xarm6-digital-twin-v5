"""The MaxArm in MuJoCo: `SimMaxArmAPI`, the twin of `RealMaxArmAPI`.

    arm = SimXArmAPI(room="maxarm_petri", render=False)   # the room's physics
    ma = SimMaxArmAPI(arm, instance="arm1")
    ma.go_home(); ma.set_position(0, -200, 70); ma.close_gripper()

It drives the room's `maxarm` robot instance on the model, data and lock that
`SimXArmAPI` already owns (the xArm6 stays controllable and idle), with the
real backend's surface: `set_position(x, y, z, yaw=)` in the ARM FRAME (mm,
left-handed, +x = the arm's right, reach along -y), `get_position`,
`go_home`, `close_gripper` / `open_gripper` (the suction cup), `set_nozzle_angle`.

What is the same as the bench, on purpose:

* every target goes through `kinematics.solve` (the floor and the vendor's
  silent clamps refuse here exactly as they do on the device);
* the arm STOPS where the bench would: the commanded point is perturbed by
  `repeatability.Repeatability` (measured gravity sag + scatter) and the
  pulses are integers, so a rehearsal meets the same misses;
* the read-back is the servos' pulses turned back into a point with
  `kinematics`, the move takes `distance / speed` of SIM time, and arrival is
  judged with `real_maxarm.ARRIVAL_TOL_MM` and one re-send, like `_await_arrival`.

What is different, and says so: the suction is a weld (like the xArm's
magnetic gripper), taken when the cup lip is within `SEAL_GAP_MM` of a
grippable object's top under the whole lip (five rays); the real cup has to press 4 mm to seal and the twin's
rigid cup does not press at all. `verify_grasp` answers from the weld, which
the real arm cannot do (no vacuum sensor): a plan that depends on it will not
transfer.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import mujoco
import numpy as np

from . import kinematics as kin
from .real_maxarm import (ARRIVAL_TOL_MM, DEFAULT_SPEED_MM_S, MAX_DURATION_MS, MIN_DURATION_MS,
                          RETRY_DURATION_MS, SETTLE_S, MaxArmError)
from .repeatability import Repeatability, exact

#: Twin joint order: base yaw, upper arm, forearm (relative), wrist level, nozzle.
JOINTS = ("j1", "j2", "j3", "j4", "j5")
ACTUATORS = ("act1", "act2", "act3", "act4", "act5")

#: The cup takes an object when its lip is this close above the object's top
#: (mm) and inside the top's footprint.
SEAL_GAP_MM = 3.0
#: ...and the surface under the cup's centre and rim must be flat to this (mm):
#: a cup half over the lid's edge, or on a dish base's bare 1 mm rim, does not seal.
SEAL_FLATNESS_MM = 1.0

#: The seal rays start this far above the cup lip (inside the cup body), m.
RAY_START_ABOVE_LIP_M = 0.004
#: A cup pressed deeper than this into an object is not sealing on it, mm
#: (the real bellows give 4-5 mm; the rigid twin cup is stopped by contact).
SEAL_PRESS_MAX_MM = 2.5
#: Control-loop slice for a paced move, s (sim time).
SLICE_S = 0.02


def vendor_to_twin(joints_deg, nozzle_deg: float = 0.0) -> tuple[float, ...]:
    """Vendor joint angles (kinematics.inverse's) -> twin joint positions, rad.
    See robots/maxarm/build_model.py for the derivation."""
    j1, j2, j3 = (float(v) for v in joints_deg)
    yaw = math.radians(j1 - 120.0)
    phi2 = math.radians(180.0 - j2)          # upper arm elevation above horizontal
    phi3 = math.radians(-j3)                 # forearm elevation (absolute: the linkage)
    q2 = phi2 - math.pi / 2                  # 0 = upper arm vertical (home)
    q3 = phi3 - phi2 + math.pi / 2           # 0 = forearm level (home)
    return (yaw, q2, q3, -phi3, math.radians(nozzle_deg))


def twin_to_vendor(q) -> tuple[float, float, float]:
    """Twin joint positions (rad; the first three used) -> vendor angles, deg."""
    yaw, q2, q3 = (float(v) for v in q[:3])
    phi2 = q2 + math.pi / 2
    phi3 = q3 + phi2 - math.pi / 2
    return (math.degrees(yaw) + 120.0, 180.0 - math.degrees(phi2), -math.degrees(phi3))


#: The twin's home posture, degrees, from the vendor's home pulses (robot.json refs it).
HOME_JOINTS_DEG = tuple(round(math.degrees(v), 6)
                        for v in vendor_to_twin(kin.pulse_to_deg(kin.HOME_PULSES)))


def arm_to_local(xyz_mm) -> np.ndarray:
    """Arm frame (mm, left-handed) -> robot body frame (m, right-handed)."""
    x, y, z = (float(v) for v in xyz_mm)
    return np.array([-x, y, z]) / 1000.0


def local_to_arm(p_m) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in p_m)
    return (-x * 1000.0, y * 1000.0, z * 1000.0)


class SimMaxArmAPI:
    """See the module docstring. Thread-safe like the real one (its own RLock
    on top of the sim's)."""

    effector = "suction"

    def __init__(self, sim, instance: str = "arm1", floor_z_mm: float = kin.DEFAULT_FLOOR_Z_MM,
                 repeat: Optional[Repeatability] = None, seed: int = 0):
        self.sim = sim
        self.model, self.data, self.lock = sim.model, sim.data, sim.lock
        self.inst = sim.room.robots[instance]
        self.instance = instance
        self.floor_z_mm = floor_z_mm
        self.repeat = repeat if repeat is not None else Repeatability(seed=seed)
        self._lock = threading.RLock()
        self._nozzle_cmd_deg: Optional[float] = None
        self._suction_cmd = False
        self._vent_open = False
        self._held: Optional[str] = None

        m = self.model
        self.jids = [m.joint(self.inst.mjcf(j)).id for j in JOINTS]
        self.qadr = [int(m.jnt_qposadr[j]) for j in self.jids]
        self.aids = [m.actuator(self.inst.mjcf(a)).id for a in ACTUATORS]
        self.base_bid = m.body(self.inst.mjcf("maxarm_base")).id
        self.cup_bid = m.body(self.inst.mjcf("cup")).id
        self.lip_sid = m.site(self.inst.mjcf("cup_lip")).id
        self.nozzle_bid = m.body(self.inst.mjcf("nozzle")).id
        # Grippable objects this instance has a weld to (the builder made one
        # per object), by logical name.
        self.welds = {name: m.equality(o.welds[instance]).id
                      for name, o in sim.room.objects.items() if instance in o.welds}
        self.body_ids = {name: m.body(o.body).id for name, o in sim.room.objects.items()}
        with self.lock:
            mujoco.mj_forward(m, self.data)
            self.base_pos = self.data.xpos[self.base_bid].copy()
            self.base_mat = self.data.xmat[self.base_bid].reshape(3, 3).copy()
        self._target_q = np.array([self.data.qpos[a] for a in self.qadr])

    # -- frames ---------------------------------------------------------------

    def arm_to_world(self, xyz_mm) -> np.ndarray:
        """Arm-frame point (mm) -> world (m)."""
        return self.base_pos + self.base_mat @ arm_to_local(xyz_mm)

    def world_to_arm(self, p_world_m) -> tuple[float, float, float]:
        return local_to_arm(self.base_mat.T @ (np.asarray(p_world_m, float) - self.base_pos))

    # -- state ----------------------------------------------------------------

    def _q(self) -> np.ndarray:
        with self.lock:
            return np.array([self.data.qpos[a] for a in self.qadr])

    def _read_pulses(self) -> list[int]:
        """The servos' positions as integer pulses: the twin of the bus read."""
        return [int(round(p)) for p in kin.deg_to_pulse(twin_to_vendor(self._q()))]

    def get_position(self):
        """(0, [x, y, z, 180, 0, yaw]) from the read-back pulses, like the real one."""
        x, y, z = kin.pulses_to_xyz(self._read_pulses())
        return 0, [x, y, z, 180.0, 0.0, self._nozzle_cmd_deg]

    def get_servo_angle(self):
        j = kin.pulse_to_deg(self._read_pulses())
        return 0, [*j, self._nozzle_cmd_deg]

    def wrist_world(self) -> np.ndarray:
        """Ground truth: the wrist point (nozzle axis at wrist height), world m."""
        with self.lock:
            return self.data.xpos[self.nozzle_bid].copy()

    def lip_world(self) -> np.ndarray:
        with self.lock:
            return self.data.site_xpos[self.lip_sid].copy()

    # -- motion ---------------------------------------------------------------

    def _sim_time(self) -> float:
        with self.lock:
            return float(self.data.time)

    def _wait_sim(self, seconds: float) -> None:
        t_end = self._sim_time() + seconds
        while self._sim_time() < t_end:
            time.sleep(0.005)

    def _set_ctrl(self, q) -> None:
        with self.lock:
            for a, v in zip(self.aids, q):
                self.data.ctrl[a] = v

    def _drive(self, q_target, duration_s: float) -> None:
        """Interpolate the joint targets over `duration_s` of SIM time."""
        q0 = self._target_q.copy()
        q1 = np.asarray(q_target, float)
        t0 = self._sim_time()
        while True:
            t = self._sim_time()
            s = min((t - t0) / max(duration_s, 1e-6), 1.0)
            self._set_ctrl(q0 + (q1 - q0) * s)
            if s >= 1.0:
                break
            time.sleep(SLICE_S / 2)
        self._target_q = q1

    def _duration_ms(self, target_xyz, speed_mm_s: float) -> int:
        start = kin.pulses_to_xyz(self._read_pulses())
        ms = 1000.0 * math.dist(start, target_xyz) / max(float(speed_mm_s), 1e-3)
        return int(min(max(ms, MIN_DURATION_MS), MAX_DURATION_MS))

    def _settled_q(self, sol: kin.Solution) -> np.ndarray:
        """Where the servos will stop for this command: the pulses of the
        perturbed point, as integers. A perturbed point the solver refuses
        (at the very edge of the envelope) settles on the exact one."""
        actual = self.repeat.settle(sol.xyz_mm, loaded=self._held is not None)
        try:
            pulses = kin.solve(actual, floor_z_mm=None).pulses
        except kin.Unreachable:
            pulses = sol.pulses
        pulses = [int(round(p)) for p in pulses]
        q = list(vendor_to_twin(kin.pulse_to_deg(pulses)))
        q[4] = self._target_q[4]
        return np.array(q)

    def _await_arrival(self, target_xyz, duration_ms: int, what: str, sol=None) -> None:
        self._wait_sim(duration_ms / 1000.0 + SETTLE_S)
        reached = kin.pulses_to_xyz(self._read_pulses())
        err = math.dist(reached, target_xyz)
        if err > ARRIVAL_TOL_MM and sol is not None:
            print(f"[SimMaxArm] {what}: {err:.1f} mm short of "
                  f"({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f}); re-sending once")
            self._drive(self._settled_q(sol), RETRY_DURATION_MS / 1000.0)
            self._wait_sim(RETRY_DURATION_MS / 1000.0 + SETTLE_S)
            reached = kin.pulses_to_xyz(self._read_pulses())
            err = math.dist(reached, target_xyz)
        if err > ARRIVAL_TOL_MM:
            raise MaxArmError(
                f"{what}: arm did not arrive. Target "
                f"({target_xyz[0]:.1f}, {target_xyz[1]:.1f}, {target_xyz[2]:.1f}), "
                f"read back ({reached[0]:.1f}, {reached[1]:.1f}, {reached[2]:.1f}), "
                f"{err:.1f} mm off (tolerance {ARRIVAL_TOL_MM}).")

    def set_position(self, x, y, z, roll=None, pitch=None, yaw=None,
                     speed=DEFAULT_SPEED_MM_S, wait=True, **kwargs) -> int:
        """Move the wrist to (x, y, z) in the arm frame, mm. Same contract as
        `RealMaxArmAPI.set_position`; returns 0 or raises MaxArmError."""
        if roll is not None and min(abs(float(roll)), abs(abs(float(roll)) - 180.0)) > 1e-6:
            raise MaxArmError(f"set_position: roll {roll} unsupported; the MaxArm "
                              f"tool always points straight down")
        if pitch is not None and abs(float(pitch)) > 1e-6:
            raise MaxArmError(f"set_position: pitch {pitch} unsupported; the MaxArm "
                              f"tool always points straight down")
        try:
            sol = kin.solve((x, y, z), self.floor_z_mm)
        except kin.Unreachable as exc:
            raise MaxArmError(f"set_position refused: {exc}") from None
        with self._lock:
            duration = self._duration_ms(sol.xyz_mm, speed)
            if yaw is not None:
                self.set_nozzle_angle(yaw, duration_ms=duration, _drive=False)
            self._drive(self._settled_q(sol), duration / 1000.0)
            if wait:
                self._await_arrival(sol.xyz_mm, duration, "set_position", sol)
        return 0

    def set_servo_angle(self, angle, speed=30.0, wait=True, **kwargs) -> int:
        a = [float(v) for v in angle]
        if len(a) not in (3, 4):
            raise MaxArmError(f"set_servo_angle: need 3 or 4 angles, got {len(a)}")
        joints = tuple(a[:3])
        xyz = kin.forward(joints)
        problems = kin.check_pulses(kin.deg_to_pulse(joints))
        if xyz[2] < self.floor_z_mm:
            problems.append(f"wrist z {xyz[2]:.1f} below the floor {self.floor_z_mm:.1f}")
        if xyz[2] > kin.Z_MAX_MM:
            problems.append(f"wrist z {xyz[2]:.1f} > {kin.Z_MAX_MM:.0f}")
        if problems:
            raise MaxArmError("set_servo_angle refused: " + "; ".join(problems))
        with self._lock:
            cur = kin.pulse_to_deg(self._read_pulses())
            biggest = max(abs(c - t) for c, t in zip(cur, joints))
            duration = int(min(max(1000.0 * biggest / max(float(speed), 1e-3),
                                   MIN_DURATION_MS), MAX_DURATION_MS))
            if len(a) == 4:
                self.set_nozzle_angle(a[3], duration_ms=duration, _drive=False)
            sol = kin.Solution(xyz, joints, kin.deg_to_pulse(joints))
            self._drive(self._settled_q(sol), duration / 1000.0)
            if wait:
                self._await_arrival(xyz, duration, "set_servo_angle", sol)
        return 0

    def set_nozzle_angle(self, angle_deg: float, duration_ms: int = 500, _drive: bool = True) -> int:
        lo, hi = kin.NOZZLE_ANGLE_RANGE_DEG
        if not (lo <= float(angle_deg) <= hi):
            raise MaxArmError(f"nozzle angle {angle_deg} outside {lo}..{hi}")
        self._nozzle_cmd_deg = float(angle_deg)
        q = self._target_q.copy()
        q[4] = math.radians(float(angle_deg))
        if _drive:
            self._drive(q, duration_ms / 1000.0)
        else:
            self._target_q[4] = q[4]        # carried by the move being sent
        return 0

    def go_home(self, wait: bool = True, **kwargs) -> int:
        return self.set_position(*kin.HOME_XYZ_MM, speed=kwargs.get("speed", 40.0), wait=wait)

    def snap_home(self) -> None:
        """Teleport to the home posture (the state a connect finds the arm in),
        exact, no pacing: for the start of a rehearsal."""
        q = np.array(vendor_to_twin(kin.pulse_to_deg(kin.HOME_PULSES)))
        with self.lock:
            for a, v in zip(self.qadr, q):
                self.data.qpos[a] = v
            for j in self.jids:
                self.data.qvel[int(self.model.jnt_dofadr[j])] = 0.0
            for a, v in zip(self.aids, q):
                self.data.ctrl[a] = v
            mujoco.mj_forward(self.model, self.data)
        self._target_q = q

    # -- suction --------------------------------------------------------------

    def _under_cup(self) -> list[tuple[Optional[str], float]]:
        """What lies under the cup: for the lip's centre and four points on the
        cup's rim (8 mm out), the object body hit by a ray cast straight down
        and the gap to it in mm (None, inf where nothing grippable is there).
        A real cup seals only on a surface that is flat under the whole lip,
        which is what five rays check and an AABB cannot (the bare rim of a
        dish base is 1 mm wide: the centre ray falls 13 mm to its floor)."""
        m, d = self.model, self.data
        body_of = {bid: name for name, bid in self.body_ids.items() if name in self.welds}
        out = []
        with self.lock:
            lip = d.site_xpos[self.lip_sid].copy()
            R = d.site_xmat[self.lip_sid].reshape(3, 3)
            geomid = np.zeros(1, dtype=np.int32)
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                # Cast from RAY_START_ABOVE_LIP_M above the lip, inside the cup
                # body (excluded), so a cup that settled a millimetre INTO the
                # lid still sees the lid: a ray that starts inside the 1 mm
                # top plate passes through it and reports the base's floor
                # (seed 4 on the twin, 2026-11-10). Negative gaps = pressing.
                start = lip + R @ np.array([dx * 0.008, dy * 0.008, RAY_START_ABOVE_LIP_M])
                dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), None, 1,
                                     self.cup_bid, geomid)
                if dist < 0 or geomid[0] < 0:
                    out.append((None, math.inf))
                    continue
                name = body_of.get(int(m.geom_bodyid[geomid[0]]))
                out.append((name, (dist - RAY_START_ABOVE_LIP_M) * 1000.0))
        return out

    def _candidates(self) -> list[tuple[str, float, float]]:
        """Objects the cup could take now: (name, gap below the lip mm, spread
        of the five gaps mm). Only objects under ALL five rays qualify."""
        hits = self._under_cup()
        names = {n for n, _ in hits}
        if len(names) != 1 or None in names:
            return []
        gaps = [g for _, g in hits]
        return [(hits[0][0], max(gaps), max(gaps) - min(gaps))]

    def close_gripper(self, wait: bool = True) -> int:
        """Pump on. Takes the object under the cup if the lip is within
        SEAL_GAP_MM of a top that is flat under the whole lip; otherwise the
        pump runs on nothing, exactly as the bench would (no error: the real
        arm cannot tell either)."""
        self._suction_cmd = True
        if self._vent_open:
            print("[SimMaxArm] suction on: the vent valve is held open, the pump runs on air")
            return 0
        if self._held is not None:
            return 0
        best = None
        for name, gap, spread in self._candidates():
            if -SEAL_PRESS_MAX_MM <= gap <= SEAL_GAP_MM and spread <= SEAL_FLATNESS_MM:
                best = (name, gap, spread)
        if best is None:
            hits = self._under_cup()
            seen = ", ".join(f"{n or 'nothing grippable'} at {g:.1f} mm" if math.isfinite(g)
                             else "nothing" for n, g in hits)
            print(f"[SimMaxArm] suction on: no seal (under the cup's centre and rim: {seen})")
            return 0
        name = best[0]
        with self.lock:
            m, d = self.model, self.data
            bid = self.body_ids[name]
            gp, gq = d.xpos[self.cup_bid].copy(), d.xquat[self.cup_bid].copy()
            cp, cq = d.xpos[bid].copy(), d.xquat[bid].copy()
            gq_inv = np.zeros(4); mujoco.mju_negQuat(gq_inv, gq)
            rel_pos = np.zeros(3); mujoco.mju_rotVecQuat(rel_pos, cp - gp, gq_inv)
            rel_quat = np.zeros(4); mujoco.mju_mulQuat(rel_quat, gq_inv, cq)
            eqid = self.welds[name]
            m.eq_data[eqid, 0:3] = 0.0
            m.eq_data[eqid, 3:6] = rel_pos
            m.eq_data[eqid, 6:10] = rel_quat
            m.eq_data[eqid, 10] = 1.0
            d.eq_active[eqid] = 1
        self._held = name
        print(f"[SimMaxArm] suction on: holding {name} (lip {best[1]:.1f} mm above its top, "
              f"surface flat within {best[2]:.1f} mm under the cup)")
        return 0

    def open_gripper(self, wait: bool = True) -> int:
        """Pump off, vent: the weld lets go. Takes the vendor's 1.1 s like the real one."""
        self._suction_cmd = False
        if self._held is not None:
            with self.lock:
                self.data.eq_active[self.welds[self._held]] = 0
            print(f"[SimMaxArm] suction off: released {self._held}")
            self._held = None
        if wait:
            self._wait_sim(1.1)
        return 0

    def vent(self, open: bool = True) -> int:
        """Valve held open (pump off, the weld lets go at once) or closed; as the real one."""
        self._vent_open = bool(open)
        return self.open_gripper(wait=False) if open else 0

    def set_gripper(self, mode: str) -> int:
        if mode != self.effector:
            raise MaxArmError(f"plan expects the {mode!r} effector; the MaxArm has "
                              f"a {self.effector!r} cup and no tool changer")
        return 0

    @property
    def held(self) -> Optional[str]:
        """What the weld holds. The REAL arm cannot answer this."""
        return self._held

    def verify_grasp(self) -> tuple[str, str]:
        if self._held:
            return "held", f"the twin's weld holds {self._held} (the real arm has no sensor for this)"
        return "empty", "nothing welded to the cup"

    def disconnect(self, release_suction: bool = False) -> None:
        if release_suction:
            self.open_gripper(wait=False)
