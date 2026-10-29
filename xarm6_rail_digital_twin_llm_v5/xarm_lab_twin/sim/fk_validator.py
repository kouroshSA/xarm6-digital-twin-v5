# sim/fk_validator.py
import mujoco
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]


def held_geoms_from_welds(model: mujoco.MjModel, gripper_body: str = "gripper") -> dict:
    """{geom name: weld name} for every geom of every body welded to the gripper."""
    out = {}
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body)
    if gid < 0:
        return out
    held = {int(model.eq_obj2id[e]): model.equality(e).name for e in range(model.neq)
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and model.eq_obj1id[e] == gid}
    for g in range(model.ngeom):
        weld = held.get(int(model.geom_bodyid[g]))
        name = model.geom(g).name
        if weld and name:
            out[name] = weld
    return out


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    achieved_pos: Optional[np.ndarray] = None
    position_error_mm: Optional[float] = None
    has_collision: bool = False
    #: Colliding (geomA, geomB) pairs, so a caller can tell a NEW collision
    #: from one that was already present before it asked for anything.
    contacts: tuple = ()


class FKValidator:
    """
    Validates against the LIVE sim state - shares model + data + lock.
    Snapshots qpos before testing candidate, restores after.
    """

    POSITION_TOLERANCE_MM = 5.0
    ARM_GEOM_NAMES = {
        "base_link", "link1_geom", "link2_geom", "link3_geom",
        "link4_geom", "link5_geom", "link6_geom", "gripper_geom",
        "carriage_geom"
    }
    # Graspable-body geom names paired with the weld constraint that activates
    # when they're held. When validating arm poses, contacts between the gripper
    # and a CURRENTLY HELD body are expected (it travels with the gripper) and
    # must be filtered out -- the FK pass updates only arm joints, not the free
    # body's qpos, so the body appears to overlap geometry that it would actually
    # move out of. Set per instance by held_geoms_from_welds(): every geom of
    # every body the gripper has a weld to. It used to be a hand-written table of
    # 53 geoms that had drifted from the scene -- the translucent cup's 13 geoms
    # and the heater-shaker's 2 LEDs were missing although both bodies had welds.
    HELD_CUBE_GEOMS: dict = {}

    def __init__(self, model: mujoco.MjModel,
                 data: mujoco.MjData,
                 lock: threading.Lock, joint_names=None, rail_joint: str = "rail",
                 ee_site: str = "end_effector", arm_geom_names=None,
                 gripper_body: str = "gripper"):
        # Names are parameters so a room can pass its robot's (prefixed) MJCF
        # names; the defaults are the lab's. The checks themselves are unchanged.
        self.model = model
        self.data  = data
        self.lock  = lock
        self.joint_ids = [model.joint(n).id for n in (joint_names or JOINT_NAMES)]
        self.rail_jid  = model.joint(rail_joint).id
        self.ee_site   = model.site(ee_site).id
        if arm_geom_names is not None:
            self.ARM_GEOM_NAMES = set(arm_geom_names)
        self.HELD_CUBE_GEOMS = held_geoms_from_welds(model, gripper_body)

    def validate(self, joint_angles_rad: np.ndarray,
                 target_pos_m: np.ndarray,
                 rail_pos_m: Optional[float] = None,
                 check_position: bool = True) -> ValidationResult:
        """Joint-limit, reachability and collision check for one configuration.

        `check_position=False` skips the FK-vs-target comparison and checks
        only limits and collision. That is what swept-path validation needs:
        the poses *between* start and target are not supposed to be at the
        target, so comparing them against it would reject every one of them.
        """
        for i, jid in enumerate(self.joint_ids):
            lo, hi = self.model.jnt_range[jid]
            if not (lo <= joint_angles_rad[i] <= hi):
                return ValidationResult(
                    is_valid=False,
                    reason=(f"Joint {i+1} angle "
                            f"{np.rad2deg(joint_angles_rad[i]):.1f}deg "
                            f"outside [{np.rad2deg(lo):.1f}deg, "
                            f"{np.rad2deg(hi):.1f}deg]")
                )

        if rail_pos_m is not None and not (0.0 <= rail_pos_m <= 0.7):
            return ValidationResult(
                is_valid=False,
                reason=f"Rail position {rail_pos_m*1000:.1f}mm outside 0-700mm"
            )

        with self.lock:
            backup_joints = np.array(
                [self.data.qpos[jid] for jid in self.joint_ids]
            )
            backup_rail = float(self.data.qpos[self.rail_jid])

            try:
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = joint_angles_rad[i]
                if rail_pos_m is not None:
                    self.data.qpos[self.rail_jid] = rail_pos_m
                mujoco.mj_forward(self.model, self.data)

                achieved = self.data.site_xpos[self.ee_site].copy()
                error_mm = np.linalg.norm(achieved - target_pos_m) * 1000.0

                if check_position and error_mm > self.POSITION_TOLERANCE_MM:
                    return ValidationResult(
                        is_valid=False,
                        reason=(f"FK position error {error_mm:.1f}mm > "
                                f"tolerance {self.POSITION_TOLERANCE_MM}mm"),
                        achieved_pos=achieved,
                        position_error_mm=error_mm
                    )

                mujoco.mj_collision(self.model, self.data)
                if self.data.ncon > 0:
                    arm_hits = []
                    for i in range(self.data.ncon):
                        c = self.data.contact[i]
                        g1 = self.model.geom(c.geom1).name
                        g2 = self.model.geom(c.geom2).name
                        if self.ARM_GEOM_NAMES & {g1, g2}:
                            other_set = {g1, g2} - self.ARM_GEOM_NAMES
                            if not other_set:
                                continue
                            other = next(iter(other_set))
                            # Ignore gripper<->held-cube contacts (see HELD_CUBE_GEOMS)
                            weld_name = self.HELD_CUBE_GEOMS.get(other)
                            if weld_name is not None:
                                try:
                                    eqid = self.model.equality(weld_name).id
                                    if self.data.eq_active[eqid]:
                                        continue
                                except KeyError:
                                    pass
                            arm_hits.append((g1, g2))
                    if arm_hits:
                        return ValidationResult(
                            is_valid=False,
                            reason=(f"Collision: {len(arm_hits)} contacts "
                                    f"({arm_hits[:3]})"),
                            achieved_pos=achieved,
                            position_error_mm=error_mm,
                            has_collision=True,
                            contacts=tuple(tuple(sorted(h)) for h in arm_hits),
                        )

                return ValidationResult(
                    is_valid=True, reason="OK",
                    achieved_pos=achieved,
                    position_error_mm=error_mm
                )

            finally:
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = backup_joints[i]
                self.data.qpos[self.rail_jid] = backup_rail
                mujoco.mj_forward(self.model, self.data)
