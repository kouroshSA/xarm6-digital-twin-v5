# sim/fk_validator.py
import mujoco
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    achieved_pos: Optional[np.ndarray] = None
    position_error_mm: Optional[float] = None
    has_collision: bool = False


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
    # move out of. Each tube has two geoms (body + cap), so both must be listed.
    HELD_CUBE_GEOMS = {
        "red_cube_geom":   "grip_red_cube",
        "green_cube_geom": "grip_green_cube",
        "blue_cube_geom":  "grip_blue_cube",
        "tube_L1_body":    "grip_tube_L1",
        "tube_L1_cap":     "grip_tube_L1",
        "tube_L2_body":    "grip_tube_L2",
        "tube_L2_cap":     "grip_tube_L2",
        "tube_L3_body":    "grip_tube_L3",
        "tube_L3_cap":     "grip_tube_L3",
        "tube_R1_body":    "grip_tube_R1",
        "tube_R1_cap":     "grip_tube_R1",
        "tube_R2_body":    "grip_tube_R2",
        "tube_R2_cap":     "grip_tube_R2",
        "tube_R3_body":    "grip_tube_R3",
        "tube_R3_cap":     "grip_tube_R3",
        # Bins (each has 5 geoms: floor + 4 walls). All need to be filtered
        # when their weld is active so the validator doesn't reject moves
        # while the bin is being pushed.
        "red_bin_floor":    "grip_red_bin",
        "red_bin_w_front":  "grip_red_bin",
        "red_bin_w_back":   "grip_red_bin",
        "red_bin_w_left":   "grip_red_bin",
        "red_bin_w_right":  "grip_red_bin",
        "green_bin_floor":    "grip_green_bin",
        "green_bin_w_front":  "grip_green_bin",
        "green_bin_w_back":   "grip_green_bin",
        "green_bin_w_left":   "grip_green_bin",
        "green_bin_w_right":  "grip_green_bin",
        "blue_bin_floor":    "grip_blue_bin",
        "blue_bin_w_front":  "grip_blue_bin",
        "blue_bin_w_back":   "grip_blue_bin",
        "blue_bin_w_left":   "grip_blue_bin",
        "blue_bin_w_right":  "grip_blue_bin",
        # Tube racks (each rack has 9 geoms: 1 base + 4 outer walls + 3
        # vertical separators + 1 horizontal separator)
        "left_rack_base":    "grip_left_rack",
        "L_outer_front":     "grip_left_rack",
        "L_outer_back":      "grip_left_rack",
        "L_outer_left":      "grip_left_rack",
        "L_outer_right":     "grip_left_rack",
        "L_sep_v1":          "grip_left_rack",
        "L_sep_v2":          "grip_left_rack",
        "L_sep_v3":          "grip_left_rack",
        "L_sep_h1":          "grip_left_rack",
        "right_rack_base":   "grip_right_rack",
        "R_outer_front":     "grip_right_rack",
        "R_outer_back":      "grip_right_rack",
        "R_outer_left":      "grip_right_rack",
        "R_outer_right":     "grip_right_rack",
        "R_sep_v1":          "grip_right_rack",
        "R_sep_v2":          "grip_right_rack",
        "R_sep_v3":          "grip_right_rack",
        "R_sep_h1":          "grip_right_rack",
    }

    def __init__(self, model: mujoco.MjModel,
                 data: mujoco.MjData,
                 lock: threading.Lock):
        self.model = model
        self.data  = data
        self.lock  = lock
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.ee_site   = model.site("end_effector").id

    def validate(self, joint_angles_rad: np.ndarray,
                 target_pos_m: np.ndarray,
                 rail_pos_m: Optional[float] = None) -> ValidationResult:
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

                if error_mm > self.POSITION_TOLERANCE_MM:
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
                            has_collision=True
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
