# sim/ik_solver.py
"""
IK solver for xArm6 (rail held fixed).

Prefers pink (https://github.com/stephane-caron/pink), a differential
IK library built on Pinocchio with native MuJoCo support. Falls back
to a damped Jacobian pseudoinverse if pink is unavailable.
"""
import threading
import numpy as np
import mujoco
from typing import Optional

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

try:
    import pink
    from pink import solve_ik
    from pink.tasks import FrameTask
    HAS_PINK = True
except ImportError:
    HAS_PINK = False
    print("[ik_solver] pink not available - using iterative Jacobian fallback. "
          "Install with: pip install pin pin-pink for faster IK.")


class IKSolver:
    """
    Wraps either pink or the iterative Jacobian fallback.

    The interface is the same regardless of backend:
        solver = IKSolver(model, data, lock)
        new_q = solver.solve(target_pos_m, current_qpos_snapshot)
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 lock: threading.Lock):
        self.model = model
        self.data  = data
        self.lock  = lock
        self.joint_ids = [model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = model.joint("rail").id
        self.ee_site   = model.site("end_effector").id
        self.backend = "pink" if HAS_PINK else "jacobian"

        if HAS_PINK:
            self._init_pink()

    def _init_pink(self):
        try:
            self._pink_config = pink.Configuration(self.model, self.data)
            self._pink_task = FrameTask(
                "end_effector",
                position_cost=1.0,
                orientation_cost=0.0,
            )
        except Exception as e:
            print(f"[ik_solver] pink init failed ({e}) - using fallback")
            self.backend = "jacobian"

    def solve(
        self,
        target_pos_m: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed_q: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for the 6 rotational joints to reach target_pos_m.

        If `target_rot` (3x3 rotation matrix) is provided, runs a weighted
        6-DOF solve that tries to honor orientation as a soft constraint
        while position remains the hard convergence criterion. When None,
        falls back to position-only.

        Operates on a snapshot of qpos so the live sim state is NOT
        corrupted - caller must hold the lock when calling, and we
        save+restore qpos around the solve.

        If `seed_q` is provided, IK iterates from that starting pose
        rather than the live qpos. Used by set_position() to retry
        from a clean configuration when the first attempt finds a
        colliding local minimum (e.g. arm bending under the bench).

        Returns: joint angles array (6,) in radians, or None if failed.
        """
        joint_qpos_backup = np.array(
            [self.data.qpos[jid] for jid in self.joint_ids]
        )
        rail_qpos_backup = float(self.data.qpos[self.rail_jid])

        try:
            if seed_q is not None:
                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = float(seed_q[i])
                mujoco.mj_forward(self.model, self.data)
            if self.backend == "pink":
                # Pink path doesn't yet plumb orientation; ignore target_rot
                # for that backend and let the Jacobian fallback handle it
                # if the caller really cares. In practice pink init fails on
                # this MuJoCo build so we always end up in _solve_jacobian.
                result = self._solve_pink(target_pos_m, max_iter, tol)
            else:
                result = self._solve_jacobian(
                    target_pos_m, max_iter, tol, target_rot=target_rot
                )
            return result
        finally:
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = joint_qpos_backup[i]
            self.data.qpos[self.rail_jid] = rail_qpos_backup
            mujoco.mj_forward(self.model, self.data)

    def _solve_pink(self, target_pos_m: np.ndarray,
                    max_iter: int, tol: float) -> Optional[np.ndarray]:
        try:
            self._pink_config.update(self.data.qpos)

            from pink.utils import SE3
            mujoco.mj_forward(self.model, self.data)
            current_rot = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
            target_se3 = SE3(rotation=current_rot, translation=target_pos_m)
            self._pink_task.set_target(target_se3)

            dt = 0.01
            for _ in range(max_iter):
                velocity = solve_ik(self._pink_config, [self._pink_task],
                                    dt, solver="quadprog")
                self._pink_config.integrate_inplace(velocity, dt)

                for i, jid in enumerate(self.joint_ids):
                    self.data.qpos[jid] = self._pink_config.q[jid]
                mujoco.mj_forward(self.model, self.data)

                err = np.linalg.norm(
                    target_pos_m - self.data.site_xpos[self.ee_site]
                )
                if err < tol:
                    return np.array(
                        [self.data.qpos[jid] for jid in self.joint_ids]
                    )
            return None
        except Exception as e:
            print(f"[ik_solver] pink failed: {e} - using fallback once")
            return self._solve_jacobian(target_pos_m, max_iter, tol)

    def _solve_jacobian(self, target_pos_m: np.ndarray,
                        max_iter: int, tol: float,
                        target_rot: Optional[np.ndarray] = None
                        ) -> Optional[np.ndarray]:
        q = np.array([self.data.qpos[jid] for jid in self.joint_ids])
        if target_rot is None:
            return self._solve_jacobian_position(q, target_pos_m,
                                                  max_iter, tol)
        # Try 6-DOF over several biased seeds. Accept the first solution
        # that meets both position and orientation tolerance; otherwise
        # accept best with position OK; otherwise fall back to position-
        # only so the motion still completes (with wrong orientation).
        best_6dof = None
        for seed in self._orientation_seeds(q, target_pos_m, target_rot):
            q_try = self._solve_jacobian_6dof(
                seed, target_pos_m, target_rot,
                max_iter=200, pos_tol=tol,
            )
            if q_try is not None:
                rot_err = self._rot_error_for(q_try, target_rot)
                pos_err = self._pos_error_for(q_try, target_pos_m)
                if pos_err < tol and rot_err < 0.05:
                    return q_try
                if (best_6dof is None or
                        (pos_err + 0.1 * rot_err) <
                        (best_6dof[1] + 0.1 * best_6dof[2])):
                    best_6dof = (q_try, pos_err, rot_err)
        if best_6dof is not None and best_6dof[1] < 0.005:
            return best_6dof[0]
        # Final fallback: position-only IK so the move completes even if
        # the requested orientation is unreachable here. Caller should
        # treat orientation arguments as best-effort.
        return self._solve_jacobian_position(q, target_pos_m, max_iter, tol)

    def _orientation_seeds(self, live_q: np.ndarray,
                            target_pos_m: np.ndarray,
                            target_rot: np.ndarray) -> list:
        """A small set of seeds biased toward common gripper orientations.

        The first seed is the live qpos so callers that already provide
        a sensible seed_q (via solve(seed_q=...)) keep their warm-start.
        The biased seeds target downward grasps (the dominant case) by
        pre-pitching joint 5 and aligning joint 1 with the xy direction
        of the target.
        """
        seeds = [live_q.copy()]
        # Base yaw to face target xy from the rail
        rail_x = float(self.data.qpos[self.rail_jid])
        base_yaw = float(np.arctan2(target_pos_m[1],
                                     target_pos_m[0] - rail_x))
        # Approximate the commanded yaw by reading column +y of target_rot
        cmd_yaw = float(np.arctan2(target_rot[1, 1], target_rot[0, 1]))
        for j2, j3 in ((0.5, -1.0), (1.0, -1.5), (0.3, -0.5)):
            seed = np.array([
                base_yaw, j2, j3, 0.0, np.deg2rad(90.0),
                cmd_yaw - base_yaw,
            ])
            for i, jid in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[jid]
                a = float(seed[i])
                while a > hi + 1e-3: a -= 2.0 * np.pi
                while a < lo - 1e-3: a += 2.0 * np.pi
                seed[i] = float(np.clip(a, lo, hi))
            seeds.append(seed)
        return seeds

    def _pos_error_for(self, q: np.ndarray,
                        target_pos_m: np.ndarray) -> float:
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = q[i]
        mujoco.mj_forward(self.model, self.data)
        return float(np.linalg.norm(
            target_pos_m - self.data.site_xpos[self.ee_site]
        ))

    def _rot_error_for(self, q: np.ndarray,
                        target_rot: np.ndarray) -> float:
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[jid] = q[i]
        mujoco.mj_forward(self.model, self.data)
        rot_cur = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
        return float(np.linalg.norm(
            _rot_mat_to_axis_angle(target_rot @ rot_cur.T)
        ))

    def _solve_jacobian_position(self, q: np.ndarray,
                                  target_pos_m: np.ndarray,
                                  max_iter: int, tol: float
                                  ) -> Optional[np.ndarray]:
        for _ in range(max_iter):
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q[i]
            mujoco.mj_forward(self.model, self.data)
            pos_cur = self.data.site_xpos[self.ee_site].copy()
            err = target_pos_m - pos_cur
            if np.linalg.norm(err) < tol:
                return q.copy()
            nv = self.model.nv
            jacp = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site)
            J = jacp[:, 1:7]
            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
            dq = J_pinv @ err
            dq = np.clip(dq, -np.deg2rad(3.0), np.deg2rad(3.0))
            q = q + dq
            for i, jid in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)
        return None

    def _solve_jacobian_6dof(self, seed_q: np.ndarray,
                              target_pos_m: np.ndarray,
                              target_rot: np.ndarray,
                              max_iter: int, pos_tol: float
                              ) -> Optional[np.ndarray]:
        """Damped least-squares IK over the full 6-DOF task.

        Returns the best-effort q (lowest combined position+orientation
        residual seen during iteration), or None if every iteration
        produced an invalid (NaN) Jacobian step.
        """
        q = seed_q.copy()
        best_q = q.copy()
        best_cost = float('inf')
        rot_tol = 0.02
        lam = 0.01

        for _ in range(max_iter):
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q[i]
            mujoco.mj_forward(self.model, self.data)
            pos_cur = self.data.site_xpos[self.ee_site].copy()
            rot_cur = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
            pos_err = target_pos_m - pos_cur
            rot_err_vec = _rot_mat_to_axis_angle(target_rot @ rot_cur.T)
            cost = float(np.linalg.norm(pos_err)
                          + 0.1 * np.linalg.norm(rot_err_vec))
            if cost < best_cost:
                best_cost = cost
                best_q = q.copy()
            if (np.linalg.norm(pos_err) < pos_tol
                    and np.linalg.norm(rot_err_vec) < rot_tol):
                return q.copy()

            err6 = np.concatenate([pos_err, rot_err_vec])
            nv = self.model.nv
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
            J6 = np.vstack([jacp[:, 1:7], jacr[:, 1:7]])
            J_pinv = J6.T @ np.linalg.inv(J6 @ J6.T + lam * np.eye(6))
            dq = J_pinv @ err6
            if not np.all(np.isfinite(dq)):
                return None
            dq = np.clip(dq, -np.deg2rad(3.0), np.deg2rad(3.0))
            q = q + dq
            for i, jid in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)

        return best_q


def _rot_mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to an axis-angle vector (axis * angle).

    Returns a length-3 vector suitable for use as an angular-velocity error
    term in differential IK. Handles the small-angle and 180-degree edge
    cases.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        diag = np.diag(R)
        i = int(np.argmax(diag))
        col = R[:, i].copy()
        col[i] += 1.0
        n = float(np.linalg.norm(col))
        if n < 1e-9:
            return np.zeros(3)
        axis = col / n
        return axis * theta
    sin_theta = float(np.sin(theta))
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * sin_theta)
    return axis * theta
