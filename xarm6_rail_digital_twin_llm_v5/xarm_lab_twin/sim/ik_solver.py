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
        max_iter: int = 100,
        tol: float = 1e-4,
        seed_q: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve IK for the 6 rotational joints to reach target_pos_m.

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
                result = self._solve_pink(target_pos_m, max_iter, tol)
            else:
                result = self._solve_jacobian(target_pos_m, max_iter, tol)
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
                        max_iter: int, tol: float) -> Optional[np.ndarray]:
        q = np.array([self.data.qpos[jid] for jid in self.joint_ids])

        for _ in range(max_iter):
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[jid] = q[i]
            mujoco.mj_forward(self.model, self.data)

            pos_cur = self.data.site_xpos[self.ee_site].copy()
            err = target_pos_m - pos_cur
            if np.linalg.norm(err) < tol:
                return q

            nv = self.model.nv
            jacp = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site)
            # Columns 1-6 = rotational joints (col 0 = rail)
            J = jacp[:, 1:7]
            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(3))
            dq = J_pinv @ err
            dq = np.clip(dq, -np.deg2rad(3.0), np.deg2rad(3.0))
            q += dq

            for i, jid in enumerate(self.joint_ids):
                lo, hi = self.model.jnt_range[jid]
                q[i] = np.clip(q[i], lo, hi)

        return None
