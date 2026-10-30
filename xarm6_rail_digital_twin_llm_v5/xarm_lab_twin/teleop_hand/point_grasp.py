# teleop_hand/point_grasp.py
"""Run a `PointTarget`: look from above, grasp with GG-CNN, lift. Blocking.

grasp   1. (rail) bring the object within the arm's comfortable reach
        2. fly to LOOK_ABOVE over the pointed position, tool pointing down
        3. capture the WRIST camera, run GG-CNN (UFACTORY's grasp network,
           vendored in `perception/grasp`)
        4. keep only grasps within GRASP_MAX_PRIOR_ERR_MM of the pointed object
           and take the NEAREST to it (not the highest quality: GG-CNN ranks
           graspability, not task relevance, and the first live run picked the
           heater-shaker's edge 40 mm from a cube it sits 9 mm beside). Then
           refuse any grasp that is nearer to a different known object than to
           the pointed one.
        5. above the grasp, descend, close, lift

It REFUSES, and says why, rather than descending on the prior: if GG-CNN
proposes nothing near the pointed object the object was seen, but nothing
verified the jaws can close there (same rule as `grasp_object`).

hover / free   fly to the target; if the arm refuses it, try the fallbacks
               (nearer along the ray) in order.

The result is graded by what the gripper holds (`arm.verify_grasp()`), never
by return codes: a sequence of rc == 0 moves that closes on the wrong cube
looks fine in every log.

Every move goes through `arm.set_position`, i.e. the same IK + collision
validators the planner's moves use, paced by `speed`.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

from sim.tool import TCP_FROM_FLANGE_M
from teleop_hand import config
from teleop_hand.pointing import PointTarget


@dataclasses.dataclass
class PointResult:
    ok: bool
    stage: str                      # where it ended: look | detect | approach | descend | close | lift | done
    message: str
    held: Optional[str] = None      # body welded to the gripper afterwards
    grasp_pose: Optional[tuple] = None


class PointGraspExecutor:
    def __init__(self, arm, camera_factory: Optional[Callable] = None,
                 detector_factory: Optional[Callable] = None,
                 abort_fn: Callable[[], bool] = lambda: False, log=print):
        self.arm = arm
        self._camera_factory = camera_factory
        self._detector_factory = detector_factory
        self._camera = None
        self._detector = None
        self.abort_fn = abort_fn
        self.log = lambda m: log(f"[POINT] {m}")

    # lazily: GG-CNN weights, torch and the EGL context cost seconds, and a
    # session that never points should not pay them. The camera is built on
    # the calling thread (EGL contexts are thread-affine) and reused.
    @property
    def camera(self):
        if self._camera is None:
            if self._camera_factory is not None:
                self._camera = self._camera_factory()
            elif hasattr(self.arm, "model") and hasattr(self.arm, "data"):
                from perception.sim_camera import SimWristCamera
                self._camera = SimWristCamera(self.arm)
            else:
                from perception.realsense_camera import RealSenseWristCamera
                self._camera = RealSenseWristCamera(arm=self.arm)
        return self._camera

    @property
    def detector(self):
        if self._detector is None:
            if self._detector_factory is not None:
                self._detector = self._detector_factory()
            else:
                from perception.grasp import GGCNNDetector
                self._detector = GGCNNDetector()
        return self._detector

    # ---- helpers ------------------------------------------------------------
    def _move(self, x, y, z, speed, yaw=0.0, roll=180.0, pitch=0.0) -> int:
        return self.arm.set_position(x=float(x), y=float(y), z=float(z), roll=roll,
                                     pitch=pitch, yaw=yaw, speed=speed, wait=True)

    def _fail(self, stage: str, msg: str) -> PointResult:
        self.log(f"stopped at {stage}: {msg}")
        return PointResult(False, stage, msg, held=self._held())

    def _held(self) -> Optional[str]:
        ok, detail = self.arm.verify_grasp()
        return detail.replace("welded to ", "") if ok == "held" else None

    def _bring_into_reach(self, target_mm: np.ndarray, base_mm: np.ndarray,
                          rail_axis: np.ndarray, rail_mm: float) -> None:
        along = float(np.dot(target_mm - base_mm, rail_axis))
        if abs(along) <= config.REACH_HALF_SPAN_MM:
            return
        want = float(np.clip(rail_mm + (along - np.sign(along) * 290.0),
                             config.RAIL_MIN_MM, config.RAIL_MAX_MM))
        self.log(f"object {along:.0f} mm along the rail from the base; rail -> {want:.0f} mm")
        self.arm.set_rail_position(want, speed_mm_s=config.RAIL_MAX_SPEED, wait=True)

    # ---- entry --------------------------------------------------------------
    def run(self, target: PointTarget, base_mm=None, rail_axis=None, rail_mm=None,
            others: Optional[dict] = None) -> PointResult:
        self.log(f"target: {target.describe()}")
        if target.kind == "grasp":
            return self._grasp(target, base_mm, rail_axis, rail_mm, others or {})
        return self._fly(target)

    def _fly(self, target: PointTarget) -> PointResult:
        speed = config.GRASP_SPEED_MM_S
        last = ""
        for p in [target.position_mm, *target.fallbacks]:
            if self.abort_fn():
                return self._fail("approach", "aborted by the operator")
            rc = self._move(p[0], p[1], p[2], speed)
            if rc == 0:
                return PointResult(True, "done", f"arrived at {np.round(p).astype(int).tolist()} mm")
            last = getattr(self.arm, "last_refusal", "") or f"rc={rc}"
            self.log(f"{np.round(p).astype(int).tolist()} refused ({last}); trying nearer")
        return self._fail("approach", f"every point along the ray was refused: {last}")

    def _grasp(self, target: PointTarget, base_mm, rail_axis, rail_mm, others) -> PointResult:
        prior = np.asarray(target.position_mm, float)
        if base_mm is not None:
            self._bring_into_reach(prior, np.asarray(base_mm, float),
                                   np.asarray(rail_axis, float), float(rail_mm))
        if self.abort_fn():
            return self._fail("look", "aborted by the operator")
        look_z = prior[2] + config.GRASP_LOOK_ABOVE_MM - TCP_FROM_FLANGE_M * 1000.0
        rc = self._move(prior[0], prior[1], look_z, config.GRASP_SPEED_MM_S)
        if rc != 0:
            return self._fail("look", f"cannot get above {target.name}: "
                                      f"{getattr(self.arm, 'last_refusal', '') or f'rc={rc}'}")

        frame = self.camera.capture()
        try:
            grasps = self.detector.detect(frame, top_k=8)
        except Exception as exc:  # noqa: BLE001 -- surfaced, not swallowed
            return self._fail("detect", f"GG-CNN failed: {exc}")
        near = []
        for g in grasps:
            if g.position_world is None:
                continue
            err = float(np.linalg.norm((np.asarray(g.position_world) * 1000.0 - prior)[:2]))
            if err <= config.GRASP_MAX_PRIOR_ERR_MM:
                near.append((g, err))
        self.log(f"GG-CNN: {len(grasps)} candidates, {len(near)} within "
                 f"{config.GRASP_MAX_PRIOR_ERR_MM:.0f} mm of {target.name}")
        if not near:
            return self._fail("detect", f"no grasp proposed on {target.name}; "
                                        f"seen is not graspable, so not descending on the prior")
        g, err = min(near, key=lambda ge: (round(ge[1] / 5.0), -ge[0].quality))
        gx, gy, gz, roll, pitch, yaw = g.to_arm_pose()
        gp = np.array([gx, gy])
        for name, pos in others.items():
            if name != target.name and np.linalg.norm(np.asarray(pos)[:2] - gp) < err:
                return self._fail("detect", f"the grasp is nearer {name} than {target.name}; "
                                            f"not descending on the wrong object")
        self.log(f"grasp at ({gx:.0f}, {gy:.0f}, {gz:.0f}) yaw {yaw:+.0f} deg, "
                 f"q={g.quality:.2f}, {err:.0f} mm from the pointed position")

        for stage, z, speed in (("approach", gz + config.GRASP_APPROACH_ABOVE_MM,
                                 config.GRASP_SPEED_MM_S),
                                ("descend", gz, config.GRASP_DESCENT_SPEED_MM_S)):
            if self.abort_fn():
                return self._fail(stage, "aborted by the operator")
            rc = self._move(gx, gy, z, speed, yaw=yaw, roll=roll, pitch=pitch)
            if rc != 0:
                return self._fail(stage, f"refused: {getattr(self.arm, 'last_refusal', '') or f'rc={rc}'}")
        rc = self.arm.close_lite6_gripper()
        if rc != 0:
            return self._fail("close", getattr(self.arm, "last_refusal", "") or f"rc={rc}")
        rc = self._move(gx, gy, gz + config.GRASP_LIFT_MM, config.GRASP_SPEED_MM_S,
                        yaw=yaw, roll=roll, pitch=pitch)
        held = self._held()
        if rc != 0:
            return PointResult(False, "lift", f"lift refused: {getattr(self.arm, 'last_refusal', '')}",
                               held=held, grasp_pose=(gx, gy, gz, roll, pitch, yaw))
        if held is None:
            return PointResult(False, "lift", "closed but nothing is held",
                               grasp_pose=(gx, gy, gz, roll, pitch, yaw))
        self.log(f"holding {held}")
        return PointResult(True, "done", f"holding {held}", held=held,
                           grasp_pose=(gx, gy, gz, roll, pitch, yaw))
