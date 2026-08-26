"""The bridge between the wrist camera and the planner.

``VisionTargeting`` owns the camera and the language targeter, and answers one
question for ``LLMBrain``: *where, in world coordinates, is the thing the task
called "the blue cube"?*

Everything expensive here is lazy. Grounding DINO takes ~3 s to construct and
~700 MB of GPU memory; a session that never says "look for" should never pay
that, so the camera and the targeter are built on the first
:meth:`locate` call and reused after.

Why the planner does not get late-bound references
--------------------------------------------------
The obvious design is a ``locate_object`` action that binds a name, and a
``move_to`` that takes ``{"ref": "target"}`` resolved at dispatch time. It was
not built that way on purpose.

``LLMBrain`` runs a plan as one shot: the LLM emits the whole JSON array, and
``scripts/validate_plan.py`` checks **every pose for reachability before any
command is dispatched**. That gate exists because on real hardware a per-command
check is worse than useless -- by the time the bad command is reached, the arm
has already executed the prefix. A ``move_to`` whose coordinates do not exist
until dispatch cannot be checked by that gate, so adding late binding would
silently punch a hole in the one safety property the gate provides.

So the vision actions are *self-contained*: each one resolves the pose, checks
it, and only then moves. Nothing moves before the pose is known, which is the
property the gate was protecting. The cost is that the planner cannot do
arbitrary arithmetic on a located position within the same plan -- it gets
offsets instead (``approach_dz_mm``, ``grasp_dz_mm``), which covers the
pick-and-place shapes this scene needs.

Located objects are also written back into the :class:`ObjectRegistry`, so the
*next* turn's prompt carries what the camera saw. That is the delivery path
that makes a located coordinate useful to the planner rather than merely
printed -- CLAUDE.md's defect class #2 is exactly a value computed correctly and
handed to nobody, and a `print()` of a grasp pose would be a textbook instance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Physical extent bounds, metres, by object word. An image carries no scale, so
# "a green cube" grounds to the green *bin* as readily as the cube -- see
# perception/language/README.md. These bounds are how a phrase gets a size.
#
# Deliberately coarse and keyed on words, not on this scene's body names: the
# point is to encode "a cube is small, a bin is not", which is true of cubes and
# bins generally, rather than to smuggle the registry back in through vision.
SIZE_HINTS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "cube":  (None, 0.08),
    "block": (None, 0.08),
    "tube":  (None, 0.05),
    "vial":  (None, 0.05),
    "cup":   (0.04, 0.20),
    "bin":   (0.06, 0.40),
    "box":   (0.06, 0.40),
    "rack":  (0.06, 0.40),
    "plate": (0.06, 0.40),
}

# Phrases added to the grounding prompt alongside whatever was asked for.
# Grounding DINO calibrates better when the alternatives are named, so listing
# the things that are NOT wanted sharpens the score for the thing that is.
DEFAULT_DISTRACTORS = ("a bin", "a cup", "a test tube", "a rack", "the bench")

# Physical cap applied when the phrase carries no size word at all. The xArm6's
# gripper spans well under 100 mm, so nothing this arm can pick up is a quarter
# of a metre across; a region that large is furniture. Belt and braces with the
# targeter's image-area guard, which catches the same failure from the other
# direction (a backdrop match that happens to measure small).
DEFAULT_MAX_SIZE_M = 0.25

# Below this range a sighting is not trustworthy enough to grasp from.
#
# The D435i's datasheet minimum is 105 mm (d435i_calib.DEPTH_MIN_M) -- below it
# there is no reading at all. The interesting failure is just ABOVE that floor:
# depth is returned, the object is measured, and everything looks fine, but the
# error is large enough to put the grasp inside the object. Observed exactly
# that: a planner that looked from ~90 mm of standoff got a grasp 3 mm low and
# drove the gripper into the cube, which the collision check caught only at the
# descent.
#
# 150 mm is a deliberately conservative floor -- a wrist camera has no reason to
# be that close to something it has not grasped yet, so refusing costs nothing
# and the planner is told to back off and look again.
MIN_RELIABLE_RANGE_MM = 150.0


@dataclass
class Sighting:
    """One resolved visual referent, in world/base millimetres."""

    description: str
    position_mm: tuple[float, float, float]
    """Object centroid, world frame."""

    size_mm: float
    """Largest measured lateral extent."""

    score: float
    """Grounding confidence."""

    grasp_pose: Optional[tuple] = None
    """``(x, y, z, roll, pitch, yaw)`` from GG-CNN, or ``None`` if it proposed
    no grasp on this object. ``None`` is meaningful: the object was seen but
    nothing verified that the gripper can close on it."""

    grasp_quality: float = 0.0
    jaw_width_mm: float = 0.0
    range_mm: float = 0.0
    """How far the object was from the camera when measured. Depth accuracy
    degrades badly below the sensor's minimum range, so this is the number that
    says whether the rest of the sighting can be believed."""
    n_pixels: int = 0
    t_wall: float = 0.0

    def summary(self) -> str:
        x, y, z = self.position_mm
        s = (f"'{self.description}' at x={x:.1f} y={y:.1f} z={z:.1f} mm "
             f"({self.size_mm:.0f} mm across, score {self.score:.2f})")
        if self.grasp_pose is not None:
            s += (f"; grasp yaw {self.grasp_pose[5]:+.1f} deg, "
                  f"jaw {self.jaw_width_mm:.0f} mm, q={self.grasp_quality:.2f}")
        else:
            s += "; no grasp proposed on it"
        return s


def size_bounds_for(description: str) -> tuple[Optional[float], Optional[float]]:
    """Physical size bounds implied by the words in ``description``.

    Longest keyword wins, so "cube" in "the blue cube" is found but "test tube
    rack" prefers "rack" over "tube".

    An unknown noun gets ``(None, DEFAULT_MAX_SIZE_M)`` rather than no bound at
    all. Leaving it unbounded was a real bug: "a rubber duck" over this bench
    grounds to a 226 mm region of the benchtop at score 0.44, and with nothing
    to reject it the planner was handed a confident sighting of an object that
    does not exist. There is no honest *lower* bound for an unknown noun, so
    that stays ``None``.
    """
    text = description.lower()
    hits = [(word, bounds) for word, bounds in SIZE_HINTS.items() if word in text]
    if not hits:
        return (None, DEFAULT_MAX_SIZE_M)
    return max(hits, key=lambda kv: len(kv[0]))[1]


class VisionTargeting:
    """Wrist-camera object location for the planner.

    Parameters
    ----------
    arm:
        A ``SimXArmAPI`` or ``RealXArmAPI``. The camera backend is chosen from
        it: an arm carrying a MuJoCo ``model``/``data`` gets the simulated
        wrist camera, anything else gets the physical D435i.
    registry:
        Optional :class:`ObjectRegistry`. When present, sightings are written
        back so the next turn's prompt carries what the camera saw.
    """

    def __init__(self, arm, registry=None, device: str = "auto"):
        self.arm = arm
        self.registry = registry
        self.device = device
        self._camera = None
        self._targeter = None
        self._observer = None
        self._observer_unavailable = False
        self.last: dict[str, Sighting] = {}
        self.unavailable_reason: Optional[str] = None

    # -- lazy construction ------------------------------------------------

    @property
    def camera(self):
        if self._camera is None:
            if hasattr(self.arm, "model") and hasattr(self.arm, "data"):
                from perception.sim_camera import SimWristCamera
                self._camera = SimWristCamera(self.arm)
            else:
                from perception.realsense_camera import RealSenseWristCamera
                self._camera = RealSenseWristCamera(arm=self.arm)
        return self._camera

    @property
    def observer(self):
        """The overhead observer, in sim. ``None`` on real hardware.

        A fixed camera needs a fixed camera: in the twin that is a MuJoCo camera
        with a known pose, on the bench it is a physical D435 whose extrinsic to
        the robot base has to be solved first. That calibration does not exist
        yet, so this returns None there rather than inventing a pose -- a survey
        from an uncalibrated observer produces confident coordinates in no
        particular frame.
        """
        if self._observer is None and not self._observer_unavailable:
            if hasattr(self.arm, "model") and hasattr(self.arm, "data"):
                from perception.scene_camera import SceneCamera
                self._observer = SceneCamera(self.arm, "cam_overhead")
            else:
                self._observer_unavailable = True
        return self._observer

    @property
    def targeter(self):
        if self._targeter is None:
            from perception.language import LanguageTargeter
            self._targeter = LanguageTargeter(device=self.device)
        return self._targeter

    def available(self) -> bool:
        """Can this session use vision at all? Cheap, and caches the reason.

        Checked before the first ``locate`` so a missing dependency reports as
        "vision unavailable: no module named transformers" rather than as a
        failed grasp.
        """
        if self.unavailable_reason is not None:
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            self.unavailable_reason = f"{exc}"
            return False
        return True

    # -- the one operation ------------------------------------------------

    def locate(self, description: str, *,
               max_size_m: Optional[float] = None,
               min_size_m: Optional[float] = None,
               want_grasp: bool = True) -> Optional[Sighting]:
        """Find ``description`` in the current wrist view.

        ``None`` means the phrase did not ground, or grounded only to things
        outside the size bounds. It never means "here is my best guess" -- a
        wrong target is the arm moving to the wrong object, so the caller must
        be able to distinguish "found" from "not found".

        Size bounds default to :func:`size_bounds_for`, so callers get the
        cube-versus-bin discrimination without having to know it exists.
        """
        if not self.available():
            return None

        auto_max, auto_min = None, None
        bounds = size_bounds_for(description)
        auto_min, auto_max = bounds[0], bounds[1]

        frame = self.camera.capture()
        target = self.targeter.target(
            description, frame,
            max_size_m=max_size_m if max_size_m is not None else auto_max,
            min_size_m=min_size_m if min_size_m is not None else auto_min,
            distractors=DEFAULT_DISTRACTORS,
            attach_grasps=want_grasp,
        )
        if target is None or target.position_world is None:
            return None

        grasp_pose = None
        quality = width_mm = 0.0
        if target.grasps:
            g = target.grasps[0]
            grasp_pose = g.to_arm_pose()
            quality = g.quality
            width_mm = g.width_m * 1000.0

        sighting = Sighting(
            description=description,
            position_mm=tuple(float(v) for v in target.position_world * 1000.0),
            size_mm=target.max_size_m * 1000.0,
            score=target.score,
            grasp_pose=grasp_pose,
            grasp_quality=quality,
            jaw_width_mm=width_mm,
            n_pixels=target.n_pixels,
            range_mm=float(target.position_cam[2]) * 1000.0,
            t_wall=time.time(),
        )
        self.last[description.strip().lower()] = sighting
        self._write_back(sighting)
        return sighting

    def survey(self, descriptions: list[str]) -> dict[str, Sighting]:
        """Locate several objects in ONE overhead look, without moving the arm.

        This is what a fixed camera buys. The wrist camera answers "where is the
        blue cube" only after the arm has gone and pointed at it, one object at a
        time; the observer answers it for everything at once, from a pose that
        does not depend on what the arm is doing.

        The coordinates are a PRIOR, not a grasp. Measured at ~8 mm in X-Y
        against the wrist camera's ~3 mm, from a metre away and across the frame
        -- good enough to aim with and to verify with, not good enough to close a
        gripper on. Every sighting is written into the registry, so the next
        turn's prompt carries what the observer saw.
        """
        cam = self.observer
        if cam is None or not self.available():
            return {}

        frame = cam.capture()
        out: dict[str, Sighting] = {}
        for description in descriptions:
            lo, hi = size_bounds_for(description)
            target = self.targeter.target(
                description, frame, min_size_m=lo, max_size_m=hi,
                distractors=DEFAULT_DISTRACTORS, attach_grasps=False)
            if target is None or target.position_world is None:
                continue
            s = Sighting(
                description=description,
                position_mm=tuple(float(v) for v in target.position_world * 1000.0),
                size_mm=target.max_size_m * 1000.0,
                score=target.score,
                grasp_pose=None,          # an observer does not propose grasps
                n_pixels=target.n_pixels,
                range_mm=float(target.position_cam[2]) * 1000.0,
                t_wall=time.time(),
            )
            out[description] = s
            self.last[description.strip().lower()] = s
            self._write_back(s)
        return out

    def _write_back(self, sighting: Sighting) -> None:
        """Record the sighting in the registry so the next turn's prompt has it.

        Kept non-fatal: a registry that refuses the write is a worse prompt next
        turn, not a failed grasp now.
        """
        if self.registry is None:
            return
        try:
            from agent.object_registry import GraspConfig, LabObject

            name = "seen:" + sighting.description.strip().lower().replace(" ", "_")
            x, y, z = (v / 1000.0 for v in sighting.position_mm)
            self.registry.register(LabObject(
                name=name,
                aliases=[sighting.description],
                position_xyz_m=[x, y, z],
                grasp=GraspConfig(approach_direction=[0, 0, -1],
                                  grip_orientation_rpy=[180, 0, 0],
                                  grip_depth=0.0,
                                  approach_standoff_mm=100.0),
                safety_notes=(f"Seen by the wrist camera, not scene ground "
                              f"truth. {sighting.size_mm:.0f} mm across, "
                              f"grounding score {sighting.score:.2f}."),
                object_type="cube",
                last_updated=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[Vision] registry write-back skipped: {exc}")

    def close(self) -> None:
        for attr in ("_camera", "_observer"):
            cam = getattr(self, attr)
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
                setattr(self, attr, None)
