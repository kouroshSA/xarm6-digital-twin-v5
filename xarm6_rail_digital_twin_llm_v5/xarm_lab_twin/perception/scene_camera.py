"""Fixed observer cameras, behind the same API as the wrist camera.

``SceneCamera`` renders one of the ``cam_overhead`` / ``cam_tripod`` cameras
defined in :mod:`perception.observer_calib` and returns the same
:class:`~perception.rgbd.RGBDFrame` as ``SimWristCamera`` and
``RealSenseWristCamera``. That is the whole point: ``LanguageTargeter``,
``GGCNNDetector`` and every consumer of a frame work through an observer with no
change at all, because none of them ever asks which camera it came from.

    from perception.scene_camera import SceneCamera
    from perception.language import LanguageTargeter

    cam    = SceneCamera(arm, "cam_overhead")
    target = LanguageTargeter().target("the blue cube", cam.capture(),
                                       max_size_m=0.08)

What an observer is for, and what it is not
-------------------------------------------
Use it to survey -- where is everything, before the arm moves -- and to verify:
did the cube end up in the cup? On real hardware that second question currently
has no answer at all, because ``physical_outcome()`` reads MuJoCo's ground
truth, which a bench does not have.

Do not use it for the grasp itself. The wrist camera measures from ~300 mm with
the target centred; an observer measures the same object from a metre away,
across the frame, and through whatever the arm is doing. The overhead camera is
also blocked by the arm during exactly the approach it is watching. Survey with
the observer, commit with the wrist.

The depth-error geometry is in ``observer_calib``'s docstring and is the reason
``cam_overhead`` is near-vertical: an oblique depth camera puts ~90% of its error
into X-Y, which is the coordinate the arm is commanded with.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - mujoco is a hard dep of the sim
    raise ImportError("SceneCamera needs mujoco installed") from exc

from . import d435i_calib as calib
from . import observer_calib
from .rgbd import RGBDFrame, pose_matrix
from .sim_camera import _MUJOCO_TO_OPTICAL, SimWristCamera


class SceneCamera(SimWristCamera):
    """A fixed camera in the scene, rendered like the wrist one.

    Subclasses ``SimWristCamera`` rather than restating it: the depth
    conditioning (far-plane masking, range clipping, quantisation, optional
    noise), the lazy GL-context handling and the frame assembly are identical
    physics and identical bugs if written twice. Only which camera is rendered,
    and the fact that there is one imager rather than two, differ.

    Parameters
    ----------
    arm:
        A ``SimXArmAPI``, or use :meth:`from_model` with a raw model/data pair.
    name:
        An observer name from ``observer_calib.OBSERVERS`` -- ``"cam_overhead"``
        or ``"cam_tripod"``.
    backdrop:
        Whether the lab room renders. Off by default, matching the rest of the
        sim; an observer looking at the wall is one place you may well want it on.
    """

    def __init__(self, arm, name: str = "cam_overhead",
                 noise_std_m: float = 0.0, seed: Optional[int] = None,
                 backdrop: bool = False):
        self.observer = observer_calib.by_name(name)
        self.name = name
        super().__init__(arm, noise_std_m=noise_std_m, seed=seed,
                         backdrop=backdrop)
        # Override the wrist camera's 640x480 render size: an observer is
        # further away and carries the 720p profile, and a renderer built at the
        # wrong size would silently resample every frame away from the
        # intrinsics that describe it.
        self.width = observer_calib.OBSERVER_INTRINSICS.width
        self.height = observer_calib.OBSERVER_INTRINSICS.height

    @classmethod
    def from_model(cls, model, data, lock=None, **kw) -> "SceneCamera":
        import threading

        shim = type("_ArmShim", (), {"model": model, "data": data,
                                     "lock": lock or threading.RLock()})()
        return cls(shim, **kw)

    def _cam_id(self, _name: str) -> int:
        """Resolve to this observer, whichever imager the base class asks for.

        A fixed observer is one physical camera, so colour and depth are the same
        MuJoCo camera -- which is also what ``align=True`` means on a real D435
        once ``rs.align`` has reprojected depth into the colour frame. The base
        class asks twice; both answers are the same id.
        """
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.name)
        if cid < 0:
            raise ValueError(
                f"scene has no camera {self.name!r}. Observer cameras are "
                f"generated into envs/lab_scene_primitive.xml -- run "
                f"`python -m perception.sync_scene`."
            )
        return cid

    # A fixed observer has no separate wide-angle depth imager to offer, so both
    # views carry the colour intrinsics. Stated rather than inherited, because
    # inheriting DEPTH_INTRINSICS here would hand out a matrix for an imager
    # this camera does not have.
    @property
    def color_intrinsics(self) -> calib.Intrinsics:
        return observer_calib.OBSERVER_INTRINSICS

    @property
    def depth_intrinsics(self) -> calib.Intrinsics:
        return observer_calib.OBSERVER_INTRINSICS

    def cam_to_world(self, aligned: bool = True) -> np.ndarray:
        """4x4, camera **optical** frame -> world. Static, but read live.

        Read from ``data`` rather than from ``observer_calib`` on purpose: if the
        scene ever disagrees with the calibration module, this returns what was
        actually rendered. A pose taken from the module would describe an image
        that does not exist.
        """
        cid = self._color_cam_id
        with self.lock:
            pos = np.array(self.data.cam_xpos[cid], dtype=np.float64)
            rot = np.array(self.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
        return pose_matrix(rot @ _MUJOCO_TO_OPTICAL, pos)

    @property
    def tilt_deg(self) -> float:
        """Degrees off vertical -- how much depth error lands in X-Y."""
        return self.observer.tilt_deg

    def __repr__(self) -> str:
        return (f"SceneCamera({self.name!r}, {self.tilt_deg:.0f} deg off "
                f"vertical, {self.observer.purpose})")


def all_observers(arm, **kw) -> dict[str, SceneCamera]:
    """One :class:`SceneCamera` per observer, keyed by name.

    Each builds its own renderer, so they share a GL context and must all be
    used from the thread that captured first -- the same constraint every
    MuJoCo camera in this package carries.
    """
    return {obs.name: SceneCamera(arm, obs.name, **kw)
            for obs in observer_calib.OBSERVERS}
