"""The simulated D435i: MuJoCo's wrist cameras behind the real camera's API.

``SimWristCamera`` returns the same :class:`~perception.rgbd.RGBDFrame` as
``perception.realsense_camera.RealSenseWristCamera``, from a camera carrying the
physical device's own factory intrinsics (see ``d435i_calib``). Code that
consumes frames should not be able to tell which it is talking to -- that is
what makes a perception routine developed in the twin worth anything on the
real cell.

What this models faithfully
---------------------------
* Intrinsics, including the real principal-point offset, via MuJoCo's exact
  ``focalpixel``/``principalpixel`` path rather than a centred ``fovy``.
* The two imagers' separate poses and their very different fields of view
  (colour 55.3 x 42.9 deg, depth 80.1 x 64.5 deg).
* Depth alignment: ``align=True`` renders depth from the *colour* camera, which
  is geometrically what librealsense's ``rs.align`` produces.
* The device's usable depth range -- returns outside ``[DEPTH_MIN_M,
  DEPTH_MAX_M]`` come back ``NaN``, as they do on the real sensor.

What it does not
----------------
Sim depth is geometrically exact. A real D435i's stereo depth is noisy, fills
in badly on specular and transparent surfaces, and drops out entirely on the
translucent cup in this scene. ``noise_std_m`` adds a crude Gaussian to close
part of that gap, but a routine that only ever works on clean sim depth is not
yet validated for the real cell. Treat the sim as the geometry check and the
real camera as the perception check.

Threading
---------
A ``mujoco.Renderer`` binds its GL context to the thread that constructs it, so
the renderer is created lazily on first capture and every later capture must
come from that same thread. ``arm.lock`` is held around ``update_scene`` (which
reads ``data``) and released before ``render()``, matching the locking pattern
``vr/stereo_renderer.py`` documents.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - mujoco is a hard dep of the sim
    raise ImportError("SimWristCamera needs mujoco installed") from exc

from . import d435i_calib as calib
from .rgbd import RGBDFrame, pose_matrix

# MuJoCo camera frame (+y up, -z forward) -> optical frame (+y down, +z forward).
# Same 180-degree flip about x that d435i_calib applies in the other direction;
# it is its own inverse.
_MUJOCO_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


class SimWristCamera:
    """Renders the wrist-mounted D435i from a running :class:`SimXArmAPI`.

    Parameters
    ----------
    arm:
        A ``SimXArmAPI``. Only ``model``, ``data`` and ``lock`` are used, so a
        bare ``(model, data)`` pair works too via :meth:`from_model`.
    noise_std_m:
        Gaussian noise added to valid depth, metres. 0 disables. The real D435i's
        RMS error grows roughly with the square of range; this flat model is a
        deliberate simplification, not a calibration.
    """

    def __init__(self, arm, noise_std_m: float = 0.0, seed: Optional[int] = None,
                 backdrop: bool = False):
        # The lab backdrop is in geom group 2 (see sim/render_options.py) and
        # is OFF by default, so the wrist camera sees the bare bench exactly as
        # it did before the room existed. Pass backdrop=True to look at the
        # furnished lab instead. It matters for perception, not just looks:
        # grounding against clutter is a different problem from grounding
        # against a void, and being able to flip between them is how you find
        # out which one a routine was relying on.
        from sim.render_options import scene_option
        self._scene_option = scene_option(backdrop=backdrop)
        self.backdrop = backdrop
        self.model = arm.model
        self.data = arm.data
        self.lock = getattr(arm, "lock", None) or threading.RLock()
        self.noise_std_m = float(noise_std_m)
        self._rng = np.random.default_rng(seed)

        self.width = calib.WIDTH
        self.height = calib.HEIGHT

        self._color_cam_id = self._cam_id(calib.COLOR_CAM_NAME)
        self._depth_cam_id = self._cam_id(calib.DEPTH_CAM_NAME)

        # Built on first capture, on whichever thread calls first (see module
        # docstring). Two renderers because MuJoCo toggles depth mode on the
        # renderer itself, and flipping it per frame costs a buffer realloc.
        self._rgb_renderer: Optional[mujoco.Renderer] = None
        self._depth_renderer: Optional[mujoco.Renderer] = None
        self._gl_thread: Optional[int] = None

    @classmethod
    def from_model(cls, model, data, lock=None, **kw) -> "SimWristCamera":
        """Build from a raw model/data pair, for scripts with no SimXArmAPI."""
        shim = type("_ArmShim", (), {"model": model, "data": data,
                                     "lock": lock or threading.RLock()})()
        return cls(shim, **kw)

    def _cam_id(self, name: str) -> int:
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(
                f"scene has no camera {name!r}. The wrist camera lives in the "
                "<body name='gripper'> subtree of envs/lab_scene_primitive.xml; "
                "if you edited that, rerun `python envs/build_mesh_scene.py`."
            )
        return cid

    # -- intrinsics -------------------------------------------------------

    @property
    def color_intrinsics(self) -> calib.Intrinsics:
        return calib.COLOR_INTRINSICS

    @property
    def depth_intrinsics(self) -> calib.Intrinsics:
        return calib.DEPTH_INTRINSICS

    # -- pose -------------------------------------------------------------

    def cam_to_world(self, aligned: bool = True) -> np.ndarray:
        """4x4 transform, camera **optical** frame -> world frame.

        ``aligned`` picks which imager, matching the ``align`` argument used at
        capture. MuJoCo's ``cam_xmat`` is the camera frame, so it is converted
        into the optical convention here rather than at any call site.
        """
        cid = self._color_cam_id if aligned else self._depth_cam_id
        with self.lock:
            pos = np.array(self.data.cam_xpos[cid], dtype=np.float64)
            rot = np.array(self.data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)
        return pose_matrix(rot @ _MUJOCO_TO_OPTICAL, pos)

    # -- capture ----------------------------------------------------------

    def _ensure_renderers(self) -> None:
        tid = threading.get_ident()
        if self._rgb_renderer is None:
            self._rgb_renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width)
            self._depth_renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width)
            self._depth_renderer.enable_depth_rendering()
            self._gl_thread = tid
        elif tid != self._gl_thread:
            print("[SimWristCamera] WARNING: capture() called from a second "
                  "thread; the GL context belongs to the thread that captured "
                  "first. Expect a crash or a blank frame.")

    def capture(self, align: bool = True) -> RGBDFrame:
        """Render one synchronised colour + depth frame.

        With ``align=True`` (the default, and what ``rs.align`` does on the real
        camera) depth is rendered from the colour camera, so both images share
        the colour intrinsics and pixel ``(u, v)`` means the same ray in each.
        With ``align=False`` depth comes from the wider depth imager and carries
        its own intrinsics -- the frame records which, so a consumer deprojecting
        it cannot pick the wrong matrix.
        """
        self._ensure_renderers()
        depth_cam = self._color_cam_id if align else self._depth_cam_id

        with self.lock:
            self._rgb_renderer.update_scene(self.data, camera=self._color_cam_id,
                                            scene_option=self._scene_option)
            self._depth_renderer.update_scene(self.data, camera=depth_cam,
                                              scene_option=self._scene_option)
            t_sim = float(self.data.time)
        color = self._rgb_renderer.render().astype(np.uint8)
        depth = self._depth_renderer.render().astype(np.float32)

        return RGBDFrame(
            color=color,
            depth=self._condition_depth(depth),
            intrinsics=self.color_intrinsics if align else self.depth_intrinsics,
            cam_to_world=self.cam_to_world(aligned=align),
            t_wall=t_sim,
            source="sim",
        )

    def _condition_depth(self, depth: np.ndarray) -> np.ndarray:
        """Make MuJoCo's ideal depth behave like the device's.

        MuJoCo returns metres along the view axis, with un-hit pixels at the far
        clip plane. Those become ``NaN`` here, as do returns outside the sensor's
        usable range -- so "no data" is never a finite number that a downstream
        deprojection would happily turn into a point.
        """
        d = depth.copy()
        far = float(self.model.stat.extent * self.model.vis.map.zfar)
        d[d >= far * 0.999] = np.nan

        if self.noise_std_m > 0.0:
            valid = np.isfinite(d)
            d[valid] += self._rng.normal(0.0, self.noise_std_m, valid.sum())

        d[(d < calib.DEPTH_MIN_M) | (d > calib.DEPTH_MAX_M)] = np.nan
        # Quantise to the device's depth units so sim and real share a
        # resolution floor; without it sim depth is smooth where real is stepped.
        valid = np.isfinite(d)
        d[valid] = np.round(d[valid] / calib.DEPTH_SCALE_M) * calib.DEPTH_SCALE_M
        return d

    def close(self) -> None:
        for r in (self._rgb_renderer, self._depth_renderer):
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
        self._rgb_renderer = self._depth_renderer = None

    def __enter__(self) -> "SimWristCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
