"""GG-CNN grasp detection on wrist-camera depth.

Turns an :class:`~perception.rgbd.RGBDFrame` into ranked antipodal grasp
candidates expressed in the robot's world frame -- ready to hand to
``SimXArmAPI.set_position``.

    from perception import SimWristCamera
    from perception.grasp import GGCNNDetector

    det    = GGCNNDetector()
    frame  = SimWristCamera(arm).capture()
    grasps = det.detect(frame)
    if grasps:
        x, y, z, roll, pitch, yaw = grasps[0].to_arm_pose()
        arm.set_position(x, y, z, roll, pitch, yaw)

Provenance
----------
GG-CNN is Douglas Morrison / ACRV-QUT (BSD-3, ``LICENSE.ggcnn``); the Cornell
weights and the xArm integration come from UFACTORY's ``ufactory_vision``
(BSD-3, ``LICENSE.ufactory``). The network definitions in ``_ggcnn_net.py`` /
``_ggcnn2_net.py`` are vendored verbatim. The weights were converted from
UFACTORY's pickled whole-model files into plain ``state_dict``s -- see
``weights/README.md`` for why and for the equivalence check.

How this differs from the upstream demo, and why
------------------------------------------------
Upstream's ``TorchGGCNN`` + ``RobotGrasp`` is a closed-loop visual-servoing
controller for a real arm: it streams poses in xArm servo mode, tracks a
running maximum across frames, and owns the pick/place state machine. None of
that transfers to the twin, which drives the arm through its own validated
motion primitives. So this module keeps the part that is the model --
depth in, grasp candidates out -- and stops there.

Four concrete changes, each for a reason worth stating:

1. **The camera transform is not re-derived.** Upstream rebuilds the
   base->flange->optical chain from euler triples at every frame. Here the
   grasp point goes through ``RGBDFrame.pixel_to_world``, the same path
   ``perception/test_projection.py`` validates against MuJoCo's ray caster.
   One transform, already checked, instead of a second one that agrees with
   the first only as long as nobody edits either.

2. **The predicted width is delivered.** Upstream computes ``width`` and
   ``depth_center``, returns them, and never reads them again -- CLAUDE.md's
   defect class #2, in the reference implementation. ``Grasp.width_m`` converts
   the network's pixel width into metres at the grasp's own depth, which is what
   a gripper aperture actually needs.

3. **Inference runs at 300x300.** Upstream feeds the full 480x480 crop.
   GG-CNN is fully convolutional so that runs, but the Cornell weights were
   trained at 300x300 and the post-filter sigmas (5.0 / 2.0 px) are tuned for
   that scale; at 480 they smooth relatively less. Upstream's own
   ``process_depth_image`` defaults to 300 -- only ``get_grasp_img`` overrides
   it. Set ``out_size=None`` to reproduce the upstream behaviour.

4. **Top-k candidates, not one tracked maximum.** Frame-to-frame max tracking
   only makes sense inside a servo loop. A planner wants options, so
   ``detect()`` returns peaks ranked by quality.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..d435i_calib import Intrinsics
from ..rgbd import RGBDFrame

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

MODELS = {
    "ggcnn": ("ggcnn_epoch_23_cornell.pt", "_ggcnn_net", "GGCNN"),
    "ggcnn2": ("ggcnn2_epoch_50_cornell.pt", "_ggcnn2_net", "GGCNN2"),
}

# GG-CNN's width output is scaled 0-1 over 0-150 px, at the network's own
# resolution. Straight from the reference implementation.
_WIDTH_SCALE_PX = 150.0


@dataclass
class Grasp:
    """One antipodal grasp candidate.

    Positions are metres. ``*_cam`` is the camera optical frame, ``*_world`` the
    robot base/world frame -- present only when the source frame carried a pose.
    """

    quality: float
    """Network confidence at this peak, 0-1. Relative, not calibrated."""

    pixel: tuple[float, float]
    """``(u, v)`` in the full-resolution source image."""

    angle_rad: float
    """Grasp angle in the image plane, radians. The gripper's closing axis."""

    width_px: float
    """Predicted jaw opening, pixels at full resolution."""

    depth_m: float
    """Depth at the grasp point, metres."""

    position_cam: np.ndarray
    """3D grasp point in the camera optical frame, metres."""

    position_world: Optional[np.ndarray] = None
    rotation_world: Optional[np.ndarray] = None
    """Full 3x3 grasp orientation in the world frame, if a pose was available."""

    width_m: float = 0.0
    """Jaw opening in metres at this grasp's depth -- what a gripper needs."""

    intrinsics: Optional[Intrinsics] = field(default=None, repr=False)

    @property
    def yaw_world_rad(self) -> Optional[float]:
        """Yaw for a top-down approach, radians, normalised to ``(-pi, 0]``.

        A parallel-jaw grasp is unchanged by rotating 180 degrees about its
        approach axis, so yaw and yaw+pi describe the same physical grasp.
        Collapsing to one representative -- the same interval UFACTORY's demo
        uses -- keeps a planner from treating them as two options, and keeps the
        wrist from taking the long way round.
        """
        if self.rotation_world is None:
            return None
        r = self.rotation_world
        pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
        sign = 1.0 if math.cos(pitch) >= 0 else -1.0
        yaw = math.atan2(r[1, 0] * sign, r[0, 0] * sign)
        if yaw < -math.pi:
            yaw += math.pi
        elif yaw > 0.0:
            yaw -= math.pi
        return yaw

    def to_arm_pose(self, degrees: bool = True) -> tuple:
        """``(x, y, z, roll, pitch, yaw)`` for ``SimXArmAPI.set_position``.

        Millimetres and degrees, with the top-down wrist orientation the twin's
        pick primitives use (roll 180, pitch 0). Raises if the grasp has no
        world pose -- there is no sensible fallback, and returning camera
        coordinates shaped like arm coordinates would be worse than failing.
        """
        if self.position_world is None:
            raise ValueError(
                "this grasp has no world pose: the frame it came from carried "
                "no cam_to_world. Capture from a camera bound to the arm."
            )
        # float(), not the raw numpy scalars: this tuple goes straight into
        # set_position and on into logs, JSON and recordings, where np.float64
        # serialises differently and reads as noise.
        x, y, z = (float(v) for v in self.position_world * 1000.0)
        yaw = self.yaw_world_rad or 0.0
        if degrees:
            return (x, y, z, 180.0, 0.0, math.degrees(yaw))
        return (x, y, z, math.pi, 0.0, yaw)


class GGCNNDetector:
    """GG-CNN inference over :class:`RGBDFrame` depth.

    Parameters
    ----------
    model:
        ``"ggcnn"`` (Cornell epoch 23, the UFACTORY demo's default) or
        ``"ggcnn2"`` (epoch 50).
    out_size:
        Network input size. Default 300, the resolution the Cornell weights were
        trained at. ``None`` uses the full crop, reproducing upstream.
    device:
        Torch device. CPU is the default and is fast enough -- the network is
        62k parameters and runs in a few milliseconds.
    """

    def __init__(self, model: str = "ggcnn", out_size: Optional[int] = 300,
                 device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GGCNNDetector needs torch. In the sim env:\n"
                "  pip install torch"
            ) from exc

        if model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}, got {model!r}")
        weight_file, net_module, net_class = MODELS[model]
        path = WEIGHTS_DIR / weight_file
        if not path.exists():
            raise FileNotFoundError(
                f"missing weights {path}. They ship with the repo; if this is a "
                f"fresh clone check git-lfs or re-run the conversion in "
                f"weights/README.md."
            )

        import importlib

        net_cls = getattr(
            importlib.import_module(f".{net_module}", __package__), net_class)
        self._torch = torch
        self.device = torch.device(device)
        self.model_name = model
        self.net = net_cls()
        # weights_only=True: these are plain tensors, so loading them cannot
        # execute code. The upstream pickles required weights_only=False, which
        # is why they were converted rather than vendored as-is.
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))
        self.net.eval().to(self.device)
        self.out_size = out_size

    # -- preprocessing ----------------------------------------------------

    @staticmethod
    def _prepare_depth(depth: np.ndarray, out_size: Optional[int]
                       ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """Centre-crop, inpaint holes, resize, and zero-centre.

        Returns ``(input, nan_mask, crop_size, y0, x0)``. The crop offsets come
        back so a peak can be mapped to a full-resolution pixel; computing them
        here and returning them keeps that mapping next to the crop that caused
        it rather than duplicated at the call site.
        """
        import cv2

        h, w = depth.shape
        crop_size = min(h, w, 500)
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        crop = depth[y0:y0 + crop_size, x0:x0 + crop_size].astype(np.float32)

        # OpenCV's inpaint misbehaves at the border, so pad by one, inpaint,
        # then strip the pad -- as the reference implementation does.
        crop = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        nan_mask = np.isnan(crop).astype(np.uint8)
        nan_mask = cv2.dilate(nan_mask, np.ones((3, 3), np.uint8), iterations=1)
        crop[nan_mask == 1] = 0.0

        # inpaint needs values in [-1, 1]; rescale and undo afterwards.
        scale = float(np.abs(crop).max()) or 1.0
        crop = cv2.inpaint(crop / scale, nan_mask, 1, cv2.INPAINT_NS) * scale
        crop = crop[1:-1, 1:-1]
        nan_mask = nan_mask[1:-1, 1:-1]

        size = out_size or crop_size
        if size != crop_size:
            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
            nan_mask = cv2.resize(nan_mask, (size, size),
                                  interpolation=cv2.INTER_NEAREST)

        # Zero-centre: GG-CNN is trained on depth relative to the scene mean, so
        # it is invariant to how far the camera happens to be.
        return (np.clip(crop - crop.mean(), -1.0, 1.0), nan_mask,
                crop_size, y0, x0)

    # -- inference --------------------------------------------------------

    def detect(self, frame: RGBDFrame, top_k: int = 3,
               min_quality: float = 0.1) -> list[Grasp]:
        """Rank grasp candidates in ``frame``, best first.

        Returns ``[]`` when nothing clears ``min_quality`` or when the peaks all
        land on pixels with no valid depth -- an empty list, never a low-quality
        guess dressed up as a detection.
        """
        import cv2
        import scipy.ndimage as ndimage
        from skimage.feature import peak_local_max

        inp, nan_mask, crop_size, y0, x0 = self._prepare_depth(
            frame.depth, self.out_size)
        size = inp.shape[0]

        tensor = self._torch.from_numpy(
            inp.reshape(1, 1, size, size).astype(np.float32)).to(self.device)
        with self._torch.no_grad():
            pred = self.net(tensor)

        quality = pred[0].cpu().numpy().squeeze()
        cos_out = pred[1].cpu().numpy().squeeze()
        sin_out = pred[2].cpu().numpy().squeeze()
        width = pred[3].cpu().numpy().squeeze() * _WIDTH_SCALE_PX

        # Inpainted holes are invented depth; suppress grasps there rather than
        # letting the network score its own interpolation.
        quality[nan_mask.astype(bool)] = 0.0

        angle = np.arctan2(sin_out, cos_out) / 2.0
        quality = ndimage.gaussian_filter(quality, 5.0)
        angle = ndimage.gaussian_filter(angle, 2.0)
        width = ndimage.gaussian_filter(width, 2.0)
        quality = np.clip(quality, 0.0, 1.0 - 1e-3)

        peaks = peak_local_max(quality, min_distance=10,
                               threshold_abs=min_quality, num_peaks=top_k)
        if peaks.shape[0] == 0:
            return []

        # Network grid -> full-resolution pixels.
        scale = crop_size / size
        grasps: list[Grasp] = []
        for r, c in peaks:
            u = c * scale + x0
            v = r * scale + y0
            iv, iu = int(round(v)), int(round(u))
            if not (0 <= iv < frame.depth.shape[0] and 0 <= iu < frame.depth.shape[1]):
                continue
            depth_m = float(frame.depth[iv, iu])
            if not np.isfinite(depth_m):
                # The peak sits on a hole. Real D435i depth is full of them, so
                # this is an ordinary outcome, not an error -- drop the candidate.
                continue

            p_cam = frame.deproject_pixel(u, v, depth_m)
            if p_cam is None:
                continue

            k = frame.intrinsics
            width_px = float(width[r, c]) * scale
            grasps.append(Grasp(
                quality=float(quality[r, c]),
                pixel=(float(u), float(v)),
                angle_rad=float(angle[r, c]),
                width_px=width_px,
                depth_m=depth_m,
                position_cam=p_cam,
                # A width in pixels means nothing to a gripper; at this depth it
                # is this many metres. Upstream stopped at the pixels.
                width_m=width_px * depth_m / k.fx,
                intrinsics=k,
                **self._world_fields(frame, p_cam, float(angle[r, c])),
            ))

        grasps.sort(key=lambda g: g.quality, reverse=True)
        return grasps

    @staticmethod
    def _world_fields(frame: RGBDFrame, p_cam: np.ndarray,
                      angle_rad: float) -> dict:
        """World-frame position and orientation, if the frame carries a pose.

        The grasp frame is the camera optical frame rotated by ``-angle`` about
        the optical z (the view axis) -- the convention UFACTORY's demo uses,
        kept so their calibration and this one describe the same thing. Composing
        it with ``cam_to_world`` is the whole transform: no separate euler chain.
        """
        if frame.cam_to_world is None:
            return {}
        c, s = math.cos(-angle_rad), math.sin(-angle_rad)
        rot_grasp_in_cam = np.array([[c, -s, 0.0],
                                     [s, c, 0.0],
                                     [0.0, 0.0, 1.0]], dtype=np.float64)
        return {
            "position_world": (frame.cam_to_world @ np.append(p_cam, 1.0))[:3],
            "rotation_world": frame.cam_to_world[:3, :3] @ rot_grasp_in_cam,
        }

    # -- visualisation ----------------------------------------------------

    def draw(self, frame: RGBDFrame, grasps: list[Grasp]) -> np.ndarray:
        """Colour image with grasp candidates drawn on. Best one in green."""
        import cv2

        img = frame.color.copy()
        for i, g in enumerate(grasps):
            u, v = g.pixel
            colour = (0, 255, 0) if i == 0 else (255, 160, 0)
            half = max(g.width_px / 2.0, 6.0)
            dx, dy = math.cos(g.angle_rad) * half, math.sin(g.angle_rad) * half
            cv2.line(img, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)),
                     colour, 2)
            cv2.circle(img, (int(u), int(v)), 4, colour, -1)
            cv2.putText(img, f"{g.quality:.2f}", (int(u) + 8, int(v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        return img
