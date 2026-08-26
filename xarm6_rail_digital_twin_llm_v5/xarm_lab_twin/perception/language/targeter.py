"""Language-conditioned targeting: say what you want, get where to grasp it.

    from perception import SimWristCamera
    from perception.language import LanguageTargeter

    t     = LanguageTargeter()
    frame = SimWristCamera(arm).capture()

    grasp = t.grasp_for("the blue cube", frame)
    if grasp:
        arm.set_position(*grasp.to_arm_pose())

Three stages, each doing only what it is good at:

1. **Ground** the phrase to image regions (Grounding DINO, open vocabulary).
2. **Physicalise** each region using depth -- separate the object from the
   surface under it, and measure how big it actually is, in metres.
3. **Select** a GG-CNN grasp that lands inside the chosen region.

Why stage 2 exists
------------------
Grounding DINO asked for "a green cube" over this bench returns the green cube
*and* the green bin, both confidently. That is not a failure of the model or the
prompt: a bin **is** a green box, and a single image carries no scale, so
nothing in the picture distinguishes a 40 mm cube from a 150 mm bin. Ranking by
grounding score alone picks the bin about as often as the cube.

Depth carries the scale that the image does not. Measuring each candidate's
physical extent turns an unresolvable language question into an easy
measurement, and ``max_size_m`` lets a caller say "a cube is small" without
anyone hard-coding this scene's objects into the perception stack.

The same step also gives the region a *mask* rather than a box, for free: pixels
inside the box at roughly the object's own depth are the object; pixels at the
bench's depth are not. That matters because a box around a small cube is mostly
bench, and a centroid over the whole box lands beside the cube rather than on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..rgbd import RGBDFrame
from .grounding import Detection, GroundingDINOGrounder

# Pixels within this much of the region's own depth count as the object rather
# than the surface behind it. 3 cm comfortably contains a cube's visible face
# and its near side while excluding the bench it sits on.
DEFAULT_DEPTH_BAND_M = 0.03

# A region needs at least this many valid depth pixels before its size and
# centroid mean anything. Below it the numbers are noise wearing a decimal point.
MIN_REGION_PIXELS = 40

# Reject a candidate whose box covers more than this fraction of the frame.
#
# Grounding DINO always returns *something* above threshold, so a phrase naming
# an object that is not present does not come back empty -- it comes back
# matched to the backdrop. Measured on this bench: "a rubber duck" grounds to a
# box covering 30% of the wrist view (the bench itself) at score 0.44, while
# "the blue cube" covers 1.3% at 0.54. Score cannot separate those; area can.
#
# This is the guard for phrases with no size prior. A caller passing
# ``max_size_m`` already rejects the bench on physical grounds; without one,
# nothing else would, and the arm would drive to the middle of the table
# believing it had found a duck.
#
# It is a heuristic about *wrist* cameras: they work at ~300 mm, where anything
# graspable is a small part of the frame. A camera close enough for a bin to
# fill the view would need this raised.
DEFAULT_MAX_AREA_FRAC = 0.20


@dataclass
class Target:
    """A grounded phrase, resolved to a physical object."""

    phrase: str
    """The phrase this was matched to."""

    score: float
    """Grounding confidence, 0-1."""

    box: tuple[float, float, float, float]
    """``(x0, y0, x1, y1)`` pixels, from the grounder."""

    mask: np.ndarray = field(repr=False)
    """Boolean ``(H, W)``: the pixels judged to be the object itself."""

    centroid_px: tuple[float, float]
    """Centroid of the masked pixels -- on the object, not the box centre."""

    position_cam: np.ndarray
    """Median 3D point of the object, camera optical frame, metres."""

    position_world: Optional[np.ndarray]
    """Same point in the robot base frame, or ``None`` with no camera pose."""

    size_m: tuple[float, float]
    """Robust lateral extent ``(width, height)`` in metres. See :meth:`max_size_m`."""

    n_pixels: int
    """How many depth pixels backed the estimate. Small means low confidence."""

    grasps: list = field(default_factory=list, repr=False)
    """GG-CNN grasps whose pixel falls inside :attr:`mask`, best first."""

    @property
    def max_size_m(self) -> float:
        """Largest lateral dimension -- the number that separates cube from bin.

        Measured from the depth points inside the mask at the 5th/95th
        percentiles, so a few stray pixels at the bench's depth cannot inflate
        it. It is a lateral extent as seen from the camera, not a true bounding
        box: an object viewed obliquely reads slightly small. Good to a few
        millimetres for anything roughly facing the camera, which is the case
        for a wrist camera looking down.
        """
        return float(max(self.size_m))


class LanguageTargeter:
    """Phrase in, grasp out, via grounding + depth + GG-CNN.

    Parameters
    ----------
    grounder:
        Any object with ``detect(image, phrases) -> list[Detection]``. Defaults
        to Grounding DINO, built lazily on first use so importing this module
        stays cheap.
    detector:
        A ``GGCNNDetector``. Also lazy. Pass ``None`` to
        :meth:`target` if you only want the object located, not grasped.
    """

    def __init__(self, grounder=None, detector=None, device: str = "auto",
                 depth_band_m: float = DEFAULT_DEPTH_BAND_M):
        self._grounder = grounder
        self._detector = detector
        self._device = device
        self.depth_band_m = depth_band_m

    @property
    def grounder(self):
        if self._grounder is None:
            self._grounder = GroundingDINOGrounder(device=self._device)
        return self._grounder

    @property
    def detector(self):
        if self._detector is None:
            from ..grasp import GGCNNDetector
            self._detector = GGCNNDetector()
        return self._detector

    # -- stage 2: depth ---------------------------------------------------

    def _physicalise(self, frame: RGBDFrame,
                     det: Detection) -> Optional[Target]:
        """Turn a 2D box into a measured object, or ``None`` if depth cannot."""
        h, w = frame.depth.shape
        x0, y0, x1, y1 = det.box
        c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c1, r1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if c1 <= c0 or r1 <= r0:
            return None

        patch = frame.depth[r0:r1, c0:c1]
        valid = np.isfinite(patch)
        if valid.sum() < MIN_REGION_PIXELS:
            return None

        # The object is the near part of the box. Using the *lower quartile* of
        # depth as the reference rather than the median matters: a box around a
        # small object is mostly the surface behind it, so the median is the
        # bench and a band around it would mask the bench instead of the object.
        near = float(np.percentile(patch[valid], 25))
        keep = valid & (np.abs(patch - near) <= self.depth_band_m)
        if keep.sum() < MIN_REGION_PIXELS:
            return None

        mask = np.zeros((h, w), dtype=bool)
        mask[r0:r1, c0:c1] = keep

        rows, cols = np.nonzero(mask)
        depths = frame.depth[rows, cols]
        k = frame.intrinsics
        xs = (cols - k.cx) * depths / k.fx
        ys = (rows - k.cy) * depths / k.fy
        pts = np.stack([xs, ys, depths], axis=1).astype(np.float64)

        # Percentile spread, not min/max: robust to the handful of pixels that
        # survive the depth band at an edge.
        lo = np.percentile(pts[:, :2], 5, axis=0)
        hi = np.percentile(pts[:, :2], 95, axis=0)
        size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))

        centroid = np.median(pts, axis=0)
        position_world = None
        if frame.cam_to_world is not None:
            position_world = (frame.cam_to_world @ np.append(centroid, 1.0))[:3]

        return Target(
            phrase=det.phrase, score=det.score, box=det.box, mask=mask,
            centroid_px=(float(cols.mean()), float(rows.mean())),
            position_cam=centroid, position_world=position_world,
            size_m=size, n_pixels=int(mask.sum()),
        )

    # -- public API -------------------------------------------------------

    def target(self, phrase: str, frame: RGBDFrame, *,
               max_size_m: Optional[float] = None,
               min_size_m: Optional[float] = None,
               max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
               distractors: Sequence[str] = (),
               attach_grasps: bool = True,
               top_k_grasps: int = 12) -> Optional[Target]:
        """Locate the single object best matching ``phrase``.

        Parameters
        ----------
        max_size_m / min_size_m:
            Physical extent bounds, metres. This is the cube-versus-bin
            discriminator -- ``max_size_m=0.08`` keeps a 40 mm cube and drops a
            150 mm bin that grounded equally well. Omit both to rank on
            grounding score and image area alone.
        max_area_frac:
            Reject candidates whose box covers more than this fraction of the
            frame. Guards against the backdrop: an absent object does not ground
            to nothing, it grounds to the bench. See
            :data:`DEFAULT_MAX_AREA_FRAC`. Pass ``1.0`` to disable.
        distractors:
            Extra phrases to ground alongside ``phrase``. Grounding DINO
            calibrates better when the prompt names the alternatives, so passing
            ``("a bin", "a cup")`` sharpens the score for the thing you want
            even though those detections are then discarded.
        attach_grasps:
            Populate ``Target.grasps`` with the GG-CNN grasps inside the region.

        Returns ``None`` when nothing grounds, or when everything that grounds
        is filtered out by the size bounds -- never a best-effort guess, because
        a wrong target here becomes the arm moving to the wrong object.
        """
        phrases = [phrase, *distractors]
        detections = self.grounder.detect(frame.color, phrases)
        if not detections:
            return None

        wanted = phrase.strip().lower().rstrip(".")
        frame_area = float(frame.color.shape[0] * frame.color.shape[1])
        targets: list[Target] = []
        for det in detections:
            # Keep only boxes matched to the phrase we asked about. Distractors
            # did their job by being in the prompt. Exact match also discards
            # Grounding DINO's merged labels ("a bin a cup"), which appear when
            # adjacent prompt phrases share attention and describe neither.
            if det.phrase.strip().lower().rstrip(".") != wanted:
                continue
            if det.area_px > max_area_frac * frame_area:
                continue
            t = self._physicalise(frame, det)
            if t is None:
                continue
            if max_size_m is not None and t.max_size_m > max_size_m:
                continue
            if min_size_m is not None and t.max_size_m < min_size_m:
                continue
            targets.append(t)

        if not targets:
            return None
        targets.sort(key=lambda t: t.score, reverse=True)
        best = targets[0]

        if attach_grasps:
            best.grasps = self._grasps_in(frame, best, top_k_grasps)
        return best

    def _grasps_in(self, frame: RGBDFrame, target: Target,
                   top_k: int) -> list:
        """GG-CNN grasps whose pixel lands on the target's mask."""
        grasps = self.detector.detect(frame, top_k=top_k)
        inside = []
        h, w = target.mask.shape
        for g in grasps:
            u, v = int(round(g.pixel[0])), int(round(g.pixel[1]))
            if 0 <= v < h and 0 <= u < w and target.mask[v, u]:
                inside.append(g)
        return inside

    def grasp_for(self, phrase: str, frame: RGBDFrame, **kwargs):
        """The one-liner: phrase in, best grasp on that object out.

        ``None`` if the phrase does not ground, if size filtering rejects
        everything, or if GG-CNN proposes no grasp on the object it found.

        That last case returns ``None`` rather than falling back to the region
        centroid, deliberately. A centroid is a *position*, not a grasp: it
        carries no jaw angle and no evidence that the gripper can close there.
        Returning one shaped like a grasp would let a caller act on a value
        nothing verified. Callers who want the position anyway should call
        :meth:`target` and read ``position_world``, which says what it is.
        """
        target = self.target(phrase, frame, **kwargs)
        if target is None or not target.grasps:
            return None
        return target.grasps[0]

    # -- visualisation ----------------------------------------------------

    def draw(self, frame: RGBDFrame, target: Optional[Target]) -> np.ndarray:
        """Colour image with the target's mask, box and best grasp drawn on."""
        import cv2

        img = frame.color.copy()
        if target is None:
            cv2.putText(img, "no target", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 80, 80), 2, cv2.LINE_AA)
            return img

        tint = np.zeros_like(img)
        tint[target.mask] = (0, 190, 255)
        img = cv2.addWeighted(img, 1.0, tint, 0.35, 0)

        x0, y0, x1, y1 = (int(v) for v in target.box)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 190, 255), 2)
        cv2.putText(img,
                    f"{target.phrase} {target.score:.2f} "
                    f"{target.max_size_m * 1000:.0f}mm",
                    (x0, max(16, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 190, 255), 1, cv2.LINE_AA)

        if target.grasps:
            from ..grasp.ggcnn import GGCNNDetector  # noqa: F401  (draw style)
            g = target.grasps[0]
            import math
            u, v = g.pixel
            half = max(g.width_px / 2.0, 6.0)
            dx, dy = math.cos(g.angle_rad) * half, math.sin(g.angle_rad) * half
            cv2.line(img, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)),
                     (0, 255, 0), 2)
            cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0), -1)
        return img
