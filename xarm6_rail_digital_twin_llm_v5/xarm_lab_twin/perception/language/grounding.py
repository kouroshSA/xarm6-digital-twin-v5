"""Open-vocabulary grounding: a phrase and an image in, boxes out.

The backend is Grounding DINO (``IDEA-Research/grounding-dino-base``), which
takes free text rather than a fixed label set -- so "the blue cube" works without
anyone having trained a blue-cube detector. It is used here purely as a
*proposal* stage: it says roughly where in the image a phrase might be, and
:mod:`perception.language.targeter` decides which proposal is actually the
object, using depth.

That split is deliberate, and the reason is visible the first time you run it.
Asked for "a green cube" over this bench, Grounding DINO returns the green cube
*and* the green bin, both with respectable scores. It is not wrong -- a bin is a
green box -- and no amount of prompt tuning reliably separates them, because the
distinction is physical size and the image alone does not carry scale. Depth
does. So this module deliberately does not try to pick a winner.

Only the vision-language part lives here; nothing in this file knows about the
robot, the scene, or grasping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-base"

# Grounding DINO expects lowercase phrases separated by ". ". The published
# examples use that format and detection quality drops noticeably without it.
_PROMPT_SEP = ". "


@dataclass
class Detection:
    """One grounded region, in image space only. No 3D, no robot frame."""

    phrase: str
    """The prompt phrase this box was matched to."""

    score: float
    """Grounding confidence, 0-1. Comparable within one call, not across models."""

    box: tuple[float, float, float, float]
    """``(x0, y0, x1, y1)`` in full-resolution pixels."""

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area_px(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def build_prompt(phrases: Sequence[str]) -> str:
    """Join phrases into Grounding DINO's expected prompt format."""
    cleaned = [p.strip().lower().rstrip(".") for p in phrases if p and p.strip()]
    if not cleaned:
        raise ValueError("no phrases to ground")
    return _PROMPT_SEP.join(cleaned) + "."


class GroundingDINOGrounder:
    """Grounding DINO behind a small, stable interface.

    Parameters
    ----------
    device:
        ``"auto"`` (default) picks CUDA when the installed torch has it and CPU
        otherwise. Measured on this machine: 0.37 s per call on the RTX 3080
        against 4.90 s on CPU. The two agree on which objects they find, to
        3e-4 px and 8e-4 in score -- close, but not bitwise, so do not assert
        exact equality across devices. CUDA costs about 3 s more at
        construction, so the win only materialises if the grounder is built
        once and reused -- which is why it is a lazy property on
        LanguageTargeter rather than a per-call object.
    box_threshold / text_threshold:
        Grounding DINO's two thresholds. Kept low-ish by default because the
        targeter re-ranks with depth afterwards; filtering hard here would throw
        away the correct object when a distractor happens to score higher.
    """

    def __init__(self, device: str = "auto", box_threshold: float = 0.25,
                 text_threshold: float = 0.25, model_id: str = MODEL_ID):
        try:
            import torch
            from transformers import (AutoModelForZeroShotObjectDetection,
                                      AutoProcessor)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "language grounding needs transformers and torch:\n"
                "  pip install transformers"
            ) from exc

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(self.device).eval()

    def detect(self, image: np.ndarray, phrases: Sequence[str],
               max_detections: int = 20) -> list[Detection]:
        """Ground every phrase in ``image`` (RGB uint8). Best first.

        Returns every surviving box rather than one per phrase -- an image can
        legitimately contain two blue cubes, and picking early would hide that
        from the caller who has the depth to tell them apart.
        """
        from PIL import Image

        prompt = build_prompt(phrases)
        pil = Image.fromarray(image)
        inputs = self.processor(images=pil, text=prompt,
                                return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil.size[::-1]],
        )[0]

        # transformers renamed this between versions: v5 returns decoded strings
        # in "text_labels" while older releases put them in "labels". Reading
        # whichever is present keeps this working across both instead of
        # producing tensor labels that stringify into nonsense.
        labels = results.get("text_labels")
        if labels is None:
            labels = results["labels"]

        detections = [
            Detection(phrase=str(label), score=float(score),
                      box=tuple(float(v) for v in box))
            for label, score, box in zip(labels, results["scores"],
                                         results["boxes"])
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:max_detections]
