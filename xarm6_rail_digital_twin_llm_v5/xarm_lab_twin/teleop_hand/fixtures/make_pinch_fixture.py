#!/usr/bin/env python3
"""Build `pinch_sequence.json`, the keypoint log test 1 runs on.

    conda activate xarm6sim_hand
    python -m teleop_hand.fixtures.make_pinch_fixture <dir-with-images>

PROVENANCE -- read before trusting the fixture:
  * `static` frames are REAL MediaPipe output: `SingleHandDetector.detect()`
    on MediaPipe's own hand test images (google-ai-edge/mediapipe test
    assets, Apache-2.0: thumb_up, pointing_up, victory, fist, right_hands,
    ...). Only the derived 21x3 joint_pos numbers are stored, not the images.
  * `sequence` frames are SYNTHETIC: one real open-hand frame with the thumb
    tip (4) and index tip (8) moved toward their midpoint along a smooth
    open -> pinch -> open profile, the other 19 keypoints untouched. It is a
    stand-in until a real pinch is recorded from the operator camera
    (Phase 1, `run_hand_tracker.py --log-keypoints`), which should replace it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "pinch_sequence.json"
IMAGES = ("thumb_up.jpg", "pointing_up.jpg", "victory.jpg", "fist.jpg",
          "right_hands.jpg", "right_hands_rotated.jpg", "left_hands.jpg")
FPS = 30.0
#: Tip distance at the pinch, metres. MediaPipe tip landmarks sit at the
#: finger centres, so touching fingertips read ~1.5 cm, not zero.
PINCH_TIP_M = 0.015


def main(img_dir: str) -> int:
    import cv2
    from teleop_hand.detector import SingleHandDetector

    static = []
    seen = set()
    for hand in ("Right", "Left"):
        for selfie in (False, True):
            det = SingleHandDetector(hand_type=hand, selfie=selfie, video=False,
                                     num_hands=2, min_detection_confidence=0.5)
            for name in IMAGES:
                p = Path(img_dir) / name
                if not p.exists() or name in seen:
                    continue
                rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                _, jp, _, _ = det.detect(rgb)
                if jp is None:
                    continue
                seen.add(name)
                static.append({"source": name, "hand_type": hand, "selfie": selfie,
                               "joint_pos": np.round(jp, 6).tolist()})
    if not static:
        raise SystemExit(f"no hands found in {img_dir}")

    # The widest real hand is the open end of the synthetic pinch.
    base = max(static, key=lambda s: np.linalg.norm(np.subtract(s["joint_pos"][4],
                                                                s["joint_pos"][8])))
    jp0 = np.asarray(base["joint_pos"], float)
    mid = (jp0[4] + jp0[8]) / 2
    half_open = (jp0[8] - jp0[4]) / 2
    open_d = 2 * np.linalg.norm(half_open)
    seq = []
    n = int(4.0 * FPS)                                   # 4 s: open, pinch, open
    for k in range(n):
        t = k / FPS
        # 1.0 open -> 0.0 pinched at t=2 s -> 1.0 open, cosine profile
        w = 0.5 * (1 + np.cos(2 * np.pi * t / 4.0))
        d = PINCH_TIP_M + (open_d - PINCH_TIP_M) * w
        jp = jp0.copy()
        u = half_open / np.linalg.norm(half_open)
        jp[4] = mid - u * d / 2
        jp[8] = mid + u * d / 2
        seq.append({"t": round(t, 4), "joint_pos": np.round(jp, 6).tolist()})

    OUT.write_text(json.dumps({
        "provenance": __doc__.split("PROVENANCE -- read before trusting the fixture:")[1].strip(),
        "fps": FPS, "pinch_tip_m": PINCH_TIP_M, "open_source": base["source"],
        "static": static, "sequence": seq}, separators=(",", ":")))
    print(f"wrote {OUT}: {len(static)} real frames, {len(seq)} synthetic frames "
          f"(open {open_d * 1000:.0f} mm from {base['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
