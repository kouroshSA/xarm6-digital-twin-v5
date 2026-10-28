# MIT License
#
# Copyright (c) 2023 Yuzhe Qin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Vendored from dexsuite/dex-retargeting
#   example/vector_retargeting/single_hand_detector.py
#   https://github.com/dexsuite/dex-retargeting/blob/3f56141bc8bd2760d5e452e382937269554ebb21/example/vector_retargeting/single_hand_detector.py
#
# CHANGES (this repo):
#   * Ported from `mp.solutions.hands` to the MediaPipe Tasks HandLandmarker.
#     mediapipe >= 0.10.30 removed `mp.solutions`, and the releases that still
#     have it need numpy < 2; this repo pins numpy 2.4.6. The model file is
#     downloaded on first use and checked against a pinned SHA-256.
#   * `detect()` returns keypoint_2d as a (21, 3) array of NORMALISED image
#     landmarks (x, y in 0..1, z relative depth) instead of a protobuf.
#   * `detect_all()` added: every hand in the frame, for the left-hand clutch
#     gesture (needs num_hands=2).
#   * Drawing re-implemented with cv2 (the solutions drawing utils are gone).
#   * The OPERATOR2MANO matrices, `parse_keypoint_3d`'s role,
#     `estimate_frame_from_hand_points` and the handedness inversion are
#     upstream's, unchanged.
"""Single-hand MediaPipe detector (vendored, see header)."""
from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np

from teleop_hand import config

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mediapipe" / "hand_landmarker.task"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the HandLandmarker model once; refuse a file whose hash is
    not the pinned one (a truncated download fails here, not as a cryptic
    TFLite error later)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hand] downloading {config.MEDIAPIPE_MODEL_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(config.MEDIAPIPE_MODEL_URL, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != config.MEDIAPIPE_MODEL_SHA256:
        raise RuntimeError(
            f"{path} has sha256 {digest}, expected {config.MEDIAPIPE_MODEL_SHA256}. "
            f"Delete it to re-download, or update the pin if Google replaced the model.")
    return path


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
        num_hands=1,
        video=True,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        self.video = video
        self._last_ts_ms = -1
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    # ---- inference -------------------------------------------------------
    def _process(self, rgb):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(rgb))
        if not self.video:
            return self.hand_detector.detect(img)
        # VIDEO mode needs strictly increasing timestamps.
        ts = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts
        return self.hand_detector.detect_for_video(img, ts)

    def detect_all(self, rgb) -> List[dict]:
        """Every detected hand: dicts with label (MediaPipe's, i.e. as if
        mirrored), score, kp2d (21,3 normalised), kp3d (21,3 world, metres)."""
        results = self._process(rgb)
        out = []
        for n, w, h in zip(results.hand_landmarks, results.hand_world_landmarks,
                           results.handedness):
            out.append({
                "label": h[0].category_name,
                "score": float(h[0].score),
                "kp2d": np.array([[p.x, p.y, p.z] for p in n], float),
                "kp3d": self.parse_keypoint_3d(w),
            })
        return out

    def detect(self, rgb, hands: Optional[List[dict]] = None):
        """Upstream's contract: (num_box, joint_pos, keypoint_2d,
        mediapipe_wrist_rot), or (0, None, None, None)."""
        hands = self.detect_all(rgb) if hands is None else hands
        if not hands:
            return 0, None, None, None

        desired = None
        for h in hands:
            if h["label"] == self.detected_hand_type:
                desired = h
                break
        if desired is None:
            return 0, None, None, None

        num_box = len(hands)
        keypoint_3d_array = desired["kp3d"] - desired["kp3d"][0:1, :]
        mediapipe_wrist_rot = self.estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, desired["kp2d"], mediapipe_wrist_rot

    def main_hand(self, hands: List[dict]) -> Optional[dict]:
        """The hand `detect` drives the arm with (same label rule)."""
        for h in hands:
            if h["label"] == self.detected_hand_type:
                return h
        return None

    def other_hand(self, hands: List[dict]) -> Optional[dict]:
        for h in hands:
            if h["label"] != self.detected_hand_type:
                return h
        return None

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: np.ndarray, img_size) -> np.ndarray:
        keypoint = np.asarray(keypoint_2d, float)[:, :2].copy()
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    # ---- drawing ---------------------------------------------------------
    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: np.ndarray, style="white"):
        import cv2
        h, w = image.shape[:2]
        px = SingleHandDetector.parse_keypoint_2d(keypoint_2d, (h, w)).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, tuple(px[a]), tuple(px[b]), (255, 255, 255), 2)
        for p in px:
            cv2.circle(image, tuple(p), 4, (48, 48, 255), -1)
        return image
