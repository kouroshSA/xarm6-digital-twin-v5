# teleop_hand/tracker.py
"""The tracker process (§2): camera -> MediaPipe -> palm pose + opening ->
HandFrame over UDP, one per camera frame (including frames with no hand).

Runs in the `xarm6sim_hand` env (MediaPipe). Imports nothing from the sim.

Sources:
  RealSenseSource  D435 colour + depth aligned to colour, spatial + temporal
                   filters. The default.
  WebcamSource     colour only: x/y from the palm's image position, z from the
                   hand's apparent size (`--calibrate-depth`). Marks frames
                   `approx`, so the receiver refuses `full` orientation.
  ReplaySource     a keypoint log (`--log-keypoints`) played back, for tests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from teleop_hand import config, gestures, palm
from teleop_hand.messages import UdpPublisher, make_frame


# ---------------------------------------------------------------------------
# sources: read() -> (rgb uint8 HxWx3, depth_m float32 HxW or None, K 3x3)
# ---------------------------------------------------------------------------
class RealSenseSource:
    def __init__(self, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT,
                 fps=config.CAMERA_FPS, serial: Optional[str] = None):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        prof = self.pipe.start(cfg)
        dev = prof.get_device()
        self.serial = dev.get_info(rs.camera_info.serial_number)
        self.usb = dev.get_info(rs.camera_info.usb_type_descriptor) \
            if dev.supports(rs.camera_info.usb_type_descriptor) else "?"
        self.depth_scale = dev.first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.0]])

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        d = self.temporal.process(self.spatial.process(d))
        rgb = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return rgb, depth, self.K

    def close(self):
        self.pipe.stop()


class WebcamSource:
    def __init__(self, index: int, width=config.CAMERA_WIDTH, height=config.CAMERA_HEIGHT):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {index}")
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height
        f = 0.9 * w            # nominal ~60 deg HFOV; only x/y scale depends on it
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        self.serial = f"webcam{index}"
        self.usb = "-"

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB), None, self.K

    def close(self):
        self.cap.release()


def webcam_palm(kp_px: np.ndarray, k: np.ndarray, depth_k: float):
    """§2.6: approximate palm position without depth."""
    size = float(np.linalg.norm(kp_px[9] - kp_px[0]))
    if size < 5:
        return None
    z = depth_k / size
    u, v = np.mean(kp_px[list(config.PALM_LANDMARKS)], axis=0)
    return palm.deproject(u, v, z, k)


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------
class HandTracker:
    def __init__(self, source, *, hand: str = "Right", aux_gesture: bool = False,
                 selfie: bool = True, publisher: Optional[UdpPublisher] = None,
                 depth_k: float = config.WEBCAM_DEPTH_K, detector=None,
                 retargeter=None, max_detect_width: int = 640, gestures_on: bool = False):
        from teleop_hand.detector import SingleHandDetector
        from teleop_hand.retarget import GripperRetargeter
        # selfie=True means "trust MediaPipe's handedness label". Upstream's
        # default (False) inverts it, assuming a mirrored image; on the D435
        # (unmirrored) that put the skeleton on the operator's LEFT hand --
        # measured 2026-10-25, and consistent with MediaPipe's own test images.
        self.source = source
        self.hand = hand
        self.aux_gesture = aux_gesture
        self.det = detector or SingleHandDetector(hand_type=hand, selfie=selfie,
                                                  num_hands=2 if aux_gesture else 1,
                                                  min_detection_confidence=0.6,
                                                  min_tracking_confidence=0.6)
        self.rt = retargeter or GripperRetargeter()
        self.pub = publisher
        self.depth_k = depth_k
        self.fist = palm.Debounce(config.FIST_DEBOUNCE_FRAMES)
        self.gestures_on = gestures_on
        self.gesture = gestures.GestureDebounce()
        self.max_detect_width = max_detect_width
        self.seq = 0
        self.fps = 0.0
        self._t_last = None
        self.last = None            # (frame, overlay dict) for --show
        self.log = None             # list of joint_pos when --log-keypoints

    def step(self):
        import cv2
        rgb, depth, k = self.source.read()
        t = time.monotonic()
        self.seq += 1
        h, w = rgb.shape[:2]
        det_in = rgb
        if w > self.max_detect_width:            # keep depth lookups at full res
            s = self.max_detect_width / w
            det_in = cv2.resize(rgb, (self.max_detect_width, int(h * s)))
        hands = self.det.detect_all(det_in)
        num, jp, kp2d, mp_rot = self.det.detect(det_in, hands)
        fist = False
        if self.aux_gesture:
            other = self.det.other_hand(hands)
            fist = self.fist.update(bool(other is not None and palm.is_fist(other["kp3d"])))

        g_label, ray_o, ray_d = gestures.NONE, None, None
        if self.gestures_on:
            main = self.det.main_hand(hands)
            raw = gestures.classify_gesture(main["kp3d"]) if main is not None else gestures.NONE
            g_label = self.gesture.update(raw)

        overlay = {"kp2d": kp2d, "pts": {}, "rot": None, "pos": None}
        if jp is None:
            f = make_frame(self.seq, t_mono=t, detected=False, hand=self.hand,
                           aux_hand_fist=fist)
        else:
            kp_px = kp2d[:, :2] * [w, h]
            opening = self.rt.opening(jp)
            from teleop_hand.retarget import pinch_distance_m
            raw_m = pinch_distance_m(jp)
            rot_mp = config.MEDIAPIPE_TO_CAMERA @ mp_rot
            if depth is not None:
                pos, rot_d, q, pts = palm.palm_from_depth(kp_px, depth, k)
                rot, src = palm.fuse(rot_mp, rot_d, q)
                approx = False
            else:
                pos, q, pts = webcam_palm(kp_px, k, self.depth_k), 0.0, {}
                rot, src, approx = rot_mp, "mediapipe", True
            if g_label == gestures.POINT and depth is not None:
                ray_o, ray_d = gestures.pointing_ray(main["kp3d"], kp_px, depth, k)
            overlay.update(pts=pts, rot=rot, pos=pos, q=q, opening=opening)
            f = make_frame(self.seq, t_mono=t, detected=True, hand=self.hand,
                           palm_pos_cam_m=pos, pos_valid=pos is not None,
                           palm_rot_cam=rot, rot_source=src, rot_quality=float(q),
                           opening=opening, opening_raw_m=raw_m,
                           aux_hand_fist=fist, confidence=1.0, approx=approx,
                           gesture=g_label, point_origin_cam_m=ray_o, point_dir_cam=ray_d)
            if self.log is not None:
                self.log.append({"t": t, "joint_pos": np.round(jp, 6).tolist(),
                                 "palm_pos": None if pos is None else np.round(pos, 5).tolist(),
                                 "rot_quality": round(float(q), 3)})
        if self.pub is not None:
            self.pub.send(f)
        if self._t_last is not None:
            dt = t - self._t_last
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-6)) if self.fps else 1.0 / max(dt, 1e-6)
        self._t_last = t
        self.last = (rgb, f, overlay, k)
        return f

    # ---- debug view ---------------------------------------------------------
    def render(self):
        import cv2
        from teleop_hand.detector import SingleHandDetector
        rgb, f, ov, k = self.last
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ov["kp2d"] is not None:
            SingleHandDetector.draw_skeleton_on_image(img, ov["kp2d"])
        if ov.get("pos") is not None:
            p = ov["pos"]
            c = (int(k[0, 0] * p[0] / p[2] + k[0, 2]), int(k[1, 1] * p[1] / p[2] + k[1, 2]))
            cv2.circle(img, c, 7, (0, 255, 255), 2)
            if ov.get("rot") is not None:
                for i, col in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    e = p + 0.06 * ov["rot"][:, i]
                    ce = (int(k[0, 0] * e[0] / e[2] + k[0, 2]), int(k[1, 1] * e[1] / e[2] + k[1, 2]))
                    cv2.line(img, c, ce, col, 2)
        txt = [f"{self.fps:4.1f} FPS  seq {f.seq}",
               "no hand" if not f.detected else
               f"open {f.opening:.2f}  q {f.rot_quality:.2f} ({f.rot_source})"
               + (f"  z {f.palm_pos_cam_m[2]:.3f} m" if f.pos_valid else "  NO DEPTH"),
               "LEFT FIST" if f.aux_hand_fist else "",
               f"gesture: {f.gesture}" + ("" if f.gesture != "point" or f.point_ray
                                          else " (no depth on knuckle)")
               if self.gestures_on else ""]
        for i, s in enumerate(txt):
            cv2.putText(img, s, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return img


def save_log(entries, path) -> None:
    Path(path).write_text(json.dumps({"fps_nominal": config.CAMERA_FPS,
                                      "frames": entries}, separators=(",", ":")))
