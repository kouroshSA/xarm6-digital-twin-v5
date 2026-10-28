# teleop_hand/messages.py
"""`HandFrame` and its localhost UDP transport. Stdlib only.

One datagram per camera frame, JSON, including frames with no hand
(`detected=false`): silence on the socket must mean "tracker dead", never
"no hand". `t_mono` is the tracker's `time.monotonic()`; both processes run on
one machine, so the receiver can subtract it from its own clock to get the
message age (and the capture-to-command latency, test 15).
"""
from __future__ import annotations

import dataclasses
import json
import socket
import time
from typing import Optional

import numpy as np

from teleop_hand import config


@dataclasses.dataclass
class HandFrame:
    seq: int = 0
    t_mono: float = 0.0
    detected: bool = False
    hand: str = "Right"
    palm_pos_cam_m: Optional[list] = None        # [x, y, z], camera frame
    pos_valid: bool = False
    palm_rot_cam: Optional[list] = None          # 3x3, camera frame
    rot_source: str = "mediapipe"                # fused | depth | mediapipe
    rot_quality: float = 0.0
    opening: Optional[float] = None              # 0 closed .. 1 open
    opening_raw_m: Optional[float] = None        # thumb-index tip distance
    aux_hand_fist: bool = False
    #: Main-hand gesture, debounced by the tracker: none | fist | point.
    #: Only set when the tracker runs with --gestures.
    gesture: str = "none"
    point_origin_cam_m: Optional[list] = None    # index MCP, camera frame
    point_dir_cam: Optional[list] = None         # unit MCP->tip, camera frame
    confidence: float = 0.0
    approx: bool = False                         # webcam: no real depth
    v: int = config.MSG_VERSION

    # ---- convenience -----------------------------------------------------
    @property
    def point_ray(self):
        """(origin, unit direction) in the camera frame, or None."""
        if self.gesture != "point" or self.point_origin_cam_m is None \
                or self.point_dir_cam is None:
            return None
        return np.asarray(self.point_origin_cam_m, float), np.asarray(self.point_dir_cam, float)

    @property
    def valid(self) -> bool:
        """A frame the mapper may move the arm on."""
        return bool(self.detected and self.pos_valid
                    and self.palm_pos_cam_m is not None)

    def pos(self) -> np.ndarray:
        return np.asarray(self.palm_pos_cam_m, float)

    def rot(self) -> Optional[np.ndarray]:
        return None if self.palm_rot_cam is None else np.asarray(self.palm_rot_cam, float)

    # ---- wire format -----------------------------------------------------
    def to_json(self) -> bytes:
        d = dataclasses.asdict(self)
        return json.dumps(d, separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> "HandFrame":
        d = json.loads(raw)
        if d.get("v") != config.MSG_VERSION:
            raise ValueError(f"HandFrame version {d.get('v')!r}, expected {config.MSG_VERSION}")
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def _as_list(x):
    return None if x is None else np.asarray(x, float).tolist()


def make_frame(seq: int, **kw) -> HandFrame:
    """Build a frame from numpy values (tracker and tests)."""
    kw.setdefault("t_mono", time.monotonic())
    for k in ("palm_pos_cam_m", "palm_rot_cam", "point_origin_cam_m", "point_dir_cam"):
        if k in kw:
            kw[k] = _as_list(kw[k])
    return HandFrame(seq=seq, **kw)


class UdpPublisher:
    def __init__(self, host: str = config.UDP_HOST, port: int = config.UDP_PORT):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: HandFrame) -> None:
        self.sock.sendto(frame.to_json(), self.addr)

    def close(self) -> None:
        self.sock.close()


class UdpReceiver:
    """Non-blocking; `poll()` drains the socket and keeps only the newest
    frame. Stale frames are worth nothing to a servo loop."""

    def __init__(self, host: str = config.UDP_HOST, port: int = config.UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.n_received = 0
        self.n_bad = 0
        self.last_rx_mono: Optional[float] = None

    def poll(self) -> Optional[HandFrame]:
        newest = None
        while True:
            try:
                raw, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            try:
                f = HandFrame.from_json(raw)
            except (ValueError, TypeError, KeyError):
                self.n_bad += 1
                continue
            self.n_received += 1
            self.last_rx_mono = time.monotonic()
            if newest is None or f.seq >= newest.seq:
                newest = f
        return newest

    def close(self) -> None:
        self.sock.close()
