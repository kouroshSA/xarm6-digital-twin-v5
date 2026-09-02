#!/usr/bin/env python3
"""Record the wrist camera -- colour and colourised depth, side by side.

    python scripts/wrist_record.py --seconds 20 --out /tmp/wrist.mp4

Or drive it from another script while the arm works:

    rec = WristRecorder("/tmp/run.mp4", arm=arm)
    rec.start()
    ...                       # move the arm, grasp, place
    frame = rec.snapshot()    # a POSED frame, for detection
    rec.stop()

Depth is rendered **red near, blue far** over a fixed range, so colours mean the
same thing in every frame of every recording and two runs can be compared by
eye. An auto-scaled colourmap would repaint the whole scene every time something
entered or left the view, which looks informative and tells you nothing.

Invalid depth is drawn BLACK rather than as some distance. The D435i drops
returns on transparent and specular surfaces -- the glass dish on this bench
reads as a hole, not as a far surface -- and painting those pixels dark blue
would put a plausible distance on the screen where the sensor has no idea.

THREADING. The recorder thread owns the camera and is the only thing that
touches it. It captures with no arm attached, so a frame grab never calls into
the xArm SDK while the main thread is commanding motion. ``snapshot()`` is the
one exception: it attaches the arm just long enough for one posed capture, and
the caller is blocked while that happens, so the SDK still sees one thread.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Optional

sys.path.insert(0, os.getcwd())

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from perception.realsense_camera import RealSenseWristCamera  # noqa: E402

#: Depth range the colourmap spans, metres. Anything nearer than NEAR or further
#: than FAR clamps.
#:
#: Tuned to the volume the wrist actually works in, not to the sensor's range.
#: A first attempt at 0.15-0.90 was technically correct and visually useless:
#: the benchtop sits ~0.39 m from the lens, which landed mid-scale, so the whole
#: frame came out one flat yellow and the objects standing on it were
#: indistinguishable from it. Ending the scale at 0.60 spends the colours on the
#: 0.2-0.5 m band where the grasping happens; the floor past the bench edge
#: (~1.3 m) clamps to blue, which is all it needs to say.
NEAR_M = 0.15
FAR_M = 0.60


def colourise_depth(depth_m: np.ndarray, near: float = NEAR_M,
                    far: float = FAR_M) -> np.ndarray:
    """Depth in metres -> RGB, red near, blue far, black where invalid."""
    valid = np.isfinite(depth_m)
    d = np.clip(depth_m, near, far)
    # 255 at `near` so JET's red end lands on close surfaces; JET runs blue->red
    # over 0->255, so the scale has to be inverted rather than the map.
    norm = np.zeros(depth_m.shape, np.uint8)
    norm[valid] = (255.0 * (far - d[valid]) / (far - near)).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)[:, :, ::-1].copy()  # BGR->RGB
    vis[~valid] = 0
    return vis


def _label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


class WristRecorder:
    """Streams the wrist camera to an mp4 in a background thread."""

    #: Frames to time before opening the writer, to learn the loop's real rate.
    #: The container stamps ONE fps and cannot be told otherwise afterwards, so
    #: a guessed value makes every playback wrong: 15 gave 11.9, and throttling
    #: down to 10 gave 8.6 because each discarded frame still costs a frame
    #: time. Measuring first and declaring what we measured is the only version
    #: that produces a file whose timing matches the run.
    CALIBRATION_FRAMES = 12

    def __init__(self, out_path: str, arm=None, fps: float = None,
                 near: float = NEAR_M, far: float = FAR_M):
        self.out_path, self.arm = out_path, arm
        self.fps = None if fps is None else float(fps)
        self.near, self.far = near, far
        # arm=None: a plain frame grab must never call the SDK, because the main
        # thread is driving the arm at the same time.
        self.cam = RealSenseWristCamera(arm=None)
        self._stop = threading.Event()
        self._snap_req = threading.Event()
        self._snap_done = threading.Event()
        self._snap_frame = None
        self._thread: Optional[threading.Thread] = None
        self.frames_written = 0
        self._writer = None
        self._t0 = 0.0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "WristRecorder":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10.0)
        self.check_rate()

    def check_rate(self) -> float:
        """Report the rate actually achieved, and say so if the file lies."""
        if not self._t0 or not self.frames_written or not self.fps:
            return 0.0
        achieved = self.frames_written / max(time.time() - self._t0, 1e-6)
        if abs(achieved - self.fps) > 0.1 * self.fps:
            print(f"[WristRecorder] WARNING: wrote {achieved:.1f} fps but the "
                  f"file declares {self.fps:.1f}; playback will be off by "
                  f"{self.fps / achieved:.2f}x. Lower --fps.")
        return achieved

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- the one place the arm is read ------------------------------------

    def snapshot(self, timeout: float = 8.0):
        """One frame carrying ``cam_to_world``, for detection. Blocks."""
        self._snap_done.clear()
        self._snap_req.set()
        if not self._snap_done.wait(timeout):
            raise TimeoutError("recorder did not deliver a posed snapshot")
        return self._snap_frame

    # -- thread -----------------------------------------------------------

    def _run(self) -> None:
        calibrated = 0
        t_cal = None
        try:
            while not self._stop.is_set():
                if self._snap_req.is_set():
                    self._snap_req.clear()
                    self.cam.arm = self.arm          # attach for one capture
                    try:
                        self._snap_frame = self.cam.capture(align=True)
                    finally:
                        self.cam.arm = None
                    self._snap_done.set()
                    continue

                frame = self.cam.capture(align=True)
                colour = np.ascontiguousarray(frame.color)
                depth = colourise_depth(frame.depth, self.near, self.far)
                _label(colour, "wrist colour")
                _label(depth, f"depth  red {self.near:.2f}m -> blue {self.far:.2f}m"
                              f"   black = no return")
                side = np.hstack([colour, depth])

                if self._writer is None:
                    # Time the full loop -- capture, colourise, stack -- before
                    # committing an fps to the container.
                    if self.fps is None:
                        if t_cal is None:
                            t_cal = time.time()
                        calibrated += 1
                        if calibrated < self.CALIBRATION_FRAMES:
                            continue
                        measured = calibrated / max(time.time() - t_cal, 1e-6)
                        self.fps = float(np.clip(measured, 3.0, 30.0))
                        print(f"[WristRecorder] measured {measured:.1f} fps; "
                              f"recording at {self.fps:.1f}")
                    h, w = side.shape[:2]
                    self._writer = cv2.VideoWriter(
                        self.out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps, (w, h))
                    if not self._writer.isOpened():
                        raise RuntimeError(
                            f"could not open {self.out_path} for writing")
                    self._t0 = time.time()
                    self.frames_written = 0
                self._writer.write(side[:, :, ::-1])   # cv2 wants BGR
                self.frames_written += 1
        finally:
            if self._writer is not None:
                self._writer.release()
            self.cam.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/wrist.mp4")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=None,
                    help="force an output fps; default measures the real one")
    args = ap.parse_args()

    rec = WristRecorder(args.out, fps=args.fps).start()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.2)
    rec.stop()
    print(f"  {rec.frames_written} frames -> {args.out} "
          f"({rec.frames_written / max(time.time()-t0, 1e-6):.1f} fps achieved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
