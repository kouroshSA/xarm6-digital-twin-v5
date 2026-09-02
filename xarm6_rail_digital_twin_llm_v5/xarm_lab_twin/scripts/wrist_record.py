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

THE VIDEO IS NOT DATA. The right-hand panel is a rendering and it cannot be
inverted back into distances: it is clipped to [NEAR_M, FAR_M], quantised to 256
levels across that span (~1.8 mm each at the default range), pushed through a
colourmap that is not cleanly invertible, and then lossily compressed, which
moves colours ACROSS the map so neighbouring JET shades can mean very different
depths. Pass ``depth_h5=`` to keep the actual depth as well -- uint16
millimetres in HDF5, which is lossless and is what anything downstream
(measurement, VLA export, re-running a detector) needs.

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
#: THESE DEFAULTS ARE FOR CLOSE GRASPING WORK. Pass your own for anything else:
#: an oblique sweep down the bench spans 166-2462 mm and clamps almost entirely
#: to blue at this range. Pick the range by MEASURING, not by eye -- histogram
#: the depth from a previous run's .h5 and give the colourmap the band the
#: content actually occupies. On the 2026-09-02 sweep that was 340-1100 mm (the
#: benchtop, 87% of pixels), with the floor at 1300-1950 and an empty gap
#: between, so 0.35-1.10 put the whole gradient on the bench and let the floor
#: clamp. Two guesses got there the wrong way first: 0.15-0.60 left everything
#: blue, 0.35-1.80 saturated the bench red.
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

    #: Rate at which raw depth is kept when ``depth_h5`` is given. 10 Hz matches
    #: the /frames group in recording.py rather than inventing a second
    #: convention; the video stays at full rate either way.
    DEPTH_HZ = 10.0

    def __init__(self, out_path: str, arm=None, fps: float = None,
                 near: float = NEAR_M, far: float = FAR_M,
                 depth_h5: Optional[str] = None, depth_hz: float = DEPTH_HZ):
        self.out_path, self.arm = out_path, arm
        self.depth_h5, self.depth_hz = depth_h5, float(depth_hz)
        self._h5 = self._h5_depth = self._h5_t = None
        self._next_depth_t = 0.0
        self.depth_frames_written = 0
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

                # Before the writer block, so the calibration below times the
                # REAL workload. Measuring the loop without the depth write and
                # then enabling it dropped the achieved rate from 30 to 17.9
                # against a container already stamped 30 -- a file claiming a
                # speed it was never recorded at, which is the exact failure
                # the calibration exists to prevent.
                self._maybe_keep_depth(frame.depth)

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
            if self._h5 is not None:
                self._h5.close()
            self.cam.close()

    # -- raw depth --------------------------------------------------------

    def _maybe_keep_depth(self, depth_m: np.ndarray) -> None:
        """Append the depth frame itself, losslessly, at ``depth_hz``."""
        if not self.depth_h5:
            return
        now = time.time()
        if now < self._next_depth_t:
            return
        self._next_depth_t = max(now, self._next_depth_t) + 1.0 / self.depth_hz

        # uint16 millimetres with 0 = no return. That is the D435i's own wire
        # format and what RGBDFrame turns into NaN on the way in, so this stores
        # exactly what the sensor said with nothing invented and nothing lost.
        mm = np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0)
        mm = np.clip(mm, 0, 65535).astype(np.uint16)

        if self._h5 is None:
            import h5py
            h, w = mm.shape
            self._h5 = h5py.File(self.depth_h5, "w")
            self._h5_depth = self._h5.create_dataset(
                "depth_mm", shape=(0, h, w), maxshape=(None, h, w),
                dtype=np.uint16, chunks=(1, h, w),
                # Level 4, not 9. Depth compresses well regardless (large flat
                # regions and a lot of exact zeros), and 9 cost enough time per
                # frame to halve the video's frame rate for a few percent of
                # file size.
                compression="gzip", compression_opts=4)
            self._h5_t = self._h5.create_dataset(
                "t_wall", shape=(0,), maxshape=(None,), dtype=np.float64,
                compression="gzip")
            self._h5_depth.attrs["units"] = "millimetres"
            self._h5_depth.attrs["invalid"] = 0
            self._h5_depth.attrs["note"] = (
                "0 means the sensor returned nothing (transparent, specular or "
                "out of range) -- it is NOT a distance of zero")
            self._h5.attrs["depth_hz"] = self.depth_hz
            self._h5.attrs["video"] = os.path.basename(self.out_path)
            self._h5.attrs["width"], self._h5.attrs["height"] = w, h

        n = self._h5_depth.shape[0]
        self._h5_depth.resize(n + 1, axis=0)
        self._h5_depth[n] = mm
        self._h5_t.resize(n + 1, axis=0)
        self._h5_t[n] = now
        self.depth_frames_written += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/wrist.mp4")
    ap.add_argument("--depth-h5", default=None,
                    help="also keep raw depth, uint16 mm, losslessly")
    ap.add_argument("--depth-hz", type=float, default=WristRecorder.DEPTH_HZ)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=None,
                    help="force an output fps; default measures the real one")
    args = ap.parse_args()

    rec = WristRecorder(args.out, fps=args.fps, depth_h5=args.depth_h5,
                        depth_hz=args.depth_hz).start()
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.2)
    rec.stop()
    print(f"  {rec.frames_written} frames -> {args.out} "
          f"({rec.frames_written / max(time.time()-t0, 1e-6):.1f} fps achieved)")
    if args.depth_h5:
        print(f"  {rec.depth_frames_written} raw depth frames -> {args.depth_h5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
