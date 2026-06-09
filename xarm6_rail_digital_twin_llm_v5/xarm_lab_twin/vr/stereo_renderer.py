# vr/stereo_renderer.py
"""Offscreen rendering of the twin to JPEG frames, mono and stereo.

One module, two entry points (mono_frame / stereo_frames), backed by a
single reused ``mujoco.Renderer`` so we never juggle multiple GL contexts.
Set ``MUJOCO_GL=egl`` (headless GPU) before importing mujoco — the server
process does this in scripts/run_vr.py.

Locking pattern (critical): hold ``arm.lock`` only around ``update_scene``
(which reads ``data``), and release it before ``render()`` (which reads the
renderer's internal scene copy). Holding the lock across ``render()`` would
needlessly stall the physics stepper for the whole GPU draw.

The render loop (``run_loop``) writes the newest frame(s) into a single-slot,
latest-wins buffer (``LatestFrame``). The server reads that slot; frames are
never queued/backlogged, so a slow client always gets the freshest frame.
"""
from __future__ import annotations

import io
import threading
import time
from typing import Optional, Tuple

import mujoco
import numpy as np
from PIL import Image

from vr import config


def _jpeg(rgb: np.ndarray, quality: int = config.JPEG_QUALITY) -> bytes:
    """Encode an HxWx3 uint8 RGB array to JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class LatestFrame:
    """Thread-safe single-slot, latest-wins frame buffer.

    Holds either the mono JPEG (``mono``) or the stereo pair
    (``left`` / ``right``), plus a monotonically increasing sequence number so
    the consumer can tell whether a new frame is available.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mono: Optional[bytes] = None
        self._left: Optional[bytes] = None
        self._right: Optional[bytes] = None
        self._seq: int = 0

    def set_mono(self, jpeg: bytes) -> None:
        with self._lock:
            self._mono = jpeg
            self._left = None
            self._right = None
            self._seq += 1

    def set_stereo(self, left: bytes, right: bytes) -> None:
        with self._lock:
            self._left = left
            self._right = right
            self._mono = None
            self._seq += 1

    def get(self) -> Tuple[int, Optional[bytes], Optional[bytes], Optional[bytes]]:
        """-> (seq, mono, left, right). Unused fields are None."""
        with self._lock:
            return self._seq, self._mono, self._left, self._right


class TwinRenderer:
    """Renders the twin to JPEG. ``mode`` is "mono" or "stereo"."""

    def __init__(self, model, data, lock,
                 mode: str = None,
                 w: int = config.FRAME_WIDTH,
                 h: int = config.FRAME_HEIGHT):
        self.model, self.data, self.lock = model, data, lock
        self.mode = mode or config.MODE
        self.w, self.h = w, h
        # Single renderer, reused across all calls / both eyes. Created
        # LAZILY on first render: a mujoco.Renderer binds its (EGL/GLFW) GL
        # context to the thread that constructs it, and render() must run on
        # that same thread. We therefore defer creation to whichever thread
        # first calls _render() — the render loop thread under run_loop(), or
        # the main thread in the smoke test — rather than __init__'s thread.
        self.r: Optional[mujoco.Renderer] = None
        self._render_thread_id: Optional[int] = None

        # Free camera for mono mode, framed like the existing passive viewer
        # (matches recording.py's default frame camera).
        self.free_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.free_cam)
        self.free_cam.lookat[:] = (0.0, 0.15, 1.0)
        self.free_cam.distance = 2.8
        self.free_cam.azimuth = 135.0
        self.free_cam.elevation = -20.0

        # Resolve stereo camera ids once (fall back gracefully if absent).
        self._cam_left_id = self._cam_id("cam_left")
        self._cam_right_id = self._cam_id("cam_right")

        self.latest = LatestFrame()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_render_ms: float = 0.0

    def _cam_id(self, name: str) -> Optional[int]:
        try:
            return self.model.camera(name).id
        except (KeyError, ValueError):
            print(f"[VR] WARNING: camera '{name}' not in scene; "
                  f"stereo mode will fall back to the free camera.")
            return None

    def _ensure_renderer(self) -> None:
        """Create the renderer on first use, bound to the calling thread's GL
        context. Warn if a second thread later tries to render through it (the
        EGL context won't be current there)."""
        tid = threading.get_ident()
        if self.r is None:
            self.r = mujoco.Renderer(self.model, height=self.h, width=self.w)
            self._render_thread_id = tid
        elif self._render_thread_id != tid:
            print("[VR] WARNING: TwinRenderer used from a second thread "
                  f"({tid} != {self._render_thread_id}); GL context is "
                  "thread-affine and this will fail.")

    def _render(self, camera) -> np.ndarray:
        self._ensure_renderer()
        # Hold the lock only for the data read in update_scene; render() reads
        # the renderer's own scene copy and must NOT hold the sim lock.
        with self.lock:
            self.r.update_scene(self.data, camera=camera)
        return self.r.render()

    def mono_frame(self) -> bytes:
        return _jpeg(self._render(self.free_cam))

    def stereo_frames(self) -> Tuple[bytes, bytes]:
        left_cam = self._cam_left_id if self._cam_left_id is not None else self.free_cam
        right_cam = self._cam_right_id if self._cam_right_id is not None else self.free_cam
        return _jpeg(self._render(left_cam)), _jpeg(self._render(right_cam))

    # ---- render loop ------------------------------------------------------
    def render_once(self) -> None:
        """Render one frame (or pair) and publish it to the latest-wins slot."""
        t0 = time.time()
        if self.mode == "stereo":
            left, right = self.stereo_frames()
            self.latest.set_stereo(left, right)
        else:
            self.latest.set_mono(self.mono_frame())
        self._last_render_ms = (time.time() - t0) * 1000.0

    def run_loop(self) -> None:
        """Blocking render loop at ``config.RENDER_HZ``. Run in its own
        thread via :meth:`start`."""
        period = 1.0 / max(config.RENDER_HZ, 1e-6)
        self._running = True
        next_t = time.time()
        while self._running:
            try:
                self.render_once()
            except Exception as e:  # noqa: BLE001 - never let the loop die
                print(f"[VR] render error: {e}")
                time.sleep(0.1)
            next_t += period
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()
        # Close the renderer on the thread that created it (GL is thread-affine).
        if self.r is not None and self._render_thread_id == threading.get_ident():
            try:
                self.r.close()
            except Exception:
                pass
            self.r = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run_loop, daemon=True)
        self._thread.start()
        print(f"[VR] render loop started ({self.mode}, "
              f"{self.w}x{self.h} @ {config.RENDER_HZ:.0f}Hz)")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            # run_loop() closes the renderer on its own thread before exiting.
            self._thread.join(timeout=2.0)
            self._thread = None
        elif self.r is not None and self._render_thread_id == threading.get_ident():
            # No loop thread (e.g. smoke test rendered on this thread): close here.
            try:
                self.r.close()
            except Exception:
                pass
            self.r = None
