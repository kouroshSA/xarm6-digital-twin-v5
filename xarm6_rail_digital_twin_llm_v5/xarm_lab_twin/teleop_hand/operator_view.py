# teleop_hand/operator_view.py
"""`--operator-view` (§5.3): the twin seen from where the operator stands.

A second, offscreen-rendered window (OpenCV). The camera sits on the
operator's side of the bench (-y) looking toward +y, so with either preset a
hand moving right moves the arm right ON SCREEN. Overlays the clutch state,
modes, rot_quality, rail position and guard status.

The Renderer owns a GL context, and GL contexts are thread-affine (the
Recorder learned this the hard way: CLAUDE.md, "Frame rendering was broken"),
so it is built INSIDE the render thread, never in __init__. The teach-handle
marker (geom group 4) is switched on for this view so the target is visible.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import mujoco
import numpy as np

from sim.render_options import scene_option

#: Free camera: behind the operator's shoulder, looking at the arm.
CAM_LOOKAT = (0.15, 0.0, 0.95)
CAM_DISTANCE = 1.9
CAM_AZIMUTH = 90.0          # looking along +y
CAM_ELEVATION = -22.0


class OperatorView:
    def __init__(self, arm, status_fn: Callable[[], dict], width=800, height=600, hz=20.0,
                 title="operator view"):
        self.arm = arm
        self.status_fn = status_fn
        self.width, self.height, self.period = width, height, 1.0 / hz
        self.title = title
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        import cv2
        try:
            renderer = mujoco.Renderer(self.arm.model, self.height, self.width)
        except Exception as e:                      # noqa: BLE001
            self.error = e
            print(f"[HAND] operator view disabled: {e}")
            return
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = CAM_LOOKAT
        cam.distance, cam.azimuth, cam.elevation = CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION
        opt = scene_option()
        opt.geomgroup[4] = 1
        while not self._stop.is_set():
            t0 = time.time()
            with self.arm.lock:
                renderer.update_scene(self.arm.data, camera=cam, scene_option=opt)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            self._overlay(img, cv2)
            cv2.imshow(self.title, img)
            cv2.waitKey(1)
            time.sleep(max(0.0, self.period - (time.time() - t0)))
        renderer.close()
        cv2.destroyWindow(self.title)

    def _overlay(self, img, cv2) -> None:
        st = self.status_fn()
        engaged = st.get("engaged")
        fault = st.get("fault")
        if fault:
            head, col = f"FAULT {fault} -- recovery key, then clutch", (0, 0, 255)
        elif st.get("silent"):
            head, col = "TRACKER DEAD? (no messages)", (0, 0, 255)
        elif st.get("blocked"):
            head, col = f"REFUSED: {st['blocked']}", (0, 0, 255)
        elif engaged:
            head, col = ("TRACKING (precision)" if st.get("precision") else "TRACKING"), (0, 200, 255)
        elif st.get("need_release"):
            head, col = "HELD -- release and press the clutch", (0, 165, 255)
        else:
            head, col = "clutch released", (180, 180, 180)
        q = st.get("rot_quality")
        o = st.get("opening")
        lines = [
            head,
            f"map {st.get('map')}  orient {st.get('orient')}  rail {st.get('rail_mode')} "
            f"{st.get('rail_mm', 0):.0f} mm",
            f"rot_quality {'-' if q is None else f'{q:.2f}'}  opening "
            f"{'-' if o is None else f'{o:.2f}'}  gripper "
            f"{'CLOSED' if st.get('gripper_closed') else 'open'}",
            ("REC" if st.get("recording") else "") + ("  rot held: " + st["rot_held"]
                                                       if st.get("rot_held") else ""),
        ]
        for i, s in enumerate(lines):
            cv2.putText(img, s, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        col if i == 0 else (255, 255, 255), 2)
