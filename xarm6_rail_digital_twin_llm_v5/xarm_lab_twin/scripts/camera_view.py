#!/usr/bin/env python3
"""Look through the physical RealSense cameras, live, while you aim them.

    python scripts/camera_view.py                    # every camera plugged in
    python scripts/camera_view.py --camera observer  # just the OT-2 one
    python scripts/camera_view.py --depth            # colour + depth side by side
    python scripts/camera_view.py --shot /tmp/aim    # one still each, no window

This is an AIMING tool, not a perception one. It does no calibration, reads no
pose, and asserts nothing about intrinsics -- ``scripts/wrist_cam_check.py``
is the tool that does all of that. What this gives you is the picture, at video
rate, with a framing grid on it, so that pointing a camera at the bench is a
thing you do by looking rather than by guessing.

Colour only by default, deliberately. The wrist D435i's depth stream hangs on a
cold start (see ``perception/realsense_camera.py``) and recovering costs ten
seconds of hardware reset -- which is worth paying to measure something, and not
worth paying to point a camera. Pass ``--depth`` when you want it.

Keys in the window: ``q``/Esc quit, ``s`` save a still of every camera,
``g`` toggle the framing grid.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass, field

import pathlib

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    sys.exit("pyrealsense2 is not installed in this environment "
             "(conda activate xarm6sim)")

import cv2

# The two bodies in this cell, by serial. A serial is the only identifier that
# survives a replug -- USB port order does not, and both devices report a name
# beginning "Intel RealSense D435", so the name does not separate them either.
KNOWN = {
    "033422072806": "wrist",      # D435i, eye-in-hand on the gripper
    "825412070500": "observer",   # D435, fixed, on top of the OT-2
}

COLOR_WH = (640, 480)
DEPTH_WH = (640, 480)
FPS = 30


@dataclass
class Cam:
    serial: str
    role: str
    product: str
    usb: str = "?"
    has_color: bool = True
    pipe: object = None
    align: object = None
    want_depth: bool = False
    last: np.ndarray = field(default=None)

    @property
    def degraded(self) -> bool:
        """True when this device cannot deliver colour.

        Asks the question that matters -- is there an RGB sensor to stream --
        rather than the proxy it usually rides on. A D435i on a USB 2 link
        exposes no colour sensor at all: not a reduced mode, the RGB Camera is
        absent from the sensor list, so every colour request fails and nothing
        downstream can build an RGBDFrame.

        Checking the link type instead would have TWO blind spots. The D435 in
        this cell runs firmware 5.9.2, which does not implement the USB-type
        query at all and so reports "?"; and a missing RGB sensor from any other
        cause -- a part-detected device, a busy sensor -- would sail through.
        This asks the consumer's question, so it catches all of them.
        """
        return not self.has_color

    @property
    def label(self) -> str:
        warn = "  !! NO COLOUR SENSOR" if self.degraded else ""
        return f"{self.role}  {self.product}  {self.serial}  USB{self.usb}{warn}"


def discover(want: str | None) -> list[Cam]:
    cams = []
    for dev in rs.context().query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        product = dev.get_info(rs.camera_info.name).replace("Intel RealSense ", "")
        role = KNOWN.get(serial, "unknown")
        if want and want not in (role, serial):
            continue
        usb = (dev.get_info(rs.camera_info.usb_type_descriptor)
               if dev.supports(rs.camera_info.usb_type_descriptor) else "?")
        has_color = any(
            p.stream_type() == rs.stream.color
            for sensor in dev.sensors for p in sensor.profiles)
        cams.append(Cam(serial=serial, role=role, product=product,
                        usb=usb, has_color=has_color))
    return sorted(cams, key=lambda c: c.role)


def start(cam: Cam, want_depth: bool) -> None:
    cfg = rs.config()
    cfg.enable_device(cam.serial)
    cfg.enable_stream(rs.stream.color, *COLOR_WH, rs.format.bgr8, FPS)
    if want_depth:
        cfg.enable_stream(rs.stream.depth, *DEPTH_WH, rs.format.z16, FPS)
    cam.pipe = rs.pipeline()
    cam.pipe.start(cfg)
    cam.want_depth = want_depth
    cam.align = rs.align(rs.stream.color) if want_depth else None


def grab(cam: Cam, timeout_ms: int = 2000) -> np.ndarray | None:
    """One displayable BGR image, colour or colour|depth, or None on timeout."""
    try:
        frames = cam.pipe.wait_for_frames(timeout_ms)
    except RuntimeError:
        return None
    if cam.align is not None:
        frames = cam.align.process(frames)
    color = frames.get_color_frame()
    if not color:
        return None
    img = np.asanyarray(color.get_data())
    if cam.want_depth:
        depth = frames.get_depth_frame()
        if depth:
            d = np.asanyarray(depth.get_data())
            vis = cv2.applyColorMap(
                cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)
            vis[d == 0] = 0          # no reading is black, not dark blue
            img = np.hstack([img, vis])
    return img


def annotate(img: np.ndarray, cam: Cam, grid: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if grid:
        # Rule-of-thirds plus centre cross: aiming needs a reference that says
        # "the bench is low and left", which a bare picture does not give you.
        for f in (1 / 3, 2 / 3):
            cv2.line(out, (int(w * f), 0), (int(w * f), h), (60, 60, 60), 1)
            cv2.line(out, (0, int(h * f)), (w, int(h * f)), (60, 60, 60), 1)
        cx, cy = w // 2, h // 2
        cv2.line(out, (cx - 14, cy), (cx + 14, cy), (0, 255, 255), 1)
        cv2.line(out, (cx, cy - 14), (cx, cy + 14), (0, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(out, cam.label, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# The live window needs a cv2 built with GUI support. `xarm6sim` installs
# opencv-python-headless, so stills work there and windows do not;
# `ufactory_vision` has the Qt build and pyrealsense2 both.
GUI_ENV = "ufactory_vision"

# A RealSense on a USB 2 link exposes no colour sensor at all, so finding a port
# that negotiates USB 3 is a precondition for everything else -- and on a machine
# with a dock, a monitor hub and both USB-A and USB-C ports, which physical hole
# gives you that is not guessable. This reads sysfs rather than opening the
# device, so it costs nothing and works while a viewer is streaming.
SYSFS_USB = "/sys/bus/usb/devices"


def usb_links() -> list[tuple[str, str, int]]:
    """(sysfs path, product, link speed Mb/s) for every Intel RealSense."""
    out = []
    for entry in sorted(pathlib.Path(SYSFS_USB).glob("*")):
        try:
            if (entry / "idVendor").read_text().strip() != "8086":
                continue
            product = (entry / "product").read_text().strip()
            speed = int((entry / "speed").read_text().strip())
        except (OSError, ValueError):
            continue
        out.append((entry.name, product, speed))
    return out


def print_links() -> bool:
    """Report each camera's link. True if every one of them is on USB 3."""
    links = usb_links()
    if not links:
        print("  no RealSense device on the bus")
        return False
    ok = True
    for path, product, speed in links:
        good = speed >= 5000
        ok &= good
        # A path with dots is behind hub(s); docks and monitor hubs are a
        # common way to land on a 480M link while holding a USB 3 cable.
        hops = path.count(".")
        via = f" via {hops} hub{'s' if hops > 1 else ''}" if hops else " direct"
        print(f"  {'OK  ' if good else 'USB2'}  {product:<40s} "
              f"{speed:>5d} Mb/s  [{path}{via}]")
    return ok


def has_window_support() -> bool:
    """True if this cv2 build can open a window.

    ``xarm6sim`` installs opencv-python-*headless*, so every GUI entry point
    raises rather than returning an error code. Saving stills works there;
    only the live window does not.
    """
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__probe__")
        return True
    except cv2.error:
        return False


def save_all(cams: list[Cam], stem: str) -> list[str]:
    written = []
    for cam in cams:
        if cam.last is None:
            continue
        path = f"{stem}_{cam.role}_{cam.serial}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, cam.last)
        written.append(path)
    return written


def _install_signal_handlers() -> None:
    """Turn SIGTERM/SIGHUP into an exception so the ``finally`` runs.

    A RealSense device whose pipeline is never stopped can be left wedged: it
    drops off the USB bus and then refuses to enumerate ("device not accepting
    address, error -71"), which no amount of software fixes -- it needs a
    physical replug. Python runs ``finally`` on SIGINT but NOT on SIGTERM, so a
    plain ``kill`` of this script would skip every ``pipe.stop()``. Raising
    KeyboardInterrupt from the handler puts SIGTERM back on the SIGINT path.
    """
    def bail(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, bail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="role (wrist/observer) or serial; default all")
    ap.add_argument("--depth", action="store_true", help="show depth beside colour")
    ap.add_argument("--shot", metavar="STEM",
                    help="save one still per camera to STEM_<role>_<serial>.png "
                         "and exit -- no window")
    ap.add_argument("--probe", action="store_true",
                    help="report each camera's USB link and exit -- use this to "
                         "find a port that negotiates USB 3")
    ap.add_argument("--watch", action="store_true",
                    help="with --probe, keep reporting as you move cables")
    ap.add_argument("--warmup", type=int, default=15,
                    help="frames to discard so auto-exposure settles (default 15)")
    args = ap.parse_args()

    _install_signal_handlers()

    if args.probe:
        if not args.watch:
            return 0 if print_links() else 1
        print("watching -- move a camera between ports; Ctrl-C to stop\n")
        last = None
        try:
            while True:
                now = usb_links()
                if now != last:
                    print_links()
                    if now and all(sp >= 5000 for _, _, sp in now):
                        print("  ^ all cameras on USB 3\n")
                    else:
                        print()
                    last = now
                time.sleep(0.5)
        except KeyboardInterrupt:
            return 0

    cams = discover(args.camera)
    if not cams:
        print("no RealSense device matched"
              + (f" {args.camera!r}" if args.camera else ""), file=sys.stderr)
        return 1

    for cam in cams:
        if cam.degraded:
            print(f"REFUSING {cam.label}\n"
                  "  This device exposes no colour sensor, so there is no image\n"
                  "  to show and nothing downstream can build an RGBDFrame.\n"
                  + ("  Cause: it is on a USB 2 link. Move it to a USB 3 port\n"
                     "  (blue SS / USB-C) with the camera's own cable, then\n"
                     "  re-run.\n" if cam.usb.startswith("2") else
                     "  The USB link is not the cause -- replug and re-check.\n"),
                  file=sys.stderr)
            continue
        start(cam, args.depth)
        print(f"streaming {cam.label}")

    cams = [c for c in cams if c.pipe is not None]
    if not cams:
        print("no usable camera", file=sys.stderr)
        return 1

    try:
        # Auto-exposure opens over the first several frames; a still grabbed
        # before it settles is darker than what the camera actually sees.
        for _ in range(args.warmup):
            for cam in cams:
                grab(cam)

        if args.shot:
            for cam in cams:
                img = grab(cam)
                if img is None:
                    print(f"  {cam.role}: no frame", file=sys.stderr)
                    continue
                cam.last = annotate(img, cam, grid=True)
            for path in save_all(cams, args.shot):
                print(f"wrote {path}")
            return 0

        if not has_window_support():
            print("this OpenCV is headless -- no live window. Either:\n"
                  f"  conda run --no-capture-output -n {GUI_ENV} \\\n"
                  f"      python scripts/camera_view.py\n"
                  "  realsense-viewer                 (Intel's own GUI)\n"
                  "or use --shot to save stills.", file=sys.stderr)
            return 2

        grid = True
        win = "RealSense — q quit · s save · g grid"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            panels = []
            for cam in cams:
                img = grab(cam)
                if img is None:
                    continue
                cam.last = annotate(img, cam, grid)
                panels.append(cam.last)
            if panels:
                width = max(p.shape[1] for p in panels)
                panels = [cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1],
                                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
                          for p in panels]
                cv2.imshow(win, np.vstack(panels))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("g"):
                grid = not grid
            if key == ord("s"):
                stem = f"/tmp/camera_view_{int(time.time())}"
                for path in save_all(cams, stem):
                    print(f"wrote {path}")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        for cam in cams:
            if cam.pipe is None:
                continue
            try:
                cam.pipe.stop()
            except (RuntimeError, KeyboardInterrupt):
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
