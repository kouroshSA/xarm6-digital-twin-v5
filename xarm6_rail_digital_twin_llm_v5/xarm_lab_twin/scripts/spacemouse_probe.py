#!/usr/bin/env python3
"""Dump raw SpaceMouse events, so the button map can be *measured* not guessed.

    python scripts/spacemouse_probe.py                 # 30 s, human-readable
    python scripts/spacemouse_probe.py --seconds 60
    python scripts/spacemouse_probe.py --dump events.jsonl

This is the first deliverable of the SpaceMouse work order, and it exists
because two facts cannot be looked up:

1. **Button indices differ per model** (Compact has 2, Pro has 15) and the
   spnav protocol carries no model string -- ``SpnavButtonEvent`` gives a bare
   ``bnum``. The only way to learn which physical key is which index is to
   press them and watch.
2. **The axis permutation into the twin frame is a guess until confirmed.**
   Vendor code ships ``[[0,0,-1],[1,0,0],[0,1,0]]``, but that encodes *their*
   bench orientation. Pushing the cap forward must move the twin's TCP +X; if
   it does not, the matrix is wrong for us.

``--dump`` writes one JSON object per event so ``teleop_sm/test_device.py``
can replay a real session with no hardware attached.

Requires spacenavd running and libspnav installed -- see docs/spacemouse_setup.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def _open_or_explain():
    """Open the spnav socket, or exit with the actual remedy.

    Both failure modes here are environmental and have specific fixes, so a
    bare traceback would send the reader to the wrong place: a missing
    ``libspnav.so`` is an apt package, while a refused connection is the
    daemon not running.
    """
    try:
        import spnav
    except OSError as exc:
        sys.exit(
            f"cannot load the spnav client library ({exc}).\n"
            f"  sudo apt install libspnav0 libspnav-dev\n"
            f"(the `spacenavd` package is the DAEMON; libspnav is the client\n"
            f" library the Python binding dlopen()s, and installing the daemon\n"
            f" alone is not enough)")
    except ImportError as exc:
        sys.exit(f"the spnav Python binding is not installed ({exc}).\n"
                 f'  pip install "spnav @ git+https://github.com/kazoo-osaro/spnav"')

    try:
        spnav.spnav_open()
    except Exception as exc:  # noqa: BLE001 - spnav raises a bare SpnavConnectionError
        sys.exit(f"spnav_open() failed ({type(exc).__name__}: {exc}).\n"
                 f"  systemctl status spacenavd     # must be active\n"
                 f"  sudo systemctl enable --now spacenavd")
    return spnav


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="how long to listen (default 30)")
    ap.add_argument("--dump", default=None,
                    help="also write every event to this file as JSONL")
    ap.add_argument("--quiet-motion", action="store_true",
                    help="only print button events (useful for mapping keys)")
    args = ap.parse_args()

    spnav = _open_or_explain()
    from spnav import SpnavMotionEvent, SpnavButtonEvent

    sink = open(args.dump, "w") if args.dump else None
    print(f"listening for {args.seconds:.0f}s -- push/pull/twist the cap, then "
          f"press every button in turn.")
    print("axes are raw spnav counts; full scale is about +/-350.\n")

    t0 = time.time()
    n_motion = 0
    buttons_seen: dict[int, int] = {}
    axis_extremes = [0] * 6

    try:
        while time.time() - t0 < args.seconds:
            ev = spnav.spnav_poll_event()
            if ev is None:
                time.sleep(0.005)
                continue
            t = time.time() - t0

            if isinstance(ev, SpnavMotionEvent):
                n_motion += 1
                axes = list(ev.translation) + list(ev.rotation)
                for i, v in enumerate(axes):
                    if abs(v) > abs(axis_extremes[i]):
                        axis_extremes[i] = v
                if sink:
                    sink.write(json.dumps({"t": round(t, 4), "type": "motion",
                                           "translation": list(ev.translation),
                                           "rotation": list(ev.rotation)}) + "\n")
                # Motion events stream at high rate; printing every one buries
                # the button presses we are here to see.
                if not args.quiet_motion and n_motion % 12 == 0:
                    tx, ty, tz, rx, ry, rz = axes
                    print(f"  [{t:5.1f}s] motion  t=({tx:5d},{ty:5d},{tz:5d})  "
                          f"r=({rx:5d},{ry:5d},{rz:5d})")

            elif isinstance(ev, SpnavButtonEvent):
                if ev.press:
                    buttons_seen[ev.bnum] = buttons_seen.get(ev.bnum, 0) + 1
                if sink:
                    sink.write(json.dumps({"t": round(t, 4), "type": "button",
                                           "bnum": ev.bnum,
                                           "press": bool(ev.press)}) + "\n")
                print(f"  [{t:5.1f}s] BUTTON  bnum={ev.bnum:2d}  "
                      f"{'PRESS' if ev.press else 'release'}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        spnav.spnav_close()
        if sink:
            sink.close()

    print(f"\n--- summary ---")
    print(f"motion events      : {n_motion}")
    print(f"per-axis extreme   : {axis_extremes}   (tx,ty,tz,rx,ry,rz)")
    if buttons_seen:
        print(f"buttons pressed    : "
              + ", ".join(f"bnum {b} x{n}" for b, n in sorted(buttons_seen.items())))
    else:
        print("buttons pressed    : NONE -- press them next time; the map needs "
              "these indices")
    if args.dump:
        print(f"event dump         : {args.dump}")
    if n_motion == 0 and not buttons_seen:
        print("\nNo events at all. The daemon is reachable but saw nothing:\n"
              "  - is the device plugged in?  lsusb | grep -i 3dconnexion\n"
              "  - does another process hold it?  spacenavd binds one client set\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
