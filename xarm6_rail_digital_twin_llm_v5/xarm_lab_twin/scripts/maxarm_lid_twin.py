#!/usr/bin/env python
"""Petri-dish lids in the MaxArm's digital twin (room `maxarm_petri`).

    python scripts/maxarm_lid_twin.py run                      # every dish: lid off, aside, back on
    python scripts/maxarm_lid_twin.py run --dishes 2 --seed 3
    python scripts/maxarm_lid_twin.py run --video out.mp4      # 1280x720 from cam_maxarm, wrist view inset
    python scripts/maxarm_lid_twin.py run --view               # the interactive viewer instead
    python scripts/maxarm_lid_twin.py look                     # where every lid is, graded

The sequence is `scripts/maxarm_lid.py`'s, rehearsed on the twin with the
bench's measured repeatability (`hardware/maxarm/repeatability.py`):

1. LID OFF: hover over the dish, lower the cup to the lid, suction, lift,
   carry to the dish's set-aside spot, lower to the table, vent IN CONTACT
   (never from height: the lid is set down, not dropped), lift.
2. LID ON: the same back, releasing `PLACE_GAP_MM` above the seated height.
3. Grade PHYSICALLY (the lid's pose against the base's): seated, perched on
   the rim, or off. A perched lid is re-seated: lifted a little, shifted by
   the measured miss, set down again (at most `RESEAT_TRIES`).

The robot REMEMBERS where each dish is: the base's position is taken once,
at the start (the real script's first look), and every later move is made
from memory, not from a picture -- with the measured scatter and sag on
every stop, which is what makes the reseat step earn its place. The only
"vision" in the twin is the grader, and the re-seat's miss measurement,
which stand in for the real script's rim-fit camera check and are taken
from the physics (said so in the log).

Heights are GEOMETRY from `hardware/maxarm/petri.py` and the measured table
contact, as the real script's are (table contact + lid height); the real cup
presses 4 mm to seal, the twin's rigid cup touches.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hardware.maxarm import kinematics as kin  # noqa: E402
from hardware.maxarm import petri  # noqa: E402
from hardware.maxarm.real_maxarm import MaxArmError  # noqa: E402
from hardware.maxarm.sim_maxarm import SimMaxArmAPI  # noqa: E402

ROOM = "maxarm_petri"
CARRY_Z = 150.0          # scripts/maxarm_lid.py CARRY_Z
APPROACH_MM = 30.0       # slow from here down
PLACE_GAP_MM = 2.0       # release this far above the resting height (user: never drop it)
TOUCH_MM = 0.5           # the rigid cup stops this far above the lid; the real one presses 4 mm
FLOOR_MARGIN_MM = 3.0    # the real script's --floor-margin-mm
RESEAT_LIFT_MM = 20.0
RESEAT_TRIES = 2
SPEED_CARRY = 80.0       # mm/s
SPEED_APPROACH = 25.0
#: Grading, mm and degrees. A seated lid's skirt surrounds the base, so its
#: centre is within the clearance; perched = still over the base but high or
#: tilted; off = anywhere else.
SEAT_XY_TOL_MM = petri.CLEARANCE_MM + 0.5
SEAT_Z_TOL_MM = 1.0
SEAT_TILT_TOL_DEG = 1.5
ASIDE_TOL_MM = 15.0

FLOOR_Z = kin.TABLE_CONTACT_Z_MM + FLOOR_MARGIN_MM
Z_SEATED_TOP = kin.TABLE_CONTACT_Z_MM + petri.SEATED_LID_TOP_MM
Z_ALONE_TOP = kin.TABLE_CONTACT_Z_MM + petri.LID_ALONE_TOP_MM


# ---------------------------------------------------------------------------
# the twin
# ---------------------------------------------------------------------------
def connect(render: bool, seed: int):
    from sim.mujoco_env import SimXArmAPI
    arm = SimXArmAPI(room=ROOM, render=render)
    ma = SimMaxArmAPI(arm, "arm1", floor_z_mm=FLOOR_Z, seed=seed)
    return arm, ma


class Dishes:
    """Ground truth about the dishes, read from the physics. The robot's
    MEMORY is filled from it once, at the start; the GRADER reads it any time."""

    def __init__(self, arm, ma):
        self.arm, self.ma = arm, ma
        self.names = sorted({o.rsplit("_", 1)[0] for o in ma.welds if o.endswith("_base")})

    def _pose(self, body_name):
        bid = self.ma.body_ids[body_name]
        with self.arm.lock:
            return self.arm.data.xpos[bid].copy(), self.arm.data.xmat[bid].reshape(3, 3).copy()

    def base_xy_arm(self, dish: str):
        p, _ = self._pose(f"{dish}_base")
        x, y, _ = self.ma.world_to_arm(p)
        return (x, y)

    def lid_xyz_arm(self, dish: str):
        p, _ = self._pose(f"{dish}_lid")
        return self.ma.world_to_arm(p)

    def lid_state(self, dish: str) -> dict:
        bp, _ = self._pose(f"{dish}_base")
        lp, lR = self._pose(f"{dish}_lid")
        dxy = (lp[:2] - bp[:2]) * 1000.0
        dz = (lp[2] - bp[2]) * 1000.0 - petri.SEATED_LID_BOTTOM_MM
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(lR[2, 2])))))
        miss = float(np.linalg.norm(dxy))
        if miss <= SEAT_XY_TOL_MM and abs(dz) <= SEAT_Z_TOL_MM and tilt <= SEAT_TILT_TOL_DEG:
            verdict = "seated"
        elif miss <= petri.LID_OD_MM / 2 + 5 and dz > -SEAT_Z_TOL_MM:
            verdict = "perched"
        else:
            verdict = "off"
        # The miss in the ARM frame (x mirrored), so a correction can be commanded.
        miss_arm = self.ma.world_to_arm(bp + np.array([dxy[0], dxy[1], 0.0]) / 1000.0)
        base_arm = self.ma.world_to_arm(bp)
        return {"verdict": verdict, "miss_mm": miss, "dz_mm": dz, "tilt_deg": tilt,
                "miss_arm_mm": (miss_arm[0] - base_arm[0], miss_arm[1] - base_arm[1])}

    def lid_flat_on_table(self, dish: str) -> dict:
        lp, lR = self._pose(f"{dish}_lid")
        x, y, z = self.ma.world_to_arm(lp)
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(lR[2, 2])))))
        return {"xy": (x, y), "z_mm": z, "tilt_deg": tilt, "flat": tilt <= SEAT_TILT_TOL_DEG and abs(z) <= 1.0}


# ---------------------------------------------------------------------------
# the moves (the real script's, on the twin's API)
# ---------------------------------------------------------------------------
def hover(ma, xy, z=CARRY_Z, speed=SPEED_CARRY):
    ma.set_position(xy[0], xy[1], z, speed=speed)


def descend_to(ma, xy, z_contact):
    """Fast to APPROACH_MM above, slow the rest (the real script's approach)."""
    ma.set_position(xy[0], xy[1], z_contact + APPROACH_MM, speed=SPEED_CARRY)
    ma.set_position(xy[0], xy[1], z_contact, speed=SPEED_APPROACH)


def pick(ma, xy, z_top, what: str) -> bool:
    """Cup onto the top at `z_top`, suction, lift. True if the twin's weld holds something."""
    hover(ma, xy)
    descend_to(ma, xy, z_top + TOUCH_MM)
    ma.close_gripper()
    ma._wait_sim(0.3)
    held = ma.held
    hover(ma, xy, speed=SPEED_APPROACH)
    print(f"  pick {what} at ({xy[0]:.0f}, {xy[1]:.0f}): {'holding ' + held if held else 'NOTHING held'}")
    return held is not None


def place(ma, xy, z_rest_top, what: str):
    """Lower the held lid until its underside is PLACE_GAP_MM above where it
    will rest, vent, lift. Never releases higher than that."""
    z = z_rest_top + PLACE_GAP_MM
    hover(ma, xy)
    descend_to(ma, xy, z)
    ma.open_gripper()           # vents and waits the vendor's 1.1 s
    ma._wait_sim(0.4)
    hover(ma, xy, speed=SPEED_APPROACH)
    print(f"  placed {what} at ({xy[0]:.0f}, {xy[1]:.0f}), released {PLACE_GAP_MM:.0f} mm up")


def abort_holding(ma, xy, z_top):
    """Something went wrong while holding: set it down where it was picked.
    The weld would carry it home otherwise, and a release at height is a drop."""
    try:
        place(ma, xy, z_top, "the lid (aborting)")
    finally:
        ma.open_gripper(wait=False)


# ---------------------------------------------------------------------------
# the task
# ---------------------------------------------------------------------------
def lid_off(ma, dishes: Dishes, dish: str, memory: dict) -> bool:
    base_xy = memory[dish]["base_xy"]
    aside = memory[dish]["aside_xy"]
    print(f"[{dish}] lid off -> aside at ({aside[0]:.0f}, {aside[1]:.0f})")
    if not pick(ma, base_xy, Z_SEATED_TOP, f"{dish}'s lid"):
        ma.open_gripper(wait=False)
        return False
    place(ma, aside, Z_ALONE_TOP, f"{dish}'s lid")
    st = dishes.lid_flat_on_table(dish)
    err = math.dist(st["xy"], aside)
    ok = st["flat"] and err <= ASIDE_TOL_MM
    print(f"  aside check (physics): lid at ({st['xy'][0]:.1f}, {st['xy'][1]:.1f}), "
          f"{err:.1f} mm from the spot, tilt {st['tilt_deg']:.1f} deg -> {'ok' if ok else 'NOT FLAT / OFF'}")
    memory[dish]["lid_aside_xy"] = aside          # what the robot remembers, not what it saw
    return ok


def lid_on(ma, dishes: Dishes, dish: str, memory: dict) -> str:
    base_xy = memory[dish]["base_xy"]
    aside = memory[dish].get("lid_aside_xy", memory[dish]["aside_xy"])
    print(f"[{dish}] lid back on at remembered ({base_xy[0]:.1f}, {base_xy[1]:.1f})")
    if not pick(ma, aside, Z_ALONE_TOP, f"{dish}'s lid"):
        ma.open_gripper(wait=False)
        return "not picked"
    place(ma, base_xy, Z_SEATED_TOP, f"{dish}'s lid")
    ma._wait_sim(0.5)
    st = dishes.lid_state(dish)
    print(f"  seat check (physics): {st['verdict']}, miss {st['miss_mm']:.1f} mm, "
          f"height {st['dz_mm']:+.1f} mm, tilt {st['tilt_deg']:.1f} deg")
    tries = 0
    target = base_xy
    while st["verdict"] == "perched" and tries < RESEAT_TRIES:
        tries += 1
        mx, my = st["miss_arm_mm"]
        # The lid sits off by (mx, my): take it where it is, put it back shifted the other way.
        lx, ly, lz = dishes.lid_xyz_arm(dish)      # lz: the lid's skirt bottom above the table
        z_top = kin.TABLE_CONTACT_Z_MM + lz + petri.LID_HEIGHT_MM   # as a WRIST height
        print(f"  reseat {tries}/{RESEAT_TRIES}: lid is ({mx:+.1f}, {my:+.1f}) mm off the base "
              "(the camera's rim-fit measurement, from the physics here)")
        ma.set_position(lx, ly, z_top + RESEAT_LIFT_MM, speed=SPEED_CARRY)
        ma.set_position(lx, ly, z_top + TOUCH_MM, speed=SPEED_APPROACH)
        ma.close_gripper()
        ma._wait_sim(0.3)
        if ma.held is None:
            print("  reseat: the cup took nothing (tilted lid); giving up")
            break
        ma.set_position(lx, ly, z_top + RESEAT_LIFT_MM, speed=SPEED_APPROACH)
        target = (target[0] - mx, target[1] - my)
        ma.set_position(target[0], target[1], Z_SEATED_TOP + RESEAT_LIFT_MM, speed=SPEED_CARRY)
        ma.set_position(target[0], target[1], Z_SEATED_TOP + PLACE_GAP_MM, speed=SPEED_APPROACH)
        ma.open_gripper()
        ma._wait_sim(0.5)
        hover(ma, target, speed=SPEED_APPROACH)
        st = dishes.lid_state(dish)
        print(f"  seat check (physics): {st['verdict']}, miss {st['miss_mm']:.1f} mm, "
              f"height {st['dz_mm']:+.1f} mm, tilt {st['tilt_deg']:.1f} deg")
    return st["verdict"]


def run(args) -> int:
    arm, ma = connect(args.view, args.seed)
    dishes = Dishes(arm, ma)
    picked = [f"dish{i}" for i in args.dishes] if args.dishes else dishes.names
    asides = petri.aside_spots_arm()
    # The robot's memory: one look at the start (here, read from the physics).
    memory = {d: {"base_xy": dishes.base_xy_arm(d), "aside_xy": asides[dishes.names.index(d)]}
              for d in picked}
    for d in picked:
        print(f"memory: {d} base at ({memory[d]['base_xy'][0]:.1f}, {memory[d]['base_xy'][1]:.1f}) mm, "
              f"aside spot ({memory[d]['aside_xy'][0]:.0f}, {memory[d]['aside_xy'][1]:.0f})")
    video = VideoWriter(arm, args.video, fps=args.fps) if args.video else None
    results = {}
    try:
        ma.snap_home()
        ma._wait_sim(0.5)
        if video:
            video.start()
            ma._wait_sim(1.0)
        for d in picked:
            results[d] = {"off": lid_off(ma, dishes, d, memory)}
        ma.go_home(speed=SPEED_CARRY)
        ma._wait_sim(0.8)
        for d in picked:
            results[d]["on"] = lid_on(ma, dishes, d, memory)
        ma.go_home(speed=SPEED_CARRY)
        ma._wait_sim(1.5)
    except MaxArmError as e:
        print(f"ABORT: {e}")
        if ma.held:
            abort_holding(ma, memory[picked[0]]["base_xy"], Z_SEATED_TOP)
        results["error"] = str(e)
    finally:
        if video:
            path = video.stop()
            print(f"video: {path} ({video.frames} frames, {video.frames / args.fps:.1f} s)")
    print("\nresult:")
    ok = True
    for d in picked:
        r = results.get(d, {})
        st = dishes.lid_state(d)
        line = f"  {d}: lid off {'ok' if r.get('off') else 'FAILED'}, back on: {r.get('on', '-')}" \
               f" (now {st['verdict']}, miss {st['miss_mm']:.1f} mm, tilt {st['tilt_deg']:.1f} deg)"
        print(line)
        ok = ok and r.get("off") is True and st["verdict"] == "seated"
    if "error" in results:
        ok = False
    print("ALL LIDS SEATED" if ok else "NOT COMPLETE")
    if args.view:
        print("viewer open; Ctrl-C to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    arm.disconnect()
    return 0 if ok else 1


def look(args) -> int:
    arm, ma = connect(False, args.seed)
    dishes = Dishes(arm, ma)
    ma._wait_sim(0.5)
    for d in dishes.names:
        st = dishes.lid_state(d)
        bx, by = dishes.base_xy_arm(d)
        print(f"{d}: base ({bx:.1f}, {by:.1f}); lid {st['verdict']}, miss {st['miss_mm']:.1f} mm, "
              f"tilt {st['tilt_deg']:.1f} deg")
    arm.disconnect()
    return 0


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------
class VideoWriter:
    """Renders cam_maxarm at `fps` of SIM time with the MaxArm's wrist camera
    inset, and pipes the frames to ffmpeg (H.264, yuv420p). Renderers live on
    the capture thread: EGL contexts are thread-affine (recording.py)."""

    def __init__(self, arm, path: str, fps: int = 30, size=(1280, 720), inset=(320, 240)):
        self.arm, self.path, self.fps, self.size, self.inset = arm, Path(path), fps, size, inset
        self.frames = 0
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Path:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        if self._error:
            print(f"[video] FAILED: {self._error}")
        return self.path

    def _loop(self):
        w, h = self.size
        iw, ih = self.inset
        try:
            main = mujoco.Renderer(self.arm.model, height=h, width=w)
            wrist = mujoco.Renderer(self.arm.model, height=ih, width=iw)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-", "-an", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(self.path)],
                stdin=subprocess.PIPE)
        except Exception as e:         # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
            return
        with self.arm.lock:
            next_t = float(self.arm.data.time)
        dt = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                with self.arm.lock:
                    t = float(self.arm.data.time)
                    if t < next_t:
                        due = False
                    else:
                        due = True
                        main.update_scene(self.arm.data, camera="cam_maxarm")
                        wrist.update_scene(self.arm.data, camera="arm1/cam_maxarm_wrist")
                if not due:
                    time.sleep(0.002)
                    continue
                frame = main.render().copy()
                small = wrist.render()
                frame[12:12 + ih + 4, 12:12 + iw + 4] = 30            # border
                frame[14:14 + ih, 14:14 + iw] = small
                proc.stdin.write(frame.tobytes())
                self.frames += 1
                next_t += dt
        except Exception as e:         # noqa: BLE001
            self._error = f"{type(e).__name__}: {e}"
        finally:
            try:
                proc.stdin.close()
                proc.wait(timeout=60)
            except Exception:          # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="lid off, aside and back on for each dish")
    r.add_argument("--dishes", type=int, nargs="*", help="dish numbers (default all)")
    r.add_argument("--seed", type=int, default=0, help="repeatability seed")
    r.add_argument("--video", help="write an mp4 of the run here (needs ffmpeg)")
    r.add_argument("--fps", type=int, default=30)
    r.add_argument("--view", action="store_true", help="interactive viewer (no EGL; not with --video)")
    r.set_defaults(fn=run)
    lk = sub.add_parser("look", help="grade every lid where it is")
    lk.add_argument("--seed", type=int, default=0)
    lk.set_defaults(fn=look)
    args = ap.parse_args(argv)
    if getattr(args, "view", False):
        os.environ.pop("MUJOCO_GL", None)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
