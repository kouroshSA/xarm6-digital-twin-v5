#!/usr/bin/env python3
"""Record a side-by-side demo: the lab scene, and what the wrist camera sees.

    python scripts/vision_demo.py --out /tmp/demo
    python scripts/vision_demo.py --out /tmp/demo --targets "the blue cube:blue_bin"

Left panel is a third-person view of the cell. Right panel is the wrist camera
at its real intrinsics, with the language-targeting overlay drawn on: the depth
mask of whatever phrase was asked for, its measured size, and the GG-CNN grasp.

Writes JPEG frames and, if ffmpeg is present, muxes them into an mp4.

Threading
---------
All rendering happens on one worker thread, which owns every GL context. The
main thread only commands motion (which blocks, because the motion primitives
pace themselves in wall-clock) and asks the worker to run a targeting pass by
setting ``request``. Two threads each holding their own EGL context works until
it doesn't; one owner is the version that keeps working.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.getcwd())
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

SCENE = "envs/lab_scene.xml"
SCENE_W, SCENE_H = 960, 720
PANEL_W, PANEL_H = 960, 720          # the wrist view is upscaled to match
FPS = 20

# How long the targeting overlay stays on screen after a look. The arm is
# stationary for this, so the frozen overlay matches what the camera can see;
# holding it through the approach would show a stale mask over a moved view.
OVERLAY_HOLD_S = 4.5


class Recorder(threading.Thread):
    """Owns the GL contexts: scene camera, wrist camera, and the targeter."""

    def __init__(self, arm, out_dir: str, device: str, backdrop: bool = False):
        super().__init__(daemon=True)
        self.backdrop = backdrop
        self.arm = arm
        self.out_dir = out_dir
        self.device = device
        self.running = True
        self.frame_idx = 0
        self.caption = "starting up"
        self.request: str | None = None      # phrase to target, set by main
        self.result = None                   # Sighting-ish, read by main
        self.result_ready = threading.Event()
        self.ready = threading.Event()
        self._overlay = None                 # last targeting overlay (RGB)
        self._overlay_until = 0.0

    # -- helpers ---------------------------------------------------------

    def _compose(self, scene_rgb, wrist_rgb):
        import cv2

        wrist = cv2.resize(wrist_rgb, (PANEL_W, PANEL_H),
                           interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((PANEL_H, SCENE_W + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :SCENE_W] = scene_rgb
        canvas[:, SCENE_W:] = wrist

        cv2.line(canvas, (SCENE_W, 0), (SCENE_W, PANEL_H), (40, 40, 40), 3)
        for x, text in ((16, "LAB SCENE"), (SCENE_W + 16, "WRIST CAMERA  D435i 640x480")):
            cv2.rectangle(canvas, (x - 8, 12), (x + 12 * len(text), 46),
                          (0, 0, 0), -1)
            cv2.putText(canvas, text, (x, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cap = self.caption
        cv2.rectangle(canvas, (0, PANEL_H - 46), (SCENE_W + PANEL_W, PANEL_H),
                      (0, 0, 0), -1)
        cv2.putText(canvas, cap, (16, PANEL_H - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (120, 230, 255), 2, cv2.LINE_AA)
        return canvas

    # -- the loop --------------------------------------------------------

    def run(self):
        import cv2
        import mujoco
        from PIL import Image

        from perception.language import LanguageTargeter
        from perception.sim_camera import SimWristCamera

        from sim.render_options import scene_option
        self._opt = scene_option(backdrop=self.backdrop)
        scene_r = mujoco.Renderer(self.arm.model, height=SCENE_H, width=SCENE_W)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.arm.model, cam)
        # Framed to take in the lab_environment backdrop -- the fume hood the
        # arm works in front of, the reagent shelving north, the window south --
        # rather than cropping to the bench as it did when there was nothing
        # behind it but void.
        cam.lookat[:] = (0.10, -0.08, 1.02)
        cam.distance, cam.azimuth, cam.elevation = 2.60, 118, -11

        wrist = SimWristCamera(self.arm, backdrop=self.backdrop)
        targeter = LanguageTargeter(device=self.device)
        _ = targeter.grounder            # pay the ~3 s model load before recording
        self.ready.set()

        period = 1.0 / FPS
        while self.running:
            t0 = time.time()

            if self.request is not None:
                phrase, self.request = self.request, None
                frame = wrist.capture()
                from agent.vision_targeting import (DEFAULT_DISTRACTORS,
                                                    size_bounds_for)
                lo, hi = size_bounds_for(phrase)
                target = targeter.target(phrase, frame, min_size_m=lo,
                                         max_size_m=hi,
                                         distractors=DEFAULT_DISTRACTORS)
                self.result = target
                if target is not None:
                    self._overlay = targeter.draw(frame, target)
                    self._overlay_until = time.time() + OVERLAY_HOLD_S
                self.result_ready.set()

            with self.arm.lock:
                scene_r.update_scene(self.arm.data, camera=cam,
                                     scene_option=self._opt)
            scene_rgb = scene_r.render()

            if self._overlay is not None and time.time() < self._overlay_until:
                wrist_rgb = self._overlay
            else:
                wrist_rgb = wrist.capture().color

            img = self._compose(scene_rgb, wrist_rgb)
            Image.fromarray(img).save(
                os.path.join(self.out_dir, f"f{self.frame_idx:05d}.jpg"),
                quality=88)
            self.frame_idx += 1

            time.sleep(max(0.0, period - (time.time() - t0)))

        scene_r.close()
        wrist.close()

    # -- main-thread API -------------------------------------------------

    def say(self, text: str):
        self.caption = text
        print(f"[demo] {text}")

    def look_for(self, phrase: str, timeout: float = 60.0):
        self.result_ready.clear()
        self.request = phrase
        if not self.result_ready.wait(timeout):
            return None
        return self.result

    def hold(self, seconds: float):
        time.sleep(seconds)



class _RecorderVision:
    """LLMBrain's vision, routed through the recorder thread.

    ``VisionTargeting`` would build its own ``SimWristCamera``, which means a
    second EGL context on the main thread while the recorder already holds one.
    That is the arrangement this script's docstring warns about. So the brain
    gets an object with the same surface whose ``locate`` hands the work to the
    thread that owns the GL context and waits for the answer -- one owner, and
    the overlay the recorder draws is the very frame the planner acted on rather
    than a re-capture that might not match.

    Subclasses the real thing so registry write-back and the size priors behave
    identically; only the capture path changes.
    """

    def __init__(self, rec, arm, registry):
        from agent.vision_targeting import VisionTargeting
        self._inner = VisionTargeting(arm, registry=registry)
        self.rec = rec

    # VisionTargeting's surface, as LLMBrain uses it.
    unavailable_reason = None

    def available(self) -> bool:
        return True

    @property
    def last(self):
        return self._inner.last

    def locate(self, description, *, max_size_m=None, min_size_m=None,
               want_grasp=True):
        import time as _t

        from agent.vision_targeting import Sighting

        target = self.rec.look_for(description)
        if target is None or target.position_world is None:
            return None

        grasp_pose = None
        quality = width_mm = 0.0
        if target.grasps:
            g = target.grasps[0]
            grasp_pose = g.to_arm_pose()
            quality = g.quality
            width_mm = g.width_m * 1000.0

        s = Sighting(
            description=description,
            position_mm=tuple(float(v) for v in target.position_world * 1000.0),
            size_mm=target.max_size_m * 1000.0,
            score=target.score,
            grasp_pose=grasp_pose,
            grasp_quality=quality,
            jaw_width_mm=width_mm,
            n_pixels=target.n_pixels,
            t_wall=_t.time(),
        )
        self._inner.last[description.strip().lower()] = s
        self._inner._write_back(s)
        return s

    def close(self):
        pass


def _caption_for(action, params) -> str:
    """A short line describing a dispatched command, for the demo's caption."""
    p = params or {}
    if action in ("grasp_object", "locate_object", "move_to_object"):
        return f'{action}: "{p.get("description", "?")}"'
    if action == "move_to":
        return (f"move_to ({p.get('x', 0):.0f}, {p.get('y', 0):.0f}, "
                f"{p.get('z', 0):.0f}) mm")
    if action == "set_rail":
        return f"rail -> {p.get('position_mm', 0):.0f} mm"
    if action == "done":
        return str(p.get("message", "done"))
    return action.replace("_", " ")


def run_llm_task(arm, rec, prompt: str, model: str = "haiku") -> int:
    """Let the planner drive, and record what it does.

    The plan is the model's, not this script's: nothing here decides the order,
    the heights or which object to look at.
    """
    from env_loader import load_env

    # Populate os.environ from .env before anthropic.Anthropic() reads it, the
    # same way run_task.py does. Without it the client constructs fine and only
    # fails at the first request, several seconds into a recording.
    load_env()

    from agent.llm_brain import LLMBrain
    from agent.object_registry import build_default_registry

    registry = build_default_registry()
    brain = LLMBrain(arm=arm, registry=registry, recorder=None, model=model)
    brain._vision = _RecorderVision(rec, arm, registry)

    original_dispatch = brain._dispatch

    def narrating_dispatch(action, params):
        rec.say(_caption_for(action, params))
        return original_dispatch(action, params)

    brain._dispatch = narrating_dispatch

    rec.say(f"planning with Claude {model}...")
    result = brain.execute_task(prompt)
    codes = [r["result"] for r in result.get("results", [])]
    failed = [c for c in codes if c != 0]

    # Grade the WORLD, not the return codes. A plan can return all zeros and
    # leave the objects somewhere else entirely -- that is the standard the
    # episode loop holds itself to, and a demo that only reported "all OK"
    # would be claiming more than it checked.
    outcome = arm.physical_outcome()
    print(f"[demo] physical outcome: {outcome}")

    headline = outcome.split(";")[0].strip() if outcome else "no outcome"
    rec.say(f"{len(codes)} commands, "
            f"{len(failed) if failed else 'all'} "
            f"{'failed' if failed else 'OK'}  |  {headline}")
    rec.hold(3.5)
    return 1 if failed else 0

def run_demo(arm, rec: Recorder, targets) -> int:
    """Home, then for each (phrase, bin): look, grasp, carry, release."""
    failures = 0

    rec.say("home pose")
    arm.go_home(wait=True)
    rec.hold(1.0)

    for phrase, bin_name in targets:
        rec.say(f'moving so the wrist camera can see the bench')
        arm.set_position(60.0, -200.0, 1180.0, 180.0, 0.0, 0.0,
                         speed=80, wait=True)
        rec.hold(0.6)

        rec.say(f'grounding "{phrase}" -> depth -> GG-CNN')
        target = rec.look_for(phrase)
        if target is None or target.position_world is None:
            rec.say(f'"{phrase}": not found')
            failures += 1
            rec.hold(1.5)
            continue

        p = target.position_world * 1000.0
        rec.say(f'"{phrase}" at ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm, '
                f'{target.max_size_m * 1000:.0f} mm across')
        rec.hold(OVERLAY_HOLD_S - 0.8)

        if not target.grasps:
            rec.say(f'no grasp proposed on "{phrase}" - refusing')
            failures += 1
            rec.hold(1.5)
            continue

        gx, gy, gz, roll, pitch, yaw = target.grasps[0].to_arm_pose()
        rec.say(f"approaching, yaw {yaw:+.0f} deg, "
                f"jaw {target.grasps[0].width_m * 1000:.0f} mm")
        arm.set_position(gx, gy, gz + 110, roll, pitch, yaw, speed=80, wait=True)
        arm.set_position(gx, gy, gz, roll, pitch, yaw, speed=45, wait=True)

        rec.say("closing gripper")
        arm.close_lite6_gripper()
        rec.hold(0.5)

        rec.say("lifting")
        arm.set_position(gx, gy, gz + 160, roll, pitch, yaw, speed=70, wait=True)

        rc, bin_pose = arm.get_body_pose(bin_name)
        if rc != 0 or bin_pose is None:
            rec.say(f"{bin_name} not in scene")
            failures += 1
            continue
        bx, by, bz = bin_pose[0], bin_pose[1], bin_pose[2]

        rec.say(f"carrying to {bin_name}")
        arm.set_position(bx, by, gz + 160, 180.0, 0.0, 0.0, speed=80, wait=True)
        arm.set_position(bx, by, bz + 120, 180.0, 0.0, 0.0, speed=60, wait=True)

        rec.say("releasing")
        arm.open_lite6_gripper()
        rec.hold(1.2)

        arm.set_position(bx, by, gz + 180, 180.0, 0.0, 0.0, speed=80, wait=True)

    rec.say("done - returning home")
    arm.go_home(wait=True)
    rec.hold(1.5)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/vision_demo")
    ap.add_argument("--targets", default="the blue cube:blue_bin,"
                                         "the green cube:green_bin")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--model", default="haiku",
                    choices=("haiku", "sonnet", "opus"),
                    help="planner model for --task. Compound tasks "
                         "(several picks in one prompt) often overrun "
                         "haiku; sonnet is the documented escalation.")
    ap.add_argument("--task", metavar="PROMPT",
                    help="hand this task to the LLM planner and record what "
                         "it does, instead of the scripted pick-and-place")
    ap.add_argument("--backdrop", action="store_true",
                    help="render the lab room behind the bench (fume hood, "
                         "shelving, window). Off by default, matching the rest "
                         "of the sim.")
    args = ap.parse_args()

    targets = [tuple(t.split(":")) for t in args.targets.split(",") if t]

    frames_dir = os.path.join(args.out, "frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    from sim.mujoco_env import SimXArmAPI

    arm = SimXArmAPI(SCENE, render=False)
    rec = Recorder(arm, frames_dir, args.device, backdrop=args.backdrop)
    rec.start()
    print("[demo] loading the grounding model...")
    rec.ready.wait()

    t0 = time.time()
    failures = (run_llm_task(arm, rec, args.task, args.model) if args.task
                else run_demo(arm, rec, targets))
    rec.running = False
    rec.join(timeout=10)
    arm.disconnect()

    wall = time.time() - t0
    print(f"[demo] {rec.frame_idx} frames over {wall:.1f}s")

    mp4 = os.path.join(args.out, "vision_demo.mp4")
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
             "-i", os.path.join(frames_dir, "f%05d.jpg"),
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
            check=True)
        print(f"[demo] wrote {mp4}")
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
    else:
        print(f"[demo] ffmpeg not found; frames are in {frames_dir}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
