#!/usr/bin/env python3
"""LLM-driven demo run with offscreen before/after renders (uses the API key).

A thin, headless-friendly wrapper over the same path as ``run_task.py``
(``LLMBrain.prepare_for_task`` + ``execute_task``) that also renders
before/after frames, so you can produce a viewable "LLM plans + drives the arm"
filmstrip without launching the interactive GLFW viewer. Handy on WSL / servers
where the passive viewer can't display.

Loads ``ANTHROPIC_API_KEY`` from ``.env`` via ``env_loader.load_env`` exactly
like the real entry points. Runs on the default (mesh) scene. Anthropic can
return transient 529 "overloaded" errors; this retries the planning call with
backoff so a busy moment doesn't abort the demo.

Usage:
    MUJOCO_GL=egl python scripts/demo_llm.py "put the green cube in the green bin"
    MUJOCO_GL=egl python scripts/demo_llm.py "..." --model sonnet --frames /tmp/demo
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_loader import load_env  # noqa: E402


def _render(arm, path, lookat=(0.0, 0.15, 0.85), dist=1.5, az=135, el=-22):
    from PIL import Image
    r = mujoco.Renderer(arm.model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    with arm.lock:
        r.update_scene(arm.data, camera=cam)
        img = r.render()
    Image.fromarray(img).save(path)
    print(f"[render] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="put the green cube in the green bin")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--scene", default="envs/lab_scene.xml",
                    help="scene to load (default: the realistic mesh scene)")
    ap.add_argument("--frames", default=None,
                    help="dir to write before/after PNGs into (skipped if omitted)")
    ap.add_argument("--retries", type=int, default=8,
                    help="planning-call retries on transient 529 overload")
    args = ap.parse_args()

    loaded = load_env()
    print("[env] loaded:", ", ".join(loaded) or "(none)",
          "| ANTHROPIC_API_KEY present:", bool(os.environ.get("ANTHROPIC_API_KEY")))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[env] no ANTHROPIC_API_KEY found in .env — cannot run the LLM.")

    from agent.llm_brain import LLMBrain, MODELS
    from agent.object_registry import build_default_registry
    from sim.mujoco_env import SimXArmAPI

    if args.frames:
        os.makedirs(args.frames, exist_ok=True)

    arm = SimXArmAPI(scene_xml=args.scene, render=False)
    brain = LLMBrain(arm=arm, registry=build_default_registry(),
                     recorder=None, model=args.model)

    print(f"\n[Task] {args.task}   (model={args.model} -> {MODELS.get(args.model)})\n")
    if args.frames:
        _render(arm, os.path.join(args.frames, "before.png"))

    brain.prepare_for_task(args.task)
    # Retry the planning/exec call with backoff — LLMBrain catches 529s and
    # returns an empty plan, so an empty result means "try again".
    result = {"commands": []}
    for attempt in range(1, args.retries + 1):
        result = brain.execute_task(args.task, dry_run=False)
        if result.get("commands"):
            print(f"[retry] got a plan on attempt {attempt}")
            break
        wait = min(4 * attempt, 20)
        print(f"[retry] empty plan (attempt {attempt}) — waiting {wait}s")
        time.sleep(wait)

    print("\n[LLM planned sequence]")
    for i, cmd in enumerate(result.get("commands", [])):
        print(f"  {i+1:2d}. {cmd['action']}  {cmd.get('params', {})}")

    for _ in range(200):  # let objects settle
        with arm.lock:
            mujoco.mj_step(arm.model, arm.data)
    if args.frames:
        _render(arm, os.path.join(args.frames, "after.png"))
    if hasattr(arm, "physical_outcome"):
        print("\n[Physical outcome]", arm.physical_outcome().splitlines()[0])
    arm.disconnect()


if __name__ == "__main__":
    main()
