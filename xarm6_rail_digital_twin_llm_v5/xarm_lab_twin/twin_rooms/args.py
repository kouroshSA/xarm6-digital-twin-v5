# twin_rooms/args.py
"""--room for entry points.

    parser = argparse.ArgumentParser()
    add_room_arg(parser)
    args = parser.parse_args()
    arm = SimXArmAPI(**sim_kwargs(args, scene_xml="envs/lab_scene.xml"))

Without --room a tool loads exactly the scene file it always did (the lab's
passthrough room is found from the file), so existing commands are unchanged.
With --room it builds that room; a tool's --scene meshes|primitive becomes the
room's variant.
"""
from __future__ import annotations

import argparse
from typing import Optional


def add_room_arg(parser: argparse.ArgumentParser, default: Optional[str] = None) -> None:
    parser.add_argument(
        "--room", default=default, metavar="ID",
        help="Room to run in (a folder under rooms/, or a template such as "
             "empty_bench; `python -m twin_rooms list` lists them). Default: the "
             "lab scene, exactly as before rooms.")


def sim_kwargs(args, scene_xml: str, variant: Optional[str] = None) -> dict:
    """SimXArmAPI keyword arguments for these args: the room when --room is
    given, otherwise the scene file the tool has always loaded."""
    room = getattr(args, "room", None)
    if room:
        return {"room": room, "variant": variant}
    return {"scene_xml": scene_xml}
