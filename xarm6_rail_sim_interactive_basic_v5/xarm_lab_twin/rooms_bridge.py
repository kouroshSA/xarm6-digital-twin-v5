# rooms_bridge.py
"""--room for the manual-control tools.

Rooms are defined once, in the LLM project (`../../xarm6_rail_digital_twin_llm_v5/
xarm_lab_twin/rooms/`); this module imports its `twin_rooms` package rather than
keeping a second copy. The default room, `basic_cubes_bins`, is this project's
own envs/basic_scene.xml loaded unchanged, so running a tool without --room
behaves exactly as before.

    args  = parse_room_args("xArm6 terminal control")
    room  = load_room_model(args.room)
    model = room.model
    model.actuator(room.mjcf("act1", "actuator"))      # 'act1' or 'arm0/act1'
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

LLM_PROJECT = (Path(__file__).resolve().parents[2]
               / "xarm6_rail_digital_twin_llm_v5" / "xarm_lab_twin")
DEFAULT_ROOM = "basic_cubes_bins"

# APPENDED, not prepended: both projects have a recording.py, and this
# project's own must keep winning. Only twin_rooms (and, for lab-robot rooms,
# the arm_backend its robot.json refers to) is taken from the LLM project.
if str(LLM_PROJECT) not in sys.path:
    sys.path.append(str(LLM_PROJECT))


@dataclass
class RoomModel:
    model: object            # mujoco.MjModel
    ctx: object              # twin_rooms.context.RoomContext
    scene_xml: str           # what loads this model on its own

    def mjcf(self, logical: str, kind: str = "body") -> str:
        return self.ctx.mjcf(logical, kind)

    def home(self):
        """(rail_mm, [j1..j6 deg]) from the room's robot type."""
        rt = self.ctx.controllable.type
        return float(rt.value("home.rail_mm")), [float(a) for a in rt.value("home.joints_deg")]


def add_room_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--room", default=DEFAULT_ROOM, metavar="ID",
                        help=f"Room to run in (default {DEFAULT_ROOM}, this project's "
                             "basic scene). Rooms are listed by `rooms` in "
                             "terminal_control.py.")


def parse_room_args(description: str = None, argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    add_room_arg(p)
    return p.parse_args(argv)


def load_room_model(room_id: str = DEFAULT_ROOM) -> RoomModel:
    from twin_rooms.builder import build_room
    model, ctx = build_room(room_id)
    return RoomModel(model, ctx, str(ctx.model_path))


def list_room_ids():
    from twin_rooms import list_rooms
    return list_rooms(include_templates=True)
