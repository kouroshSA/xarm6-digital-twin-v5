"""Rooms: declarative scenes (environment + robots + objects) that every tool
can run in, chosen with --room. See rooms/README.md."""
from twin_rooms.room import (DEFAULT_ROOM, ROBOTS_DIR, ROOMS_DIR, ObjectSpec,
                             Pose, RobotSpec, RoomDef, list_rooms, load_room,
                             load_room_dir)
from twin_rooms.schema import RoomError

__all__ = ["DEFAULT_ROOM", "ROBOTS_DIR", "ROOMS_DIR", "ObjectSpec", "Pose",
           "RobotSpec", "RoomDef", "RoomError", "list_rooms", "load_room",
           "load_room_dir"]
