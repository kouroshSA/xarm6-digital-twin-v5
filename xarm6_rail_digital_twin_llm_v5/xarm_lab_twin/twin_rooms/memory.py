# twin_rooms/memory.py
"""Where a room's cross-session agent memory lives.

    lessons.md       per-episode lessons            (gitignored)
    reviews.md       Opus session reviews
    world_model.md   cross-task invariants
    objects.json     object-registry seed           (gitignored)

The lab (lab_a2s2) and any anonymous scene-file room keep using the files at
the project root and agent/objects.json, exactly where they have always been,
so the lab's accumulated memory is untouched. Every other room gets its own
folder, memory/<room_id>/, so lessons learned on one bench never leak into a
prompt about another.
"""
from __future__ import annotations

from pathlib import Path

from twin_rooms.room import DEFAULT_ROOM, PROJECT_ROOT

MEMORY_DIR = PROJECT_ROOT / "memory"


def uses_lab_memory(room_id: str) -> bool:
    """The lab, and scene files loaded without a room (the pre-rooms path)."""
    return room_id == DEFAULT_ROOM or room_id.startswith("scene_")


def memory_dir(room_id: str) -> Path:
    return PROJECT_ROOT if uses_lab_memory(room_id) else MEMORY_DIR / room_id


def memory_file(room_id: str, name: str) -> Path:
    """Path of memory file `name` (lessons.md, reviews.md, world_model.md,
    objects.json) for `room_id`. Does not create anything."""
    if name == "objects.json" and uses_lab_memory(room_id):
        return PROJECT_ROOT / "agent" / "objects.json"
    return memory_dir(room_id) / name


def room_id_of(arm) -> str:
    """The arm's room id; an arm with no room (real hardware) is in the lab."""
    room = getattr(arm, "room", None)
    return getattr(room, "room_id", DEFAULT_ROOM)


def memory_file_for(arm, name: str, create: bool = False) -> Path:
    """memory_file() for the arm's room. Writers pass create=True to make the
    room's memory folder (never needed for the lab); readers do not, so
    building a prompt never leaves an empty folder behind."""
    path = memory_file(room_id_of(arm), name)
    if create and not uses_lab_memory(room_id_of(arm)):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
