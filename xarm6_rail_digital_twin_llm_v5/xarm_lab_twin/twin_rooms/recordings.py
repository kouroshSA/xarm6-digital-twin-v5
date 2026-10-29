# twin_rooms/recordings.py
"""Rooms in recordings: what a session records about its room, and how a
replay gets the room back.

    fields = room_fields(ctx)          # -> metadata.json / trajectory.h5 attrs
    model, ctx, warnings = replay_room(metadata)

A recording made before rooms has no room_id; it is replayed in the project's
default room (lab_a2s2 here, basic_cubes_bins in the manual-control project).
If the room's definition has changed since the recording (hash mismatch), the
replay warns and goes ahead: the joint trajectory is still meaningful even if
an object has moved.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from twin_rooms.room import DEFAULT_ROOM


def room_fields(ctx) -> dict:
    """The room facts a recording carries. Empty values when ctx is None."""
    if ctx is None:
        return {"room_id": "", "room_hash": "", "room_variant": "", "robots": []}
    return {
        "room_id": ctx.room_id,
        "room_hash": ctx.hash,
        "room_variant": ctx.variant or "",
        "robots": [{"instance": r.instance, "type": r.type.type,
                    "end_effector": r.spec.end_effector or r.type.end_effectors["default"],
                    "controllable": r.spec.controllable}
                   for r in ctx.robots.values()],
    }


def context_for_recorder(model, scene_xml) -> Optional[object]:
    """The room of a model a Recorder was given without one (the pre-rooms
    call). None if the scene file cannot be matched -- never raises, because a
    recording must not fail over metadata."""
    try:
        from pathlib import Path
        from twin_rooms.builder import context_for_scene
        from twin_rooms.room import PROJECT_ROOT
        p = Path(scene_xml)
        if not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p          # tools pass project-relative paths
        return context_for_scene(p, model)
    except Exception:  # noqa: BLE001
        return None


def replay_room(meta: dict, default_room: str = DEFAULT_ROOM,
                override: Optional[str] = None) -> Tuple[object, object, List[str]]:
    """(model, ctx, warnings) to replay a session with metadata `meta`."""
    from twin_rooms.builder import build_room
    warnings: List[str] = []
    rid = override or meta.get("room_id") or default_room
    variant = None if override else (meta.get("room_variant") or None)
    if not override and not meta.get("room_id"):
        warnings.append(f"recording has no room_id (made before rooms): replaying in "
                        f"{default_room!r}")
        if str(meta.get("scene_xml", "")).endswith("lab_scene_primitive.xml"):
            variant = "primitive"
    model, ctx = build_room(rid, variant=variant)
    recorded = meta.get("room_hash")
    if recorded and not override and recorded != ctx.hash:
        warnings.append(f"room {rid!r} has changed since this recording (hash {recorded} "
                        f"-> {ctx.hash}); replaying anyway")
    return model, ctx, warnings
