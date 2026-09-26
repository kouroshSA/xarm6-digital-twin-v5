# teleop_sm/buttons.py
"""Semantic actions, and which physical button raises them per device model.

Button indices differ between models -- the Compact has 2, the Pro reports 15
-- and the spnav protocol carries no model string: `SpnavButtonEvent` gives a
bare `bnum`. So the mapping cannot be autodetected from the wire, and is
selected by the `--device {compact,pro}` flag instead.

Adding a third device is one dict in `BUTTON_MAPS` plus an entry in
`KEYBOARD_FALLBACK` if it is short of buttons.

THE PRO MAP IS PROVISIONAL. `spacemouse_probe.py` is the instrument that
settles it: press each key, read the `bnum`. Until that has been run on the
physical device, treat the Pro indices below as a hypothesis -- they follow
3Dconnexion's documented ordering, which spacenavd may remap ("Device
046d:c62b reports 15 buttons before disjointed button remapping" is in the
daemon's own log, and "before remapping" is doing real work in that sentence).
The Compact's two buttons are unambiguous.
"""
from __future__ import annotations

from enum import Enum


class Action(Enum):
    """What the operator wants, independent of which key produced it."""

    GRIPPER_TOGGLE = "gripper_toggle"
    MODE_CYCLE = "mode_cycle"
    RAIL_MODE = "rail_mode"          # held, not edge-triggered
    RECORD_TOGGLE = "record_toggle"
    RESET_SCENE = "reset_scene"
    HOME = "home"


#: Actions that are meaningful while HELD rather than on the rising edge.
HELD_ACTIONS = frozenset({Action.RAIL_MODE})


#: model -> {button index: Action}, for actions fired on the RISING EDGE.
#:
#: SpaceMouse Pro: MENU=0, FIT=1, TOP=2, RIGHT=4, FRONT=5, ROLL_CW=8, ESC=22,
#: ALT=23, SHIFT=24, CTRL=25, and 1..4 = 12..15 in 3Dconnexion's own
#: numbering. spacenavd renumbers to a dense 0..14 -- its log says
#: "reports 15 buttons before disjointed button remapping" -- which is exactly
#: why these are PROVISIONAL until spacemouse_probe.py has been run.
BUTTON_MAPS: dict[str, dict[int, Action]] = {
    "compact": {
        0: Action.GRIPPER_TOGGLE,
        1: Action.MODE_CYCLE,
    },
    "pro": {
        0: Action.GRIPPER_TOGGLE,
        1: Action.MODE_CYCLE,
        3: Action.RECORD_TOGGLE,
        4: Action.RESET_SCENE,
        5: Action.HOME,
    },
}

#: model -> {button index: Action}, for actions that are meaningful WHILE HELD.
#:
#: Separate from BUTTON_MAPS because one physical key can carry both, and on
#: the Compact it must: with two buttons there is no room for a dedicated rail
#: key, so button 1 is MODE_CYCLE when tapped and RAIL_MODE when held. Tapping
#: vs. holding is a distinction an operator can actually make, and the
#: receiver resolves the overload by suppressing the edge action whenever the
#: same key is being held.
#:
#: The Pro has room for a dedicated key, so its rail button appears only here.
HOLD_MAPS: dict[str, dict[int, Action]] = {
    "compact": {
        1: Action.RAIL_MODE,
    },
    "pro": {
        2: Action.RAIL_MODE,
    },
}

#: Actions with no button on a given model fall back to a keyboard key.
#: Kept live on the Pro too -- it costs nothing and makes testing without the
#: device possible.
KEYBOARD_FALLBACK: dict[str, Action] = {
    "r": Action.RECORD_TOGGLE,
    "x": Action.RESET_SCENE,
    "h": Action.HOME,
    "g": Action.GRIPPER_TOGGLE,
    "m": Action.MODE_CYCLE,
}


def map_for(model: str) -> dict[int, Action]:
    """Edge-action map for ``model``, or raise with the list of known models."""
    try:
        return BUTTON_MAPS[model]
    except KeyError:
        raise ValueError(
            f"unknown SpaceMouse model {model!r}; "
            f"known: {sorted(BUTTON_MAPS)}") from None


def hold_map_for(model: str) -> dict[int, Action]:
    """Held-action map for ``model``. Empty is legitimate."""
    if model not in BUTTON_MAPS:
        raise ValueError(
            f"unknown SpaceMouse model {model!r}; "
            f"known: {sorted(BUTTON_MAPS)}")
    return HOLD_MAPS.get(model, {})


def describe(model: str) -> str:
    """Human-readable banner of the active map, printed at startup.

    An operator who has just plugged in a device needs to know what each key
    does on *that* device; printing it beats reading the source.
    """
    lines = [f"SpaceMouse '{model}' button map:"]
    for idx, action in sorted(map_for(model).items(), key=lambda kv: kv[0]):
        also = hold_map_for(model).get(idx)
        extra = f"   (hold -> {also.value})" if also is not None else ""
        lines.append(f"    button {idx:<2d} -> {action.value}{extra}")
    for idx, action in sorted(hold_map_for(model).items(), key=lambda kv: kv[0]):
        if idx not in map_for(model):
            lines.append(f"    button {idx:<2d} -> {action.value}  (hold)")

    bound = set(map_for(model).values()) | set(hold_map_for(model).values())
    spare = [(k, a) for k, a in KEYBOARD_FALLBACK.items() if a not in bound]
    if spare:
        lines.append("  keyboard (actions with no button on this model):")
        for key, action in spare:
            lines.append(f"    '{key}' -> {action.value}")
    lines.append("  keyboard always available: "
                 + ", ".join(f"'{k}'={a.value}" for k, a in KEYBOARD_FALLBACK.items()))
    return "\n".join(lines)
