# teleop_kin/keys.py
"""Viewer key handler for teach mode.

Registered with `SimXArmAPI.register_key_handler`, so it sees each key before
the default nudge keymap. Keys in KEYMAP are consumed (the arrows then no
longer nudge the double-clicked body); every other key falls through, so R/E/T
and friends keep working.

The handler only QUEUES the action: it runs on the viewer thread, and home,
reset and the rail step block for a paced move.
"""
from __future__ import annotations

from teleop_kin import config
from teleop_kin.receiver import KinAction

KEYMAP = {
    config.KEY_UP: KinAction.GRIPPER_OPEN,
    config.KEY_DOWN: KinAction.GRIPPER_CLOSE,
    config.KEY_LEFT: KinAction.RAIL_MINUS,
    config.KEY_RIGHT: KinAction.RAIL_PLUS,
    config.KEY_INSERT: KinAction.RECORD_TOGGLE,
    config.KEY_DELETE: KinAction.DISCARD_TAKE,
    config.KEY_HOME: KinAction.HOME,
    config.KEY_END: KinAction.RESET_SCENE,
}

KEY_TABLE = """\
  Mouse  double-click the orange ball to select it, then
         Ctrl + right-drag   move it      (the arm follows)
         Ctrl + left-drag    rotate it
         red ball = out of reach; let go and it snaps back to the arm
  Up     open gripper            Down   close gripper
  Left   rail -{step:.0f} mm            Right  rail +{step:.0f} mm
  Insert start / stop a take     Delete discard current (or last) take
  Home   arm to home pose        End    reset the scene"""


def make_handler(receiver):
    """-> callable(keycode) -> bool for `arm.register_key_handler`."""
    def _handler(keycode: int) -> bool:
        action = KEYMAP.get(int(keycode))
        if action is None:
            return False
        receiver.post(action)
        return True
    return _handler


def describe(rail_step_mm: float = config.RAIL_STEP_MM) -> str:
    return KEY_TABLE.format(step=rail_step_mm)
