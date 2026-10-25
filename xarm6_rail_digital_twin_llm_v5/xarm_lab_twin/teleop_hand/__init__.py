"""Whole-arm hand-tracking teleop (bare hand in front of a camera).

Two processes. The TRACKER (`scripts/run_hand_tracker.py`, MediaPipe +
dex-retargeting, its own conda env) turns camera frames into `HandFrame`
messages on localhost UDP. The TELEOP side (`scripts/run_handteleop_sim.py`,
`scripts/run_handteleop_real.py`) maps them onto the arm, rail and gripper and
records ordinary `Recorder` sessions. See README.md.

Nothing here imports MediaPipe at module level: the teleop side runs in the
plain `xarm6sim` env.
"""
