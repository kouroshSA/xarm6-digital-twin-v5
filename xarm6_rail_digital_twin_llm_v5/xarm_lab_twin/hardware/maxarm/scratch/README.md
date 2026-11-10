# MaxArm one-off scripts

The throwaway scripts behind measurements in `../README.md` and `../NOTES_NEXT.md`,
kept for reference, not maintained. Each one connects to the board and several
MOVE the arm: read before running, user at the arm. Run from `xarm_lab_twin/`.
Some still name `~/Models/maxarm_calib/` (now `../data/`).

- `teach_check.py LOG.json`: countdown + buzzer, torque off 30 s, 25 Hz read-back, torque back on in place.
- `load_state.py`: reads each servo's load state (bus-servo command 32).
- `low_hover.py`: hovers the cup 3 mm over pick run 3's point (floor guard lowered for one move).
- `dbg_look.py`, `look.py`: wrist views + Grounding DINO / tag detection debugging.
- `floor_daemon.py`, `floor_cmds.txt`, `floor_log.txt`: the stepwise table-contact measurement.
- `backup_board.py OUT_DIR`: copies every file off the board, checked against the board's SHA-256.
- `connect_check.py`, `listen.py`, `catboard.py`, `kincheck.py`, `timer_probe*.py`,
  `frame_photos.py`, `frame_timed.py`, `collect.log`: first-day bring-up (port, stock program timers, frame, first calibration log).
