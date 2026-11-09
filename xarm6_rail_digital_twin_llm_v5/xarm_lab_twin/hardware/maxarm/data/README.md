# MaxArm session data

Measurements and pictures from the device sessions (2026-11-07 to 11-09), kept
in the repo so work can continue on another machine. `scripts/maxarm_hand_eye.py`
(`--out`) and `scripts/maxarm_pick_place.py` write here now; until 2026-11-09
they wrote to `~/Models/maxarm_calib/`.

| file | what |
|---|---|
| `maxarm_tag_40mm.pdf` / `.png` | the calibration tag (DICT_4X4_50 id 0, 40 mm); print at 100% |
| `tag_place.json` | where the tag was laid (under the cup, z = table contact), as first recorded |
| `hand_eye_samples_20261108_001807.json` | the hand-eye views the calibration was solved from (`resolve` re-solves these) |
| `hand_eye_calib.before_resolve.json` | `../hand_eye_calib.json` before the 5 mm tag correction (2026-11-09) |
| `pick_place_before/look/after.png` | annotated wrist views from the pick-and-place runs; `_dbg`, `_after_raw` unannotated |
| `teach_check_samples.json` | 25 Hz servo read-back with torque off (the hand-guiding check; the arm did not move) |
| `first_views/` | first wrist-camera pictures (2026-11-07/08): home, base turned +-60, looks at z 160-210, auto-exposure stuck dark |

Not here: the board's own files (stock `main.py`, vendor libraries, `.rob`
actions), backed up to `~/Models/maxarm_board_backup/2026-11-07/` on the lab PC.
They are Hiwonder's code and this repo is mirrored publicly, so they stay out.
