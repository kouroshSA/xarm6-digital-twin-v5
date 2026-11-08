# MaxArm: temporary notes for the next session (2026-11-09)

Working notes, kept in the repo so a session on another device can pick them
up. Fold them into `README.md` (or delete them) once the work below is done.
State at writing: `feat/maxarm` on dev; the hand-eye calibration, wrist-camera
frames and arrival tolerance are measured and committed; pick and place placed
1 of 3 runs (see README, "Open, needs the device").

## 1. Manual teaching: hand-guiding does NOT work; teach by jogging

Measured 2026-11-09 with the user at the arm:

- Torque off works: `BusServo.unload` on servos 1-3, confirmed by reading each
  servo's load state (protocol command 32, not wrapped by the vendor's
  `BusServo.py`): 1/1/1 -> 0/0/0 -> 1/1/1. With torque off the arm only settles
  2-3 mm onto its gears.
- **But the arm is too stiff to move by hand with torque off** (user, twice,
  30 s windows, beeps from the board's buzzer marking start/end). The gear
  trains hold it, the same reason it does not drop when powered off. So the
  vendor's `ESPMax.teaching_mode()` does not make it hand-guidable either.
- Useful anyway: servo read-back runs at ~25 Hz over USB (1,362 reads, 0
  failures), and torque can be restored IN PLACE with no snap: send a move to
  the pulses just read (a bare `load` drives to the last commanded target).
- To correct: `RealMaxArmAPI.disconnect`'s docstring says torque off "would
  drop the arm onto the table". It does not (user: the powered-off arm holds;
  measured above). Fix the wording when this area is next touched.

Plan (not started): **jog teaching**. The operator drives the arm, the
recorder logs read-back (25 Hz), pump state, nozzle angle as commanded, and
wrist-camera frames; replay goes through `set_position` (floor guard, arrival
check). Inputs, in order: keyboard jog first (no hardware), then the SpaceMouse
(`teleop_sm/`, 3 translation axes map onto x/y/z, a button for the pump), then
possibly hand tracking (`teleop_hand/`, needs its own camera calibration). Follow
`teleop_kin/`'s exit discipline: every exit path leaves the arm holding where it
is and the pump as it was.

## 2. Digital twin of the MaxArm

No public model exists (searched 2026-11-09): no URDF/MJCF/ROS package from
Hiwonder for the MaxArm (they publish URDFs only for their ROS arms), only a
Webots model in a UVG thesis with no files found. Hiwonder's download pack for
the kit may have structural drawings or CAD -- not checked. So build one:

- Primitive geometry first, like the xArm twin started; real meshes can replace
  it later without touching anything else.
- Kinematics from `kinematics.py` (link lengths L0 84, L1 8.2, L2 128, L3 138,
  L4 16.8 mm, from the vendor's `espmax.py`), including the parallel linkage
  that keeps the tool level. Grade the twin's motion against `kinematics.py`
  (and through it against the board), never against itself.
- Measured facts to carry over (single source: the code, not copies):
  left-handed arm frame (+x = arm's right seen from behind, reach along -y);
  table contact at wrist z 52 (`TABLE_CONTACT_Z_MM`), floor 62; arrival
  tolerance 10 mm; the wrist camera mount `hand_eye_calib.json` (camera fixed to
  the platform, NOT the nozzle); nozzle -90..90 deg, no read-back.
- Suction cup as a weld, like the xArm's magnetic gripper, plus the wrist D435
  as a sim camera at the calibrated mount so `maxarm_pick_place.py` can be
  rehearsed in the twin before hardware.

### Main use: Petri dish lids (user, 2026-11-09)

The robot's job will be to **lift the lid off a standard 90 mm Petri dish and
set it aside, or put the lid back on**. Build the twin around that:

- Model a 90 mm dish as base + lid (separate bodies; the lid rests loosely on
  the base). Standard dimensions to MEASURE on the real dishes, not assume:
  lid outer diameter (~90-92 mm), lid height (~8-10 mm), base height
  (~14-15 mm), and how much the lid overlaps the base.
- The cup takes the lid by its flat top centre. Smooth polystyrene should seal
  far better than the wooden test cubes did (run 3's no-hold).
- Tasks to support: "lid off and set aside", "lid back on", probably over
  several dishes; grade physically (lid on its own base, centred within the
  overlap; a lid set aside lies flat, not on another dish).
- Perception: a dish lid is clear plastic, so the wrist D435's depth is
  unreliable on it (the xArm cell measured dropouts on transparent surfaces),
  and from the arm's reach the lid top is inside the depth minimum anyway. Use
  geometry like `maxarm_pick_place.py` does: known heights + colour/edge in the
  image (the dish rim), fitted on a known plane.

### Where: a separate room, on the existing bench

Make it its own room for development (`rooms/<id>/room.json`, `twin_rooms`),
on the same bench the main rooms use, with the xArm6 on its rail there too.
Today's rooms: `lab_a2s2` (default, passthrough), `lab_a2s2_rebuilt`,
`simready_demo` (both built rooms on the lab bench), `basic_cubes_bins`. A built
room on the lab bench (like `lab_a2s2_rebuilt` / `simready_demo`) is the model:
add a MaxArm robot type under `robots/` (like `lite6`), place it on the bench at
true scale (it is small next to the xArm6), and add the dishes as objects. Keep
the lab's own rooms unchanged.
