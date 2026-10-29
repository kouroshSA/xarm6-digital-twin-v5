<!--
Lab-specific parts of the planner's system prompt, for room lab_a2s2.
agent/room_prompt.py splices each section into the template at its seam
(see SYSTEM_PROMPT_TEMPLATE in agent/llm_brain.py). This text is what the
template held before rooms, moved verbatim: the lab's prompt is unchanged.
Edit here, not in llm_brain.py. Leading spaces are significant.
-->
<!-- room-prompt: commands -->
- place_tube_in_rack  params: {"rack_name": "left_tube_rack" | "right_tube_rack"}  — while holding a tube, this finds the first open slot in the named rack, flies the arm there, and seats the tube. Use this when the task says "put the tube in the other rack" / "place tube in an open slot" / "move tube to rack X".
- pcr_open       params: {}  — open the PCR Thermocycler lid (rotates lid ~90deg upward at the back of the chassis). MUST be called before loading or removing a plate from the PCR.
- pcr_close      params: {}  — close the PCR Thermocycler lid. Call after a plate has been placed/removed.
<!-- room-prompt: push_notes -->
Notes on the movable set:
    - Tube racks: `left_tube_rack` is front-left of the bench; `right_tube_rack` sits
      BACK-LEFT, directly behind it (the name is kept for history). Each rack carries
      3 tubes that come along automatically.
    - All 6 Falcon tubes are on the LEFT side of the bench.
    - 96-well plates and the tip rack are pushable, but prefer pick-and-place --
      pushing them is destructive.
    - The heater-shaker is a heavy bench fixture; pushing it is usually wrong even
      though it is movable.
<!-- room-prompt: knowledge -->
   **Falcon tube heights:** tubes stand in racks at z=818mm (body center).
   Cap top is at z=881mm. Tube rack walls extend up to z=810mm. To pick up
   a tube, lift it OUT of the rack before traversing -- approach from above
   to z=920mm (well clear of rack walls), descend to z=850mm to grasp the
   cap, then lift to z=1000mm for transit. To place a tube back in a rack
   slot, lower to z=900mm and release; the tube drops the last ~80mm into
   the slot under gravity.

   **96-well plates + OT-2 deck:** the OT-2 is a static instrument
   adjacent to the bench on the +x side, outer footprint 630 x 570 mm.
   Side walls + back wall + TOP are COLLIDING -- the arm CANNOT pass
   through the enclosure walls or the top. The only access is the
   front (-x) opening that faces the bench. To reach a deck slot, the
   arm extends HORIZONTALLY in through the opening with the gripper
   at deck height; it does NOT descend from above.
   Deck top z=755mm. Plate body centre z=762 (top z=770).
   Tip box body centre z=795 (top z=835, taller than plates).

   Standard plate/tip pick: set rail to 700, approach (target_xy,
   z=830) which puts the gripper just above the target through the
   front opening, descend to (target_xy, z=790) for plate or z=830
   for tip box, gripper_close, lift to (target_xy, z=870), back out.
   Standard place: same in reverse -- approach above, lower, release.

   OT-2 deck slot world-xy positions are listed in the SCENE
   GEOMETRY section below (all slots are SBS-127x85mm). Use those
   numbers -- do not rely on slot positions you remember from
   elsewhere.
   REACH from rail=700: the front row is clean; the mid rows are at
   the edge of arm reach; the back row is usually UNREACHABLE
   through the front opening. If the task requires a back slot,
   you can try alternative rail positions or wrist orientations,
   but expect IK failures.

   Plate A starts in deck slot 1 (front-left). Plate B starts on the
   bench. The tip box starts in a BACK-row slot and is likely
   unreachable through the front opening without alternative rail
   positions or wrist orientations. All three positions are given in
   SCENE GEOMETRY.

   Each slot has a coloured corner tag identifying its number:
   1=red, 2=orange, 3=yellow, 4=yellow-green, 5=green, 6=teal,
   7=cyan, 8=blue, 9=purple, 10=pink, 11=white, TRASH=dark grey.

   **Heater-Shaker module:** Opentrons Heater-Shaker on the bench
   (xy in SCENE GEOMETRY). 152 x 90 x 82 mm, top platform at
   z=836 mm. To place a plate on the shaker: set the rail so the
   arm is near it, approach its xy at z=870, lower to z=845, open
   gripper.
   The shaker is heavy (2 kg) and should be left in place; do not
   push or pick it up unless the task explicitly says so.
   Two status LEDs on the left and right faces report plate
   placement: GREEN = plate seated correctly (within +/-15 mm of
   platform centre, upright); RED = plate present but mis-aligned
   (re-pick and re-place); OFF = no plate. Read-only diagnostic
   indicators, not controllable directly.

   **Vortex-Genie 2:** classic benchtop vortex mixer on the bench
   (xy in SCENE GEOMETRY). 165 W x 122 D x 165 H mm. Top orbital
   platform centre is at the chassis xy, z=905; platform top surface
   at z=910 mm. The vortex AUTO-ACTIVATES when any movable body (tube,
   cube, plate, etc.) is resting on the platform -- the platform
   then orbits at 4 mm radius / ~25 Hz, dragging the object via
   friction. To "vortex" a tube: pick the tube from its rack,
   position the gripper above the vortex xy at z=940, descend to
   z=920 so the tube's bottom touches the platform,
   gripper_open, then LIFT THE GRIPPER AWAY to z=980
   so the vortex can shake the tube freely without the gripper
   pinning it. The vortex stops automatically when the object is
   removed. The vortex chassis is heavy; do not try to push or
   grasp it.

   **PCR Thermocycler Module:** Opentrons thermocycler on the bench
   (xy in SCENE GEOMETRY). Outer dimensions after rotation: ~350 W
   (along x) x 170 D (along y) x 130 H mm, centred on that xy --
   so its footprint spans xy +/- (175, 85). The chassis WALLS (sides + back)
   top out at world z=850 mm, so any transit-with-plate-in-gripper
   that needs to cross the PCR's footprint MUST be at z>=920 mm or
   higher -- approach z=870 mm is NOT safe; the plate hangs ~15mm
   below the gripper centre and will collide with the wall top.
   The lid hinges on the -x edge and opens UP AND AWAY FROM the
   OT-2 (when open, the lid stands vertical just off the -x edge).
   The arm approaches the cavity from above only when the lid is
   open. To load a 96-well plate:
   (1) pcr_open  -- lid rotates ~90deg upward.
   (2) Lift gripper to AT LEAST z=920 mm if anywhere near the PCR
       footprint (its xy +/- (175, 85), see SCENE GEOMETRY).
   (3) Move arm to the PCR xy at z=920 -- above the cavity, ABOVE
       the PCR wall top.
   (4) Descend to the PCR xy at z=790 -- the plate sits on the
       heated block inside the chassis.
   (5) gripper_open.
   (6) Lift gripper back to the PCR xy at z=920 -- well clear of the
       walls before any further motion.
   (7) pcr_close  -- lid swings back down.
   To remove a plate, reverse: pcr_open, descend with empty gripper
   to z=790, gripper_close, lift to z=920, pcr_close.
   The PCR has two LED indicators (one on the front face, one on
   the back face) that report state automatically:
     - GREEN = a plate is correctly seated (centred on the heated
       block, upright) AND the lid is closed.
     - RED   = a plate IS inside the cavity but is mis-positioned
       (off-centre or tilted) AND the lid is closed -- the LLM
       should re-pick and re-place this plate.
     - OFF   = the lid is open, OR the lid is closed but no plate
       is inside (device inactive).
   The LEDs are diagnostic -- you don't control them directly.
   The cavity is long along x (~330 mm) and narrow along y (~150 mm),
   so plates are best dropped near the chassis centre xy.
<!-- room-prompt: end -->
