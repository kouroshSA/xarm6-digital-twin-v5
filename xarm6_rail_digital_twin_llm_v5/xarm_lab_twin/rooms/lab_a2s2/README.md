# lab_a2s2 — A2S2 wet lab bench (the original lab)

The scene every tool has always used: the L-shaped bench, the xArm6 on its 700 mm
rail, cubes, bins, tube racks and the instruments (OT-2, heater-shaker, vortex,
PCR). This room is a **passthrough**: it loads `envs/lab_scene.xml` (or
`envs/lab_scene_primitive.xml` for `--scene primitive`) unchanged, and only adds
what the scene file cannot say — object labels and types, cameras, and notes for
the agent. Without `--room`, every entry point runs here, exactly as before.
Edit the scene in `envs/lab_scene_primitive.xml` as always (see CLAUDE.md).

The planner's lab knowledge (tube racks, OT-2 access, heater-shaker, vortex, PCR)
is in [`agent_prompt.md`](agent_prompt.md), spliced into the system prompt by
`agent/room_prompt.py`. It is the text the prompt template held before rooms,
moved verbatim, so the lab's prompt did not change. Things worth knowing that it
covers: the two red cubes straddle the rail, so "the red cube" is ambiguous (say
front or back); there is no red bin; the OT-2 is reached only through its front
opening from rail 700; the PCR lid must be opened (`pcr_open`) before loading a
plate.
