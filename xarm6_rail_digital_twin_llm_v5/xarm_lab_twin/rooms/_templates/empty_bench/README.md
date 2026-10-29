# empty_bench (template) — the robot on its bench, nothing else

The lab bench (main section plus the L-shaped OT-2 extension), the xArm6 on its
700 mm rail with the rail LED strips, the observer cameras, and no objects. The
bench and cameras are extracted from `envs/lab_scene.xml` when the room is built
and the robot comes from `robots/xarm6_rail`, so a change to the lab's bench or
arm reaches every room made from this template. Start a new room with
`python -m twin_rooms new <id> --from empty_bench`.
