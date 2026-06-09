# vr/ — Meta Quest 3 teleoperation of the MuJoCo digital twin.
#
# Sim-only WebXR teleop: the operator drives the simulated xArm6 with the
# Touch controllers and sees the twin through the headset (mono or stereo).
# Reuses the existing SimXArmAPI -> IKSolver -> ctrl -> MuJoCo -> Recorder
# path; the only new thing is the source of the EE target (a human hand
# instead of the LLM). See vr/README.md and scripts/run_vr.py.
