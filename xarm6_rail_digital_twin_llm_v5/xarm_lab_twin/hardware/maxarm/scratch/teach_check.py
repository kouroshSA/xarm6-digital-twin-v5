"""Torque off, stream read-back while the user moves the arm, torque back on in place."""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI
from hardware.maxarm import kinematics as kin

WINDOW_S = 30.0
COUNTDOWN_S = 10
LOG = sys.argv[1]
arm = RealMaxArmAPI("/dev/ttyUSB0")
samples, fails = [], 0
p = arm._read_pulses()
print("start pulses", p, "xyz", [round(v, 1) for v in kin.pulses_to_xyz(p)], flush=True)
arm._call("_r(lambda: globals().__setitem__('_bz', __import__('Buzzer').Buzzer()))", "buzzer")
def beep(ms=150):
    arm._call(f"_r(_bz.setBuzzer, {ms})", "beep")
for k in range(COUNTDOWN_S, 0, -1):
    print(f"torque off in {k}", flush=True)
    time.sleep(1.0)
try:
    beep(400)
    for i in (1, 2, 3):
        arm._call(f"_r(_bs.unload, {i})", "unload")
    print("TORQUE OFF - move the arm now", flush=True)
    t0 = time.monotonic(); last = 0
    while time.monotonic() - t0 < WINDOW_S:
        t = time.monotonic() - t0
        try:
            p = arm._read_pulses()
        except Exception as exc:  # noqa: BLE001
            fails += 1
            continue
        xyz = kin.pulses_to_xyz(p)
        samples.append({"t": t, "pulses": p, "xyz": xyz})
        if t - last >= 1.0:
            last = t
            print(f"{t:5.1f}s pulses {p} xyz {[round(v, 1) for v in xyz]}", flush=True)
finally:
    # Torque back on WHERE IT IS: a run to the read pulses re-enables each servo
    # without moving it (a bare load would drive to the last commanded target).
    p = arm._read_pulses()
    arm._call(f"_r(_move, {p[0]}, {p[1]}, {p[2]}, 500)", "hold in place")
    time.sleep(0.8)
    beep(150); time.sleep(0.3); beep(150)
    q = arm._read_pulses()
    print("TORQUE ON at", p, "-> read", q, "xyz", [round(v, 1) for v in kin.pulses_to_xyz(q)], flush=True)
    arm.disconnect()
    n = len(samples)
    span = samples[-1]["t"] - samples[0]["t"] if n > 1 else 0
    moved = max(abs(a - b) for s in samples for a, b in zip(s["pulses"], samples[0]["pulses"])) if n else 0
    print(f"{n} samples in {span:.1f} s = {n / span if span else 0:.1f} Hz, {fails} failed reads, "
          f"largest pulse change {moved}", flush=True)
    json.dump(samples, open(LOG, "w"))
