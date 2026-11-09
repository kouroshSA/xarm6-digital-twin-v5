import sys, os, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI
from hardware.maxarm import kinematics as kin
for p in [(60, -180, 150), (-60, -180, 150)]:
    s = kin.solve(p); print("plan", p, "pulses", [round(v) for v in s.pulses])
t0 = time.monotonic(); T = lambda: f"{time.monotonic()-t0:5.1f}s"
arm = RealMaxArmAPI("/dev/ttyUSB0")
print(T(), "connected; stock loop", arm.stock_loop.split(';')[1].strip(), "pulses", arm.connect_pulses, flush=True)
try:
    arm.go_home(); print(T(), "home", arm.get_position(), flush=True)
    arm.set_position(60, -180, 150, speed=30); print(T(), "AT x=+60 -> holding 10 s", arm.get_position(), flush=True)
    time.sleep(10)
    arm.set_position(-60, -180, 150, speed=30); print(T(), "AT x=-60 -> holding 10 s", arm.get_position(), flush=True)
    time.sleep(10)
    arm.go_home(); print(T(), "home", arm.get_position(), flush=True)
finally:
    arm.disconnect()
