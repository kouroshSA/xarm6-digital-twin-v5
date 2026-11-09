import os, sys, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI
X, Y = 29.9, -215.6           # run 3's commanded hover point
arm = RealMaxArmAPI("/dev/ttyUSB0")
try:
    print("start", [round(v, 1) for v in arm.get_position()[1][:3]])
    arm.set_position(X, Y, 62.0, speed=30)
    arm.floor_z_mm = 54.0      # this one move only; contact measured at 52 at (0,-200)
    arm.set_position(X, Y, 55.0, speed=10)
    time.sleep(0.4)
    print("low hover read-back", [round(v, 1) for v in arm.get_position()[1][:3]])
finally:
    arm.disconnect()
