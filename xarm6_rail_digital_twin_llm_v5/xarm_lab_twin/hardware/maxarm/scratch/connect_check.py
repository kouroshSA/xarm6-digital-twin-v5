import sys, os, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI
for n in range(3):
    arm = RealMaxArmAPI("/dev/ttyUSB0")
    print(f"connect {n+1}: stock loop -> {arm.stock_loop}; pulses {arm.connect_pulses}, supply {arm.supply_mv}")
    time.sleep(1.0)
    print("   1 s later:", arm.repl.exec("print('tim' in globals(), [repr(Timer(i)) for i in (2, 3)])" if False else
          "from machine import Timer\nprint('tim' in globals(), repr(Timer(2)), repr(Timer(3)))").strip())
    arm.disconnect()
