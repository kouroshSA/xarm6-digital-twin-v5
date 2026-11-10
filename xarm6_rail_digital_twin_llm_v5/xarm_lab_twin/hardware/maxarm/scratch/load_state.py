import os, sys, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI
arm = RealMaxArmAPI("/dev/ttyUSB0")
arm.repl.exec("""
import time as _t
def _load_state(i):
    _bs.uart.read()
    b = [0x55, 0x55, i, 3, 32]
    b.append(_bs.check_sum(b))
    _bs.send_data(b)
    t0 = _t.ticks_ms()
    while _bs.uart.any() == 0:
        if _t.ticks_ms() - t0 > 50:
            return None
    if _bs.servo_receive_handle():
        return _bs.rx_buf[5]
    return 'bad'
""")
st = lambda: [arm._call(f"_r(_load_state, {i})", "load state") for i in (1, 2, 3)]
try:
    print("torque on now, load state:", st())
    for i in (1, 2, 3):
        arm._call(f"_r(_bs.unload, {i})", "unload")
    time.sleep(0.3)
    print("after unload, load state:", st())
finally:
    p = arm._read_pulses()
    arm._call(f"_r(_move, {p[0]}, {p[1]}, {p[2]}, 500)", "hold in place")
    time.sleep(0.8)
    print("after run-in-place, load state:", st(), "pulses", arm._read_pulses())
    arm.disconnect()
