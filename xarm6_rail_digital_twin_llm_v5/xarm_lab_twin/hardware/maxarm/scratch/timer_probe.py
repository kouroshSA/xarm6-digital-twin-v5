import sys, os
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink, RawRepl
l = SerialLink("/dev/ttyUSB0"); r = RawRepl(l); r.enter()
print(r.exec("print('tim' in globals(), repr(globals().get('tim')))"))
print(r.exec("from machine import Timer\nt2 = Timer(2)\nprint(repr(t2), t2 is tim if 'tim' in globals() else None)"))
print(r.exec("import time\na = tim.value(); time.sleep_ms(50); print('value', a, tim.value())"))
print(r.exec("tim.deinit(); print(repr(tim))"))
print(r.exec("import time\na = tim.value(); time.sleep_ms(50); print('value after', a, tim.value())"))
r.exit(); l.close()
