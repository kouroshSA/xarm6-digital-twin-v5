import sys, os
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink, RawRepl
l = SerialLink("/dev/ttyUSB0"); r = RawRepl(l); r.enter()
print(r.exec("import sys, gc, time\nfrom machine import Timer\nprint([repr(Timer(i)) for i in range(4)])"))
print(r.exec("print(sorted(sys.modules.keys()))"))
print(r.exec("print(sorted(k for k in globals() if not k.startswith('__')))"))
print(r.exec("gc.collect(); a=gc.mem_free(); time.sleep_ms(1500); b=gc.mem_free(); print('mem_free', a, b, 'delta', a-b)"))
r.exit(); l.close()
