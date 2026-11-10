import sys, os
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink, RawRepl
l = SerialLink("/dev/ttyUSB0"); r = RawRepl(l); r.enter()
for f in sys.argv[1:]:
    print(f"===== {f}"); print(r.exec(f"print(open({f!r}).read())", 10))
r.exit(); l.close()
