import sys, os, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink
l = SerialLink("/dev/ttyUSB0"); buf = b""; t0 = time.monotonic()
while b">>> " not in buf and time.monotonic() - t0 < 8: buf += l.read_available()
print("booted in %.1f s" % (time.monotonic() - t0))
def say(line):
    l.write(line.encode() + b"\r"); time.sleep(0.6); out = b""
    for _ in range(20): out += l.read_available()
    print(out.decode(errors="replace").strip())
say("from machine import Timer; print([repr(Timer(i)) for i in range(4)], 'tim' in globals(), 'main' in globals())")
say("print(tim)")
l.close()
