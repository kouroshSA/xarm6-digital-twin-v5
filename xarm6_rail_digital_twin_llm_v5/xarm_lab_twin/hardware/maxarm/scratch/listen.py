import sys, time, os
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink
t0 = time.monotonic(); l = SerialLink("/dev/ttyUSB0")
buf = b""
while time.monotonic() - t0 < 6:
    d = l.read_available()
    if d: print(f"{time.monotonic()-t0:5.2f} {d!r}"); buf += d
l.write(b"\r\x03\x03"); time.sleep(0.5); print("after ^C:", l.read_available() + l.read_available())
l.write(b"\r\x01"); time.sleep(0.5); print("after ^A:", l.read_available() + l.read_available())
l.write(b"\x02"); time.sleep(0.3); print("after ^B:", l.read_available() + l.read_available())
l.close()
