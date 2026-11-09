import sys, os, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink
l = SerialLink("/dev/ttyUSB0"); buf = b""; t0 = time.monotonic()
while b">>> " not in buf and time.monotonic() - t0 < 8: buf += l.read_available()
def send(b, wait=0.6):
    l.write(b); time.sleep(wait); out = b""
    for _ in range(20): out += l.read_available()
    return out
Q = b"print('tim' in globals(), 'main' in globals(), 'arm' in globals(), repr(__import__('machine').Timer(2)))\r"
print("fresh      :", send(Q))
print("after ^C^C :", send(b"\r\x03\x03"))
print("check      :", send(Q))
print("after ^A   :", send(b"\r\x01"))
print("raw exec   :", send(Q.rstrip(b"\r") + b"\x04", 1.0))
print("raw exec 2 :", send(Q.rstrip(b"\r") + b"\x04", 1.0))
send(b"\x02"); l.close()
