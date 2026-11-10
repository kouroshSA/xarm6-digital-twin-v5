import sys, os, time
sys.path.insert(0, os.getcwd())
from hardware.maxarm.real_maxarm import RealMaxArmAPI, MaxArmError
CMDS = sys.argv[1]
X, Y = 0.0, -200.0
arm = RealMaxArmAPI("/dev/ttyUSB0")
arm.floor_z_mm = None                     # the floor is what is being measured
def where():
    p = arm.get_position()[1]; return f"read-back ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})"
def go(z, speed):
    try:
        arm.set_position(X, Y, z, speed=speed); return f"z={z:.0f} OK, {where()}"
    except MaxArmError as e:
        return f"z={z:.0f} DID NOT ARRIVE ({e}); {where()}"
z = 120.0
print("connected; stock loop", arm.stock_loop.split(';')[1].strip(), flush=True)
print(go(z, 30), flush=True)
while z > 90:
    z -= 5; print(go(z, 10), flush=True)
print("READY at z=90: waiting for commands", flush=True)
seen = 0
try:
    while True:
        lines = open(CMDS).read().splitlines()
        for c in lines[seen:]:
            c = c.strip(); seen += 1
            if c.startswith("d"):
                n = int(c[1:] or 1)
                for _ in range(n):
                    if z - 2 < 20: print("z=20 floor of the search; not going lower", flush=True); break
                    z -= 2; print(go(z, 10), flush=True)
            elif c == "touch":
                print(f"CONTACT at commanded wrist z = {z:.0f}; {where()}", flush=True)
                print(go(z + 30, 20), flush=True); raise SystemExit
            elif c == "abort":
                print(go(z + 30, 20), flush=True); raise SystemExit
        time.sleep(0.2)
finally:
    arm.disconnect(); print("disconnected (servos keep holding)", flush=True)
