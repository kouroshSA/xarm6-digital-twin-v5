import sys, os, json
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink, RawRepl
from hardware.maxarm import kinematics as kin
l = SerialLink("/dev/ttyUSB0"); r = RawRepl(l); r.enter()
r.exec("import __espmax, ujson")
def b(expr): return json.loads(r.exec(f"print(ujson.dumps(list({expr})))").strip())
for p in [(120,-180,85), (150,-150,150), (200,10,100)]:
    bi = b(f"__espmax.inverse({kin.L4}, {tuple(map(float,p))})")
    print("inverse", p, "board", [round(v,3) for v in bi], "port same-args", [round(v,3) for v in kin.inverse(p)])
    bf = b(f"__espmax.forward({kin.L4}, {tuple(bi)})")
    print("   forward(board joints) board", [round(v,2) for v in bf], "port", [round(v,2) for v in kin.forward(bi)])
r.exit(); l.close()
