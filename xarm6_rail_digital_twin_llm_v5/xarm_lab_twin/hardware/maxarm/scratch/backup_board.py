import sys, os, json, hashlib, binascii
sys.path.insert(0, os.getcwd())
from hardware.maxarm.transport import SerialLink, RawRepl
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
l = SerialLink("/dev/ttyUSB0"); r = RawRepl(l, timeout_s=20); r.enter()
r.exec("import os, ubinascii, uhashlib, ujson")
walk = r.exec("""
def _w(d):
    out = []
    for n in os.listdir(d):
        p = (d.rstrip('/') + '/' + n) if d != '' else n
        st = os.stat(p)
        if st[0] & 0x4000: out += _w(p)
        else: out.append([p, st[6]])
    return out
print(ujson.dumps(_w('')))
""")
files = json.loads(walk.strip())
manifest = []
for path, size in files:
    data = b""
    while len(data) < size:
        h = r.exec(f"f=open({path!r},'rb'); f.seek({len(data)}); print(ubinascii.hexlify(f.read(1024)).decode()); f.close()").strip()
        if not h: break
        data += binascii.unhexlify(h)
    bh = r.exec(f"h=uhashlib.sha256(); f=open({path!r},'rb')\nwhile True:\n    b=f.read(1024)\n    if not b: break\n    h.update(b)\nf.close(); print(ubinascii.hexlify(h.digest()).decode())").strip()
    local = hashlib.sha256(data).hexdigest()
    ok = len(data) == size and local == bh
    dst = os.path.join(out, path); os.makedirs(os.path.dirname(dst) or out, exist_ok=True)
    open(dst, "wb").write(data)
    manifest.append({"path": path, "size": size, "sha256": local, "board_sha256": bh, "ok": ok})
    print(f"  {'OK ' if ok else 'BAD'} {path:22s} {size:7d} B  {local[:12]}")
info = r.exec("import sys; print(sys.version, sys.implementation, os.uname())").strip()
json.dump({"board": info, "port": "/dev/ttyUSB0", "files": manifest}, open(os.path.join(out, "MANIFEST.json"), "w"), indent=2)
r.exit(); l.close()
print(f"{sum(m['ok'] for m in manifest)}/{len(manifest)} files verified -> {out}")
