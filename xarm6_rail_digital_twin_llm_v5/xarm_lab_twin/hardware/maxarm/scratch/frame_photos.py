import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np, pyrealsense2 as rs
from PIL import Image, ImageDraw
from hardware.maxarm.real_maxarm import RealMaxArmAPI
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
pipe.start(cfg)
for _ in range(30): pipe.wait_for_frames()          # auto-exposure settles
def shot(name, label):
    for _ in range(5): f = pipe.wait_for_frames()
    im = Image.fromarray(np.asanyarray(f.get_color_frame().get_data()).copy())
    ImageDraw.Draw(im).text((12, 12), label, fill=(255, 255, 0))
    im.save(f"{out}/{name}.png"); print("saved", name, flush=True); return im
arm = RealMaxArmAPI("/dev/ttyUSB0")
ims = []
try:
    arm.go_home(); ims.append(shot("1_home", f"home {tuple(round(v) for v in arm.get_position()[1][:3])}"))
    arm.set_position(60, -180, 150, speed=30); time.sleep(0.5)
    ims.append(shot("2_x_plus60", f"x=+60 {tuple(round(v) for v in arm.get_position()[1][:3])}"))
    arm.set_position(-60, -180, 150, speed=30); time.sleep(0.5)
    ims.append(shot("3_x_minus60", f"x=-60 {tuple(round(v) for v in arm.get_position()[1][:3])}"))
    arm.go_home()
finally:
    arm.disconnect(); pipe.stop()
w, h = ims[0].size
sheet = Image.new("RGB", (w * 3 // 2, h // 2))
for i, im in enumerate(ims): sheet.paste(im.resize((w // 2, h // 2)), (i * w // 2, 0))
sheet.save(f"{out}/sheet.png"); print("sheet done")
