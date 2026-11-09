import sys, os, time, json
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
import scripts.maxarm_hand_eye as m
from teleop_hand.hand_calib import detect_tag
tag = json.load(open(os.path.expanduser("~/Models/maxarm_calib/tag_place.json")))["tag_xyz_arm_mm"]
cam, _ = m._camera()
for _ in range(60): cam.capture()                     # let auto-exposure settle
arm = m._arm()
out = "/tmp/claude-1000/-home-ksa-Models-xarm6-digital-twin-v5/0b7f2466-01e3-444a-bf2b-0a36a5020733/scratchpad/maxarm_frame"
try:
    for z in (210, 185, 160):
        arm.set_position(tag[0], tag[1], z, speed=40); time.sleep(0.5)
        for _ in range(10): f = cam.capture()
        o = detect_tag(f.color, f.intrinsics.matrix, 0.04, f.depth)
        print(f"z={z}: brightness {f.color.mean():.0f}, depth median {np.nanmedian(f.depth):.3f} m, tag",
              None if o is None else f"uv {np.round(o.corners_px.mean(0)).tolist()} cam {np.round(o.pos_cam_m*1000,1).tolist()} mm reproj {o.reproj_px:.2f}", flush=True)
        Image.fromarray(f.color).save(f"{out}/look_z{z}.png")
finally:
    arm.disconnect(); cam.close()
