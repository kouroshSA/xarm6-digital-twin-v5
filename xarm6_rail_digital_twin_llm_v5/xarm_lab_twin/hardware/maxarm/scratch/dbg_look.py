import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, cv2
from hardware.maxarm.real_maxarm import RealMaxArmAPI
from hardware.maxarm.wrist_camera import MaxArmWristCamera
from perception.language import LanguageTargeter
arm = RealMaxArmAPI("/dev/ttyUSB0")
arm.set_position(75.0, -210.0, 200.0, speed=40); time.sleep(0.4)
cam = MaxArmWristCamera(arm)
try:
    f = cam.capture()
    cv2.imwrite(os.path.expanduser("~/Models/maxarm_calib/pick_place_dbg.png"), f.color[:, :, ::-1])
    lt = LanguageTargeter()
    for kw in [dict(max_size_m=0.06), dict(), dict(max_area_frac=1.0)]:
        t = lt.target("a green cube", f, distractors=["a red cube", "a paper marker"], attach_grasps=False, **kw)
        print(kw, None if t is None else (round(t.score, 2), [int(v) for v in t.box],
              np.round(np.array(t.size_m) * 1000).tolist(), np.round(t.position_world * 1000, 1).tolist()))
    dets = lt.grounder.detect(f.color, ["a green cube", "a red cube", "a paper marker"]) if hasattr(lt, "grounder") else None
    print("raw:", dets)
finally:
    arm.disconnect(); cam.close()
