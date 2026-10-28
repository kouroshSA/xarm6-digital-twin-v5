# teleop_hand/config.py
"""Tunables for hand-tracking teleop. Runner flags overwrite a few of these.

Workspace box and rail limits are NOT here: they come from `vr.config`, the
same numbers SpaceMouse, VR and kinesthetic teaching use.
"""
from __future__ import annotations

import numpy as np

from vr.config import RAIL_MAX_MM, RAIL_MIN_MM  # noqa: F401  (re-exported)

# ---- transport ------------------------------------------------------------
UDP_HOST = "127.0.0.1"
UDP_PORT = 47100
MSG_VERSION = 2

# ---- tracker --------------------------------------------------------------
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
#: MediaPipe landmarks whose depth gives the palm: wrist and the four MCPs.
PALM_LANDMARKS = (0, 5, 9, 13, 17)
DEPTH_WINDOW_PX = 5                 # median window around each landmark
MIN_VALID_PALM_POINTS = 3
#: Plane-fit RMS residual -> rot_quality: <= GOOD is 1.0, >= BAD is 0.0.
ROT_QUALITY_GOOD_MM = 4.0
ROT_QUALITY_BAD_MM = 15.0
#: Below this quality the tracker publishes MediaPipe's frame unfused.
FUSE_MIN_QUALITY = 0.5
#: MediaPipe world-landmark axes -> OpenCV camera axes (x right, y down,
#: z away from the camera). MEASURED identity in Phase 0 (README).
MEDIAPIPE_TO_CAMERA = np.eye(3)
FIST_DEBOUNCE_FRAMES = 3
#: Webcam fallback: z from apparent hand size, z = K / (wrist->middle MCP px).
#: --calibrate-depth replaces it; this default is a 0.6 m hand ~ 95 px.
WEBCAM_DEPTH_K = 0.6 * 95.0
MEDIAPIPE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                       "hand_landmarker/hand_landmarker/float16/latest/"
                       "hand_landmarker.task")
MEDIAPIPE_MODEL_SHA256 = ("fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1")

# ---- gripper (§3) ---------------------------------------------------------
#: Direct-pinch reference: thumb-tip to index-tip distance (m) that reads as
#: fully pinched / fully spread. Phase 2 calibrates these per operator.
PINCH_CLOSED_M = 0.02
PINCH_OPEN_M = 0.10
#: Binary effectors (twin weld, BIO, vacuum): close below, open above.
GRIP_CLOSE_BELOW = 0.30
GRIP_OPEN_ABOVE = 0.60
#: Real standard gripper: send only when the opening moved this much...
REAL_GRIP_MIN_CHANGE = 0.03
#: ...and no more often than this.
REAL_GRIP_MAX_HZ = 10.0
#: Freeze the arm target while the opening changes faster than this (1/s).
PINCH_FREEZE = True
PINCH_FREEZE_RATE = 1.5
#: Hold the freeze this long after the rate falls back, so the tail of the
#: close (the hand settling) is not tracked either.
PINCH_FREEZE_HOLD_S = 0.15

# ---- mapping (§4) ---------------------------------------------------------
MAP_MODE = "relative"               # relative | absolute
ORIENT_MODE = "locked"              # locked | yaw | full
RAIL_MODE = "edge"                  # edge | keys
CLUTCH = "key"                      # key | pedal | gesture
SCALE = 1.0
ROT_SCALE = 1.0
PRECISION_POS_FACTOR = 0.3
PRECISION_ROT_FACTOR = 0.5
ROT_DEADBAND_DEG = 5.0
#: Once past the deadband, rotation tracks fully until it falls back under
#: this, so crossing the band never produces a 5 deg step each way.
ROT_DEADBAND_EXIT_DEG = 2.5
YAW_MIN_QUALITY = 0.3
FULL_MIN_QUALITY = 0.5

#: Camera frame -> robot world frame. Robot world: x along the rail (the
#: operator's right), y away from the operator toward the bench, z up.
#: Camera (OpenCV): x image-right, y image-down, z out of the lens.
PRESETS = {
    # Camera in front of the operator, looking back at them. Image-right is
    # the operator's LEFT, so x is mirrored; toward the camera is toward the
    # robot (+y).
    "facing": np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, -1.0, 0.0]]),
    # Camera over the operator's shoulder, looking where they look.
    "behind": np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0]]),
}
PRESET = "facing"

# ---- filtering and dropouts (§4.4) ----------------------------------------
#: The position filter runs in MILLIMETRES: beta is Hz of extra cutoff per
#: mm/s. In metres the spec's beta=0.02 adds 0.02 Hz at 1 m/s -- nothing --
#: and the filter degenerates to a fixed 1 Hz low-pass, ~160 ms behind a
#: moving hand at any speed.
ONE_EURO_MIN_CUTOFF = 1.0           # Hz
ONE_EURO_BETA = 0.02                # Hz per mm/s
ONE_EURO_D_CUTOFF = 1.0             # Hz
ROT_LPF_CUTOFF_HZ = 3.0
JUMP_POS_MM = 50.0
JUMP_ROT_DEG = 30.0
DROPOUT_S = 0.15
SILENCE_S = 0.5

# ---- rail (§4.5) ----------------------------------------------------------
#: The arm's comfortable reach along the rail, either side of its base. The
#: workspace box spans the whole rail, so the edge-push needs a band that
#: moves WITH the base; this is it. Home already puts the tool ~290 mm along
#: the rail from the base, so the band must be wider than that or the rail
#: would start creeping the moment the clutch engages at home.
REACH_HALF_SPAN_MM = 400.0
EDGE_MARGIN_MM = 30.0
EDGE_GAIN = 2.0                     # rail mm/s per mm of overshoot
RAIL_MAX_SPEED = 150.0              # mm/s
RAIL_STEP_MM = 25.0                 # --rail keys
REAL_RAIL_MAX_HZ = 10.0

# ---- absolute mode (§4.1) -------------------------------------------------
#: Arm-relative target box, mm: x about the base along the rail, y/z world.
#: The calibrated hand box maps linearly onto this. It sits inside the shared
#: workspace box and around home (tool ~290 mm along the rail from the base,
#: y -53, z 1063 at rail 350), and its x extent stays inside the rail reach
#: band (REACH_HALF_SPAN_MM + EDGE_MARGIN_MM), or an absolute target would
#: start the edge-push. test_mapping checks that relation.
ABSOLUTE_BOX_MM = {"x": (150.0, 400.0), "y": (-200.0, 250.0), "z": (850.0, 1100.0)}

# ---- limits and guards (§4.7) ---------------------------------------------
FREQUENCY = 60.0
MAX_POS_SPEED = 150.0               # mm/s, on the commanded target
MAX_ROT_SPEED = 90.0                # deg/s
SOLVE_MIN_POS_MM = 0.5
SOLVE_MIN_ROT_DEG = 0.5
BRANCH_FLIP_DEG = 20.0
WRIST_SINGULAR_DEG = 5.0
JOINT_LIMIT_MARGIN_DEG = 3.0
#: An IK answer is refused if FK of it misses the target by more than this.
#: IKSolver returns rotation-missed solutions as successes (guards.py).
POSE_TOL_MM = 2.0
POSE_TOL_DEG = 2.0
CLAMP_TOLERANCE_MM = 1.0
IK_RETRY_PERIOD_S = 0.25
IK_WARN_PERIOD_S = 1.0

# ---- display marker (teach-handle mocap body) -----------------------------
RGBA_TRACKING = (1.0, 0.55, 0.0, 0.6)    # orange
RGBA_RELEASED = (0.55, 0.55, 0.55, 0.45) # grey
RGBA_BLOCKED = (1.0, 0.1, 0.1, 0.7)      # red
RGBA_PRECISION = (0.15, 0.45, 1.0, 0.6)  # blue
RGBA_GESTURE = (0.2, 0.8, 0.3, 0.6)      # green: fist-follow / point target

# ---- keys -----------------------------------------------------------------
CLUTCH_KEY = "ctrl_r"               # pynput Key name; --clutch-key overrides
PEDAL_KEY = "f9"                    # most USB pedals can be set to send F-keys
PRECISION_KEY = "shift_r"
RECOVER_KEY = "f12"                 # real arm: explicit error recovery
GLFW_LEFT, GLFW_RIGHT = 263, 262
GLFW_INSERT, GLFW_DELETE = 260, 261
GLFW_HOME, GLFW_END = 268, 269

# ---- real arm (§6) --------------------------------------------------------
REAL_STREAM_MODE = 7                # cartesian online trajectory planning
REAL_STREAM_HZ = 25.0
REAL_COLLISION_SENSITIVITY = 3      # 0..5
REAL_ERR_POLL_S = 0.2

# ---- recording ------------------------------------------------------------
RECORD = True
TASK_LABEL_SIM = "handtrack_sim"
TASK_LABEL_REAL = "handtrack_real"

# ---- gestures: fist-follow and point-to-grasp ------------------------------
#: Both are active only while the teleop clutch is RELEASED, so a fisted or
#: pointing hand can never fight the clutch-driven mapper (a fist closes the
#: pinch, which would read as "close the gripper" under the clutch).
GESTURE_DEBOUNCE_FRAMES = 3         # tracker: consecutive frames before a change
#: Index counts as extended when MCP->PIP and PIP->tip differ by less than this.
POINT_STRAIGHT_DEG = 40.0

# Fist-follow.
FOLLOW_STANDOFF_MM = (100.0, 150.0)  # the EE stays between these from the fist
FOLLOW_ENGAGE_S = 0.25              # fist held this long before the arm moves
FOLLOW_Z_MIN_MM = 870.0             # bench top is 750; keep the tool well above
FOLLOW_JUMP_MM = 120.0              # one-frame fist jump treated as an outlier
FOLLOW_JUMP_ACCEPT_FRAMES = 3       # ...unless this many frames agree on it
FOLLOW_MAX_SPEED_MM_S = 150.0

# Pointing.
POINT_DWELL_S = 1.0                 # ray must hold still this long to commit
POINT_STEADY_DEG = 4.0              # ...within this angle of its running mean
POINT_CONE_DEG = 8.0                # an object within this of the ray is "pointed at"
POINT_STANDOFF_MM = 80.0            # stop this far short of a surface the ray hits
POINT_FLOOR_CLEARANCE_MM = 100.0    # free-space targets stay this far above the bench
POINT_REACH_MM = 550.0              # free-space targets within this of the arm base
POINT_MAX_RAY_M = 2.5

# Point-to-grasp sequence (wrist camera + GG-CNN).
GRASP_LOOK_ABOVE_MM = 300.0         # flange height above the object for the look
GRASP_APPROACH_ABOVE_MM = 100.0
GRASP_MAX_PRIOR_ERR_MM = 25.0       # GG-CNN grasp must be this near the pointed object
GRASP_LIFT_MM = 120.0
GRASP_SPEED_MM_S = 80.0
GRASP_DESCENT_SPEED_MM_S = 50.0
