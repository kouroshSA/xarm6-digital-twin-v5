"""Intel RealSense D435i wrist camera — the single source of truth.

Every number describing the wrist camera lives here exactly once: the stream
config, the colour/depth intrinsics, the depth->colour extrinsic, and the
flange->camera hand-eye transform. The sim camera, the real camera driver and
the scene XML all derive from this module rather than repeating it.

That is deliberate. This repo's standing defect class #1 is "one fact, several
copies, and the copies drift" (see CLAUDE.md). A camera is unusually prone to
it: intrinsics naturally want to be written into the scene XML, into the
deprojection code, and into whatever consumes point clouds. Here they are
written once. ``scripts/sim_checks.py::check_wrist_camera_matches_calib`` ties
the scene XML's copy back to this module so the two cannot silently diverge,
and ``python -m perception.d435i_calib --emit-xml`` regenerates the XML block
mechanically instead of by hand.

Provenance
----------
* **Intrinsics / extrinsics** — read off the physical device on 2026-08-25 via
  ``rs-enumerate-devices -c``. Serial ``027422071693``, firmware 5.11.1.100.
  These are Intel's factory calibration, not a re-calibration of ours.
* **Hand-eye (``EULER_FLANGE_TO_COLOR_OPT``)** — UFACTORY's published figure for
  their xArm camera stand, lifted verbatim from
  ``ufactory_vision/ggcnn_grasping_demo/example/realsense_d435/run_rs_d435_grasp.py``
  (``EULER_EEF_TO_COLOR_OPT``, BSD-3). It is the transform from the xArm6 tool
  flange (``link_eef``, i.e. zero TCP offset) to the colour *optical* frame.

  **Assumption worth knowing:** this presumes the camera is on UFACTORY's
  standard stand. If the real rig uses a different bracket, this is the one
  constant to re-measure -- everything else here is the camera's own factory
  data and stays valid. Re-measuring it changes this file only.

Frame conventions
-----------------
Two conventions meet in this module and mixing them up is the classic wrist-cam
bug, so they are named explicitly everywhere:

* **optical** (RealSense / ROS ``*_optical_frame``, and OpenCV): ``+x`` right,
  ``+y`` down, ``+z`` forward into the scene. All intrinsics below are in this
  convention.
* **MuJoCo camera**: ``+x`` right, ``+y`` up, ``-z`` forward. A MuJoCo camera
  looks down its own ``-z``.

The two differ by a 180-degree rotation about ``x``; ``mujoco_mount_quat()``
applies it. Nothing else in the codebase should hand-roll that flip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Device ---------------------------------------------------------------
DEVICE_NAME = "Intel RealSense D435I"
DEVICE_SERIAL = "027422071693"
DEVICE_FIRMWARE = "5.11.1.100"
CALIB_READ_DATE = "2026-08-25"

# --- Stream configuration -------------------------------------------------
# 640x480 @ 30 Hz on both streams. Chosen because it is the only resolution the
# colour and depth sensors share natively at 4:3, so depth->colour alignment
# involves no rescaling, and because it is what ufactory_vision's GGCNN demo
# runs. The intrinsics below are specific to this resolution -- RealSense
# reports different fx/fy per profile, so changing WIDTH/HEIGHT invalidates
# COLOR_INTRINSICS and DEPTH_INTRINSICS. Re-read them from the device if you do.
WIDTH = 640
HEIGHT = 480
FPS = 30

# Depth is quantised in units of this many metres (D400 default depth scale).
DEPTH_SCALE_M = 0.001

# Beyond this the D435i's stereo depth is too noisy to act on; the sim renderer
# clips to the same range so sim and real degrade the same way.
DEPTH_MIN_M = 0.105   # datasheet min-Z for D435 at 640x480
DEPTH_MAX_M = 3.0


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in the OpenCV/optical convention (+x right, +y down)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 camera matrix K, as GGCNN and OpenCV expect it."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def fovy_deg(self) -> float:
        """Vertical field of view, for MuJoCo's ``fovy`` attribute."""
        return math.degrees(2.0 * math.atan(self.height / (2.0 * self.fy)))

    @property
    def fovx_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.fx)))


# Factory intrinsics at 640x480. Distortion coefficients are all zero on this
# unit (Intel ships the D400 colour stream rectified), which is why nothing here
# undistorts: there is nothing to undo. If a future device reports non-zero
# Inverse-Brown-Conrady coeffs, the sim's ideal pinhole stops being a faithful
# twin and the real frames need undistorting before they are comparable.
COLOR_INTRINSICS = Intrinsics(
    width=WIDTH, height=HEIGHT,
    fx=610.523071289062, fy=610.968688964844,
    cx=318.129516601562, cy=246.347259521484,
)

# The depth (stereo) imager is much wider-angle than the colour one: 80.1 x 64.5
# degrees against 55.3 x 42.9. So an unaligned depth frame sees well beyond the
# colour frame's edges, and the sim models the two as separate cameras rather
# than one shared frustum.
DEPTH_INTRINSICS = Intrinsics(
    width=WIDTH, height=HEIGHT,
    fx=380.666198730469, fy=380.666198730469,
    cx=324.783020019531, cy=240.828155517578,
)

# Depth -> colour extrinsic, straight from the device. The rotation is within
# 0.3 degrees of identity, so the two imagers are effectively parallel and the
# only meaningful term is the 14.7 mm stereo-to-RGB baseline along +x.
#
# NOTE: when frames are captured with ``align=True`` the depth image has already
# been reprojected into the colour frame by librealsense, so the *aligned* depth
# map must be deprojected with COLOR_INTRINSICS, not DEPTH_INTRINSICS. Getting
# this backwards produces a point cloud with a ~15 mm lateral bias that looks
# entirely plausible -- exactly the kind of quietly-wrong value CLAUDE.md warns
# about. ``RGBDFrame.intrinsics`` carries the right one so callers never choose.
_DEPTH_TO_COLOR_ROT_RAW = np.array(
    [[0.999977, 0.00451854, 0.00500225],
     [-0.00451371, 0.999989, -0.000976488],
     [-0.00500661, 0.000953887, 0.999987]],
    dtype=np.float64,
)


def _nearest_rotation(m: np.ndarray) -> np.ndarray:
    """Project a matrix onto the nearest true rotation (SVD / orthogonal Procrustes).

    ``rs-enumerate-devices`` prints extrinsics to six significant figures, and a
    rounded rotation matrix is not quite a rotation: this one's determinant is
    0.9999994 and it sits 0.045 degrees from the nearest orthonormal matrix.
    That is small, but it is not nothing -- it propagates into the depth
    camera's scene pose as a real angular error, and it makes "does the scene
    match the calibration?" un-answerable, because converting to a quaternion
    silently projects while comparing against the raw matrix does not.

    Projecting once, here, means every consumer sees the same valid rotation.
    """
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:  # reflection, not a rotation
        u[:, -1] *= -1
        r = u @ vt
    return r


DEPTH_TO_COLOR_ROT = _nearest_rotation(_DEPTH_TO_COLOR_ROT_RAW)
DEPTH_TO_COLOR_TRANS = np.array(
    [0.0147364465519786, 8.33612575661391e-05, 0.00028126998222433],
    dtype=np.float64,
)

# --- Hand-eye -------------------------------------------------------------
# xArm6 tool flange -> colour optical frame, as [x, y, z, roll, pitch, yaw]
# in metres and radians (ROS 'sxyz' convention, i.e. R = Rz(yaw) Ry(pitch) Rx(roll)).
#
# Sanity check that this is the right way round, and worth keeping: the optical
# +z (view direction) works out to [-0.004, -0.009, 1.000] in flange
# coordinates -- within half a degree of the flange's own +z. A wrist camera
# should look where the gripper reaches, and this one does. If a re-calibration
# ever produces a view axis that is not near flange +z, it is wrong.
EULER_FLANGE_TO_COLOR_OPT = (
    0.067052239, -0.0311387575, 0.021611456,     # xyz, metres
    -0.004202176, -0.00848499, 1.5898775,        # rpy, radians
)

# The MuJoCo ``gripper`` body sits this far out along link6's +z. The scene
# places the camera under that body (it is rigidly fixed to the flange, and
# living in the gripper subtree means ``build_mesh_scene.py`` copies it verbatim
# into the generated mesh scene -- one copy, both scenes). So every pose this
# module hands to the XML is rebased by subtracting it.
GRIPPER_BODY_OFFSET_M = np.array([0.0, 0.0, 0.06], dtype=np.float64)

# Scene XML names, referenced by the sim camera and the parity check.
COLOR_CAM_NAME = "cam_wrist_color"
DEPTH_CAM_NAME = "cam_wrist_depth"
CAMERA_BODY_NAME = "wrist_camera"

# D435 housing, from Intel's datasheet: 90 mm wide x 25 mm tall x 25 mm deep.
# Half-extents, ordered in the MuJoCo *camera* frame (+x right, +y up, -z
# forward) because that is the frame the wrist_camera body carries.
#
# The geom is contype/conaffinity 0, so it never generates contacts and cannot
# invalidate an existing collision-free trajectory. It does carry the device's
# real 72 g, which is NOT inert: it takes the gripper subtree from 190 g to
# 262 g. That is deliberate -- a real D435i bolted to the flange really does add
# that mass, and a twin that renders the camera but pretends it is weightless is
# lying about the arm. It is 5% of the xArm6's 5 kg rated payload, so it changes
# nothing about reachability, but if a future trajectory is ever tuned to the
# gram, this is where the extra 72 g came from.
HOUSING_MASS_KG = 0.072
HOUSING_HALF_EXTENTS_M = (0.045, 0.0125, 0.0125)

# The housing is pushed back along the camera's +z (i.e. *away* from the view
# direction, since MuJoCo cameras look down -z) by its own half-depth, so its
# front face lands on the optical plane rather than straddling it.
#
# This is not cosmetic tidying. With the box centred on the optical origin the
# camera sits inside its own housing: MuJoCo's near clip plane happens to hide
# that from the renderer, so colour and depth look perfectly fine, but anything
# that does not respect the near plane -- ray casting, a point-cloud collision
# query -- hits the housing 12.5 mm in front of the lens and believes it. A
# discrepancy the images cannot show you is exactly the kind worth designing out
# rather than remembering.
#
# The extra 1 mm past the half-depth keeps the front face strictly *behind* the
# optical plane rather than exactly on it. With the face at z=0 a ray cast from
# the camera origin starts precisely on the box surface, and MuJoCo's ray caster
# reports a hit on the housing -- a boundary case that makes any
# visibility-from-the-camera query answer "my own body".
HOUSING_Z_OFFSET_M = HOUSING_HALF_EXTENTS_M[2] + 0.001


def _euler_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS 'sxyz' euler -> rotation matrix. Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def flange_to_color_optical() -> tuple[np.ndarray, np.ndarray]:
    """(R, t) taking a point in the colour optical frame to the flange frame."""
    x, y, z, roll, pitch, yaw = EULER_FLANGE_TO_COLOR_OPT
    return _euler_to_mat(roll, pitch, yaw), np.array([x, y, z], dtype=np.float64)


def flange_to_depth_optical() -> tuple[np.ndarray, np.ndarray]:
    """(R, t) for the depth (left-IR) optical frame, in the flange frame."""
    r_color, t_color = flange_to_color_optical()
    # colour = depth + t_d2c, so depth sits t_d2c *behind* colour.
    return r_color @ DEPTH_TO_COLOR_ROT.T, t_color - r_color @ DEPTH_TO_COLOR_TRANS


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), MuJoCo's quaternion order."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# optical (+y down, +z forward) -> MuJoCo camera (+y up, -z forward).
_OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def mujoco_mount_quat(optical_rot: np.ndarray) -> np.ndarray:
    """Convert an optical-frame rotation into MuJoCo camera-frame (w,x,y,z)."""
    return _mat_to_quat(optical_rot @ _OPTICAL_TO_MUJOCO)


def sensor_size_m(intr: Intrinsics) -> tuple[float, float]:
    """A virtual sensor size for MuJoCo's ``sensorsize`` attribute.

    MuJoCo refuses ``focalpixel``/``principalpixel`` without a physical sensor
    size, but the pixel-denominated values it derives from them are invariant to
    the pitch we pick: it stores ``focal_len = focalpixel * sensorsize /
    resolution`` and inverts the same ratio at render time. We use a 3 um pitch,
    which puts the virtual sensor in the right ballpark for a real D435 imager.
    """
    pitch = 3.0e-6
    return intr.width * pitch, intr.height * pitch


def principal_pixel(intr: Intrinsics) -> tuple[float, float]:
    """OpenCV ``(cx, cy)`` -> MuJoCo's ``principalpixel`` pair.

    The conversion is not the obvious ``cx - width/2``. Both parts of that are
    wrong, and neither is guessable -- they were measured, by rendering markers
    at known camera-frame positions and finding where they actually landed
    (``perception.test_projection`` re-runs that measurement as a test):

    * **Origin.** ``principalpixel="0 0"`` puts the optical axis at pixel
      ``((W-1)/2, (H-1)/2)``, not ``(W/2, H/2)``. OpenCV and RealSense index
      pixel *centres* from 0, so their centre is ``(W-1)/2`` too; using ``W/2``
      introduces a half-pixel bias.
    * **Sign.** Increasing ``principalpixel`` moves the axis towards *lower*
      u and v -- in **both** axes. It is tempting to reason that only y flips,
      because MuJoCo's image y runs up while OpenCV's runs down; that reasoning
      gives the right answer for y and the wrong one for x.

    A wrong value here is a pure translation of the image: focal length still
    checks out, straight lines stay straight, and every self-consistent
    round-trip through the same intrinsics still closes. It only shows up
    against an independent source of truth, which is why the projection test
    compares against MuJoCo's ray caster rather than against itself.
    """
    return ((intr.width - 1) / 2.0 - intr.cx,
            (intr.height - 1) / 2.0 - intr.cy)


def mujoco_camera_attrs(intr: Intrinsics) -> dict[str, str]:
    """Camera XML attributes reproducing ``intr`` exactly under MuJoCo.

    Uses MuJoCo's exact-intrinsics path (``resolution`` + ``sensorsize`` +
    ``focalpixel`` + ``principalpixel``) rather than ``fovy``, so the sim
    reproduces the real principal-point offset instead of assuming a perfectly
    centred one.
    """
    sw, sh = sensor_size_m(intr)
    px, py = principal_pixel(intr)
    return {
        "resolution": f"{intr.width} {intr.height}",
        "sensorsize": f"{sw:.9f} {sh:.9f}",
        "focalpixel": f"{intr.fx:.6f} {intr.fy:.6f}",
        "principalpixel": f"{px:.6f} {py:.6f}",
    }


def scene_camera_poses() -> dict[str, dict[str, object]]:
    """Pose + intrinsics for each wrist camera, rebased into the gripper body.

    This is what the scene XML must contain, and what the parity check in
    ``scripts/sim_checks.py`` compares the XML against.
    """
    out: dict[str, dict[str, object]] = {}
    for name, (rot, trans), intr in (
        (COLOR_CAM_NAME, flange_to_color_optical(), COLOR_INTRINSICS),
        (DEPTH_CAM_NAME, flange_to_depth_optical(), DEPTH_INTRINSICS),
    ):
        out[name] = {
            "pos": trans - GRIPPER_BODY_OFFSET_M,
            "quat": mujoco_mount_quat(rot),
            "intrinsics": intr,
        }
    return out


def emit_scene_xml(indent: str = " " * 24) -> str:
    """Render the ``<body name="wrist_camera">`` block for the scene XML.

    Paste the output into ``envs/lab_scene_primitive.xml`` inside
    ``<body name="gripper">``, then rerun ``python envs/build_mesh_scene.py``.
    Generating it beats typing it: the numbers stay tied to this module, and
    ``check_wrist_camera_matches_calib`` proves they still are.
    """
    poses = scene_camera_poses()
    hx, hy, hz = HOUSING_HALF_EXTENTS_M
    colour = poses[COLOR_CAM_NAME]
    cp = colour["pos"]
    cq = colour["quat"]

    lines = [
        f'{indent}<!-- Intel RealSense D435i wrist camera (eye-in-hand).',
        f'{indent}     GENERATED by perception/d435i_calib.py::emit_scene_xml() --',
        f'{indent}     do not retype these numbers, regenerate them. Pose is the',
        f'{indent}     UFACTORY camera-stand hand-eye calibration, rebased from the',
        f'{indent}     flange into this gripper body. Intrinsics are the physical',
        f'{indent}     device\'s factory values (serial {DEVICE_SERIAL}) at',
        f'{indent}     {WIDTH}x{HEIGHT}. The housing geom is contype/conaffinity 0,',
        f'{indent}     so the camera is visible but physically inert. -->',
        f'{indent}<body name="{CAMERA_BODY_NAME}"',
        f'{indent}      pos="{cp[0]:.9f} {cp[1]:.9f} {cp[2]:.9f}"',
        f'{indent}      quat="{cq[0]:.9f} {cq[1]:.9f} {cq[2]:.9f} {cq[3]:.9f}">',
        f'{indent}  <geom name="d435i_housing" type="box"',
        f'{indent}        size="{hx:.4f} {hy:.4f} {hz:.4f}"',
        f'{indent}        pos="0 0 {HOUSING_Z_OFFSET_M:.4f}"',
        f'{indent}        rgba="0.10 0.10 0.12 1" mass="{HOUSING_MASS_KG}"',
        f'{indent}        contype="0" conaffinity="0"/>',
    ]

    # The stand that holds the camera off the flange. Cosmetic (mass 0,
    # non-colliding) but not decoration: without it the housing hangs in space
    # beside the wrist in every render, which reads as a bug in the model.
    #
    # Its pose is DERIVED from the same hand-eye constants as the camera rather
    # than typed next to them -- a bracket carrying its own copy of the offset
    # would keep pointing at where the camera used to be after a re-calibration.
    b_pos, b_quat, b_half = _bracket_in_camera_frame()
    lines += [
        f'{indent}  <geom name="d435i_bracket" type="box"',
        f'{indent}        size="{b_half:.5f} 0.006 0.004"',
        f'{indent}        pos="{b_pos[0]:.6f} {b_pos[1]:.6f} {b_pos[2]:.6f}"',
        f'{indent}        quat="{b_quat[0]:.6f} {b_quat[1]:.6f} '
        f'{b_quat[2]:.6f} {b_quat[3]:.6f}"',
        f'{indent}        rgba="0.30 0.30 0.33 1" mass="0"',
        f'{indent}        contype="0" conaffinity="0"/>',
    ]
    r_parent = _quat_to_mat(cq)
    for name in (COLOR_CAM_NAME, DEPTH_CAM_NAME):
        entry = poses[name]
        # Rebase again: the cameras are children of wrist_camera, which already
        # carries the colour pose, so each camera's own offset is relative to it.
        # MuJoCo reads a child's pos in the PARENT's rotated frame, so the
        # gripper-frame difference has to come back through r_parent -- differencing
        # alone silently puts the depth imager 14.7 mm along the wrong axis.
        rel_pos = r_parent.T @ (entry["pos"] - cp)
        rel_rot = r_parent.T @ _quat_to_mat(entry["quat"])
        rq = _mat_to_quat(rel_rot)
        attrs = mujoco_camera_attrs(entry["intrinsics"])
        lines += [
            f'{indent}  <camera name="{name}"',
            f'{indent}          pos="{rel_pos[0]:.9f} {rel_pos[1]:.9f} {rel_pos[2]:.9f}"',
            f'{indent}          quat="{rq[0]:.9f} {rq[1]:.9f} {rq[2]:.9f} {rq[3]:.9f}"',
            f'{indent}          resolution="{attrs["resolution"]}"',
            f'{indent}          sensorsize="{attrs["sensorsize"]}"',
            f'{indent}          focalpixel="{attrs["focalpixel"]}"',
            f'{indent}          principalpixel="{attrs["principalpixel"]}"/>',
        ]
    lines.append(f'{indent}</body>')
    return "\n".join(lines)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _bracket_in_camera_frame() -> tuple[np.ndarray, np.ndarray, float]:
    """Pose and half-length of the camera stand, in the camera's own frame.

    The stand runs from the camera back to the flange axis at the camera's own
    height. Returns ``(pos, quat, half_length)`` for a box whose local +x lies
    along that run, so the geom follows automatically if the hand-eye
    calibration changes.
    """
    rot, trans = flange_to_color_optical()
    # A point on the flange axis, level with the camera: same z, no offset.
    to_axis_flange = np.array([-trans[0], -trans[1], 0.0], dtype=np.float64)
    r_mj = rot @ _OPTICAL_TO_MUJOCO
    v = r_mj.T @ to_axis_flange          # same vector, camera frame
    length = float(np.linalg.norm(v))

    # Rotation taking local +x onto v.
    x_axis = v / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(x_axis @ helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(helper, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    quat = _mat_to_quat(np.column_stack([x_axis, y_axis, z_axis]))

    # Centred halfway along the run, and set back behind the lens plane so the
    # stand does not intrude into the camera's own view.
    pos = v / 2.0 + np.array([0.0, 0.0, HOUSING_Z_OFFSET_M + 0.004])
    return pos, quat, length / 2.0


def summary() -> str:
    """Human-readable dump, used by the check scripts and worth eyeballing."""
    r_c, t_c = flange_to_color_optical()
    return "\n".join([
        f"{DEVICE_NAME}  serial={DEVICE_SERIAL}  fw={DEVICE_FIRMWARE}",
        f"  streams        : colour + depth {WIDTH}x{HEIGHT} @ {FPS} Hz",
        f"  colour intrin  : fx={COLOR_INTRINSICS.fx:.3f} fy={COLOR_INTRINSICS.fy:.3f} "
        f"cx={COLOR_INTRINSICS.cx:.3f} cy={COLOR_INTRINSICS.cy:.3f}",
        f"  colour FOV     : {COLOR_INTRINSICS.fovx_deg:.2f} x "
        f"{COLOR_INTRINSICS.fovy_deg:.2f} deg",
        f"  depth intrin   : fx={DEPTH_INTRINSICS.fx:.3f} fy={DEPTH_INTRINSICS.fy:.3f} "
        f"cx={DEPTH_INTRINSICS.cx:.3f} cy={DEPTH_INTRINSICS.cy:.3f}",
        f"  flange->colour : t={np.round(t_c, 6).tolist()} m",
        f"  view axis      : {np.round(r_c[:, 2], 6).tolist()} (flange frame; "
        f"should be ~[0,0,1])",
    ])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit-xml", action="store_true",
                    help="print the scene XML block for lab_scene_primitive.xml")
    args = ap.parse_args()
    print(emit_scene_xml() if args.emit_xml else summary())
