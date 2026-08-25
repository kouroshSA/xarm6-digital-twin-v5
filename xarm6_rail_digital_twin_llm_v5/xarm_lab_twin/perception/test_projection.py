"""Does the sim camera's pinhole model actually match what MuJoCo renders?

Run: ``MUJOCO_GL=egl python -m perception.test_projection``

Why this exists rather than a unit test of the maths
----------------------------------------------------
Every intrinsic in ``d435i_calib`` is used twice: once by MuJoCo, to render, and
once by ``RGBDFrame``, to deproject. Testing either alone proves nothing --
they can agree with themselves and disagree with each other, which is the "value
computed correctly and delivered to nobody" shape CLAUDE.md warns about. A
principal-point sign error, in particular, is a few-pixel bias that no
self-consistent round-trip would ever surface: deproject-then-reproject returns
the pixel you started with whether the sign is right or wrong.

So this compares against a third, independent source of truth: MuJoCo's own ray
caster. For a grid of pixels we

1. deproject the *rendered* depth into a world point, and
2. cast a ray from the camera along the direction our intrinsics imply, letting
   ``mj_ray`` report where the scene geometry actually is,

and require the two to land on the same point. Renderer, ray caster and our
pinhole model all have to agree, and the ray caster does not read our intrinsics
at all -- so a wrong principal point shows up immediately as a bias that grows
towards the image edges.

The suite also checks the two facts that make the camera the *right* camera:
that MuJoCo recovers the physical device's fx/fy/cx/cy, and that the mount
reproduces UFACTORY's hand-eye calibration relative to the flange.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402  (must follow the MUJOCO_GL default)

from . import d435i_calib as calib  # noqa: E402
from .sim_camera import SimWristCamera  # noqa: E402

SCENE = os.path.join(os.path.dirname(__file__), "..", "envs", "lab_scene.xml")

# A ray and a rendered pixel should agree to far better than a millimetre; the
# only real source of disagreement is depth quantisation, which _condition_depth
# deliberately imposes at 1 mm to match the device.
POINT_TOL_M = 2.0e-3
INTRINSICS_TOL_PX = 0.01
POSE_TOL_M = 1.0e-6


def _pose_for_a_useful_view(model, data) -> None:
    """Bend the arm so the wrist camera looks at the bench rather than the sky.

    Any pose with geometry in frame does; this one just reliably puts the bench
    and its objects inside the colour frustum.
    """
    for name, deg in (("joint1", 0.0), ("joint2", -35.0), ("joint3", -60.0),
                      ("joint4", 0.0), ("joint5", 95.0), ("joint6", 0.0)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = np.radians(deg)
    mujoco.mj_forward(model, data)


def check_intrinsics_round_trip(model) -> list[str]:
    """MuJoCo must store exactly the intrinsics the physical device reports."""
    failures = []
    for cam_name, intr in ((calib.COLOR_CAM_NAME, calib.COLOR_INTRINSICS),
                           (calib.DEPTH_CAM_NAME, calib.DEPTH_INTRINSICS)):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        res = model.cam_resolution[cid]
        ss = model.cam_sensorsize[cid]
        fx_len, fy_len, cx_len, cy_len = model.cam_intrinsic[cid]

        fx = fx_len / ss[0] * res[0]
        fy = fy_len / ss[1] * res[1]
        # Inverse of d435i_calib.principal_pixel -- see that docstring for why
        # the origin is (res-1)/2 and why both axes are negated.
        cx = (res[0] - 1) / 2.0 - cx_len / ss[0] * res[0]
        cy = (res[1] - 1) / 2.0 - cy_len / ss[1] * res[1]

        for label, got, want in (("fx", fx, intr.fx), ("fy", fy, intr.fy),
                                 ("cx", cx, intr.cx), ("cy", cy, intr.cy)):
            if abs(got - want) > INTRINSICS_TOL_PX:
                failures.append(
                    f"{cam_name}.{label}: scene has {got:.4f}, device reports "
                    f"{want:.4f} (delta {got - want:+.4f} px)")
    return failures


def check_mount_matches_handeye(model, data) -> list[str]:
    """The camera must sit where UFACTORY's hand-eye calibration says it does."""
    failures = []
    l6 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    r_flange = data.xmat[l6].reshape(3, 3)

    for cam_name, (r_want, t_want) in (
        (calib.COLOR_CAM_NAME, calib.flange_to_color_optical()),
        (calib.DEPTH_CAM_NAME, calib.flange_to_depth_optical()),
    ):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        t_got = r_flange.T @ (data.cam_xpos[cid] - data.xpos[l6])
        if not np.allclose(t_got, t_want, atol=POSE_TOL_M):
            failures.append(
                f"{cam_name} sits at {np.round(t_got * 1000, 4).tolist()} mm in "
                f"the flange frame; calibration says "
                f"{np.round(t_want * 1000, 4).tolist()} mm")

        # cam_xmat is the MuJoCo camera frame; convert to optical to compare.
        r_got = (r_flange.T @ data.cam_xmat[cid].reshape(3, 3)) @ np.diag([1.0, -1.0, -1.0])
        if not np.allclose(r_got, r_want, atol=1e-6):
            ang = np.degrees(np.arccos(np.clip(
                (np.trace(r_got.T @ r_want) - 1.0) / 2.0, -1.0, 1.0)))
            failures.append(f"{cam_name} orientation is {ang:.4f} deg off the "
                            f"hand-eye calibration")
    return failures


def check_render_matches_raycast(model, data, cam) -> list[str]:
    """The load-bearing test. Rendered depth vs MuJoCo's ray caster."""
    frame = cam.capture(align=True)
    c2w = frame.cam_to_world
    origin = c2w[:3, 3]
    rot = c2w[:3, :3]
    k = frame.intrinsics

    # A grid spanning the frame, including near the edges where a principal-point
    # error is largest. Skipping the outer 40 px avoids grazing hits where a
    # sub-pixel disagreement turns into a large depth difference.
    us = np.linspace(40, k.width - 40, 9)
    vs = np.linspace(40, k.height - 40, 7)

    geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)

    errors = []
    compared = 0
    for u in us:
        for v in vs:
            p_render = frame.pixel_to_world(u, v)
            if p_render is None:
                continue  # nothing in range on this ray; not a failure

            d_cam = np.array([(u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0])
            d_world = rot @ (d_cam / np.linalg.norm(d_cam))
            dist = mujoco.mj_ray(model, data, origin, d_world,
                                 geomgroup, 1, -1, geomid)
            if dist < 0 or geomid[0] < 0:
                continue  # ray caster found nothing; rendered depth may be a
                          # different geom group, so not comparable
            p_ray = origin + dist * d_world
            errors.append(np.linalg.norm(p_render - p_ray))
            compared += 1

    if compared < 20:
        return [f"only {compared} pixels had both a rendered depth and a ray "
                f"hit; the test pose probably points at empty space"]

    errors = np.array(errors)
    # A handful of pixels legitimately straddle a silhouette edge, where the
    # renderer and the ray caster can pick different geoms. Judge on the bulk.
    p95 = float(np.percentile(errors, 95))
    if p95 > POINT_TOL_M:
        return [f"rendered depth and ray cast disagree: 95th-percentile "
                f"distance {p95 * 1000:.2f} mm over {compared} pixels "
                f"(max {errors.max() * 1000:.2f} mm). A systematic bias here "
                f"means the pinhole model and MuJoCo's camera differ -- check "
                f"the principalpixel sign in d435i_calib.mujoco_camera_attrs."]

    print(f"  render vs raycast: {compared} pixels, median "
          f"{np.median(errors) * 1000:.3f} mm, p95 {p95 * 1000:.3f} mm")
    return []


def check_principal_sign_is_detectable(model, data, cam) -> list[str]:
    """Prove the previous check would actually catch a flipped principal point.

    A test that passes for the right reason and also for the wrong one is not a
    test. This deliberately corrupts cy by twice the real offset -- the exact
    error a sign flip produces -- and requires the comparison to fail.
    """
    frame = cam.capture(align=True)
    good = frame.intrinsics
    flipped = calib.Intrinsics(
        good.width, good.height, good.fx, good.fy,
        good.cx, good.height - good.cy,      # mirror cy about the centre
    )
    if abs(flipped.cy - good.cy) < 1.0:
        return ["principal point is too close to centred for this check to "
                "mean anything; skip rather than trust it"]

    frame.intrinsics = flipped
    c2w = frame.cam_to_world
    origin, rot = c2w[:3, 3], c2w[:3, :3]
    geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
    geomid = np.zeros(1, dtype=np.int32)

    errors = []
    for u in np.linspace(40, good.width - 40, 9):
        for v in np.linspace(40, good.height - 40, 7):
            p_render = frame.pixel_to_world(u, v)
            if p_render is None:
                continue
            d_cam = np.array([(u - flipped.cx) / flipped.fx,
                              (v - flipped.cy) / flipped.fy, 1.0])
            d_world = rot @ (d_cam / np.linalg.norm(d_cam))
            dist = mujoco.mj_ray(model, data, origin, d_world,
                                 geomgroup, 1, -1, geomid)
            if dist < 0 or geomid[0] < 0:
                continue
            errors.append(np.linalg.norm(p_render - (origin + dist * d_world)))

    if not errors or float(np.percentile(errors, 95)) <= POINT_TOL_M:
        return ["a deliberately flipped principal point still passed the "
                "render-vs-raycast check, so that check proves nothing"]
    print(f"  negative control: flipped cy gives p95 "
          f"{np.percentile(errors, 95) * 1000:.1f} mm — the check has teeth")
    return []


def main() -> int:
    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    data = mujoco.MjData(model)
    _pose_for_a_useful_view(model, data)
    cam = SimWristCamera.from_model(model, data)

    failures: list[str] = []
    for label, fn in (
        ("intrinsics round-trip", lambda: check_intrinsics_round_trip(model)),
        ("mount vs hand-eye", lambda: check_mount_matches_handeye(model, data)),
        ("render vs raycast", lambda: check_render_matches_raycast(model, data, cam)),
        ("negative control", lambda: check_principal_sign_is_detectable(model, data, cam)),
    ):
        errs = fn()
        print(f"{'FAIL' if errs else 'PASS'}  {label}")
        for e in errs:
            print(f"      {e}")
        failures += errs

    cam.close()
    print()
    print(f"{len(failures)} failure(s)" if failures else "all projection checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
