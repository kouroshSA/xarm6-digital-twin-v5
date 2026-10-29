"""Room builder: composed rooms reproduce the lab's robot exactly, builtin
objects reproduce the lab's objects, two robots coexist, and a grippable
object can actually be carried.

    MUJOCO_GL=egl python -m twin_rooms.test_builder

The carry test is graded physically (the object rises and travels with the
rail), with a negative control: the same motion without the weld leaves it on
the bench. "A weld was created" would pass for a weld between the wrong bodies.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import mujoco
import numpy as np

from twin_rooms import RoomError, load_room_dir
from twin_rooms.builder import build_room, build_room_def
from twin_rooms.robot import load_robot_type
from twin_rooms.room import ROOMS_DIR

LAB_XML = (ROOMS_DIR.parent / "envs" / "lab_scene.xml").resolve()
LAB = mujoco.MjModel.from_xml_path(str(LAB_XML))
XARM = load_robot_type("xarm6_rail", check_model=False)


def _subtree(m, root):
    out = []
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = int(m.body_parentid[x])
        if x == root:
            out.append(b)
    return out


def _robot_bodies(m, prefix=""):
    ids = []
    for b in XARM.model["bodies"] + XARM.model["operator_aids"]:
        ids += _subtree(m, m.body(prefix + b).id)
    return ids


def _test_room(objects, robots=None, cameras=None, extra=None):
    """A composed room in a temp dir, built on the lab bench."""
    tmp = Path(tempfile.mkdtemp())
    rd = tmp / "t_room"
    rd.mkdir()
    d = {"id": "t_room", "name": "test", "description": "",
         "environment": {"extract_from": str(LAB_XML), "bodies": ["bench", "observers"]},
         "robots": robots or [{"instance": "arm0", "type": "xarm6_rail"}],
         "objects": objects, "cameras": cameras or []}
    d.update(extra or {})
    (rd / "room.json").write_text(json.dumps(d))
    return tmp, rd


def _build(objects, **kw):
    tmp, rd = _test_room(objects, **kw)
    try:
        return build_room_def(load_room_dir(rd), write=False)
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# the robot, extracted
# ---------------------------------------------------------------------------
def test_empty_bench_holds_the_robot_and_nothing_else():
    m, ctx = build_room("empty_bench", write=False)
    top = sorted(m.body(i).name for i in range(1, m.nbody) if m.body_parentid[i] == 0)
    assert top == ["arm0/led_strip_A", "arm0/led_strip_B", "arm0/rail_track",
                   "arm0/teach_handle", "arm0/vr_head", "bench", "observers"], top
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert not free, "an empty bench has no loose objects"
    assert ctx.objects == {}


def test_extracted_robot_is_the_lab_robot():
    """Every robot body, geom, joint and actuator equal to the lab's, and in
    the same place. Compared property by property, so a dropped default class
    or a wrong frame cannot pass."""
    m, ctx = build_room("empty_bench", write=False)
    d, dl = mujoco.MjData(m), mujoco.MjData(LAB)
    mujoco.mj_forward(m, d)
    mujoco.mj_forward(LAB, dl)
    lab_ids = _robot_bodies(LAB)
    assert len(lab_ids) == len(_robot_bodies(m, "arm0/"))
    for bl in lab_ids:
        name = LAB.body(bl).name
        b = m.body("arm0/" + name).id
        assert np.allclose(d.xpos[b], dl.xpos[bl], atol=1e-9), name
        assert np.allclose(d.xquat[b], dl.xquat[bl], atol=1e-9), name
        assert np.isclose(m.body_mass[b], LAB.body_mass[bl]), name
        assert np.allclose(m.body_inertia[b], LAB.body_inertia[bl]), name
    for gl in range(LAB.ngeom):
        if LAB.geom_bodyid[gl] not in lab_ids:
            continue
        g = m.geom("arm0/" + LAB.geom(gl).name).id
        for attr in ("geom_type", "geom_size", "geom_friction", "geom_contype",
                     "geom_conaffinity", "geom_group", "geom_rgba", "geom_solref", "geom_solimp"):
            assert np.allclose(getattr(m, attr)[g], getattr(LAB, attr)[gl]), (LAB.geom(gl).name, attr)
        assert np.allclose(d.geom_xpos[g], dl.geom_xpos[gl], atol=1e-9), LAB.geom(gl).name
    for jl in range(LAB.njnt):
        if LAB.jnt_bodyid[jl] not in lab_ids:
            continue
        j = m.joint("arm0/" + LAB.joint(jl).name).id
        for attr in ("jnt_type", "jnt_range", "jnt_axis"):
            assert np.allclose(getattr(m, attr)[j], getattr(LAB, attr)[jl]), (LAB.joint(jl).name, attr)
        assert np.isclose(m.dof_damping[m.jnt_dofadr[j]], LAB.dof_damping[LAB.jnt_dofadr[jl]])
        assert np.isclose(m.dof_armature[m.jnt_dofadr[j]], LAB.dof_armature[LAB.jnt_dofadr[jl]])
    lab_acts = [a for a in range(LAB.nu) if LAB.jnt_bodyid[LAB.actuator_trnid[a, 0]] in lab_ids]
    assert m.nu == len(lab_acts) == 9, (m.nu, len(lab_acts))
    for al in lab_acts:
        a = m.actuator("arm0/" + LAB.actuator(al).name).id
        for attr in ("actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                     "actuator_forcerange", "actuator_ctrllimited", "actuator_forcelimited"):
            assert np.allclose(getattr(m, attr)[a], getattr(LAB, attr)[al]), (LAB.actuator(al).name, attr)
    assert m.nexclude == LAB.nexclude
    for attr in ("timestep", "gravity", "integrator", "cone", "impratio"):
        assert np.allclose(getattr(m.opt, attr), getattr(LAB.opt, attr)), attr
    for c in ("cam_wrist_color", "cam_wrist_depth"):
        assert np.allclose(d.cam_xpos[m.camera("arm0/" + c).id], dl.cam_xpos[LAB.camera(c).id], atol=1e-9)
        assert np.isclose(m.cam_fovy[m.camera("arm0/" + c).id], LAB.cam_fovy[LAB.camera(c).id])
    # the bench too
    for name in ("bench", "observers"):
        assert np.allclose(d.xpos[m.body(name).id], dl.xpos[LAB.body(name).id])


def test_operator_aids_stay_mocap():
    m, _ = build_room("empty_bench", write=False)
    for n in ("arm0/teach_handle", "arm0/vr_head"):
        assert m.body_mocapid[m.body(n).id] >= 0, n


def test_build_file_loads_on_its_own():
    m, ctx = build_room("empty_bench")
    assert ctx.model_path == ROOMS_DIR / "_templates" / "empty_bench" / "build" / "scene.xml"
    m2 = mujoco.MjModel.from_xml_path(str(ctx.model_path))
    assert (m2.nbody, m2.njnt, m2.nu, m2.ngeom, m2.nmesh) == (m.nbody, m.njnt, m.nu, m.ngeom, m.nmesh)


def test_logical_names():
    m, ctx = build_room("empty_bench", write=False)
    assert ctx.mjcf("joint1", "joint") == "arm0/joint1"
    assert ctx.mjcf("end_effector", "site") == "arm0/end_effector"
    assert ctx.mjcf("cam_wrist_color", "camera") == "arm0/cam_wrist_color"
    assert ctx.mjcf("bench") == "bench"
    assert ctx.logical("arm0/act_rail") == "act_rail"
    assert ctx.robot().joint_names[0] == "arm0/joint1"


def test_passthrough_rooms_are_the_scene_unchanged():
    m, ctx = build_room("lab_a2s2")
    assert ctx.model_path == LAB_XML
    assert (m.nbody, m.njnt, m.nu, m.neq) == (LAB.nbody, LAB.njnt, LAB.nu, LAB.neq)
    assert ctx.mjcf("joint1", "joint") == "joint1"
    # welds are discovered, including the irregular names
    assert ctx.objects["left_tube_rack"].welds == {"arm0": "grip_left_rack"}
    assert ctx.objects["red_cube_front"].welds == {"arm0": "grip_red_cube_front"}
    assert ctx.objects["vortex_genie"].welds == {}
    assert not (ROOMS_DIR / "lab_a2s2" / "build").exists(), "a passthrough room writes no build"


def test_primitive_variant_builds():
    m, ctx = build_room("empty_bench", variant="primitive", write=False)
    assert m.nmesh == 0 and ctx.mjcf("joint6", "joint") == "arm0/joint6"


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------
LAB_TWINS = {"cube": "green_cube", "bin": "green_bin", "tube_rack": "left_tube_rack",
             "tube": "tube_L1"}


def _geoms_rel(m, b):
    """Geoms of body b as sorted (type, size, local pos, rgba-alpha, contype,
    friction, solref, mass-bearing) rows, independent of names."""
    rows = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b:
            rows.append(tuple(np.round(np.concatenate([[m.geom_type[g]], m.geom_size[g], m.geom_pos[g],
                                                       [m.geom_contype[g], m.geom_conaffinity[g]],
                                                       m.geom_friction[g], m.geom_solref[g],
                                                       m.geom_solimp[g][:3]]), 6)))
    return sorted(rows)


def test_builtin_defaults_reproduce_the_lab_objects():
    """Each builtin with default params must be its lab original, geom by geom
    (shape, size, placement, contact parameters) and in mass and inertia.
    This is what ties builtins.DEFAULTS to the scene XML."""
    objs = [{"name": f"b_{k}", "source": f"builtin:{k}",
             "pose": {"pos": [-0.6 + 0.3 * i, -0.3, 0.8]}} for i, k in enumerate(LAB_TWINS)]
    m, ctx = _build(objs)
    for i, (kind, lab_name) in enumerate(LAB_TWINS.items()):
        b = m.body(ctx.mjcf(f"b_{kind}")).id
        bl = LAB.body(lab_name).id
        assert _geoms_rel(m, b) == _geoms_rel(LAB, bl), kind
        assert np.isclose(m.body_mass[b], LAB.body_mass[bl]), (kind, m.body_mass[b], LAB.body_mass[bl])
        assert np.allclose(m.body_inertia[b], LAB.body_inertia[bl], rtol=1e-6), kind
        assert np.allclose(m.body_ipos[b], LAB.body_ipos[bl], atol=1e-9), kind


def test_objects_get_prefixed_names_poses_and_joints():
    objs = [{"name": "red_cube", "source": "builtin:cube", "params": {"rgba": [1, 0, 0, 1]},
             "pose": {"pos": [0.1, -0.2, 0.765], "euler": [0, 0, 30]}},
            {"name": "tray", "source": "builtin:bin", "dynamic": False,
             "pose": {"pos": [0.3, -0.3, 0.75]}}]
    m, ctx = _build(objs)
    assert ctx.mjcf("red_cube") == "obj/red_cube"
    assert ctx.mjcf("red_cube_geom", "geom") == "obj/red_cube_geom"
    assert m.joint("obj/red_cube_joint").type == mujoco.mjtJoint.mjJNT_FREE
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = m.body("obj/red_cube").id
    assert np.allclose(d.xpos[b], [0.1, -0.2, 0.765])
    yaw = np.degrees(np.arctan2(d.xmat[b][3], d.xmat[b][0]))
    assert np.isclose(yaw, 30.0), yaw
    assert m.body_jntnum[m.body("obj/tray").id] == 0, "static object must have no joint"
    assert ctx.objects["tray"].welds == {}, "a static object gets no grasp weld"


def test_objects_settle_on_the_bench():
    """Drop test: builtins placed at their resting z stay there, upright, and
    come to rest (nothing explodes, sinks or drifts)."""
    objs = [{"name": "c", "source": "builtin:cube", "pose": {"pos": [0.0, -0.25, 0.765]}},
            {"name": "bn", "source": "builtin:bin", "pose": {"pos": [0.2, -0.3, 0.75]}},
            {"name": "rk", "source": "builtin:tube_rack", "pose": {"pos": [-0.3, -0.3, 0.755]}},
            {"name": "tb", "source": "builtin:tube", "pose": {"pos": [-0.32, -0.28, 0.8175]}}]  # tube in slot col2,row2
    m, ctx = _build(objs)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    start = {o: d.xpos[m.body(ctx.mjcf(o)).id].copy() for o in ("c", "bn", "rk", "tb")}
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for o, p0 in start.items():
        b = m.body(ctx.mjcf(o)).id
        drift = np.linalg.norm(d.xpos[b] - p0)
        assert drift < 0.003, (o, drift)
        assert d.xmat[b][8] > 0.999, (o, "tipped over")
        assert np.linalg.norm(d.cvel[b]) < 1e-2, (o, d.cvel[b])


def test_mjcf_object_file_attached_twice():
    tmp = Path(tempfile.mkdtemp())
    try:
        frag = tmp / "t_room" / "parts"
        frag.mkdir(parents=True)
        (frag / "block.xml").write_text(
            '<mujoco><worldbody><body name="block"><geom name="g" type="box" size="0.02 0.02 0.02" '
            'mass="0.05"/></body></worldbody></mujoco>')
        objs = [{"name": f"blk{i}", "source": "mjcf:parts/block.xml",
                 "pose": {"pos": [0.1 * i, -0.3, 0.77]}} for i in (1, 2)]
        _, rd = _test_room(objs)
        shutil.move(str(rd / "room.json"), str(frag.parent / "room.json"))
        m, ctx = build_room_def(load_room_dir(frag.parent), write=False)
        assert ctx.mjcf("blk1") == "obj/blk1" and ctx.mjcf("blk2/g", "geom") == "obj/blk2/g"
        assert m.joint("obj/blk2_joint").type == mujoco.mjtJoint.mjJNT_FREE
        assert ctx.objects["blk1"].welds == {"arm0": "arm0/grip_blk1"}
    finally:
        shutil.rmtree(tmp)


def test_simready_object_not_yet_imported_is_refused_by_name():
    try:
        _build([{"name": "tray", "source": "simready:plastic_tray", "pose": {"pos": [0, -0.3, 0.76]}}])
        raise AssertionError("built a room with a SimReady asset that does not exist")
    except RoomError as e:
        assert "'plastic_tray' is not imported" in e.problems[0], e.problems


def test_name_clash_is_refused():
    try:
        _build([{"name": "gripper", "source": "builtin:cube", "pose": {"pos": [0, -0.3, 0.77]}}])
        raise AssertionError("built a room where 'gripper' means two bodies")
    except RoomError as e:
        assert any("body 'gripper' would mean both" in p for p in e.problems), e.problems


# ---------------------------------------------------------------------------
# grasp welds: the consumer
# ---------------------------------------------------------------------------
def _carry(weld_on: bool):
    """Put a cube in the jaws, optionally activate its weld the way
    SimXArmAPI.gripper_close does, then drive the rail 200 mm. Returns the
    cube's and the gripper's displacement."""
    m, ctx = _build([{"name": "cube", "source": "builtin:cube",
                      "pose": {"pos": [0, -0.25, 0.765]}}])
    d = mujoco.MjData(m)
    arm = ctx.robot()
    home = np.deg2rad(XARM.value("home.joints_deg"))
    rail_q = XARM.value("home.rail_mm") / 1000.0
    jids = [m.joint(n).id for n in arm.joint_names]
    acts = [m.actuator(n).id for n in arm.actuator_names]
    rail_j = m.joint(arm.mjcf("rail")).id
    rail_a = m.actuator(arm.mjcf("act_rail")).id
    for j, a, q in zip(jids, acts, home):
        d.qpos[m.jnt_qposadr[j]] = q
        d.ctrl[a] = q
    d.qpos[m.jnt_qposadr[rail_j]] = rail_q
    d.ctrl[rail_a] = rail_q
    mujoco.mj_forward(m, d)
    # place the cube between the jaws: 30 mm below the EE site
    site = d.site_xpos[m.site(arm.mjcf("end_effector")).id].copy()
    cj = m.joint("obj/cube_joint")
    d.qpos[cj.qposadr[0]:cj.qposadr[0] + 3] = site - [0, 0, 0.03]
    mujoco.mj_forward(m, d)
    cube = m.body("obj/cube").id
    g = m.body(arm.mjcf("gripper")).id
    p0, g0 = d.xpos[cube].copy(), d.xpos[g].copy()
    if weld_on:
        eq = m.equality(ctx.objects["cube"].welds["arm0"]).id
        gq_inv = np.zeros(4)
        mujoco.mju_negQuat(gq_inv, d.xquat[g])
        rel = np.zeros(3)
        mujoco.mju_rotVecQuat(rel, d.xpos[cube] - d.xpos[g], gq_inv)
        rq = np.zeros(4)
        mujoco.mju_mulQuat(rq, gq_inv, d.xquat[cube])
        m.eq_data[eq, 0:3], m.eq_data[eq, 3:6], m.eq_data[eq, 6:10], m.eq_data[eq, 10] = 0, rel, rq, 1
        d.eq_active[eq] = 1
    for k in range(int(1.5 / m.opt.timestep)):
        d.ctrl[rail_a] = rail_q + 0.2 * min(1.0, k * m.opt.timestep / 1.0)
        mujoco.mj_step(m, d)
    return d.xpos[cube] - p0, d.xpos[g] - g0


def test_grippable_object_is_carried_by_its_weld():
    """The cube must move WITH the gripper (same displacement within 5 mm), and
    the gripper must actually have travelled. The bare PD arm sags a little
    under the load, which moves both together and is not a carry failure."""
    cube, grip = _carry(weld_on=True)
    assert grip[0] > 0.15, f"the rail did not move the gripper: {grip}"
    assert np.linalg.norm(cube - grip) < 0.005, (cube, grip)


def test_without_the_weld_the_cube_is_left_behind():
    """Negative control for the test above: same motion, weld off."""
    cube, grip = _carry(weld_on=False)
    assert grip[0] > 0.15, grip
    assert np.linalg.norm(cube - grip) > 0.1, (cube, grip)


def test_two_robots_of_one_type():
    robots = [{"instance": "arm0", "type": "xarm6_rail"},
              {"instance": "arm1", "type": "xarm6_rail", "controllable": False,
               "base_pose": {"pos": [0, 1.2, 0], "euler": [0, 0, 180]}}]
    m, ctx = _build([{"name": "cube", "source": "builtin:cube",
                      "pose": {"pos": [0, -0.25, 0.765]}}], robots=robots)
    assert m.nu == 18
    assert ctx.robot("arm1").mjcf("joint1") == "arm1/joint1"
    assert ctx.mjcf("joint1", "joint") == "arm0/joint1"
    assert not ctx.has("arm1/joint1", "joint"), "non-controllable robots stay out of the flat map"
    assert ctx.objects["cube"].welds == {"arm0": "arm0/grip_cube", "arm1": "arm1/grip_cube"}
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b0, b1 = d.xpos[m.body("arm0/xarm_base").id], d.xpos[m.body("arm1/xarm_base").id]
    # yaw 180 about the world origin, then +1.2 m in y
    assert np.allclose(b1, [-b0[0], 1.2 - b0[1], b0[2]], atol=1e-9), (b0, b1)


def test_room_camera_looks_at_its_target():
    cams = [{"name": "cam_side", "pos": [1.0, -1.0, 1.4], "lookat": [0, 0, 0.8], "fovy": 50}]
    m, ctx = _build([], cameras=cams)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = m.camera(ctx.mjcf("cam_side", "camera")).id
    fwd = -d.cam_xmat[c].reshape(3, 3)[:, 2]
    want = np.array([0, 0, 0.8]) - [1.0, -1.0, 1.4]
    assert np.allclose(fwd, want / np.linalg.norm(want), atol=1e-6)
    up = d.cam_xmat[c].reshape(3, 3)[:, 1]
    assert up[2] > 0, "camera image is upside down"


def test_hash_covers_robot_json():
    _, a = build_room("empty_bench", write=False)
    _, b = build_room("empty_bench", write=False)
    assert a.hash == b.hash
    _, c = build_room("lab_a2s2")
    assert c.hash != a.hash


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
