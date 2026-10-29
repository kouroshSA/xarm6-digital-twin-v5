"""The lab rebuilt in the room format (lab_a2s2_rebuilt) against the lab
itself (lab_a2s2, envs/lab_scene.xml loaded unchanged).

    MUJOCO_GL=egl python -m twin_rooms.test_lab_rebuilt

Spec step 6: same body, joint and actuator counts, every body's initial world
position within 1 mm, and a fixed scripted pick-and-place producing the same
outcome in both. The pick is graded on physical_outcome() and on which body
the weld holds, through the real SimXArmAPI in each room.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim.mujoco_env import SimXArmAPI  # noqa: E402
from twin_rooms.builder import build_room  # noqa: E402
from twin_rooms.room import PROJECT_ROOT  # noqa: E402

LAB = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "envs" / "lab_scene.xml"))


def _models():
    m, ctx = build_room("lab_a2s2_rebuilt", write=False)
    return m, ctx


def test_counts_match_the_lab():
    m, _ = _models()
    for attr in ("nbody", "njnt", "nu", "neq", "ncam", "nq", "nv", "nmocap", "nexclude"):
        assert getattr(m, attr) == getattr(LAB, attr), (attr, getattr(m, attr), getattr(LAB, attr))


def test_every_body_within_1mm_and_same_orientation():
    m, ctx = _models()
    d, dl = mujoco.MjData(m), mujoco.MjData(LAB)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_kinematics(LAB, dl)
    lab_names = {LAB.body(i).name for i in range(1, LAB.nbody)}
    rebuilt = {ctx.logical(m.body(i).name): i for i in range(1, m.nbody)}
    assert set(rebuilt) == lab_names, (sorted(lab_names - set(rebuilt)),
                                       sorted(set(rebuilt) - lab_names))
    worst = 0.0
    for name, i in rebuilt.items():
        j = LAB.body(name).id
        err = float(np.linalg.norm(d.xpos[i] - dl.xpos[j]))
        worst = max(worst, err)
        assert err < 0.001, (name, err)
        assert abs(abs(float(np.dot(d.xquat[i], dl.xquat[j]))) - 1) < 1e-6, (name, "orientation")
        assert np.isclose(m.body_mass[i], LAB.body_mass[j], rtol=1e-4), (name, "mass")
        assert np.allclose(m.body_inertia[i], LAB.body_inertia[j], rtol=1e-4, atol=1e-9), (name, "inertia")
    print(f"    worst body position error {worst * 1000:.4f} mm")


def test_same_welds_actuators_and_joint_ranges():
    m, ctx = _models()
    lab_eq = {LAB.body(int(LAB.eq_obj2id[e])).name for e in range(LAB.neq)}
    rb_eq = {ctx.logical(m.body(int(m.eq_obj2id[e])).name) for e in range(m.neq)}
    assert rb_eq == lab_eq, sorted(rb_eq ^ lab_eq)
    for a in range(LAB.nu):
        name = LAB.actuator(a).name
        b = m.actuator(ctx.mjcf(name, "actuator")).id
        assert np.allclose(m.actuator_gainprm[b], LAB.actuator_gainprm[a]), name
        assert np.allclose(m.actuator_ctrlrange[b], LAB.actuator_ctrlrange[a]), name
    for j in range(LAB.njnt):
        name = LAB.joint(j).name
        k = m.joint(ctx.mjcf(name, "joint")).id
        assert np.allclose(m.jnt_range[k], LAB.jnt_range[j]), name


def _scripted_pick(arm):
    """green cube -> green bin, a fixed script; returns (physical_outcome, held-during-carry)."""
    arm.reset_scene()
    time.sleep(0.5)
    with arm.lock:
        mujoco.mj_forward(arm.model, arm.data)
        c = arm.data.xpos[arm.model.body(arm._mj("green_cube")).id] * 1000
        b = arm.data.xpos[arm.model.body(arm._mj("green_bin")).id] * 1000
    arm.set_rail_position(350, wait=True)
    arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True)
    arm.set_position(c[0], c[1], c[2] + 30, roll=180, wait=True)
    arm.close_lite6_gripper()
    arm.set_position(c[0], c[1], c[2] + 100, roll=180, wait=True)
    with arm.lock:
        held = bool(arm.data.eq_active[arm.weld_eqids["green_cube"]])
    arm.set_position(b[0], b[1], b[2] + 120, roll=180, wait=True)
    arm.open_lite6_gripper()
    time.sleep(1.0)
    return arm.physical_outcome(), held


def test_scripted_pick_and_place_same_outcome():
    outcomes = {}
    for label, kw in (("lab", dict(scene_xml="envs/lab_scene.xml")),
                      ("rebuilt", dict(room="lab_a2s2_rebuilt"))):
        arm = SimXArmAPI(render=False, **kw)
        try:
            outcomes[label] = _scripted_pick(arm)
        finally:
            arm.disconnect()
    (lab_out, lab_held), (rb_out, rb_held) = outcomes["lab"], outcomes["rebuilt"]
    print(f"    lab:     {lab_out}\n    rebuilt: {rb_out}")
    assert lab_held and rb_held, (lab_held, rb_held)
    assert "green_cube in green_bin" in lab_out and "green_cube in green_bin" in rb_out
    facts = lambda s: {f.strip() for f in s.split(";") if " in " in f or "fell" in f}  # noqa: E731
    assert facts(lab_out) == facts(rb_out), (lab_out, rb_out)


def test_room_json_is_what_the_generator_writes():
    r = subprocess.run([sys.executable, "scripts/make_lab_rebuilt.py", "--check"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


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
    sys.stdout.flush()
    os._exit(1 if failed else 0)
