"""SimReady importer tests. Offline by default.

    MUJOCO_GL=egl python -m simready_import.test_importer [--network]

USD stages are BUILT IN THE TEST with pxr, so each case states exactly what
the reader must recover (a 10 cm cube authored in centimetres and Y-up must
come out 0.1 m and Z-up). The committed assets are then placed in a room and
picked through SimXArmAPI, graded by the weld that holds the object and how
far it rose -- not by return codes. --network also fetches a real asset end
to end and requires the conversion to reproduce the committed files exactly.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from simready_import import ASSETS_DIR, manifest  # noqa: E402
from simready_import.mjcf_writer import write_mjcf  # noqa: E402
from simready_import.usd_reader import ArticulatedAssetError, read_usd  # noqa: E402

LAB_XML = (ASSETS_DIR.parent.parent / "envs" / "lab_scene.xml").resolve()


# ---------------------------------------------------------------------------
# stage builders
# ---------------------------------------------------------------------------
def _stage(path: Path, mpu=1.0, up="Z"):
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(st, mpu)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.y if up == "Y" else UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(st, "/Root")
    st.SetDefaultPrim(root.GetPrim())
    return st, root


def _box_mesh(st, path, lo, hi):
    """Closed axis-aligned box as a 12-triangle mesh."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    pts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
           (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([i for q in quads for i in q])
    return m


def _u_mesh(st, path):
    """A U channel (concave): base 0.12 x 0.06 x 0.01, two 0.01-thick walls 0.06 high,
    built as one closed mesh from three boxes' faces would not be manifold, so the
    U profile is extruded along y as a single watertight prism."""
    prof = [(0.0, 0.0), (0.12, 0.0), (0.12, 0.06), (0.11, 0.06), (0.11, 0.01),
            (0.01, 0.01), (0.01, 0.06), (0.0, 0.06)]          # x, z (counter-clockwise)
    n = len(prof)
    pts = [(x, 0.0, z) for x, z in prof] + [(x, 0.06, z) for x, z in prof]
    faces, counts = [], []
    for i in range(n):                                        # side walls
        j = (i + 1) % n
        faces += [i, j, n + j, n + i]
        counts.append(4)
    faces += list(range(n - 1, -1, -1))                       # y=0 cap (outward -y)
    counts.append(n)
    faces += list(range(n, 2 * n))                            # y=0.06 cap
    counts.append(n)
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    return m


# ---------------------------------------------------------------------------
def test_cube_in_centimetres_and_y_up():
    tmp = Path(tempfile.mkdtemp())
    try:
        st, root = _stage(tmp / "cube.usda", mpu=0.01, up="Y")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        # a 10 x 30 x 10 cm box, tall along Y (the up axis), resting on y = 0
        m = _box_mesh(st, "/Root/box", (-5, 0, -5), (5, 30, 5))
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        c = UsdGeom.Cube.Define(st, "/Root/marker")                # a 4 cm primitive cube
        c.CreateSizeAttr(4.0)
        UsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(0, 50, 0))  # 50 cm UP
        st.Save()
        a = read_usd(tmp / "cube.usda")
        lo, hi = a.bounds()
        size = hi - lo
        assert np.allclose(size, [0.10, 0.10, 0.30], atol=1e-6), size        # metres, tall in Z
        assert abs(lo[2]) < 1e-9
        prim = a.body.primitives[0]
        assert prim.kind == "box" and np.allclose(prim.size, [0.02] * 3), prim
        assert np.allclose(prim.pos, [0, 0, 0.5], atol=1e-9), prim.pos       # Y-up 50 cm -> z 0.5 m
        assert a.meters_per_unit == 0.01 and a.up_axis == "Y"
    finally:
        shutil.rmtree(tmp)


def test_mass_and_physics_material():
    tmp = Path(tempfile.mkdtemp())
    try:
        st, root = _stage(tmp / "m.usda")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        m = _box_mesh(st, "/Root/box", (0, 0, 0), (0.05, 0.05, 0.05))
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(m.GetPrim())
        mass.CreateMassAttr(0.5)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.025, 0.025, 0.01))
        mat = UsdShade.Material.Define(st, "/Root/phys")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateDynamicFrictionAttr(0.7)
        pm.CreateRestitutionAttr(0.3)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        st.Save()
        a = read_usd(tmp / "m.usda")
        assert abs(a.body.mass - 0.5) < 1e-9 and abs(a.body.friction - 0.7) < 1e-6
        assert any("restitution" in w for w in a.warnings), a.warnings
        w = write_mjcf(a, "massbox", tmp / "out")
        mm = mujoco.MjModel.from_xml_path(str(w.xml))
        b = mm.body("massbox").id
        assert abs(mm.body_mass[b] - 0.5) < 1e-6
        # authored COM (0.025, 0.025, 0.01) -> the origin is the bbox centre (0.025)^3
        assert np.allclose(mm.body_ipos[b], [0, 0, 0.01 - 0.025], atol=1e-6), mm.body_ipos[b]
        coll = [g for g in range(mm.ngeom) if mm.geom_contype[g]]
        assert coll and all(abs(mm.geom_friction[g][0] - 0.7) < 1e-6 for g in coll)
        assert all(mm.geom_contype[g] == 0 for g in range(mm.ngeom) if mm.geom_group[g] == 2)
    finally:
        shutil.rmtree(tmp)


def test_concave_u_shape_is_decomposed():
    tmp = Path(tempfile.mkdtemp())
    try:
        st, root = _stage(tmp / "u.usda")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        m = _u_mesh(st, "/Root/u")
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        st.Save()
        a = read_usd(tmp / "u.usda")
        w = write_mjcf(a, "ushape", tmp / "out")
        assert w.n_collision > 1, w.warnings
        # the pieces must leave the U's channel empty: their volume is far below
        # the volume of the U's convex hull (a single hull would fill it)
        import trimesh
        vols = [trimesh.load(p).volume for p in sorted((tmp / "out").glob("collision_*.obj"))]
        assert sum(vols) < 0.6 * (0.12 * 0.06 * 0.06), (len(vols), sum(vols))
    finally:
        shutil.rmtree(tmp)


def test_joints_articulated_refused_fixed_merged():
    tmp = Path(tempfile.mkdtemp())
    try:
        for kind in ("revolute", "fixed"):
            st, root = _stage(tmp / f"{kind}.usda")
            for i, name in enumerate(("a", "b")):
                x = UsdGeom.Xform.Define(st, f"/Root/{name}")
                UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
                m = _box_mesh(st, f"/Root/{name}/mesh", (0.05 * i, 0, 0), (0.05 * i + 0.04, 0.04, 0.04))
                UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.FixedJoint).Define(
                st, "/Root/joint")
            j.CreateBody0Rel().SetTargets([Sdf.Path("/Root/a/mesh")])   # a prim INSIDE the body
            j.CreateBody1Rel().SetTargets([Sdf.Path("/Root/b/mesh")])
            st.Save()
        try:
            read_usd(tmp / "revolute.usda")
            raise AssertionError("a revolute joint was imported as rigid")
        except ArticulatedAssetError as e:
            assert "--rigid" in str(e)
        assert read_usd(tmp / "revolute.usda", rigid=True).body.parts
        a = read_usd(tmp / "fixed.usda")
        assert any("merged" in w for w in a.warnings) and len(a.bodies) == 1
    finally:
        shutil.rmtree(tmp)


def test_no_collision_falls_back_to_visual_and_texture_survives():
    tmp = Path(tempfile.mkdtemp())
    try:
        from PIL import Image
        Image.new("RGB", (8, 8), (200, 30, 30)).save(tmp / "red.png")
        st, root = _stage(tmp / "t.usda")
        m = _box_mesh(st, "/Root/box", (0, 0, 0), (0.04, 0.04, 0.04))
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.faceVarying)
        pv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)] * 6)
        mat = UsdShade.Material.Define(st, "/Root/look")
        sh = UsdShade.Shader.Define(st, "/Root/look/surf")
        sh.CreateIdAttr("UsdPreviewSurface")
        tex = UsdShade.Shader.Define(st, "/Root/look/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("red.png")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        st.Save()
        a = read_usd(tmp / "t.usda")
        assert any("no CollisionAPI" in w for w in a.warnings)
        assert any(p.collision for p in a.body.parts)
        assert any("default density" in w for w in a.warnings)
        w = write_mjcf(a, "texbox", tmp / "out")
        mm = mujoco.MjModel.from_xml_path(str(w.xml))
        assert mm.ntex == 1 and mm.nmat == 1
        assert abs(mm.body_mass[mm.body("texbox").id] - 1000 * 0.04 ** 3) < 1e-6
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# the committed assets, in a room
# ---------------------------------------------------------------------------
PICKS = json.loads((ASSETS_DIR / "picks.json").read_text())


def test_committed_assets_match_their_manifest():
    """Every file the manifest lists is present and unedited."""
    for p in PICKS["assets"]:
        e = manifest.entry(p["slug"])
        assert e and e["revision"] == PICKS["revision"] and e["license"] == "CC-BY-4.0", p["slug"]
        for name, digest in e["files"].items():
            assert manifest.sha256(ASSETS_DIR / p["slug"] / name) == digest, (p["slug"], name)


def _room(objects):
    tmp = Path(tempfile.mkdtemp())
    rd = tmp / "t_room"
    rd.mkdir()
    (rd / "room.json").write_text(json.dumps({
        "id": "t_room", "name": "t", "description": "",
        "environment": {"extract_from": str(LAB_XML), "bodies": ["bench", "observers"]},
        "robots": [{"instance": "arm0", "type": "xarm6_rail"}], "objects": objects}, indent=2))
    return tmp, rd


def test_assets_rest_and_a_bad_placement_is_reverted():
    from simready_import import cli
    tmp, rd = _room([])
    try:
        for i, p in enumerate(PICKS["assets"]):
            rc = cli.main(["place", p["slug"], "--room", str(rd), "--name", f"o{i}",
                           "--pos", str(-0.15 + 0.22 * i), "-0.27"])
            assert rc == 0, p["slug"]
        before = (rd / "room.json").read_text()
        rc = cli.main(["place", "cupcake_g01", "--room", str(rd), "--name", "flying",
                       "--pos", "0.0", "-0.4", "1.10"])          # 35 cm in the air: falls
        assert rc == 1 and (rd / "room.json").read_text() == before
    finally:
        shutil.rmtree(tmp)


def test_imported_cucumber_is_picked_by_the_pick_primitive():
    """Graded by the weld holding the cucumber and how far it rose."""
    from sim.mujoco_env import SimXArmAPI
    tmp, rd = _room([{"name": "cucumber", "source": "simready:cucumber_a01", "dynamic": True,
                      "pose": {"pos": [-0.1, -0.25, 0.775], "euler": [0, 0, 90]}}])
    arm = SimXArmAPI(room=str(rd), render=False)
    try:
        arm.reset_scene()
        time.sleep(1.5)
        with arm.lock:
            mujoco.mj_forward(arm.model, arm.data)
            c = arm.data.xpos[arm.model.body(arm._mj("cucumber")).id] * 1000
        arm.set_rail_position(float(np.clip(c[0] + 350, 0, 700)), wait=True)
        assert arm.set_position(c[0], c[1], c[2] + 130, roll=180, wait=True) == 0
        assert arm.set_position(c[0], c[1], c[2] + 30, roll=180, wait=True) == 0
        assert arm.close_lite6_gripper() == 0, arm.last_refusal
        arm.set_position(c[0], c[1], c[2] + 130, roll=180, wait=True)
        with arm.lock:
            held = bool(arm.data.eq_active[arm.weld_eqids["cucumber"]])
            z = arm.data.xpos[arm.model.body(arm._mj("cucumber")).id][2] * 1000
        assert held and z - c[2] > 50, (held, z - c[2])
    finally:
        arm.disconnect()
        shutil.rmtree(tmp)


def test_network_fetch_reproduces_the_committed_asset():
    if "--network" not in sys.argv:
        print("    (skipped: pass --network)")
        return
    from simready_import.fetch import fetch
    p = next(a for a in PICKS["assets"] if a["slug"] == "bottled_drink_e01")
    tmp = Path(tempfile.mkdtemp())
    try:
        res = fetch(PICKS["repo_id"], p["root_usd"], PICKS["revision"][:12], cache_dir=tmp)
        assert res.revision == PICKS["revision"] and res.license == "CC-BY-4.0"
        w = write_mjcf(read_usd(res.root), p["slug"], tmp / "out")
        committed = manifest.entry(p["slug"])["files"]
        for name, digest in committed.items():
            assert manifest.sha256(tmp / "out" / name) == digest, name
    finally:
        shutil.rmtree(tmp)


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
