# simready_import/usd_reader.py
"""USD stage -> neutral dataclasses (no MuJoCo here).

    asset = read_usd("…/sm_food_produce_cucumber_a01_01.usda")
    asset.bodies[0].parts       # triangle meshes in METRES, Z-UP, asset-root frame

What is read, and how:
  units / axes   UsdGeom.GetStageMetersPerUnit / GetStageUpAxis; points are
                 scaled to metres and a Y-up stage is rotated to Z-up.
  transforms     every prim's local-to-world, made relative to the default prim,
                 is baked into its points (instanced meshes included).
  meshes         UsdGeom.Mesh, fan-triangulated; UVs (primvars:st, vertex or
                 faceVarying) kept per triangle corner; GeomSubsets split a mesh
                 by bound material.
  primitives     UsdGeom.Cube / Sphere / Cylinder / Capsule -> PrimitiveGeom.
  collision      prims with UsdPhysics.CollisionAPI, with the MeshCollisionAPI
                 approximation; none at all -> the visual meshes, with a warning.
  mass           UsdPhysics.MassAPI mass / density / centerOfMass; else the
                 bound physics material's density; else default_density (warned).
  friction       the bound physics material's dynamicFriction (restitution is
                 logged and ignored).
  materials      UsdPreviewSurface diffuseColor / opacity, or its UsdUVTexture
                 file; MDL-only -> displayColor or grey, with a warning.
  variants       a "MuJoCo" physics variant set is switched to "Enabled" when the
                 asset has one; failing that a "physics" set to "physics" (the
                 ForgeCAD packaging); see PHYSICS_VARIANTS.
                 asset has one (SimReady assets do), so the MuJoCo-authored
                 physics is what is read.
  articulation   rigid bodies connected by FIXED joints are one body (a "link").
                 Revolute and prismatic joints between links are refused
                 (ArticulatedAssetError) unless rigid=True (weld everything into
                 one body) or articulated=True: then each link is a BodyData,
                 the first is the root, and every other one carries the
                 JointData that moves it against its parent -- axis and anchor
                 in the asset frame, limits in radians or metres.
  semantics      semantics:labels:* on the default prim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class ArticulatedAssetError(ValueError):
    """The asset has moving joints; v1 imports rigid objects only."""


@dataclass
class MaterialInfo:
    name: str
    rgba: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    texture: Optional[Path] = None          # diffuse/albedo image, resolved on disk


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray                    # (N, 3) metres, asset frame, Z up
    faces: np.ndarray                       # (M, 3) int
    uvs: Optional[np.ndarray] = None        # (M, 3, 2) per triangle corner
    material: Optional[str] = None
    visual: bool = True
    collision: bool = False
    approximation: str = "none"


@dataclass
class PrimitiveGeom:
    name: str
    kind: str                               # box | sphere | cylinder | capsule
    size: Tuple[float, ...]                 # MuJoCo convention (half sizes)
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]  # w x y z
    material: Optional[str] = None
    visual: bool = True
    collision: bool = False


@dataclass
class JointData:
    name: str
    kind: str                               # hinge | slide
    axis: Tuple[float, float, float]        # unit vector, asset frame
    anchor: Tuple[float, float, float]      # a point on the axis, asset frame (m)
    range: Optional[Tuple[float, float]]    # radians (hinge) or metres (slide); None = free
    damping: Optional[float] = None         # authored drive damping, as found


@dataclass
class BodyData:
    name: str
    parts: List[MeshPart] = field(default_factory=list)
    primitives: List[PrimitiveGeom] = field(default_factory=list)
    mass: Optional[float] = None            # kg, if authored
    center_of_mass: Optional[Tuple[float, float, float]] = None
    density: Optional[float] = None         # kg/m^3, if authored and mass is not
    friction: Optional[float] = None
    parent: Optional[str] = None            # parent link's name; None for the root
    joint: Optional[JointData] = None       # what moves this link against its parent


@dataclass
class AssetData:
    source: Path
    meters_per_unit: float
    up_axis: str
    variant: str                            # "MuJoCo=Enabled" or "default"
    bodies: List[BodyData]
    materials: Dict[str, MaterialInfo]
    labels: Dict[str, List[str]]
    warnings: List[str] = field(default_factory=list)

    @property
    def body(self) -> BodyData:
        return self.bodies[0]

    def bounds(self, visual: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pts = [p.vertices for b in self.bodies for p in b.parts
               if (p.visual if visual else p.collision)]
        if not pts:
            pts = [p.vertices for b in self.bodies for p in b.parts]
        allp = np.concatenate(pts) if pts else np.zeros((1, 3))
        return allp.min(axis=0), allp.max(axis=0)


# ---------------------------------------------------------------------------
def _matrix(gf_mat) -> np.ndarray:
    # Gf matrices are row-vector convention: p' = p * M
    return np.array(gf_mat, dtype=float)


def _apply(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    hom = np.hstack([points, np.ones((len(points), 1))])
    return (hom @ m)[:, :3]


def _triangulate(counts, indices) -> Tuple[np.ndarray, np.ndarray]:
    """Fan triangulation. Returns (triangles of vertex indices, the face index
    each triangle came from, and the corner positions in the flat index list)."""
    tris, owner, corners = [], [], []
    k = 0
    for f, n in enumerate(counts):
        for i in range(1, n - 1):
            tris.append((indices[k], indices[k + i], indices[k + i + 1]))
            corners.append((k, k + i, k + i + 1))
            owner.append(f)
        k += n
    return (np.array(tris, dtype=np.int64).reshape(-1, 3),
            np.array(owner, dtype=np.int64), np.array(corners, dtype=np.int64).reshape(-1, 3))


def _mat_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix (column-vector convention) -> unit quaternion w, x, y, z."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (R[j, i] + R[i, j]) / s
        v[k] = (R[k, i] + R[i, k]) / s
        q = ((R[k, j] - R[j, k]) / s, *v)
    q = np.array(q) / np.linalg.norm(q)
    return tuple(float(x) for x in (q if q[0] >= 0 else -q))


def _input(shader, name):
    """A shader input, or None when the shader does not have it (pxr hands back
    an invalid input that throws on use)."""
    inp = shader.GetInput(name)
    return inp if inp and inp.GetAttr().IsValid() else None


def _purpose(prim) -> str:
    return UsdGeom.Imageable(prim).ComputePurpose() if prim.IsA(UsdGeom.Imageable) else "default"


def _material_info(mat: UsdShade.Material, stage_dir: Path, warnings: List[str]) -> MaterialInfo:
    info = MaterialInfo(name=mat.GetPrim().GetName())
    surface = mat.ComputeSurfaceSource()[0]
    preview = None
    if surface and surface.GetIdAttr().Get() == "UsdPreviewSurface":
        preview = surface
    else:
        for p in Usd.PrimRange(mat.GetPrim(), Usd.TraverseInstanceProxies()):
            sh = UsdShade.Shader(p)
            if p.IsA(UsdShade.Shader) and sh.GetIdAttr().Get() == "UsdPreviewSurface":
                preview = sh
                break
    if preview is None:
        warnings.append(f"material {info.name}: no UsdPreviewSurface (MDL only?); using grey")
        return info
    rgb = (0.7, 0.7, 0.7)
    alpha = 1.0
    diff = _input(preview, "diffuseColor")
    if diff is not None:
        if diff.HasConnectedSource():
            src = diff.GetConnectedSources()[0][0].source
            tex = UsdShade.Shader(src.GetPrim())
            f = _input(tex, "file")
            if f is not None and f.Get() is not None:
                ap = f.Get()
                resolved = ap.resolvedPath or str((stage_dir / ap.path).resolve())
                if Path(resolved).is_file():
                    info.texture = Path(resolved)
                else:
                    warnings.append(f"material {info.name}: texture {ap.path} not found")
            fb = _input(tex, "fallback")
            if fb is not None and fb.Get() is not None:
                rgb = tuple(fb.Get())[:3]
            rgb = (1.0, 1.0, 1.0) if info.texture else rgb
        elif diff.Get() is not None:
            rgb = tuple(float(c) for c in diff.Get())
    op = _input(preview, "opacity")
    if op is not None and not op.HasConnectedSource() and op.Get() is not None:
        alpha = float(op.Get())
    info.rgba = (float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha)
    return info


def _physics_material(prim) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(dynamicFriction, density, restitution) of the physics material bound to prim."""
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")[0]
    if not mat or not mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        return None, None, None
    m = UsdPhysics.MaterialAPI(mat.GetPrim())
    get = lambda a: a.Get() if a and a.HasAuthoredValue() else None  # noqa: E731
    return get(m.GetDynamicFrictionAttr()), get(m.GetDensityAttr()), get(m.GetRestitutionAttr())


def _owning_body(target: Sdf.Path, bodies: Dict[str, str]) -> Optional[str]:
    p = target
    while p != Sdf.Path.absoluteRootPath and not p.isEmpty:
        if str(p) in bodies:
            return str(p)
        p = p.GetParentPath()
    return None


def _link_name(path: str) -> str:
    """/RootNode/Geometry/door_01_obj_00 -> door_01"""
    n = Sdf.Path(path).name
    for suffix in ("_obj_00", "_obj_01", "_obj"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    return n


#: Physics variants tried in order; the first the asset has is selected.
PHYSICS_VARIANTS = (("MuJoCo", "Enabled"), ("physics", "physics"))


def read_usd(path, rigid: bool = False, articulated: bool = False,
             exclude: Tuple[str, ...] = (),
             default_density: float = 1000.0,
             variant: Optional[Tuple[str, str]] = None) -> AssetData:
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD stage {path}")
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    warnings: List[str] = []
    used_variant = "default"
    for vset, value in ([variant] if variant else PHYSICS_VARIANTS):
        if root.GetVariantSets().HasVariantSet(vset):
            vs = root.GetVariantSets().GetVariantSet(vset)
            if value in vs.GetVariantNames():
                vs.SetVariantSelection(value)
                used_variant = f"{vset}={value}"
                break

    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    to_z = Y_UP_TO_Z_UP if up == "Y" else np.eye(3)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inv = np.linalg.inv(_matrix(cache.GetLocalToWorldTransform(root)))

    def to_asset(points_local: np.ndarray, prim) -> np.ndarray:
        m = _matrix(cache.GetLocalToWorldTransform(prim)) @ root_inv
        return (_apply(points_local, m) * mpu) @ to_z.T

    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    if exclude:
        # Prims the caller asked to leave out (and everything under them): a
        # loose accessory packaged with an instrument, which one asset -- one
        # body tree -- cannot carry as a separate free object.
        dropped = [p for p in prims if p.GetName() in exclude]
        missing = sorted(set(exclude) - {p.GetName() for p in dropped})
        if missing:
            raise ValueError(f"{path.name}: --exclude names no prim: {missing}")
        cut = [p.GetPath() for p in dropped]
        prims = [p for p in prims if not any(p.GetPath().HasPrefix(c) for c in cut)
                 and not (p.IsA(UsdPhysics.Joint) and any(
                     t.HasPrefix(c) for c in cut for r in (UsdPhysics.Joint(p).GetBody0Rel(),
                                                          UsdPhysics.Joint(p).GetBody1Rel())
                     for t in r.GetTargets()))]
        warnings.append(f"excluded {[str(c) for c in cut]} (and the joints to them)")

    # ---- articulation -------------------------------------------------------
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims) \
            and not (rigid or articulated):
        raise ArticulatedAssetError(f"{path.name}: has an ArticulationRootAPI (a jointed "
                                    "asset) -- pass --articulated to keep its joints, or "
                                    "--rigid to import it as one rigid body")
    parent = {str(b.GetPath()): str(b.GetPath()) for b in bodies}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    movable = []
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        if not p.IsA(UsdPhysics.FixedJoint):
            if articulated and not rigid:
                movable.append(p)
            elif not rigid:
                raise ArticulatedAssetError(
                    f"{path.name}: joint {p.GetName()} is a {p.GetTypeName()} -- pass "
                    "--articulated to keep its joints, or --rigid to weld everything "
                    "into one body")
            continue
        t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        # A joint may target a prim INSIDE a rigid body (SimReady targets the
        # mesh); it belongs to the nearest enclosing rigid body.
        a = _owning_body(t0[0], parent) if t0 else None
        b = _owning_body(t1[0], parent) if t1 else None
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups = {find(k) for k in parent}
    n_fixed = sum(1 for p in prims if p.IsA(UsdPhysics.FixedJoint))
    tree = _link_tree(movable, groups, find, parent, to_asset, cache, root_inv, mpu, to_z,
                      path.name) if movable else None
    if len(groups) > 1 and not rigid and tree is None:
        raise ArticulatedAssetError(f"{path.name}: {len(groups)} rigid bodies not joined by "
                                    "fixed joints; pass --rigid to import them as one body")
    if n_fixed:
        warnings.append(f"{n_fixed} fixed joint(s): merged into "
                        + ("their links" if tree else "one rigid body"))

    # ---- geometry -----------------------------------------------------------
    # One BodyData per link (articulated), else everything in one body.
    if tree:
        order, links, names = tree
        link_bodies = {g: BodyData(name=names[g], parent=links[g][0], joint=links[g][1])
                       for g in order}
        for g in order:
            if link_bodies[g].parent is not None:
                link_bodies[g].parent = link_bodies[link_bodies[g].parent].name
        warnings.append(f"{len(movable)} moving joint(s): {len(order)} links, root "
                        f"{link_bodies[order[0]].name}")
    else:
        order, link_bodies = [None], {None: BodyData(name=root.GetName())}

    def link_of(prim) -> Optional[str]:
        if tree is None:
            return None
        owner = _owning_body(prim.GetPath(), parent)
        return find(owner) if owner else order[0]

    per_link = {g: ([], [], [], []) for g in order}     # masses, coms, densities, frictions
    materials: Dict[str, MaterialInfo] = {}
    stage_dir = path.parent

    def material_of(prim) -> Optional[str]:
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if not mat:
            return None
        key = str(mat.GetPath())
        if key not in materials:
            materials[key] = _material_info(mat, stage_dir, warnings)
        return key

    any_collision = any(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
    for p in prims:
        g = link_of(p)
        body = link_bodies[g]
        masses, coms, densities, frictions = per_link[g]
        purpose = _purpose(p)
        is_coll = p.HasAPI(UsdPhysics.CollisionAPI)
        if is_coll:
            ce = UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
            is_coll = ce is not False
        visual = purpose in ("default", "render")
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts:
                continue
            verts = to_asset(np.array(pts, dtype=float), p)
            tris, owner, corners = _triangulate(list(counts), list(idx))
            uvs = None
            st = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
            if st and st.HasValue():
                vals = np.array(st.ComputeFlattened(), dtype=float) if st.GetIndices() \
                    else np.array(st.Get(), dtype=float)
                interp = st.GetInterpolation()
                if interp == UsdGeom.Tokens.faceVarying and len(vals) == len(idx):
                    uvs = vals[corners]
                elif interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) \
                        and len(vals) == len(pts):
                    uvs = vals[tris]
            approx = "none"
            if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() or "none")
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            pieces = []
            if subsets and visual:
                taken = np.zeros(len(counts), dtype=bool)
                for s in subsets:
                    faces = np.array(s.GetIndicesAttr().Get() or [], dtype=np.int64)
                    taken[faces] = True
                    sel = np.isin(owner, faces)
                    pieces.append((f"{p.GetName()}_{s.GetPrim().GetName()}", sel,
                                   material_of(s.GetPrim()) or material_of(p)))
                rest = ~np.isin(owner, np.nonzero(taken)[0])
                if rest.any():
                    pieces.append((p.GetName(), rest, material_of(p)))
            else:
                pieces.append((p.GetName(), np.ones(len(tris), dtype=bool), material_of(p)))
            for name, sel, mat in pieces:
                body.parts.append(MeshPart(
                    name=name, vertices=verts, faces=tris[sel],
                    uvs=uvs[sel] if uvs is not None else None, material=mat,
                    visual=visual, collision=False, approximation=approx))
            if is_coll or (not any_collision and visual):
                # one collision part per prim, not per material subset
                body.parts.append(MeshPart(name=f"{p.GetName()}_collision", vertices=verts,
                                           faces=tris, visual=False, collision=True,
                                           approximation=approx))
        elif p.IsA(UsdGeom.Gprim) and p.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Capsule"):
            m = _matrix(cache.GetLocalToWorldTransform(p)) @ root_inv
            scale = np.linalg.norm(m[:3, :3], axis=1)
            rot = (m[:3, :3] / scale[:, None]).T          # column-vector rotation
            rot = to_z @ rot
            pos = (to_z @ (m[3, :3] * mpu))
            quat = _mat_to_quat(rot)
            s = scale * mpu
            kind = p.GetTypeName()
            if kind == "Cube":
                size = tuple((UsdGeom.Cube(p).GetSizeAttr().Get() or 2.0) / 2.0 * s)
                kind = "box"
            elif kind == "Sphere":
                size = ((UsdGeom.Sphere(p).GetRadiusAttr().Get() or 1.0) * s.max(),)
                kind = "sphere"
            else:
                g = UsdGeom.Cylinder(p) if kind == "Cylinder" else UsdGeom.Capsule(p)
                r = (g.GetRadiusAttr().Get() or 1.0) * s[0]
                h = (g.GetHeightAttr().Get() or 2.0) * s[2]
                if (g.GetAxisAttr().Get() or "Z") != "Z":
                    warnings.append(f"{p.GetName()}: non-Z {kind.lower()} axis not supported; "
                                    "treated as Z")
                size = (r, h / 2.0)
                kind = kind.lower()
            body.primitives.append(PrimitiveGeom(
                name=p.GetName(), kind=kind, size=tuple(float(v) for v in size),
                pos=tuple(float(v) for v in pos), quat=tuple(float(v) for v in quat),
                material=material_of(p), visual=visual, collision=is_coll or not any_collision))
        # ---- mass / friction ------------------------------------------------
        if p.HasAPI(UsdPhysics.MassAPI):
            ma = UsdPhysics.MassAPI(p)
            mass = ma.GetMassAttr().Get()
            dens = ma.GetDensityAttr().Get()
            if mass and mass > 0:
                masses.append(float(mass))
                com = ma.GetCenterOfMassAttr().Get()
                c = to_asset(np.array([list(com)], dtype=float), p)[0] if com is not None \
                    and tuple(com) != (-float("inf"),) * 3 else None
                coms.append(c)
            elif dens and dens > 0:
                densities.append(float(dens))
        if is_coll:
            fr, dn, rest = _physics_material(p)
            if fr is not None:
                frictions.append(float(fr))
            if dn:
                densities.append(float(dn))
            if rest:
                warnings.append(f"{p.GetName()}: restitution {rest:.2f} ignored (MuJoCo "
                                "contacts are not restitution-based)")

    if not any(b.parts or b.primitives for b in link_bodies.values()):
        raise ValueError(f"{path.name}: no mesh or primitive geometry under {root.GetPath()}")
    if not any_collision:
        warnings.append("no CollisionAPI prims: the visual geometry is used for collision")
    for g in order:
        body = link_bodies[g]
        masses, coms, densities, frictions = per_link[g]
        if masses:
            body.mass = float(sum(masses))
            if all(c is not None for c in coms):
                body.center_of_mass = tuple(float(v) for v in
                                            np.average(np.array(coms), axis=0, weights=masses))
        elif densities:
            body.density = float(np.mean(densities))
        else:
            body.density = default_density
            warnings.append(f"{body.name}: no mass or density authored: default density "
                            f"{default_density} kg/m^3")
        if frictions:
            body.friction = float(np.mean(frictions))
    empty = [link_bodies[g].name for g in order
             if not (link_bodies[g].parts or link_bodies[g].primitives)]
    if empty:
        raise ValueError(f"{path.name}: link(s) {empty} have no geometry")

    labels: Dict[str, List[str]] = {}
    for a in root.GetAttributes():
        n = a.GetName()
        if n.startswith("semantics:labels:") and a.HasAuthoredValue():
            labels[n.split(":", 2)[2]] = [str(v) for v in (a.Get() or [])]
    return AssetData(source=path, meters_per_unit=mpu, up_axis=up, variant=used_variant,
                     bodies=[link_bodies[g] for g in order], materials=materials, labels=labels,
                     warnings=list(dict.fromkeys(warnings)))


def _quat_matrix(q) -> np.ndarray:
    """Gf.Quatf/Quatd -> 3x3 rotation, column-vector convention."""
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _link_tree(movable, groups, find, parent, to_asset, cache, root_inv, mpu, to_z, fname):
    """Revolute/prismatic joints -> (links in parent-first order, {link: (parent, JointData)}).

    A link is a fixed-joint group (named by its union-find root). The root link
    is the one that is never a joint's body1 -- body0 is the parent side in
    SimReady's assets -- falling back to the most-connected one. A joint met
    from its body1 side is reversed (axis negated, limits swapped and negated).
    """
    edges = []
    named = {}                    # link -> the rigid body its joints name (door_01, not shelf_01)
    for p in movable:
        j = UsdPhysics.Joint(p)
        t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        a = _owning_body(t0[0], parent) if t0 else None
        b = _owning_body(t1[0], parent) if t1 else None
        if a is None or b is None:
            raise ArticulatedAssetError(f"{fname}: joint {p.GetName()} connects to the world "
                                        "or to a non-rigid prim; not supported")
        named.setdefault(find(a), a)
        named[find(b)] = b
        kind = "hinge" if p.IsA(UsdPhysics.RevoluteJoint) else \
            "slide" if p.IsA(UsdPhysics.PrismaticJoint) else None
        if kind is None:
            raise ArticulatedAssetError(f"{fname}: joint {p.GetName()} is a {p.GetTypeName()}; "
                                        "only revolute and prismatic joints are supported")
        axis_tok = str(p.GetAttribute("physics:axis").Get() or "X")
        e = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[axis_tok]
        lp0 = p.GetAttribute("physics:localPos0").Get()
        lr0 = p.GetAttribute("physics:localRot0").Get()
        frame = cache.GetLocalToWorldTransform(Usd.Stage.GetPrimAtPath(p.GetStage(), t0[0]))
        m = _matrix(frame) @ root_inv                       # row-vector convention
        anchor = to_asset(np.array([list(lp0 if lp0 is not None else (0, 0, 0))], float),
                          Usd.Stage.GetPrimAtPath(p.GetStage(), t0[0]))[0]
        v = (_quat_matrix(lr0) @ e) if lr0 is not None else e
        v = (v @ m[:3, :3]) * mpu
        scale = float(np.linalg.norm(v)) or 1.0
        axis = to_z @ (v / scale)
        lo = p.GetAttribute("physics:lowerLimit").Get()
        hi = p.GetAttribute("physics:upperLimit").Get()
        rng = None
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            rng = (float(np.deg2rad(lo)), float(np.deg2rad(hi))) if kind == "hinge" \
                else (float(lo) * scale, float(hi) * scale)
        damp = p.GetAttribute(f"drive:{'angular' if kind == 'hinge' else 'linear'}"
                              ":physics:damping").Get()
        edges.append((find(a), find(b), JointData(
            name=_link_name(str(p.GetPath())).replace("joint_", "").replace("_joint_00", ""),
            kind=kind, axis=tuple(float(x) for x in axis), anchor=tuple(float(x) for x in anchor),
            range=rng, damping=float(damp) if damp is not None else None)))
    children = {b for _, b, _ in edges}
    candidates = [g for g in groups if g not in children]
    if len(candidates) != 1:
        degree = {g: sum(g in (a, b) for a, b, _ in edges) for g in groups}
        candidates = [max(groups, key=lambda g: (degree[g], g))]
    root = candidates[0]
    links = {root: (None, None)}
    order = [root]
    frontier = [root]
    while frontier:
        g = frontier.pop(0)
        for a, b, jd in edges:
            if a == g and b not in links:
                links[b] = (g, jd)
            elif b == g and a not in links:
                lo_hi = None if jd.range is None else (-jd.range[1], -jd.range[0])
                links[a] = (g, JointData(jd.name, jd.kind, tuple(-x for x in jd.axis),
                                         jd.anchor, lo_hi, jd.damping))
            else:
                continue
            child = b if a == g else a
            order.append(child)
            frontier.append(child)
    missing = sorted(set(groups) - set(links))
    if missing:
        raise ArticulatedAssetError(f"{fname}: link(s) {[_link_name(g) for g in missing]} "
                                    "are not connected to the root by any joint")
    names = {g: _link_name(named.get(g, g)) for g in order}
    return order, links, names
