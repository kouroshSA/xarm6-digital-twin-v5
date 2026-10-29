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
                 asset has one (SimReady assets do), so the MuJoCo-authored
                 physics is what is read.
  articulation   rigid bodies connected by FIXED joints are one body; any other
                 joint, an ArticulationRootAPI, or unconnected rigid bodies are
                 refused (ArticulatedAssetError) unless rigid=True.
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
class BodyData:
    name: str
    parts: List[MeshPart] = field(default_factory=list)
    primitives: List[PrimitiveGeom] = field(default_factory=list)
    mass: Optional[float] = None            # kg, if authored
    center_of_mass: Optional[Tuple[float, float, float]] = None
    density: Optional[float] = None         # kg/m^3, if authored and mass is not
    friction: Optional[float] = None


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


def read_usd(path, rigid: bool = False, default_density: float = 1000.0,
             variant: Optional[Tuple[str, str]] = ("MuJoCo", "Enabled")) -> AssetData:
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD stage {path}")
    root = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()))
    warnings: List[str] = []
    used_variant = "default"
    if variant and root.GetVariantSets().HasVariantSet(variant[0]):
        vs = root.GetVariantSets().GetVariantSet(variant[0])
        if variant[1] in vs.GetVariantNames():
            vs.SetVariantSelection(variant[1])
            used_variant = f"{variant[0]}={variant[1]}"

    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    to_z = Y_UP_TO_Z_UP if up == "Y" else np.eye(3)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inv = np.linalg.inv(_matrix(cache.GetLocalToWorldTransform(root)))

    def to_asset(points_local: np.ndarray, prim) -> np.ndarray:
        m = _matrix(cache.GetLocalToWorldTransform(prim)) @ root_inv
        return (_apply(points_local, m) * mpu) @ to_z.T

    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))

    # ---- articulation -------------------------------------------------------
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims) and not rigid:
        raise ArticulatedAssetError(f"{path.name}: has an ArticulationRootAPI (a jointed "
                                    "asset); v1 imports rigid objects only -- pass --rigid "
                                    "to import it as one rigid body")
    parent = {str(b.GetPath()): str(b.GetPath()) for b in bodies}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        if not p.IsA(UsdPhysics.FixedJoint):
            if not rigid:
                raise ArticulatedAssetError(
                    f"{path.name}: joint {p.GetName()} is a {p.GetTypeName()}; v1 imports "
                    "rigid objects only -- pass --rigid to weld everything into one body")
            continue
        t0, t1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        # A joint may target a prim INSIDE a rigid body (SimReady targets the
        # mesh); it belongs to the nearest enclosing rigid body.
        a = _owning_body(t0[0], parent) if t0 else None
        b = _owning_body(t1[0], parent) if t1 else None
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups = {find(k) for k in parent}
    if len(groups) > 1 and not rigid:
        raise ArticulatedAssetError(f"{path.name}: {len(groups)} rigid bodies not joined by "
                                    "fixed joints; pass --rigid to import them as one body")
    n_fixed = sum(1 for p in prims if p.IsA(UsdPhysics.FixedJoint))
    if n_fixed:
        warnings.append(f"{n_fixed} fixed joint(s): merged into one rigid body")

    # ---- geometry -----------------------------------------------------------
    body = BodyData(name=root.GetName())
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
    masses, coms, densities, frictions = [], [], [], []
    for p in prims:
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

    if not body.parts and not body.primitives:
        raise ValueError(f"{path.name}: no mesh or primitive geometry under {root.GetPath()}")
    if not any_collision:
        warnings.append("no CollisionAPI prims: the visual geometry is used for collision")
    if masses:
        body.mass = float(sum(masses))
        if all(c is not None for c in coms):
            body.center_of_mass = tuple(float(v) for v in
                                        np.average(np.array(coms), axis=0, weights=masses))
    elif densities:
        body.density = float(np.mean(densities))
    else:
        body.density = default_density
        warnings.append(f"no mass or density authored: default density {default_density} kg/m^3")
    if frictions:
        body.friction = float(np.mean(frictions))

    labels: Dict[str, List[str]] = {}
    for a in root.GetAttributes():
        n = a.GetName()
        if n.startswith("semantics:labels:") and a.HasAuthoredValue():
            labels[n.split(":", 2)[2]] = [str(v) for v in (a.Get() or [])]
    return AssetData(source=path, meters_per_unit=mpu, up_axis=up, variant=used_variant,
                     bodies=[body], materials=materials, labels=labels,
                     warnings=list(dict.fromkeys(warnings)))
