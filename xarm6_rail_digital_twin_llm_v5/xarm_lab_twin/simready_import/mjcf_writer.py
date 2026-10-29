# simready_import/mjcf_writer.py
"""Neutral dataclasses (usd_reader) -> an MJCF fragment plus mesh files.

    result = write_mjcf(asset, "cucumber_a01", out_dir)

Writes into out_dir (emptied first):
  visual_<k>.obj      one per visual part, with UVs (decimated above a triangle
                      budget -- decimation drops UVs, so that part goes untextured)
  collision_<k>.obj   convex pieces: one hull per collision part, or CoACD
                      pieces when the USD asks for convexDecomposition or the
                      part is clearly concave (MuJoCo collides every mesh as its
                      convex hull, so a concave bin or tray must be split)
  <name>.png          diffuse textures, downscaled to texture_max_px
  <slug>.xml          assets + ONE body: visual geoms contype=0 conaffinity=0
                      group=2; collision geoms group=3, invisible

The body origin is moved to the CENTRE of the visual bounding box -- the
convention of the lab's cubes, which the grasp gate relies on (it measures
the jaws against the body origin; a bottom origin read as "object 95 mm below
the jaws"). To set an object on a surface, pose it at z = surface + height/2;
`simready_import place` does that by default. Mass and centre of
mass are the USD's when authored (inertia from the collision hulls about that
centre), otherwise density x hull volume.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

#: Fraction of a part's hull volume that must be EMPTY before it counts as
#: concave and is decomposed (a cup: large; a bottle: ~0).
CONCAVITY_THRESHOLD = 0.15

#: Collision geoms get condim 6 (torsional + rolling friction). A convex mesh
#: touches a flat surface at very few points, and a round object (the
#: cucumber) otherwise rocks on its curved side indefinitely: measured, it
#: was still rocking after 2 s with rolling friction 1e-4 and came to rest in
#: 1.3 s at 0.005 m (real produce and plastics resist rolling by deforming).
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.005


@dataclass
class WriteResult:
    xml: Path
    bbox_mm: np.ndarray            # size x, y, z
    mass: float
    n_visual: int
    n_collision: int
    origin_shift: np.ndarray       # added to every point (metres)
    warnings: List[str] = field(default_factory=list)


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, uvs=None) -> None:
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if uvs is None:
            for a, b, c in faces + 1:
                f.write(f"f {a} {b} {c}\n")
            return
        for tri in uvs:
            for u, v in tri:
                f.write(f"vt {u:.6f} {v:.6f}\n")
        for t, (a, b, c) in enumerate(faces + 1):
            k = 3 * t + 1
            f.write(f"f {a}/{k} {b}/{k + 1} {c}/{k + 2}\n")


def _compact(verts: np.ndarray, faces: np.ndarray):
    used, inv = np.unique(faces, return_inverse=True)
    return verts[used], inv.reshape(faces.shape)


def _decimate(verts, faces, target: int):
    import fast_simplification
    ratio = 1.0 - target / len(faces)
    v, f = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32),
                                        target_reduction=ratio)
    return np.asarray(v, float), np.asarray(f, np.int64)


def _hulls(verts, faces, approximation: str, max_hulls: int, warnings: List[str], name: str):
    mesh = trimesh.Trimesh(verts, faces, process=True)
    hull = mesh.convex_hull
    concave = False
    if approximation == "convexDecomposition":
        concave = True
    elif approximation in ("none", "meshSimplification", "sdf") and mesh.is_watertight \
            and hull.volume > 0:
        concave = (1.0 - mesh.volume / hull.volume) > CONCAVITY_THRESHOLD
    if approximation in ("boundingCube", "boundingSphere"):
        warnings.append(f"{name}: approximation {approximation} written as a convex hull")
    if not concave:
        return [hull]
    import coacd
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                            threshold=0.05, max_convex_hull=max_hulls)
    hulls = [trimesh.Trimesh(v, f).convex_hull for v, f in parts]
    warnings.append(f"{name}: concave -> {len(hulls)} convex pieces (CoACD)")
    return hulls[:max_hulls]


def _texture(src: Path, dst: Path, max_px: int) -> None:
    from PIL import Image
    im = Image.open(src)
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px))
    im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(dst)


def write_mjcf(asset, slug: str, out_dir: Path, *, max_visual_tris: int = 40000,
               max_hulls: int = 16, texture_max_px: int = 1024,
               torsional_friction: float = TORSIONAL_FRICTION,
               rolling_friction: float = ROLLING_FRICTION) -> WriteResult:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    warnings: List[str] = []
    body = asset.body

    lo, hi = asset.bounds(visual=True)
    shift = -(lo + hi) / 2.0

    # ---- visual ------------------------------------------------------------
    textures: Dict[str, str] = {}
    materials: Dict[str, str] = {}
    lines_asset, lines_body = [], []
    for key, m in asset.materials.items():
        mname = f"{slug}_mat{len(materials)}"
        materials[key] = mname
        attrs = f'rgba="{m.rgba[0]:.4f} {m.rgba[1]:.4f} {m.rgba[2]:.4f} {m.rgba[3]:.4f}"'
        if m.texture is not None:
            if str(m.texture) not in textures:
                tname = f"{slug}_tex{len(textures)}"
                _texture(m.texture, out_dir / f"{tname}.png", texture_max_px)
                textures[str(m.texture)] = tname
                lines_asset.append(f'    <texture name="{tname}" type="2d" file="{tname}.png"/>')
            attrs += f' texture="{textures[str(m.texture)]}"'
        lines_asset.append(f'    <material name="{mname}" {attrs} specular="0.3" shininess="0.3"/>')

    vis = [p for p in body.parts if p.visual]
    total = sum(len(p.faces) for p in vis)
    n_vis = 0
    for p in vis:
        verts, faces = _compact(p.vertices + shift, p.faces)
        uvs = p.uvs
        if total > max_visual_tris:
            target = max(200, int(len(faces) * max_visual_tris / total))
            verts, faces = _decimate(verts, faces, target)
            uvs = None
            warnings.append(f"{p.name}: decimated to {len(faces)} triangles (texture dropped)")
        name = f"{slug}_visual{n_vis}"
        _write_obj(out_dir / f"visual_{n_vis}.obj", verts, faces, uvs)
        lines_asset.append(f'    <mesh name="{name}" file="visual_{n_vis}.obj"/>')
        mat = materials.get(p.material)
        if mat and uvs is None and asset.materials[p.material].texture is not None:
            mat_attr = f'rgba="{0.7} {0.7} {0.7} 1"'
        else:
            mat_attr = f'material="{mat}"' if mat else 'rgba="0.7 0.7 0.7 1"'
        lines_body.append(f'      <geom name="{slug}_visual{n_vis}" type="mesh" mesh="{name}" '
                          f'{mat_attr} contype="0" conaffinity="0" group="2" density="0"/>')
        n_vis += 1

    # ---- collision ---------------------------------------------------------
    hulls = []
    for p in (q for q in body.parts if q.collision):
        hulls += _hulls(p.vertices + shift, p.faces, p.approximation, max_hulls, warnings, p.name)
    hulls = hulls[:max_hulls]
    prim_lines = []
    for g in body.primitives:
        if not g.collision and not g.visual:
            continue
        pos = np.array(g.pos) + shift
        size = " ".join(f"{s:.6f}" for s in g.size)
        cls = ('group="3" rgba="0.5 0.5 0.5 0"' if g.collision else
               'contype="0" conaffinity="0" group="2" density="0"')
        prim_lines.append(f'      <geom type="{g.kind}" size="{size}" pos="{pos[0]:.6f} {pos[1]:.6f} '
                          f'{pos[2]:.6f}" quat="{" ".join(f"{q:.6f}" for q in g.quat)}" {cls}/>')

    volume = sum(h.volume for h in hulls)
    if body.mass is not None:
        mass = body.mass
    else:
        mass = (body.density or 1000.0) * volume
    friction = (f' condim="6" friction="{body.friction if body.friction else 1.0:.3f} '
                f'{torsional_friction:g} {rolling_friction:g}"')
    for k, h in enumerate(hulls):
        _write_obj(out_dir / f"collision_{k}.obj", np.asarray(h.vertices), np.asarray(h.faces))
        lines_asset.append(f'    <mesh name="{slug}_collision{k}" file="collision_{k}.obj"/>')
        lines_body.append(f'      <geom name="{slug}_collision{k}" type="mesh" mesh="{slug}_collision{k}" '
                          f'group="3" rgba="0.5 0.5 0.5 0" density="0"{friction}/>')
    lines_body += prim_lines

    # ---- inertia: authored mass + COM, shape from the hulls -----------------
    inertial = ""
    if hulls and volume > 0:
        dens = mass / volume
        com_h = sum(h.center_mass * h.volume for h in hulls) / volume
        I = np.zeros((3, 3))
        for h in hulls:
            Ih = h.moment_inertia * dens          # about the hull's own COM (unit density)
            d = h.center_mass - com_h
            I += Ih + dens * h.volume * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        # The authored centre of mass (a real object is not uniformly dense)
        # with the uniform-hull inertia tensor taken about its own centroid:
        # an approximation, but the tensor's shape is the object's.
        com = (np.array(body.center_of_mass) + shift) if body.center_of_mass is not None \
            else com_h
        full = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
        inertial = (f'      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{mass:.6f}" '
                    f'fullinertia="{" ".join(f"{v:.9g}" for v in full)}"/>')
    elif prim_lines:
        warnings.append("primitive collision only: MuJoCo computes inertia from the primitives")
        lines_body = [ln.replace('rgba="0.5 0.5 0.5 0"', f'rgba="0.5 0.5 0.5 0" density='
                                 f'"{body.density or 1000.0}"') for ln in lines_body]
    else:
        raise ValueError(f"{slug}: no collision geometry")

    xml = out_dir / f"{slug}.xml"
    xml.write_text("\n".join([
        f'<mujoco model="simready_{slug}">',
        f'  <!-- GENERATED by simready_import from {asset.source.name}; do not edit.',
        f'       Origin = centre of the visual bounding box. -->',
        '  <compiler angle="radian"/>',
        "  <asset>", *lines_asset, "  </asset>",
        "  <worldbody>",
        f'    <body name="{slug}">',
        *([inertial] if inertial else []),
        *lines_body,
        "    </body>",
        "  </worldbody>",
        "</mujoco>", ""]))
    size_mm = (hi - lo) * 1000.0
    return WriteResult(xml=xml, bbox_mm=size_mm, mass=float(mass), n_visual=n_vis,
                       n_collision=len(hulls), origin_shift=shift, warnings=warnings)
