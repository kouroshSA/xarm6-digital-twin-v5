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
                      group=1 (VISUAL_GROUP); collision geoms group=3, invisible

Articulated assets (usd_reader articulated=True) become a body tree: the root
link at the asset's bbox centre, each other link a child body at its own bbox
centre carrying a hinge or slide joint at the USD anchor. Every collision part
of a link is CoACD-decomposed (SimReady's MuJoCo variant authors one convex
hull per mesh, which fills a fridge's interior solid). Handles (`handles=`,
asset-frame segments) become capsule sites "<link>_handle" in their link;
the builder welds the gripper to them.

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

#: Geom group of the visual meshes. NOT 2: the lab reserves group 2 for its
#: room backdrop, which the viewer, every offscreen renderer and mj_ray hide
#: by default (sim/render_options.py) -- in group 2 every imported object was
#: invisible to the live viewer, to recorded frames and to the wrist camera.
#: Group 1 is shown by MuJoCo's default MjvOption, which all three start from.
VISUAL_GROUP = 1

#: Joint defaults for articulated links. A hinge or slide joint with no loss
#: drifts on every contact; frictionloss holds a door or drawer where it was
#: left, as a real seal or runner does. SimReady's authored drive damping
#: (0.1-1, PhysX units) is recorded in the manifest, not used.
HINGE_DAMPING, HINGE_FRICTIONLOSS = 2.0, 1.0          # N m s/rad, N m
SLIDE_DAMPING, SLIDE_FRICTIONLOSS = 20.0, 2.0         # N s/m, N
JOINT_ARMATURE = 0.05
#: frictionloss is a soft constraint, so a steady load creeps through it: a
#: toaster lever (11 g, vertical slide) slid 7 mm in 5 s and to its stop
#: within a minute. A vertical slide gets a return spring to its rest
#: position instead, as the real part has (gravity then holds it ~2 mm low).
VERTICAL_SLIDE_STIFFNESS = 50.0                       # N/m

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
    links: Dict[str, dict] = field(default_factory=dict)   # articulated: per-link summary


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


def _degenerate(verts, faces) -> bool:
    """True for a mesh MuJoCo cannot take a volume of (flat, or nearly so)."""
    ext = np.ptp(verts, axis=0)
    if ext.min() < 1e-4:
        return True
    try:
        return trimesh.Trimesh(verts, faces, process=False).convex_hull.volume < 1e-9
    except Exception:  # noqa: BLE001 -- qhull refuses coplanar input
        return True


def _decimate(verts, faces, target: int):
    import fast_simplification
    ratio = 1.0 - target / len(faces)
    v, f = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32),
                                        target_reduction=ratio)
    return np.asarray(v, float), np.asarray(f, np.int64)


def _hulls(verts, faces, approximation: str, max_hulls: int, warnings: List[str], name: str,
           force_decompose: bool = False):
    mesh = trimesh.Trimesh(verts, faces, process=True)
    hull = mesh.convex_hull
    concave = False
    if force_decompose:
        concave = True
    elif approximation == "convexDecomposition":
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
    if len(hulls) > 1:
        warnings.append(f"{name}: concave -> {len(hulls)} convex pieces (CoACD)")
    return hulls[:max_hulls]


#: Support plates (articulated assets): upward-facing patches at least this
#: big, flat to this tolerance and filling this much of their own outline
#: get a thin convex plate under them. CoACD's pieces left holes in a
#: crisper drawer's floor -- a box dropped mid-drawer fell through the drawer
#: and the fridge -- and the authored single hull fills the drawer solid. The
#: fill test keeps a drawer's rim (a thin ring) from becoming a lid.
SUPPORT_MIN_AREA_M2 = 0.004
SUPPORT_MIN_FILL = 0.6
SUPPORT_THICKNESS_M = 0.004
SUPPORT_FLAT_M = 0.006
#: A floor meets its walls in a curved fillet, which is not flat and gets no
#: plate; CoACD left a 30 mm gap there in the crisper drawer, and a 30 mm box
#: fell through it. Plates are grown by this much, clipped to the link's own
#: footprint so they never poke out of its walls.
SUPPORT_GROW_M = 0.020


def _clip_to_box(poly, lo, hi):
    """Sutherland-Hodgman: a convex polygon (N, 2) clipped to an axis box."""
    out = [tuple(p) for p in poly]
    for axis, bound, keep_below in ((0, lo[0], False), (0, hi[0], True),
                                    (1, lo[1], False), (1, hi[1], True)):
        src, out = out, []
        inside = (lambda p: p[axis] <= bound) if keep_below else (lambda p: p[axis] >= bound)
        for i, cur in enumerate(src):
            prev = src[i - 1]
            if inside(cur) != inside(prev):
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append(tuple(prev[k] + t * (cur[k] - prev[k]) for k in range(2)))
            if inside(cur):
                out.append(cur)
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def _support_plates(verts, faces) -> list:
    """Thin convex plates under the large, flat, filled upward-facing patches
    of a mesh (floors, shelves), largest first."""
    from scipy.spatial import ConvexHull
    mesh = trimesh.Trimesh(verts, faces, process=True)
    foot_lo, foot_hi = mesh.bounds[0][:2], mesh.bounds[1][:2]
    sel = np.nonzero(mesh.face_normals[:, 2] > 0.95)[0]
    if not len(sel):
        return []
    sub = mesh.submesh([sel], append=True, repair=False)
    comps = trimesh.graph.connected_components(sub.face_adjacency,
                                               nodes=np.arange(len(sub.faces)), min_len=1)
    plates = []
    for comp in comps:
        tri = sub.triangles[comp]
        z = tri[:, :, 2]
        area = float(sub.area_faces[comp].sum())
        if np.ptp(z) > SUPPORT_FLAT_M or area < SUPPORT_MIN_AREA_M2:
            continue
        xy = tri[:, :, :2].reshape(-1, 2)
        try:
            outline = ConvexHull(xy)
        except Exception:  # noqa: BLE001 -- degenerate (collinear) patch
            continue
        if area / outline.volume < SUPPORT_MIN_FILL:        # 2-D hull: volume is area
            continue
        top = float(z.max())
        poly = xy[outline.vertices]
        centre = poly.mean(axis=0)
        r = np.linalg.norm(poly - centre, axis=1, keepdims=True)
        poly = _clip_to_box(centre + (poly - centre) * (1 + SUPPORT_GROW_M / np.maximum(r, 1e-6)),
                            foot_lo, foot_hi)
        if len(poly) < 3:
            continue
        pts = np.vstack([np.c_[poly, np.full(len(poly), top)],
                         np.c_[poly, np.full(len(poly), top - SUPPORT_THICKNESS_M)]])
        plates.append((area, trimesh.Trimesh(pts).convex_hull))
    return [p for _, p in sorted(plates, key=lambda t: -t[0])]


def _texture(src: Path, dst: Path, max_px: int) -> None:
    from PIL import Image
    im = Image.open(src)
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px))
    im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB").save(dst)


def write_mjcf(asset, slug: str, out_dir: Path, *, max_visual_tris: int = 40000,
               max_hulls: int = 16, texture_max_px: int = 1024,
               torsional_friction: float = TORSIONAL_FRICTION,
               rolling_friction: float = ROLLING_FRICTION,
               handles: Optional[Dict[str, tuple]] = None) -> WriteResult:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    warnings: List[str] = []
    articulated = len(asset.bodies) > 1
    handles = dict(handles or {})
    unknown = sorted(set(handles) - {b.name for b in asset.bodies})
    if unknown:
        raise ValueError(f"{slug}: handle(s) on unknown link(s) {unknown}; links: "
                         f"{[b.name for b in asset.bodies]}")

    lo, hi = asset.bounds(visual=True)
    shift = -(lo + hi) / 2.0

    # ---- materials (shared by every link) ----------------------------------
    textures: Dict[str, str] = {}
    materials: Dict[str, str] = {}
    lines_asset: List[str] = []
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

    # The triangle budget is for the whole asset, shared out by triangle count.
    total = sum(len(p.faces) for b in asset.bodies for p in b.parts if p.visual)
    counters = {"vis": 0, "col": 0}
    origins, body_lines, links = {}, {}, {}
    total_mass = 0.0
    for b in asset.bodies:
        if b.parent is None:
            origin = -shift
        else:
            pts = [p.vertices[np.unique(p.faces)] for p in b.parts if p.visual] or \
                  [p.vertices[np.unique(p.faces)] for p in b.parts]
            allp = np.concatenate(pts)
            origin = (allp.min(axis=0) + allp.max(axis=0)) / 2.0
        origins[b.name] = origin
        lines, mass = _body_geoms(b, slug, out_dir, -origin, lines_asset, materials, asset,
                                  total, max_visual_tris, max_hulls, warnings, counters,
                                  torsional_friction, rolling_friction,
                                  force_decompose=articulated)
        total_mass += mass
        if b.name in handles:
            lines.append(_handle_site(b.name, handles[b.name], origin))
        if b.joint is not None:
            j = b.joint
            anchor = np.asarray(j.anchor) - origin
            rng = j.range
            if rng is not None and not (rng[0] <= 0.0 <= rng[1]):
                warnings.append(f"{b.name}: joint range {np.round(rng, 4).tolist()} excludes the "
                                "rest position 0; widened to include it")
                rng = (min(rng[0], 0.0), max(rng[1], 0.0))
            lim = (f'limited="true" range="{rng[0]:.6f} {rng[1]:.6f}"' if rng is not None
                   else 'limited="false"')
            damp, loss = ((HINGE_DAMPING, HINGE_FRICTIONLOSS) if j.kind == "hinge"
                          else (SLIDE_DAMPING, SLIDE_FRICTIONLOSS))
            spring = ""
            if j.kind == "slide" and abs(j.axis[2]) > 0.5:
                spring = f' stiffness="{VERTICAL_SLIDE_STIFFNESS:g}"'
                warnings.append(f"{b.name}: vertical slide -> return spring "
                                f"{VERTICAL_SLIDE_STIFFNESS:g} N/m to its rest position")
            lines.insert(0, f'      <joint name="{b.name}_joint" type="{j.kind}" '
                            f'pos="{anchor[0]:.6f} {anchor[1]:.6f} {anchor[2]:.6f}" '
                            f'axis="{j.axis[0]:.6f} {j.axis[1]:.6f} {j.axis[2]:.6f}" {lim} '
                            f'damping="{damp:g}" frictionloss="{loss:g}" armature="{JOINT_ARMATURE:g}"'
                            f'{spring}/>')
            links[b.name] = {"parent": b.parent, "joint": f"{b.name}_joint", "kind": j.kind,
                             "range": list(rng) if rng is not None else None,
                             "authored_damping": j.damping, "mass_kg": round(mass, 4),
                             "handle": b.name in handles}
        body_lines[b.name] = lines

    def emit(name, depth, parent_origin):
        pad = "  " * depth
        rel = origins[name] - parent_origin
        bname = slug if depth == 2 else name
        pos = "" if depth == 2 else f' pos="{rel[0]:.6f} {rel[1]:.6f} {rel[2]:.6f}"'
        out = [f'{pad}<body name="{bname}"{pos}>']
        out += [pad + ln[4:] if depth > 2 else ln for ln in body_lines[name]]
        for c in asset.bodies:
            if c.parent == name:
                out += emit(c.name, depth + 1, origins[name])
        out.append(f"{pad}</body>")
        return out

    root = asset.bodies[0].name
    tree = emit(root, 2, np.zeros(3))
    # MuJoCo filters parent-child contacts only when the parent is not welded
    # to the world -- and a fixed fridge's root is. So a door against the
    # body it hangs on needs an exclude, as do siblings that meet (two doors
    # at a seam). Deeper parent-child pairs (a flap on a door) are filtered
    # already. The room builder renames the root and rewrites these.
    parent_of = {b.name: b.parent for b in asset.bodies}
    root_name = asset.bodies[0].name
    excludes = [(a, b) for a, b in
                (_self_contacts(lines_asset, tree, out_dir, slug) if articulated else [])
                if slug in (a, b) or not (parent_of.get(a) == b or parent_of.get(b) == a)
                or root_name in (parent_of.get(a), parent_of.get(b))]
    if excludes:
        warnings.append(f"{len(excludes)} link pair(s) touch at rest: contact excluded "
                        f"({', '.join(f'{a}/{b}' for a, b in excludes)})")
    xml = out_dir / f"{slug}.xml"
    xml.write_text("\n".join([
        f'<mujoco model="simready_{slug}">',
        f'  <!-- GENERATED by simready_import from {asset.source.name}; do not edit.',
        f'       Origin = centre of the visual bounding box. -->',
        '  <compiler angle="radian"/>',
        "  <asset>", *lines_asset, "  </asset>",
        "  <worldbody>", *tree, "  </worldbody>",
        *(["  <contact>", *[f'    <exclude body1="{a}" body2="{b}"/>' for a, b in excludes],
           "  </contact>"] if excludes else []),
        "</mujoco>", ""]))
    size_mm = (hi - lo) * 1000.0
    return WriteResult(xml=xml, bbox_mm=size_mm, mass=float(total_mass),
                       n_visual=counters["vis"], n_collision=counters["col"],
                       origin_shift=shift, warnings=warnings, links=links)


def _handle_site(link: str, handle, origin) -> str:
    """A handle as a capsule site: centre at the segment's middle, local z
    along the bar, local x the OUTWARD normal (the side the gripper comes
    from; default the asset's +x, SimReady furniture's front), half-length
    in size[1]. The simulator reads all three back from the site frame."""
    p0, p1 = (np.asarray(v, float) - origin for v in handle[:2])
    normal = np.asarray(handle[2] if len(handle) > 2 else (1.0, 0.0, 0.0), float)
    z = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    x = normal - z * np.dot(normal, z)
    if np.linalg.norm(x) < 1e-6:
        raise ValueError(f"handle {link}: its normal is parallel to the bar")
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    q = np.zeros(4)
    import mujoco
    mujoco.mju_mat2Quat(q, R.ravel())
    c = (p0 + p1) / 2
    return (f'      <site name="{link}_handle" type="capsule" '
            f'size="0.012 {np.linalg.norm(p1 - p0) / 2:.5f}" '
            f'pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}" '
            f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" group="4" rgba="1 0.5 0 0.6"/>')


def _primitive_volume(g) -> float:
    """Volume of a PrimitiveGeom (MuJoCo half-size conventions)."""
    if g.kind == "box":
        return 8.0 * g.size[0] * g.size[1] * g.size[2]
    if g.kind == "sphere":
        return 4.0 / 3.0 * np.pi * g.size[0] ** 3
    r, h = g.size[0], g.size[1]
    cyl = np.pi * r * r * 2.0 * h
    return cyl + (4.0 / 3.0 * np.pi * r ** 3 if g.kind == "capsule" else 0.0)


def _self_contacts(lines_asset, tree, out_dir: Path, slug: str):
    """Link pairs of this asset in contact at qpos0 (siblings that overlap by
    construction, e.g. a drawer resting on a shelf it slides over)."""
    import mujoco
    probe = out_dir / "_probe.xml"
    probe.write_text("\n".join(['<mujoco>', '  <compiler angle="radian"/>', "  <asset>",
                                *lines_asset, "  </asset>", "  <worldbody>", *tree,
                                "  </worldbody>", "</mujoco>"]))
    try:
        m = mujoco.MjModel.from_xml_path(str(probe))
    finally:
        probe.unlink()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = (m.body(int(m.geom_bodyid[g])).name for g in (c.geom1, c.geom2))
        if b1 != b2:
            pairs.add(tuple(sorted((b1, b2))))
    return sorted(pairs)


def _body_geoms(body, slug, out_dir, shift, lines_asset, materials, asset, total,
                max_visual_tris, max_hulls, warnings, counters, torsional_friction,
                rolling_friction, force_decompose=False):
    """Geoms + inertial for one body (points moved by `shift` into its frame).
    Appends mesh assets to lines_asset; returns (body lines, mass)."""
    lines_body = []
    vis = [p for p in body.parts if p.visual]
    for p in vis:
        verts, faces = _compact(p.vertices + shift, p.faces)
        uvs = p.uvs
        if total > max_visual_tris:
            target = max(200, int(len(faces) * max_visual_tris / total))
            if target < len(faces):           # a part already under its share stays as is
                verts, faces = _decimate(verts, faces, target)
                uvs = None
                warnings.append(f"{p.name}: decimated to {len(faces)} triangles (texture dropped)")
        k = counters["vis"]
        name = f"{slug}_visual{k}"
        _write_obj(out_dir / f"visual_{k}.obj", verts, faces, uvs)
        # MuJoCo computes a volume for every mesh, even a massless visual one,
        # and refuses a flat one (a 2-triangle decal); "shell" accepts it.
        flat = _degenerate(verts, faces)
        lines_asset.append(f'    <mesh name="{name}" file="visual_{k}.obj"'
                           f'{" inertia=" + chr(34) + "shell" + chr(34) if flat else ""}/>')
        mat = materials.get(p.material)
        if mat and uvs is None and asset.materials[p.material].texture is not None:
            mat_attr = f'rgba="{0.7} {0.7} {0.7} 1"'
        else:
            mat_attr = f'material="{mat}"' if mat else 'rgba="0.7 0.7 0.7 1"'
        lines_body.append(f'      <geom name="{slug}_visual{k}" type="mesh" mesh="{name}" '
                          f'{mat_attr} contype="0" conaffinity="0" group="{VISUAL_GROUP}" density="0"/>')
        counters["vis"] += 1

    # The piece cap bounds what CoACD GENERATES. Convex pieces the asset
    # authors itself (a Palatial clock ships 118 convexHull colliders) are all
    # kept: capping them by volume dropped the clock's legs, the smallest
    # pieces and the ones it stands on.
    tagged = []                                    # (piece, generated?) in part order
    for p in (q for q in body.parts if q.collision):
        pieces = _hulls(p.vertices + shift, p.faces, p.approximation, max_hulls, warnings, p.name,
                        force_decompose=force_decompose)
        tagged += [(h, len(pieces) > 1) for h in pieces]
    generated = [h for h, gen in tagged if gen]
    if len(generated) > max_hulls:
        warnings.append(f"{body.name}: {len(generated)} generated convex pieces, kept the "
                        f"{max_hulls} largest")
        keep = {id(h) for h in sorted(generated, key=lambda h: -h.volume)[:max_hulls]}
        tagged = [(h, gen) for h, gen in tagged if not gen or id(h) in keep]
    hulls = [h for h, _ in tagged]
    if force_decompose:                        # articulated: floors and shelves hold things
        cparts = [q for q in body.parts if q.collision]
        if cparts:
            offs = np.cumsum([0] + [len(q.vertices) for q in cparts])
            plates = _support_plates(np.concatenate([q.vertices for q in cparts]) + shift,
                                     np.concatenate([q.faces + o for q, o in zip(cparts, offs)]))
            if plates:
                warnings.append(f"{body.name}: {len(plates)} support plate(s) under its "
                                "floors/shelves")
            hulls += plates
    prim_lines = []
    for g in body.primitives:
        if not g.collision and not g.visual:
            continue
        pos = np.array(g.pos) + shift
        size = " ".join(f"{s:.6f}" for s in g.size)
        cls = ('group="3" rgba="0.5 0.5 0.5 0"' if g.collision else
               f'contype="0" conaffinity="0" group="{VISUAL_GROUP}" density="0"')
        prim_lines.append(f'      <geom type="{g.kind}" size="{size}" pos="{pos[0]:.6f} {pos[1]:.6f} '
                          f'{pos[2]:.6f}" quat="{" ".join(f"{q:.6f}" for q in g.quat)}" {cls}/>')

    volume = sum(h.volume for h in hulls)
    if body.mass is not None:
        mass = body.mass
    else:
        mass = (body.density or 1000.0) * volume
    friction = (f' condim="6" friction="{body.friction if body.friction else 1.0:.3f} '
                f'{torsional_friction:g} {rolling_friction:g}"')
    for h in hulls:
        k = counters["col"]
        _write_obj(out_dir / f"collision_{k}.obj", np.asarray(h.vertices), np.asarray(h.faces))
        lines_asset.append(f'    <mesh name="{slug}_collision{k}" file="collision_{k}.obj"/>')
        lines_body.append(f'      <geom name="{slug}_collision{k}" type="mesh" mesh="{slug}_collision{k}" '
                          f'group="3" rgba="0.5 0.5 0.5 0" density="0"{friction}/>')
        counters["col"] += 1
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
        # MuJoCo computes the inertia from the primitives; an AUTHORED mass is
        # kept by choosing the density that gives it (it used to be ignored,
        # and the incubator's 47 kg shell came out at whatever 1000 kg/m^3 made).
        vol = sum(_primitive_volume(g) for g in body.primitives if g.collision)
        dens = (body.mass / vol) if (body.mass is not None and vol > 0) \
            else (body.density or 1000.0)
        mass = dens * vol
        warnings.append(f"{body.name}: primitive collision only; density {dens:.0f} kg/m^3 "
                        f"gives {mass:.2f} kg")
        lines_body = [ln.replace('rgba="0.5 0.5 0.5 0"', f'rgba="0.5 0.5 0.5 0" density='
                                 f'"{dens:.6g}"') for ln in lines_body]
    else:
        raise ValueError(f"{slug}: {body.name} has no collision geometry")
    return ([inertial] if inertial else []) + lines_body, float(mass)
