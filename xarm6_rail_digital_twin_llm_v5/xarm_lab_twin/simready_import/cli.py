# simready_import/cli.py
"""python -m simready_import <command>   (run from xarm_lab_twin/)

  search <query> [--limit N] [--dataset REPO ...]
        SimReady assets whose folder names match, with dataset, root USD, size
        and licence.
  fetch <repo_id> <path_in_repo> [--revision SHA] [--slug S] [--rigid]
        download at a pinned commit, check every dependency resolves, convert
        to assets/simready/<slug>/ and record it in assets/simready/manifest.json.
        Prints the licence and the bounding box in mm.
  fetch --picks
        fetch every asset in assets/simready/picks.json
  place <slug> --room ID --name NAME --pos X Y [Z] [--euler R P Y] [--fixed]
        [--label TEXT] [--type T] [--keep]
        add it to the room's room.json, then DROP-TEST it (build, simulate
        2 s, report pose, penetration, rest). A failing drop test reverts the
        placement unless --keep. Without Z the object stands on the bench
        (bench top 0.750 + half its height; the origin is its bbox centre).
  remove <name> --room ID
  list [--room ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from simready_import import ASSETS_DIR, manifest

BENCH_TOP_Z = 0.750


# ---------------------------------------------------------------------------
# fetch + convert
# ---------------------------------------------------------------------------
def default_slug(folder: str) -> str:
    return folder.split("/")[-1].lower().replace("-", "_")


def import_asset(repo_id: str, path_in_repo: str, revision: Optional[str] = None,
                 slug: Optional[str] = None, rigid: bool = False, max_hulls: int = 16,
                 max_tris: int = 40000) -> dict:
    from simready_import.fetch import fetch
    from simready_import.mjcf_writer import ROLLING_FRICTION, TORSIONAL_FRICTION, write_mjcf
    from simready_import.usd_reader import read_usd
    res = fetch(repo_id, path_in_repo, revision)
    slug = slug or default_slug(res.folder)
    print(f"[fetch] {repo_id}@{res.revision[:8]} {path_in_repo}")
    print(f"[fetch] LICENSE: {res.license}  ({'; '.join(res.license_detail)})")
    asset = read_usd(res.root, rigid=rigid)
    out = ASSETS_DIR / slug
    w = write_mjcf(asset, slug, out, max_visual_tris=max_tris, max_hulls=max_hulls)
    entry = manifest.record(
        slug, repo_id=repo_id, path_in_repo=path_in_repo, revision=res.revision,
        license=res.license, source_units_m=asset.meters_per_unit,
        source_up_axis=asset.up_axis, variant=asset.variant,
        options={"rigid": rigid, "max_hulls": max_hulls, "max_visual_tris": max_tris,
                 "torsional_friction": TORSIONAL_FRICTION, "rolling_friction": ROLLING_FRICTION},
        asset_dir=out, bbox_mm=w.bbox_mm, mass_kg=w.mass, labels=asset.labels)
    print(f"[import] {slug}: bbox {w.bbox_mm[0]:.1f} x {w.bbox_mm[1]:.1f} x {w.bbox_mm[2]:.1f} mm, "
          f"{w.mass * 1000:.0f} g, {w.n_visual} visual / {w.n_collision} collision mesh(es)")
    for msg in asset.warnings + w.warnings:
        print(f"[import] note: {msg}")
    print(f"[import] -> {w.xml}")
    return entry


def cmd_fetch(a) -> int:
    if a.picks:
        picks = json.loads((ASSETS_DIR / "picks.json").read_text())
        for p in picks["assets"]:
            import_asset(picks["repo_id"], p["root_usd"], picks["revision"], slug=p["slug"],
                         rigid=a.rigid)
        return 0
    if not (a.repo_id and a.path_in_repo):
        print("fetch needs <repo_id> <path_in_repo>, or --picks")
        return 2
    import_asset(a.repo_id, a.path_in_repo, a.revision, a.slug, a.rigid, a.max_hulls, a.max_tris)
    return 0


def cmd_search(a) -> int:
    from simready_import.discover import DEFAULT_DATASETS, HfFolderSearch
    hits = HfFolderSearch(a.dataset or DEFAULT_DATASETS).search(a.query, a.limit)
    if not hits:
        print(f"no SimReady asset folder matches {a.query!r}")
        return 1
    print(f"{'dataset':24s} {'file':78s} {'MB':>6s}  license")
    for h in hits:
        print(f"{h.repo_id:24s} {h.root_usd or h.folder + '/ (no root USD)':78s} "
              f"{h.size_mb or 0:6.1f}  {h.license}")
    print(f"(revision {hits[0].revision}; fetch with --revision {hits[0].revision[:12]})")
    return 0


# ---------------------------------------------------------------------------
# rooms
# ---------------------------------------------------------------------------
def _room_json(room_id: str) -> Path:
    from twin_rooms.room import ROOMS_DIR, TEMPLATES
    for d in (ROOMS_DIR / room_id, ROOMS_DIR / TEMPLATES / room_id, Path(room_id)):
        if (d / "room.json").is_file():
            return d / "room.json"
    raise SystemExit(f"no room {room_id!r}")


@dataclass
class DropReport:
    ok: bool
    final_pos_mm: np.ndarray
    moved_mm: float
    drop_mm: float
    max_penetration_mm: float
    speed: float
    problems: List[str]


def drop_test(room_dir: Path, name: str, seconds: float = 2.0) -> DropReport:
    """Build the room, let `name` settle for `seconds`, and judge it."""
    import mujoco
    from twin_rooms.builder import build_room
    model, ctx = build_room(str(room_dir), write=False)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    bid = model.body(ctx.objects[name].body).id
    geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    p0 = d.xpos[bid].copy()
    # cvel is (angular rad/s, linear m/s); judged separately -- a small object
    # can spin fast while barely moving.
    pen, lin_max, ang_max = 0.0, 0.0, 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in geoms or c.geom2 in geoms:
                pen = max(pen, -float(c.dist))
        lin_max = max(lin_max, float(np.linalg.norm(d.cvel[bid][3:])))
        ang_max = max(ang_max, float(np.linalg.norm(d.cvel[bid][:3])))
    p = d.xpos[bid].copy()
    speed = float(np.linalg.norm(d.cvel[bid][3:]))
    spin = float(np.linalg.norm(d.cvel[bid][:3]))
    problems = []
    if not np.all(np.isfinite(p)) or lin_max > 2.0 or ang_max > 50.0 \
            or np.linalg.norm(p - p0) > 0.5:
        problems.append(f"exploded (peak {lin_max:.1f} m/s, {ang_max:.0f} rad/s)")
    if p[2] < p0[2] - 0.03 or p[2] < 0.3:
        problems.append(f"fell through / off (z {p0[2]*1000:.0f} -> {p[2]*1000:.0f} mm)")
    if np.linalg.norm((p - p0)[:2]) > 0.02:
        problems.append(f"drifted {np.linalg.norm((p - p0)[:2])*1000:.0f} mm sideways")
    if pen > 0.005:
        problems.append(f"penetrates {pen*1000:.1f} mm")
    if speed > 0.01 or spin > 0.2:
        problems.append(f"not at rest after {seconds:.0f} s ({speed:.3f} m/s, {spin:.2f} rad/s)")
    return DropReport(not problems, p * 1000, float(np.linalg.norm(p - p0)) * 1000,
                      float(p0[2] - p[2]) * 1000, pen * 1000, speed, problems)


def _label(slug: str) -> str:
    words = [w for w in slug.split("_") if not (len(w) <= 3 and any(c.isdigit() for c in w))]
    return " ".join(words) or slug


def cmd_place(a) -> int:
    if not (ASSETS_DIR / a.slug / f"{a.slug}.xml").is_file():
        print(f"asset {a.slug!r} is not imported; `python -m simready_import fetch ...` first")
        return 1
    rj = _room_json(a.room)
    before = rj.read_text()
    d = json.loads(before)
    if any(o["name"] == a.name for o in d["objects"]):
        print(f"room {a.room!r} already has an object {a.name!r}")
        return 1
    ent = manifest.entry(a.slug) or {}
    # The asset's origin is its bounding-box centre: on the bench is half its
    # height above the bench top (upright; pass Z for anything else).
    rest_z = BENCH_TOP_Z + ent.get("bbox_mm", [0, 0, 0])[2] / 2000.0 + 0.001
    pos = list(a.pos) + ([round(rest_z, 4)] if len(a.pos) == 2 else [])
    aliases = [v for vs in ent.get("labels", {}).values() for v in vs
               if not v.startswith("Q")][:2]
    obj = {"name": a.name, "source": f"simready:{a.slug}",
           "pose": {"pos": pos, **({"euler": list(a.euler)} if a.euler else {})},
           "dynamic": not a.fixed, "type": a.type, "label": a.label or _label(a.slug),
           "grippable": not a.fixed}
    if aliases:
        obj["aliases"] = aliases
    d["objects"].append(obj)
    rj.write_text(json.dumps(d, indent=2) + "\n")
    try:
        from twin_rooms import load_room_dir
        load_room_dir(rj.parent)
        rep = drop_test(rj.parent, a.name)
    except Exception as e:  # noqa: BLE001
        rj.write_text(before)
        print(f"[place] could not build the room with {a.name}: {e}; placement reverted")
        return 1
    print(f"[drop test] {a.name} in {d['id']}: final ({rep.final_pos_mm[0]:.0f}, "
          f"{rep.final_pos_mm[1]:.0f}, {rep.final_pos_mm[2]:.0f}) mm, moved {rep.moved_mm:.1f} mm "
          f"(dropped {rep.drop_mm:.1f}), max penetration {rep.max_penetration_mm:.2f} mm, "
          f"speed {rep.speed:.4f} -> {'AT REST' if rep.ok else 'FAIL: ' + '; '.join(rep.problems)}")
    if not rep.ok and not a.keep:
        rj.write_text(before)
        print("[place] placement reverted (pass --keep to keep it anyway)")
        return 1
    return 0


def cmd_remove(a) -> int:
    rj = _room_json(a.room)
    d = json.loads(rj.read_text())
    n = len(d["objects"])
    d["objects"] = [o for o in d["objects"] if o["name"] != a.name]
    if len(d["objects"]) == n:
        print(f"room {a.room!r} has no object {a.name!r}")
        return 1
    rj.write_text(json.dumps(d, indent=2) + "\n")
    print(f"removed {a.name} from {d['id']}")
    return 0


def cmd_list(a) -> int:
    if a.room:
        d = json.loads(_room_json(a.room).read_text())
        rows = [o for o in d["objects"] if o["source"].startswith("simready:")]
        for o in rows:
            print(f"  {o['name']:20s} {o['source']:32s} pos {o['pose']['pos']}  "
                  f"{'dynamic' if o.get('dynamic', True) else 'fixed'}")
        if not rows:
            print(f"  (no SimReady objects in {a.room})")
        return 0
    for slug, e in manifest.load().items():
        bb = e["bbox_mm"]
        print(f"  {slug:22s} {bb[0]:6.1f} x {bb[1]:6.1f} x {bb[2]:6.1f} mm  {e['mass_kg']*1000:6.0f} g  "
              f"{e['license']:10s} {e['repo_id']}@{e['revision'][:8]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m simready_import", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--dataset", action="append")
    s.set_defaults(fn=lambda a: cmd_search(argparse.Namespace(**{**vars(a), "query": " ".join(a.query)})))
    f = sub.add_parser("fetch")
    f.add_argument("repo_id", nargs="?")
    f.add_argument("path_in_repo", nargs="?")
    f.add_argument("--revision")
    f.add_argument("--slug")
    f.add_argument("--rigid", action="store_true", help="import a jointed asset as one rigid body")
    f.add_argument("--picks", action="store_true", help="fetch assets/simready/picks.json")
    f.add_argument("--max-hulls", type=int, default=16)
    f.add_argument("--max-tris", type=int, default=40000)
    f.set_defaults(fn=cmd_fetch)
    pl = sub.add_parser("place")
    pl.add_argument("slug")
    pl.add_argument("--room", required=True)
    pl.add_argument("--name", required=True)
    pl.add_argument("--pos", type=float, nargs="+", required=True, metavar="M")
    pl.add_argument("--euler", type=float, nargs=3, metavar="DEG")
    pl.add_argument("--fixed", action="store_true")
    pl.add_argument("--label")
    pl.add_argument("--type", default="other",
                    choices=["cube", "bin", "container", "rack", "tube", "plate",
                             "instrument", "fixture", "other"])
    pl.add_argument("--keep", action="store_true")
    pl.set_defaults(fn=cmd_place)
    r = sub.add_parser("remove")
    r.add_argument("name")
    r.add_argument("--room", required=True)
    r.set_defaults(fn=cmd_remove)
    ls = sub.add_parser("list")
    ls.add_argument("--room")
    ls.set_defaults(fn=cmd_list)
    a = p.parse_args(argv)
    if a.cmd == "place" and len(a.pos) not in (2, 3):
        p.error("--pos takes X Y or X Y Z (metres)")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
