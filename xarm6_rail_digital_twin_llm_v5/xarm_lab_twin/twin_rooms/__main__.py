"""python -m twin_rooms <command>

    list                           rooms and templates
    new <id> --from <template>     start a room from a template (default empty_bench)
    build <id> [--variant V]       build it; writes rooms/<id>/build/scene.xml
    validate [<id> ...]            check room.json files (all rooms by default)
    view <id> [--variant V]        open the room in the MuJoCo viewer
    export-usd <id> [--out DIR]    build it, then convert build/scene.xml to USD with
                                   mujoco-usd-converter (its own env; see rooms/README.md)

Run from xarm_lab_twin/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from twin_rooms import RoomError, list_rooms, load_room
from twin_rooms.room import ROOMS_DIR, TEMPLATES


def cmd_list(_args) -> int:
    for rid in list_rooms(include_templates=True):
        try:
            room = load_room(rid.split("/")[-1])
        except RoomError as e:
            print(f"  {rid:<26} INVALID: {e.problems[0]}")
            continue
        kind = "passthrough" if room.is_passthrough else "composed"
        robots = ", ".join(f"{r.instance}:{r.type}" for r in room.robots)
        print(f"  {rid:<26} {kind:<11} {len(room.objects):>2} objects  {robots}  -- {room.name}")
    return 0


def cmd_new(args) -> int:
    src = ROOMS_DIR / TEMPLATES / args.template
    dst = ROOMS_DIR / args.id
    if not (src / "room.json").is_file():
        print(f"no template {args.template!r}; templates: "
              f"{[r.split('/')[-1] for r in list_rooms(include_templates=True) if r.startswith(TEMPLATES)]}")
        return 1
    if dst.exists():
        print(f"{dst} already exists")
        return 1
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build"))
    d = json.loads((dst / "room.json").read_text())
    d["id"], d["name"] = args.id, args.name or args.id.replace("_", " ")
    # Paths in a template are relative to rooms/_templates/<t>/; one level less here.
    env = d.get("environment")
    if isinstance(env, dict) and env.get("extract_from", "").startswith("../../../"):
        env["extract_from"] = env["extract_from"][3:]
    (dst / "room.json").write_text(json.dumps(d, indent=2) + "\n")
    load_room(args.id)                                   # validates
    print(f"created {dst} from {args.template}; edit room.json, then "
          f"`python -m twin_rooms build {args.id}`")
    return 0


def cmd_build(args) -> int:
    from twin_rooms.builder import build_room
    model, ctx = build_room(args.id, variant=args.variant)
    print(f"{ctx.room_id}: {model.nbody - 1} bodies, {model.njnt} joints, {model.nu} actuators, "
          f"{model.neq} welds, {len(ctx.objects)} objects; hash {ctx.hash}")
    print(f"  {ctx.model_path}")
    return 0


def cmd_validate(args) -> int:
    ids = args.ids or [r.split("/")[-1] for r in list_rooms(include_templates=True)]
    bad = 0
    for rid in ids:
        try:
            load_room(rid)
            print(f"  ok       {rid}")
        except RoomError as e:
            bad += 1
            print(f"  INVALID  {e}")
    return 1 if bad else 0


def cmd_view(args) -> int:
    from twin_rooms.builder import build_room
    model, ctx = build_room(args.id, variant=args.variant)
    import mujoco.viewer
    print(f"viewing {ctx.room_id} ({ctx.model_path}); close the window to exit")
    mujoco.viewer.launch(model)
    return 0


#: mujoco-usd-converter pins its own MuJoCo (0.8.0 needs 3.16.x; this env runs
#: 3.8.1), so it lives in a separate environment and is run as a program.
CONVERTER_ENV = "XARM_USD_CONVERTER"


def cmd_export_usd(args) -> int:
    import os
    import shutil
    import subprocess
    from twin_rooms.builder import build_room
    exe = args.converter or os.environ.get(CONVERTER_ENV) or shutil.which("mujoco_usd_converter")
    if not exe:
        print("mujoco-usd-converter is not installed. It needs its own MuJoCo version, so "
              "install it in a separate env:\n"
              "  conda create -n xarm6sim_usd python=3.11 && conda activate xarm6sim_usd\n"
              "  pip install mujoco-usd-converter\n"
              f"then pass --converter $(which mujoco_usd_converter), or set {CONVERTER_ENV}.")
        return 2
    _, ctx = build_room(args.id, variant=args.variant)
    scene = ctx.model_path        # passthrough: its scene file; composed: build/scene.xml
    out = Path(args.out or (ctx.room.dir / "build" / "usd"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"converting {scene} -> {out}")
    r = subprocess.run([exe, str(scene), str(out)])
    if r.returncode == 0:
        print("note: MuJoCo-specific settings (the weld-constraint grasps, solver options) are "
              "carried in MjcPhysics schemas and may behave differently in other runtimes")
    return r.returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m twin_rooms", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    n = sub.add_parser("new")
    n.add_argument("id")
    n.add_argument("--from", dest="template", default="empty_bench")
    n.add_argument("--name")
    n.set_defaults(fn=cmd_new)
    for name, fn in (("build", cmd_build), ("view", cmd_view)):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.add_argument("--variant", help="meshes | primitive")
        s.set_defaults(fn=fn)
    e = sub.add_parser("export-usd")
    e.add_argument("id")
    e.add_argument("--variant")
    e.add_argument("--out")
    e.add_argument("--converter", help=f"the mujoco_usd_converter program (or ${CONVERTER_ENV})")
    e.set_defaults(fn=cmd_export_usd)
    v = sub.add_parser("validate")
    v.add_argument("ids", nargs="*")
    v.set_defaults(fn=cmd_validate)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except RoomError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
