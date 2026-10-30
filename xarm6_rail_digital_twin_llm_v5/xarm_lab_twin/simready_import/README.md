# SimReady importer

Converts NVIDIA SimReady assets (OpenUSD, hosted as Hugging Face datasets) into
MuJoCo objects that any room can use as `"source": "simready:<slug>"`.

Install once into `xarm6sim`: `pip install -r requirements-simready.txt`
(x86_64 and aarch64 wheels for all of it). Run from `xarm_lab_twin/`.

```bash
python -m simready_import search tray
python -m simready_import fetch nvidia/simready-assets \
    food_produce_cucumber_a01/simready_usd/sm_food_produce_cucumber_a01_01.usda \
    --revision a666cb4381ba --slug cucumber_a01
python -m simready_import fetch --picks          # everything in assets/simready/picks.json
python -m twin_rooms new my_room
python -m simready_import place cucumber_a01 --room my_room --name cucumber --pos -0.1 -0.25
python -m simready_import list [--room my_room]
python -m simready_import remove cucumber --room my_room
```

## What each step does

| step | module | |
|---|---|---|
| search | `discover.py` | Matches the query against SimReady asset folder names (`food_produce_cucumber_a01`), then shows the root USD, size and licence. The backend is pluggable (`SearchBackend`); there is no NVIDIA USD Search backend yet, because its endpoint has not been verified. |
| fetch | `fetch.py` | Resolves the revision to a **commit sha**, downloads the whole asset folder (the USD references layers, payloads and textures) into `assets/simready/cache/<org>/<dataset>/<sha>/` (gitignored), and requires every dependency to resolve (`UsdUtils.ComputeAllDependencies`). Prints the licence from the dataset card and the asset's own `LICENSE.txt` and packaging file. |
| read | `usd_reader.py` | Produces neutral dataclasses with no MuJoCo involved. Converts to metres and Z-up and bakes transforms. Reads triangles, UVs, GeomSubsets, primitives, collision prims and their approximation, mass/centre-of-mass/density, friction, and UsdPreviewSurface colour and textures. Selects the asset's **MuJoCo physics variant**. Merges bodies held by fixed joints; any other joint, or an articulation root, is refused unless `--rigid` is given. |
| write | `mjcf_writer.py` | Writes `assets/simready/<slug>/`: `visual_*.obj` (decimated above a budget), `collision_*.obj` (one hull, or CoACD pieces for concave parts), textures, and `<slug>.xml`. The fragment has one body (or, with `--articulated`, a body per link): visual geoms in group 1 that do not collide, and invisible collision geoms in group 3. Not group 2: the lab hides that group (its backdrop) in the viewer, the renderers and ray casts. |
| record | `manifest.py` | Adds an entry to `assets/simready/manifest.json`: repo, path, sha, licence, source units and up-axis, options, bounding box, mass, and a sha256 for every file. |
| place | `cli.py` | Adds the object to the room's `room.json` and runs a **drop test**: build, simulate 2 s, then report the final pose, penetration and whether it is at rest. A failure (it explodes, falls through, drifts more than 20 mm, penetrates more than 5 mm, or is still moving) reverts the placement, unless `--keep` is given. |

## Articulated assets, handles, other datasets

- `--articulated` keeps revolute/prismatic joints: each fixed-joint group is a
  link (named after the body its joints name), the root at the asset's bbox
  centre, each child at its own, joints at the USD anchors with limits in rad/m.
  Joint damping and friction loss are ours (`HINGE_*`, `SLIDE_*`); a vertical
  slide gets a return spring, because friction loss creeps under gravity.
- Every collision part of an articulated asset is CoACD-decomposed (the MuJoCo
  variant's single hull fills a fridge solid), and large flat filled upward
  patches get a 4 mm support plate, grown 20 mm and clipped to the link.
- `--handle LINK=X,Y,Z:X,Y,Z[:NX,NY,NZ]` (asset USD frame, metres; the normal is
  the side the gripper comes from, default +x) writes a capsule site
  `<link>_handle`; the room builder adds a CONNECT per gripper.
- Contacts between links that touch at rest are excluded; the builder renames
  the root in those excludes. `--exclude PRIM` drops a loose accessory (one asset
  is one body tree).
- Physics variant: `MuJoCo=Enabled`, else `physics=physics` (ForgeCAD).
  `picks.json` entries may carry their own `repo_id`, `revision`, `max_hulls`,
  `max_visual_tris`, `handles`, `exclude`.
- The piece cap (`--max-hulls`) bounds CoACD's output only; convex colliders an
  asset authors are all kept (capping by volume dropped a clock's legs).
- Decimation drops UVs, so a mesh over budget loses its texture. Raise
  `max_visual_tris` for an asset whose look matters (the clock keeps 134k).
- `ATTRIBUTION.md` is generated from the manifest on every import.

## Conventions and decisions

- **The body origin is the centre of the visual bounding box.** That is the lab's
  cube convention, and the grasp gate relies on it, because it measures the jaws
  against the body origin. Without `Z`, `place` stands the object on the bench.
- **The asset's MuJoCo variant is used.** SimReady assets ship a `MuJoCo`
  physics variant: a convex-hull approximation per mesh, plus authored mass and
  centre of mass. These are read in preference to the PhysX variant, which asks
  for SDF collision.
- **Collision geoms use `condim="6"` with rolling friction 0.005.** A convex mesh
  touches a flat surface at very few points. With almost no rolling resistance,
  the cucumber was still rocking after 2 s; at 0.005 it is at rest in 1.3 s.
- **Restitution is logged and ignored.** MuJoCo contacts are not
  restitution-based.
- **Licences.** Each imported asset's licence is in the manifest. The
  `nvidia/simready-assets` dataset is CC-BY-4.0: commercial use is allowed, and
  attribution is required.

## Limits found on the first real assets

- A **tall object** cannot be grasped from above by the twin's magnetic
  gripper. The grasp gate requires the object's centre to be within 45 mm below
  the jaws. The 231 mm bottle rests and can be pushed, but not picked from the
  top. A real gripper would take it by the neck.
- The **cupcake** (74.7 mm at its narrowest) is at the 75 mm jaw limit of the
  standard gripper. The bio gripper spans 130 mm.

Tests: `MUJOCO_GL=egl python -m simready_import.test_importer` (offline: USD
stages built in the test, plus the committed assets). Add `--network` to also
fetch a real asset end to end.
