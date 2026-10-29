# simready_import/manifest.py
"""assets/simready/manifest.json: one entry per imported asset slug, recording
where it came from (repo, path, the pinned revision sha), its licence, the
source's units and up axis, the conversion options, and a sha256 of every file
written -- so an imported asset can be traced back and checked for edits."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from simready_import import ASSETS_DIR, MANIFEST


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path = MANIFEST) -> Dict[str, dict]:
    return json.loads(path.read_text()) if path.is_file() else {}


def save(entries: Dict[str, dict], path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(entries.items())), indent=2) + "\n")


def record(slug: str, *, repo_id: str, path_in_repo: str, revision: str, license: str,
           source_units_m: float, source_up_axis: str, variant: str, options: dict,
           asset_dir: Path, bbox_mm, mass_kg: float, labels: dict,
           manifest_path: Path = MANIFEST) -> dict:
    files = {p.name: sha256(p) for p in sorted(asset_dir.iterdir()) if p.is_file()}
    entry = {
        "repo_id": repo_id, "path_in_repo": path_in_repo, "revision": revision,
        "license": license, "source_meters_per_unit": source_units_m,
        "source_up_axis": source_up_axis, "physics_variant": variant,
        "options": options, "bbox_mm": [round(float(v), 1) for v in bbox_mm],
        "mass_kg": round(float(mass_kg), 5), "labels": labels, "files": files,
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries = load(manifest_path)
    entries[slug] = entry
    save(entries, manifest_path)
    return entry


def entry(slug: str, manifest_path: Path = MANIFEST) -> Optional[dict]:
    return load(manifest_path).get(slug)


def asset_dir(slug: str) -> Path:
    return ASSETS_DIR / slug
