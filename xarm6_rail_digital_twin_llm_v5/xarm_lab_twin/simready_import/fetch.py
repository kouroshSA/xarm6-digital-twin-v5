# simready_import/fetch.py
"""Download a SimReady asset at a pinned revision, check it is complete, and
record where it came from.

    res = fetch("nvidia/simready-assets",
                "food_produce_cucumber_a01/simready_usd/sm_food_produce_cucumber_a01_01.usda",
                revision="a666cb43...")

- The revision is resolved to a COMMIT SHA before anything is downloaded, and
  the download is stored under that sha: a branch name would silently change
  what "the same asset" means.
- USD references other layers, payloads and textures, so the whole asset folder
  is fetched (snapshot_download with allow_patterns; thumbnails skipped), then
  every dependency of the root layer is resolved with
  UsdUtils.ComputeAllDependencies. Anything unresolved fails the fetch.
- The licence is read from the dataset card and the asset folder's own
  .metadata/LICENSE.txt / packaging json; the caller prints it (some SimReady
  datasets are non-commercial).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from simready_import import CACHE_DIR


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    repo_id: str
    path_in_repo: str
    revision: str                 # commit sha
    root: Path                    # local root USD
    folder: str                   # asset folder in the repo
    license: str
    license_detail: List[str]


def asset_folder(path_in_repo: str) -> str:
    """The asset's folder: everything before `simready_usd/` (SimReady layout),
    else the file's directory."""
    parts = path_in_repo.split("/")
    if "simready_usd" in parts:
        return "/".join(parts[:parts.index("simready_usd")])
    return "/".join(parts[:-1])


def resolve_revision(repo_id: str, revision: Optional[str], api=None) -> str:
    from huggingface_hub import HfApi
    api = api or HfApi()
    return api.dataset_info(repo_id, revision=revision).sha


def check_dependencies(root: Path) -> List[str]:
    """Unresolved dependencies of a USD layer (empty when complete)."""
    from pxr import Sdf, UsdUtils
    _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(root)))
    return [str(u) for u in unresolved]


def read_license(repo_id: str, sha: str, local_folder: Path, api=None) -> (str, List[str]):
    from huggingface_hub import HfApi
    api = api or HfApi()
    detail = []
    info = api.dataset_info(repo_id, revision=sha)
    card = (info.card_data or {}).get("license") if info.card_data else None
    if card:
        detail.append(f"dataset card: {card}")
    pkg = local_folder / "com.nvidia.simready.packaging.json"
    pkg_lic = json.loads(pkg.read_text()).get("license") if pkg.is_file() else None
    if pkg_lic:
        detail.append(f"asset packaging: {pkg_lic}")
    txt = local_folder / ".metadata" / "LICENSE.txt"
    if txt.is_file():
        first = next((ln.strip() for ln in txt.read_text(errors="replace").splitlines()
                      if ln.strip()), "")
        detail.append(f"asset LICENSE.txt: {first}")
    lic = (pkg_lic or card or "unknown")
    return str(lic).upper(), detail


def fetch(repo_id: str, path_in_repo: str, revision: Optional[str] = None,
          cache_dir: Path = CACHE_DIR, api=None) -> FetchResult:
    from huggingface_hub import snapshot_download
    sha = resolve_revision(repo_id, revision, api)
    folder = asset_folder(path_in_repo)
    local = Path(cache_dir) / repo_id / sha
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type="dataset", revision=sha, local_dir=str(local),
                      allow_patterns=[f"{folder}/**"],
                      ignore_patterns=["*/.thumbs/*", "*.png.png"])
    root = local / path_in_repo
    if not root.is_file():
        raise FetchError(f"{path_in_repo} is not in {repo_id}@{sha[:8]}")
    missing = check_dependencies(root)
    if missing:
        raise FetchError(f"{path_in_repo}: {len(missing)} dependency(ies) did not resolve "
                         f"locally: {missing[:5]}")
    lic, detail = read_license(repo_id, sha, local / folder, api)
    return FetchResult(repo_id, path_in_repo, sha, root, folder, lic, detail)
