# simready_import/discover.py
"""Search for SimReady assets.

The search backend is pluggable: SearchBackend.search(query, limit) -> [Hit].
HfFolderSearch, the only one so far, matches the query against the asset
folder names of SimReady datasets on Hugging Face (nvidia/simready-assets has
~2,700 folders such as `food_produce_cucumber_a01`), then fills in each hit's
root USD, size and licence. An NVIDIA USD Search API backend can be added as
another SearchBackend; none is wired in, because its endpoint is not verified.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from simready_import import CACHE_DIR

DEFAULT_DATASETS = ("nvidia/simready-assets",)


@dataclass
class Hit:
    repo_id: str
    revision: str            # commit sha the hit was found at
    folder: str
    root_usd: Optional[str]  # path in repo
    size_mb: Optional[float]
    license: str
    score: float


class SearchBackend:
    def search(self, query: str, limit: int = 10) -> List[Hit]:
        raise NotImplementedError


def tokens(name: str) -> List[str]:
    """'food_beverage_bottledDrink_e01' -> ['food', 'beverage', 'bottled', 'drink', 'e01']"""
    out = []
    for part in re.split(r"[_\-\s/.]+", name):
        out += re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+[a-z]*|[a-z]\d+", part) or [part]
    return [t.lower() for t in out if t]


def _singular(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def score(query: str, folder: str) -> float:
    """Fraction of query words found among the folder's words (prefix match, so
    'micro' finds 'microplate'), with a small bonus for shorter names."""
    q = [_singular(t) for t in tokens(query)]
    f = [_singular(t) for t in tokens(folder)]
    if not q:
        return 0.0
    hits = sum(1 for t in q if any(w == t or w.startswith(t) or t.startswith(w) and len(w) > 3
                                   for w in f))
    return hits / len(q) + 0.01 / (1 + len(f)) if hits else 0.0


class HfFolderSearch(SearchBackend):
    def __init__(self, datasets: Sequence[str] = DEFAULT_DATASETS, api=None):
        from huggingface_hub import HfApi
        self.api = api or HfApi()
        self.datasets = list(datasets)

    def _index(self, repo_id: str):
        info = self.api.dataset_info(repo_id)
        sha = info.sha
        lic = (info.card_data or {}).get("license") if info.card_data else None
        lic = lic or next((t.split(":", 1)[1] for t in (info.tags or [])
                           if t.startswith("license:")), "unknown")
        cache = CACHE_DIR / "index" / f"{repo_id.replace('/', '__')}@{sha}.json"
        if cache.is_file():
            folders = json.loads(cache.read_text())
        else:
            folders = [e.path for e in self.api.list_repo_tree(repo_id, repo_type="dataset",
                                                               revision=sha, recursive=False)
                       if not e.path.startswith(".") and "." not in e.path]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(folders))
        return sha, str(lic).upper() if lic else "unknown", folders

    def _details(self, repo_id: str, sha: str, folder: str):
        files = list(self.api.list_repo_tree(repo_id, path_in_repo=folder, repo_type="dataset",
                                             revision=sha, recursive=True))
        size = sum(getattr(f, "size", 0) or 0 for f in files) / 1e6
        paths = [f.path for f in files]
        root = None
        meta = f"{folder}/.metadata/com.nvidia.simready.root_usds.json"
        if meta in paths:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id, meta, repo_type="dataset", revision=sha)
            entries = json.loads(Path(p).read_text()).get("entries", [])
            root = f"{folder}/{entries[0]}" if entries else None
        if root is None:
            usds = sorted(p for p in paths if p.endswith((".usd", ".usda", ".usdc", ".usdz"))
                          and "/payloads/" not in p and "/runnables/" not in p)
            root = usds[0] if usds else None
        return root, size

    def search(self, query: str, limit: int = 10) -> List[Hit]:
        hits: List[Hit] = []
        for repo in self.datasets:
            sha, lic, folders = self._index(repo)
            ranked = sorted(((score(query, f), f) for f in folders), reverse=True)
            for s, folder in ranked[:limit]:
                if s <= 0:
                    break
                hits.append(Hit(repo, sha, folder, None, None, lic, s))
        hits.sort(key=lambda h: -h.score)
        hits = hits[:limit]
        for h in hits:
            h.root_usd, h.size_mb = self._details(h.repo_id, h.revision, h.folder)
        return hits
