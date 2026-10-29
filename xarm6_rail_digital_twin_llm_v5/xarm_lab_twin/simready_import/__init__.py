"""Import NVIDIA SimReady (OpenUSD) assets into rooms as MuJoCo objects.

    python -m simready_import search tray
    python -m simready_import fetch nvidia/simready-assets <folder>/simready_usd/<file>.usd --revision <sha>
    python -m simready_import place <slug> --room <id> --name <instance> --pos x y z
    python -m simready_import list [--room <id>]
    python -m simready_import remove <instance> --room <id>

See simready_import/README.md. usd_reader (USD -> dataclasses) and mjcf_writer
(dataclasses -> MJCF + meshes) do not import each other.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "simready"
CACHE_DIR = ASSETS_DIR / "cache"
MANIFEST = ASSETS_DIR / "manifest.json"
