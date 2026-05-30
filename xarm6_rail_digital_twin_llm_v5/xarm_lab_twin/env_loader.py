"""
Minimal .env loader (no python-dotenv dependency).
Reads KEY=VALUE pairs from .env in the project root and sets them in os.environ
if not already set. Skips blank lines and lines starting with '#'.
"""
import os
from pathlib import Path


def load_env(env_path: str = None, override: bool = False) -> dict:
    """Load .env file into os.environ. Returns dict of loaded keys."""
    if env_path is None:
        # Default: .env next to this file (project root)
        env_path = Path(__file__).resolve().parent / ".env"
    else:
        env_path = Path(env_path)

    loaded = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip exactly one matched pair of surrounding quotes. The old
        # .strip("'").strip('"') chomped unmatched/nested quotes that are
        # legitimate value characters (e.g. an API key containing a quote).
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val
            loaded[key] = val
    return loaded


if __name__ == "__main__":
    # CLI mode: `python env_loader.py` prints what the .env file declares,
    # whether or not it was already present in the shell environment.
    # (Values are masked so secrets don't leak to screen.)
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        print(f"[env_loader] No .env found at {env_path}.")
    else:
        loaded = load_env(override=True)
        if not loaded:
            print(f"[env_loader] .env at {env_path} contains no usable keys.")
        else:
            print(f"[env_loader] {len(loaded)} variable(s) declared in .env:")
            for k, v in loaded.items():
                masked = (v[:8] + "..." + v[-4:]) if len(v) > 16 else "***"
                print(f"  {k} = {masked}  (len={len(v)})")
