"""Convert UFACTORY's pickled GG-CNN models into plain state dicts.

    python -m perception.grasp.convert_weights --src ~/Models/ufactory_vision
    python -m perception.grasp.convert_weights --verify-only

The shipped weights in ``weights/`` were produced by this script. It exists so
the conversion is reproducible and auditable rather than a step someone once ran
by hand, and so re-running it against a newer upstream is a single command.

Why convert at all
------------------
UFACTORY ships ``torch.save(model)`` of the whole module, not its parameters.
Loading that requires ``weights_only=False``, which unpickles arbitrary Python
objects -- it executes code from the file. It also hard-codes the class path
``models.ggcnn.GGCNN``, so the load only works if a package called ``models``
happens to be importable, which upstream arranges with a ``sys.path.append``
into its own tree. Both properties are fine in a demo repo and bad in a
dependency.

A ``state_dict`` is plain tensors. It loads under ``weights_only=True`` (no code
execution), and it binds to the vendored ``_ggcnn_net.py`` / ``_ggcnn2_net.py``
by shape rather than by pickle path, so nothing outside this package needs to
exist at import time.

The conversion preserves the network exactly: ``--verify-only`` re-checks that
the state dict reloaded into a fresh net reproduces the original pickled model's
output bit-for-bit on random input. When it was first run, both models matched
to 0.0.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"

# (upstream filename, our filename, vendored module, class name)
CONVERSIONS = [
    ("ggcnn_epoch_23_cornell", "ggcnn_epoch_23_cornell.pt", "_ggcnn_net", "GGCNN"),
    ("epoch_50_cornell", "ggcnn2_epoch_50_cornell.pt", "_ggcnn2_net", "GGCNN2"),
]


def _install_models_alias() -> None:
    """Make ``models.ggcnn`` / ``models.ggcnn2`` resolve to OUR vendored copies.

    The pickles name their classes under a top-level ``models`` package. Rather
    than putting upstream's directory on ``sys.path`` -- which would unpickle
    against code we are not shipping -- point that name at the files in this
    package, so what gets loaded is what gets vendored.
    """
    pkg = types.ModuleType("models")
    pkg.__path__ = []  # namespace-ish; we register the submodules by hand
    sys.modules["models"] = pkg
    for mod_name, local in (("models.ggcnn", "_ggcnn_net.py"),
                            ("models.ggcnn2", "_ggcnn2_net.py")):
        spec = importlib.util.spec_from_file_location(mod_name, HERE / local)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


def _equivalent(original, rebuilt, torch) -> float:
    """Max absolute difference between two nets on the same random input."""
    original.eval()
    rebuilt.eval()
    x = torch.from_numpy(
        np.random.default_rng(0).normal(size=(1, 1, 300, 300)).astype(np.float32))
    with torch.no_grad():
        a = torch.cat([t.flatten() for t in original(x)])
        b = torch.cat([t.flatten() for t in rebuilt(x)])
    return float((a - b).abs().max())


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/home/ksa/Models/ufactory_vision",
                    help="ufactory_vision checkout holding the pickled models")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the shipped weights against upstream; write nothing")
    args = ap.parse_args()

    src_dir = Path(args.src).expanduser() / "ggcnn_grasping_demo" / "models"
    if not src_dir.is_dir():
        print(f"upstream models not found at {src_dir}")
        return 1

    _install_models_alias()
    WEIGHTS.mkdir(exist_ok=True)
    failures = 0

    for src_name, out_name, mod_name, cls_name in CONVERSIONS:
        src = src_dir / src_name
        if not src.exists():
            print(f"SKIP {src_name}: not in {src_dir}")
            continue

        # weights_only=False is unavoidable HERE -- reading upstream's pickle is
        # the whole point. It is why the output is a state dict, so that no
        # consumer ever has to do this again.
        original = torch.load(src, map_location="cpu", weights_only=False)
        dest = WEIGHTS / out_name
        if not args.verify_only:
            torch.save(original.state_dict(), dest)

        if not dest.exists():
            print(f"FAIL {out_name}: not present and --verify-only was given")
            failures += 1
            continue

        net_cls = getattr(
            importlib.import_module(f".{mod_name}", __package__), cls_name)
        rebuilt = net_cls()
        rebuilt.load_state_dict(
            torch.load(dest, map_location="cpu", weights_only=True))

        delta = _equivalent(original, rebuilt, torch)
        params = sum(p.numel() for p in rebuilt.parameters())
        status = "OK  " if delta == 0.0 else "FAIL"
        if delta != 0.0:
            failures += 1
        print(f"{status} {out_name:30s} {params:>7d} params  "
              f"max|upstream - ours| = {delta:.3e}  "
              f"{dest.stat().st_size // 1024} KB")

    print()
    print(f"{failures} failure(s)" if failures
          else "shipped weights are equivalent to upstream's")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
