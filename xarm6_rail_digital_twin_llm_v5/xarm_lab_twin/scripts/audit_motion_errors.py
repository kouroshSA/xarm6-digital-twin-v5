#!/usr/bin/env python3
"""Find motion commands that can fail and still report success.

This is the defect class that produced the rail bug: `hardware/real_arm.py` caught
an `AttributeError` from a renamed SDK method, logged a warning, and returned `0` —
the success code — so every caller believed a 700 mm rail move had happened when the
arm had not moved at all.

The audit is AST-based rather than grep-based, because the thing that matters is
*what an exception handler returns inside a motion function*, which a text search
cannot see. For every function whose name looks like it commands motion (or any
function in hardware/), every `try/except` is inspected and the handler is reported
if it:

  - returns a success-shaped value (`0`, `True`, or a tuple/list starting with `0`), or
  - swallows the exception entirely (`pass`) without re-raising.

Both mean a failed command becomes an apparently successful one.

    python scripts/audit_motion_errors.py            # human-readable report
    python scripts/audit_motion_errors.py --json     # machine-readable
    python scripts/audit_motion_errors.py --check    # exit 1 if any finding

Findings that are genuinely fine — an optional, non-motion side-effect where
continuing really is correct — go in ALLOWLIST with a reason, so the exception is
recorded rather than argued about again later.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

# Directories worth auditing. Excludes vendored assets and caches.
SCAN_DIRS = [".", "agent", "sim", "hardware", "scripts", "vr"]
SKIP_PARTS = {"__pycache__", ".git", "recordings", "envs", "assets", "node_modules"}

# Function names that command motion or actuate a tool.
MOTION_NAME = re.compile(
    r"(^|_)(set|move|goto|go|servo|grip|gripper|rail|push|place|home|wave|open|close|"
    r"nudge|reset|jog|lift|descend|retreat)", re.IGNORECASE)

# Every function in these directories is treated as motion-adjacent.
ALWAYS_AUDIT_DIRS = ("hardware",)

# Method names that actually command the arm. Matched against the *attribute* of a
# call node (`arm.set_rail_position(...)`), never against source text — matching
# text made `self.model.geom("gripper_geom")` look like a gripper command.
MOTION_CALL_ATTRS = {
    "set_position", "set_rail_position", "set_servo_angle", "set_servo_angle_j",
    "set_servo_cartesian", "set_linear_motor_pos", "set_linear_track_pos",
    "open_gripper", "close_gripper", "open_lite6_gripper", "close_lite6_gripper",
    "open_bio_gripper", "close_bio_gripper", "set_gripper_position",
    "set_vacuum_gripper", "push_object", "place_tube_in_rack", "go_home",
    "wave_goodbye", "set_pcr_lid", "nudge_body", "rail_home",
}

# Tests legitimately catch expected exceptions and return a success marker; that is
# the assertion, not a swallowed failure. Audited separately by their own asserts.
TEST_FILE = re.compile(r"(^|/)(test_[^/]+|[^/]+_test)\.py$")

# (file, function, handler_lineno) -> why this one is acceptable.
ALLOWLIST: dict[tuple[str, str, int], str] = {}


def _is_success_value(node: ast.AST | None) -> str | None:
    """Describe the returned value if it looks like a success code, else None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if node.value is True:
            return "True"
        # `return 0` — the classic. Note `False`/`None` are not success-shaped.
        if node.value == 0 and node.value is not False:
            return "0"
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        first = node.elts[0]
        if isinstance(first, ast.Constant) and first.value == 0 and first.value is not False:
            return f"({first.value}, ...)"
    return None


def _handler_reraises(handler: ast.ExceptHandler) -> bool:
    return any(isinstance(n, ast.Raise) for n in ast.walk(handler))


def _handler_is_bare_pass(handler: ast.ExceptHandler) -> bool:
    return all(isinstance(stmt, ast.Pass) for stmt in handler.body)


def _calls_motion(node: ast.Try) -> str | None:
    """Name of the motion method called in this try body, if any.

    Inspects Call nodes rather than source text: `self.model.geom("gripper_geom")`
    contains the word 'gripper' but commands nothing.
    """
    for stmt in node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                fname = (sub.func.attr if isinstance(sub.func, ast.Attribute)
                         else getattr(sub.func, "id", None))
                if fname in MOTION_CALL_ATTRS:
                    return fname
    return None


def audit_file(path: str) -> list[dict]:
    if TEST_FILE.search(path.replace(os.sep, "/")):
        return []
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as exc:
        return [{"file": path, "function": "<module>", "line": 0,
                 "kind": "unreadable", "detail": str(exc)}]
    return audit_source(src, path)


def audit_source(src: str, path: str) -> list[dict]:
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as exc:
        return [{"file": path, "function": "<module>", "line": exc.lineno or 0,
                 "kind": "parse_error", "detail": str(exc)}]

    in_always = any(os.path.normpath(path).startswith(d + os.sep) or
                    os.path.dirname(os.path.normpath(path)) == d
                    for d in ALWAYS_AUDIT_DIRS)
    findings = []
    # Enclosing function name for each Try node, for reporting.
    owner: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Try):
                    owner.setdefault(id(node), fn.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        fn_name = owner.get(id(node), "<module>")
        called = _calls_motion(node)

        # Audit a try block if it either sits in a motion-named function (or under
        # hardware/), or actually calls a motion method from anywhere.
        if not (called or in_always or MOTION_NAME.search(fn_name)):
            continue

        for handler in node.handlers:
            if _handler_reraises(handler):
                continue
            exc_name = (ast.unparse(handler.type) if handler.type else "bare except")
            key = (path, fn_name, handler.lineno)
            why = f" wrapping {called}()" if called else ""

            for stmt in ast.walk(handler):
                if isinstance(stmt, ast.Return):
                    shape = _is_success_value(stmt.value)
                    if shape:
                        findings.append({
                            "file": path, "function": fn_name, "line": stmt.lineno,
                            "kind": "returns_success_on_exception",
                            "detail": f"`except {exc_name}`{why} returns {shape}",
                            "allowlisted": ALLOWLIST.get(key),
                        })
                        break
            else:
                # Swallowing only matters when a motion call is what was swallowed.
                if called and _handler_is_bare_pass(handler):
                    findings.append({
                        "file": path, "function": fn_name, "line": handler.lineno,
                        "kind": "swallows_motion_failure",
                        "detail": f"`except {exc_name}: pass`{why} with no re-raise",
                        "allowlisted": ALLOWLIST.get(key),
                    })
    return findings


def collect() -> list[dict]:
    seen, out = set(), []
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x not in SKIP_PARTS]
            if d == "." and os.path.normpath(root) != ".":
                continue  # repo root: top level only, subdirs handled explicitly
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                p = os.path.normpath(os.path.join(root, f))
                if p in seen:
                    continue
                seen.add(p)
                out.extend(audit_file(p))
    return out


# Snippets the detector must classify correctly. A clean audit only means something
# if the detector can still catch the bug — an always-empty check is worse than none,
# because it reads as evidence of safety.
SELF_TEST_CASES = [
    # (label, source, must_flag)
    ("the original real_arm rail bug", '''
def set_rail_position(self, position_mm, speed_mm_s=50.0, wait=True):
    try:
        return self.arm.set_linear_track_pos(position_mm, speed=speed_mm_s, wait=wait)
    except AttributeError:
        print("unavailable")
        self._rail_pos_mm = position_mm
        return 0
''', True),
    ("dispatcher swallowing a motion failure", '''
def dispatch(self, cmd):
    try:
        self.arm.set_position(**cmd)
    except Exception:
        pass
''', True),
    ("success tuple returned on exception", '''
def move_to(self, x, y, z):
    try:
        return self.arm.set_position(x=x, y=y, z=z)
    except Exception:
        return 0, None
''', True),
    ("correctly returns failure code", '''
def nudge_body(self, name):
    try:
        return self.model.body(name).id
    except (KeyError, ValueError):
        print("not in scene")
        return 1
''', False),
    ("re-raises after logging", '''
def set_rail_position(self, pos):
    try:
        return self.arm.set_linear_motor_pos(pos)
    except Exception as exc:
        print(exc)
        raise
''', False),
    ("cosmetic asset lookup, not motion", '''
def _resolve(self):
    for n in ("gripper_geom", "finger_left_geom"):
        try:
            self._gids.append(self.model.geom(n).id)
        except Exception:
            pass
''', False),
]


def run_self_test() -> int:
    failures = 0
    for label, src, must_flag in SELF_TEST_CASES:
        found = [f for f in audit_source(src, "selftest.py") if not f.get("allowlisted")]
        ok = bool(found) == must_flag
        verb = "flags" if must_flag else "ignores"
        print(f"  {'PASS' if ok else 'FAIL'}  {verb}: {label}"
              + ("" if ok else f"  (got {len(found)} finding(s))"))
        failures += not ok
    print(f"\n  {len(SELF_TEST_CASES) - failures} passed, {failures} failed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any non-allowlisted finding remains")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the detector still catches the bug it looks for")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    findings = collect()
    live = [f for f in findings if not f.get("allowlisted")]
    waived = [f for f in findings if f.get("allowlisted")]

    if args.json:
        json.dump({"summary": {"findings": len(live), "allowlisted": len(waived)},
                   "findings": findings}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if not live:
            print("  no motion command returns success on an exception path")
        for f in live:
            print(f"  {f['file']}:{f['line']}  {f['function']}()")
            print(f"      {f['kind']}: {f['detail']}")
        for f in waived:
            print(f"  [waived] {f['file']}:{f['line']} {f['function']}() — {f['allowlisted']}")
        print(f"\n  {len(live)} finding(s), {len(waived)} allowlisted")

    return 1 if (args.check and live) else 0


if __name__ == "__main__":
    sys.exit(main())
