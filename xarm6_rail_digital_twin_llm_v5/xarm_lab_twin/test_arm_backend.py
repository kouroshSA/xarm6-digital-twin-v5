"""Parity test for the ArmBackend contract (B2).

Every contract method must be either implemented or explicitly declared
unsupported on both backends. A method that is merely *absent* is the failure
this guards: the shared stack would call it and die with AttributeError partway
through a plan, after part of that plan had already executed on a real arm.

Hardware-free — the classes are inspected, never instantiated, so no arm, no
controller container and no MuJoCo context are needed.

    python -m test_arm_backend
"""
from __future__ import annotations

import sys

from arm_backend import (ARM_BACKEND_METHODS, backend_report, missing_methods,
                         render_parity_table, unsupported_reason)


def _backends():
    from hardware.real_arm import RealXArmAPI
    from sim.mujoco_env import SimXArmAPI
    return [SimXArmAPI, RealXArmAPI]


def test_no_missing_methods():
    bad = []
    for cls in _backends():
        for name in missing_methods(cls):
            bad.append(f"{cls.__name__}.{name}")
    if bad:
        return ("FAIL  contract methods neither implemented nor declared "
                f"unsupported: {bad}")
    return f"PASS  both backends cover all {len(ARM_BACKEND_METHODS)} contract methods"


def test_unsupported_declarations_have_reasons():
    """An unsupported method must say why. 'Not implemented' with no reason sends
    the next reader to the source to work out whether it is a gap or a decision."""
    thin = []
    for cls in _backends():
        for name, (status, detail) in backend_report(cls).items():
            if status == "unsupported" and len(detail.strip()) < 25:
                thin.append(f"{cls.__name__}.{name}: {detail!r}")
    if thin:
        return f"FAIL  unsupported declarations without a useful reason: {thin}"
    return "PASS  every unsupported declaration carries a reason"


def test_unsupported_methods_raise():
    """They must raise, not return a benign value.

    A scene manipulation that quietly returns 0 on hardware would let a plan
    report success while the bench was untouched.
    """
    for cls in _backends():
        for name, (status, _) in backend_report(cls).items():
            if status != "unsupported":
                continue
            fn = getattr(cls, name)
            # Call unbound with a dummy self; the raiser never touches it.
            try:
                fn(object())
            except NotImplementedError:
                continue
            except Exception as exc:  # noqa: BLE001
                return (f"FAIL  {cls.__name__}.{name} raised "
                        f"{type(exc).__name__}, expected NotImplementedError")
            return f"FAIL  {cls.__name__}.{name} did not raise"
    return "PASS  every unsupported method raises NotImplementedError"


def test_something_is_actually_shared():
    """Guard against a contract that is vacuously satisfied by declaring
    everything unsupported on one side."""
    reports = {c.__name__: backend_report(c) for c in _backends()}
    shared = [n for n in ARM_BACKEND_METHODS
              if all(r[n][0] == "implemented" for r in reports.values())]
    if len(shared) < 10:
        return (f"FAIL  only {len(shared)} methods implemented on both backends; "
                f"the contract has stopped meaning anything")
    return f"PASS  {len(shared)} methods genuinely implemented on both backends"


TESTS = [test_no_missing_methods,
         test_unsupported_declarations_have_reasons,
         test_unsupported_methods_raise,
         test_something_is_actually_shared]


def main() -> int:
    failures = 0
    for fn in TESTS:
        try:
            line = fn()
        except Exception as exc:  # noqa: BLE001
            line = f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}"
        if line.startswith("FAIL"):
            failures += 1
        print("  " + line)
    print()
    print(render_parity_table(_backends()))
    print(f"\n  {len(TESTS) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
