# teleop_sm/test_device.py
"""Unit tests for teleop_sm/device.py and buttons.py.

    python -m teleop_sm.test_device       # plain-python, prints PASS/FAIL
    pytest teleop_sm/test_device.py

MUST PASS WITH NO HARDWARE ATTACHED and with no spacenavd running: every test
drives `SpaceMouseDevice` through its `fake_events` path, which never imports
spnav. Follows vr/test_transforms.py conventions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from teleop_sm import config
from teleop_sm.buttons import Action, BUTTON_MAPS, describe, map_for
from teleop_sm.device import (SpaceMouseDevice, _FakeButtonEvent,
                              _FakeMotionEvent)


def _dev(events):
    """Device with a canned event stream, already ingested. No thread, no daemon."""
    d = SpaceMouseDevice(fake_events=events)
    for ev in events:
        d._ingest(ev, _FakeMotionEvent, _FakeButtonEvent)
    return d


# ---------------------------------------------------------------------------
# normalisation and deadzone
# ---------------------------------------------------------------------------
def test_normalisation_scales_by_max_value():
    half = config.MAX_VALUE / 2.0
    d = _dev([_FakeMotionEvent((half, 0, 0), (0, 0, 0))])
    s = d.state()
    assert abs(s[0] - 0.5) < 1e-9, f"expected 0.5, got {s[0]}"
    assert np.allclose(s[1:], 0.0)


def test_overshoot_is_clipped():
    """A hard shove exceeds MAX_VALUE. It must not exceed the speed cap.

    The receiver multiplies this by mm/s, so an un-clipped 1.4 would silently
    drive 40% faster than the operator asked for.
    """
    d = _dev([_FakeMotionEvent((config.MAX_VALUE * 1.5, 0, 0), (0, 0, 0))])
    assert d.state()[0] == 1.0, d.state()[0]

    d = _dev([_FakeMotionEvent((-config.MAX_VALUE * 3.0, 0, 0), (0, 0, 0))])
    assert d.state()[0] == -1.0, d.state()[0]


def test_deadzone_zeroes_small_deflection():
    """Below the deadzone must be exactly 0.0, not merely small.

    An integrating jog turns any standing offset into continuous drift, so
    "nearly zero" is not good enough -- the arm would walk while the
    operator's hand rests on the cap.
    """
    tiny = config.MAX_VALUE * (config.DEADZONE * 0.5)
    d = _dev([_FakeMotionEvent((tiny, -tiny, tiny), (tiny, tiny, -tiny))])
    s = d.state()
    assert np.all(s == 0.0), f"deadzone leaked: {s}"


def test_deadzone_passes_real_deflection():
    big = config.MAX_VALUE * (config.DEADZONE * 4.0)
    d = _dev([_FakeMotionEvent((big, 0, 0), (0, 0, 0))])
    assert d.state()[0] > 0.0


def test_all_six_axes_are_exposed():
    """The vendor teleop zeroes Z and drops rotation entirely. We must not."""
    m = config.MAX_VALUE
    d = _dev([_FakeMotionEvent((m, m, m), (m, m, m))])
    s = d.state()
    assert np.allclose(s, 1.0), f"some axis was dropped or zeroed: {s}"
    assert s[2] != 0.0, "tz was zeroed -- that is the vendor bug we exist to avoid"
    assert np.any(s[3:] != 0.0), "rotation axes were dropped"


# ---------------------------------------------------------------------------
# axis permutation
# ---------------------------------------------------------------------------
def test_permutation_applies_to_both_triples():
    """Translation and rotation are permuted separately by the same matrix."""
    m = config.MAX_VALUE
    d = _dev([_FakeMotionEvent((m, 0, 0), (0, m, 0))])
    got = d.state_twin()

    expect = np.empty(6)
    expect[:3] = config.AXIS_PERMUTATION @ np.array([1.0, 0, 0])
    expect[3:] = config.AXIS_PERMUTATION @ np.array([0, 1.0, 0])
    expect *= config.AXIS_SIGNS
    assert np.allclose(got, expect), f"{got} != {expect}"


def test_permutation_is_orthonormal():
    """Whatever the final matrix turns out to be, it must be a rotation.

    A non-orthonormal permutation would scale or shear the jog: pushing
    diagonally would move faster than pushing along an axis, which feels
    broken and is hard to diagnose by hand.
    """
    m = config.AXIS_PERMUTATION
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-12), m @ m.T
    assert abs(abs(np.linalg.det(m)) - 1.0) < 1e-12, np.linalg.det(m)


def test_state_twin_preserves_magnitude():
    m = config.MAX_VALUE
    d = _dev([_FakeMotionEvent((m, m, 0), (0, 0, 0))])
    assert abs(np.linalg.norm(d.state_twin()[:3])
               - np.linalg.norm(d.state()[:3])) < 1e-12


# ---------------------------------------------------------------------------
# buttons
# ---------------------------------------------------------------------------
def test_button_edge_fires_once():
    d = _dev([_FakeButtonEvent(0, True)])
    assert d.button_edge(0) is True, "first read must see the edge"
    assert d.button_edge(0) is False, "edge must be consumed"


def test_button_repeat_press_is_not_a_second_edge():
    """spacenavd can repeat a press. A repeat must not fire the action twice
    -- on GRIPPER_TOGGLE that would open and immediately re-close."""
    d = _dev([_FakeButtonEvent(0, True), _FakeButtonEvent(0, True)])
    assert d.button_edge(0) is True
    assert d.button_edge(0) is False, "repeat press produced a spurious edge"


def test_button_held_is_level_not_edge():
    d = _dev([_FakeButtonEvent(1, True)])
    assert d.button_held(1) is True
    assert d.button_held(1) is True, "held must stay true while down"
    d._ingest(_FakeButtonEvent(1, False), _FakeMotionEvent, _FakeButtonEvent)
    assert d.button_held(1) is False


def test_release_then_press_is_a_new_edge():
    d = _dev([_FakeButtonEvent(0, True)])
    assert d.button_edge(0) is True
    for ev in (_FakeButtonEvent(0, False), _FakeButtonEvent(0, True)):
        d._ingest(ev, _FakeMotionEvent, _FakeButtonEvent)
    assert d.button_edge(0) is True, "a fresh press after release must re-fire"


def test_pending_edges_drains():
    d = _dev([_FakeButtonEvent(0, True), _FakeButtonEvent(3, True)])
    assert d.pending_edges() == {0, 3}
    assert d.pending_edges() == set(), "drain must clear"


# ---------------------------------------------------------------------------
# button maps
# ---------------------------------------------------------------------------
def test_known_models_have_maps():
    for model in ("compact", "pro"):
        assert map_for(model), model


def test_unknown_model_raises_with_the_options():
    try:
        map_for("spaceball")
    except ValueError as exc:
        assert "compact" in str(exc) and "pro" in str(exc), exc
    else:
        raise AssertionError("unknown model must raise")


def test_compact_covers_the_essential_actions():
    """With two buttons, gripper and mode must still be reachable; the rest
    fall back to the keyboard."""
    actions = set(BUTTON_MAPS["compact"].values())
    assert Action.GRIPPER_TOGGLE in actions
    assert Action.MODE_CYCLE in actions


def test_every_map_is_injective():
    """Two buttons raising the same action is almost certainly a typo."""
    for model, mapping in BUTTON_MAPS.items():
        vals = list(mapping.values())
        assert len(vals) == len(set(vals)), f"{model} maps a duplicate action: {vals}"


def test_describe_mentions_each_bound_button():
    text = describe("pro")
    for idx in BUTTON_MAPS["pro"]:
        assert f"button {idx}" in text, f"banner omits button {idx}:\n{text}"


# ---------------------------------------------------------------------------
# dump round-trip
# ---------------------------------------------------------------------------
def test_events_from_dump_round_trip(tmp_path=None):
    """A probe dump must rebuild into replayable events -- that is how a real
    hardware session becomes a fixture for these tests."""
    import json
    import tempfile

    from teleop_sm.device import events_from_dump

    # Deflection must clear the deadzone, or state() correctly reports 0.0 and
    # the round-trip assertion below would be testing the wrong thing.
    tx = int(config.MAX_VALUE * 0.5)
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(json.dumps({"t": 0.1, "type": "motion",
                             "translation": [tx, 0, 0],
                             "rotation": [0, 0, 0]}) + "\n")
        fh.write(json.dumps({"t": 0.2, "type": "button",
                             "bnum": 4, "press": True}) + "\n")
        path = fh.name

    evs = events_from_dump(path)
    os.unlink(path)

    assert len(evs) == 2, evs
    assert evs[0].translation == (tx, 0, 0)
    assert evs[1].bnum == 4 and evs[1].press is True

    d = _dev(evs)
    assert d.button_edge(4) is True
    assert abs(d.state()[0] - tx / config.MAX_VALUE) < 1e-12, d.state()[0]


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
