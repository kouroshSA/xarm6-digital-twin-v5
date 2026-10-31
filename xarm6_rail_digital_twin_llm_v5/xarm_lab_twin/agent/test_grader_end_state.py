"""The grader judges where things END, in the scene's own words.

Live runs of "open the incubator door, put the cucumber and the pastry in,
close the door; same for the clock and the cabinet" did everything and were
graded failures, three different ways: the door the task closes again was
still expected open; "clock moved" was asked for beside "clock in cabinet";
and a closed door ("incubator Door closed") is not a fact physical_outcome
reports at all. Pure Python: no sim, no API.

    python -m agent.test_grader_end_state
"""
from __future__ import annotations

import sys
import traceback

from agent.dynamic_grader import _drop_reclosed
from agent.outcome_checker import _matches_with_swap

TASK = ("open the incubator door, then grab the cucumber and put it in the incubator, then "
        "grab the pastry and put it in the incubator, then close the incubator door. After "
        "that, open the cabinet door, then get the clock and put it in the cabinet, then "
        "close the cabinet door and return to the home position")
# the intake appends scene facts; their "open or close" must not count
FACTS = (" Measured from the scene: incubator: it has parts operate_part can open or close "
         "by their handles: incubator.Door; cabinet: open or close cabinet.door_01")
DONE = ("cucumber in incubator; cupcake in incubator; clock in cabinet; "
        "bottle farther from clock")


def test_a_reclosed_door_is_not_expected_open():
    spec = _drop_reclosed(TASK + FACTS, ("all", [
        "incubator Door open", "cucumber in incubator", "cupcake in incubator",
        "incubator Door open", "cabinet door_01 open", "clock in cabinet"]))
    assert spec == ("all", ["cucumber in incubator", "cupcake in incubator",
                            "clock in cabinet"]), spec


def test_a_door_left_open_is_still_expected_open():
    spec = _drop_reclosed("open the incubator door" + FACTS, ("all", ["incubator Door open"]))
    assert spec == ("all", ["incubator Door open"]), spec


def test_closed_is_written_as_not_open():
    spec = _drop_reclosed(TASK, ("all", ["cucumber in incubator", "incubator Door closed"]))
    assert spec == ("all", ["cucumber in incubator", "NOT incubator Door open"]), spec
    assert _matches_with_swap(DONE, "NOT incubator Door open")
    assert not _matches_with_swap(DONE + "; incubator Door open 6 deg", "NOT incubator Door open")


def test_moved_beside_in_is_dropped_and_bare_moved_matches_in():
    spec = _drop_reclosed(TASK, ("all", ["clock in cabinet", "clock moved"]))
    assert spec == ("all", ["clock in cabinet"]), spec
    assert _matches_with_swap(DONE, "clock moved")            # it went INTO something
    assert not _matches_with_swap(DONE, "keyboard moved")     # it did not


def test_closer_to_is_met_by_in():
    assert _matches_with_swap(DONE, "clock closer to cabinet")
    assert not _matches_with_swap(DONE, "clock closer to incubator")


TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
