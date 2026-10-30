"""What the episode loop learns from a refused move with an object in hand.

Run: ``python -m agent.test_failure_analysis``  (no sim, no model)

The refusal "the carried X would hit Y" means two different things. Refused on
the way DOWN to the destination, the spot is occupied; refused on a traverse,
the path is blocked. The analyser used to answer both with "lift z to >= ...",
and on a live run the planner duly released the cucumber 70 mm above the bench
beside the cupcake: three "successes" that were drops. So the checks are on
the constraint text the next episode is actually given (the consumer), not on
which branch ran.
"""
from __future__ import annotations

import sys
import traceback

from agent.episode_loop import EpisodeContext, analyse_command_failure

REFUSAL = ("move refused: path blocked at 92% along the path: the carried "
           "cucumber would hit cupcake/cupcake_g01_collision0")
FAILED = {"action": "move_to", "result": 2}


def _move(x, y, z):
    return {"action": "move_to", "params": {"x": x, "y": y, "z": z, "roll": 180,
                                            "pitch": 0, "yaw": 0}}


class _Obj:
    position_xyz_m = [0.12, -0.25, 0.789]


class _Registry:
    def find(self, name):
        return _Obj() if name == "cupcake" else None


def test_refused_descent_says_the_spot_is_occupied():
    c = analyse_command_failure(FAILED, _move(51, -264, 865), registry=_Registry(),
                                refusal=REFUSAL, prev_command=_move(51, -264, 900))
    assert "OCCUPIED" in c and "cupcake at (120, -250)" in c, c
    assert "Lift z" not in c and "not fix this" in c, c


def test_refused_traverse_says_go_higher():
    c = analyse_command_failure(FAILED, _move(51, -264, 900), registry=_Registry(),
                                refusal=REFUSAL, prev_command=_move(-99, -264, 900))
    assert "carry it higher" in c and "OCCUPIED" not in c, c


def test_lab_geom_names_resolve_to_the_object():
    lab = "move refused: the carried green_cube would hit red_cube_front_geom"
    c = analyse_command_failure(FAILED, _move(0, -250, 800), refusal=lab,
                                prev_command=_move(0, -250, 900))
    assert "into red_cube_front:" in c, c


def test_without_a_refusal_the_old_advice_stands():
    c = analyse_command_failure(FAILED, _move(51, -264, 865))
    assert "Lift z to >= 915" in c, c


def test_the_constraint_reaches_the_next_prompt():
    ctx = EpisodeContext(task="move the cucumber")
    ctx.add_constraint(analyse_command_failure(
        FAILED, _move(51, -264, 865), registry=_Registry(), refusal=REFUSAL,
        prev_command=_move(51, -264, 900)))
    block = ctx.constraints_block()
    assert "OCCUPIED" in block and "dropped object is not placed" in block, block


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
