"""Confirmed deviations of the pinned engine from the 2025 (6th edition) rules.

Each test asserts the *correct* rule and is marked strict xfail. When the engine
or adapter is patched, the test starts passing, strict xfail turns that into a
failure, and the marker and docs/rules-matrix.md row must be updated together.
Row IDs (D-xx) refer to docs/rules-matrix.md.
"""

from itertools import pairwise
from typing import Any

import pytest
from catanatron import Action, ActionType, Color
from catanatron.apply_action import yield_resources
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import WOOD
from catanatron.state_functions import get_player_freqdeck

from tests.rules.helpers import force_turn, new_game, past_setup, set_hand


def deviation(row: str, reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"{row}: {reason}")


@deviation("D-01", "Game.copy() shares the RNG object with the original")
def test_d01_copy_does_not_perturb_original_rng() -> None:
    game = new_game(7)
    before = game.random.getstate()
    game.copy().play()
    assert game.random.getstate() == before


@deviation("D-02", "win is checked for every player after every action")
def test_d02_only_active_player_can_win() -> None:
    game = past_setup(8)
    state = game.state
    inactive_seat = (state.current_turn_index + 1) % 4
    state.player_state[f"P{inactive_seat}_ACTUAL_VICTORY_POINTS"] = 10
    assert game.winning_color() is None


@deviation("D-03", "no single-affected-player exception on bank shortage")
def test_d03_bank_shortage_single_player_gets_remainder() -> None:
    board = new_game(4).state.board
    tile = next(t for t in board.map.land_tiles.values() if t.resource == WOOD)
    node = next(iter(tile.nodes.values()))
    board.build_settlement(Color.RED, node, initial_build_phase=True)
    board.build_city(Color.RED, node)
    payout, _ = yield_resources(board, [1, 19, 19, 19, 19], tile.number)
    assert payout[Color.RED] == [1, 0, 0, 0, 0]


def _route_cut_geometry(land: set[int]) -> tuple[list[int], int, int]:
    """A 6-road path and a 2-road approach that lets an opponent settle its midpoint."""

    def paths(start: int, length: int) -> Any:
        stack = [[start]]
        while stack:
            path = stack.pop()
            if len(path) == length:
                yield path
                continue
            stack.extend(
                [*path, n] for n in STATIC_GRAPH.neighbors(path[-1]) if n in land and n not in path
            )

    for start in sorted(land):
        for path in paths(start, 7):
            for m in STATIC_GRAPH.neighbors(path[3]):
                if m not in land or m in path:
                    continue
                for k in STATIC_GRAPH.neighbors(m):
                    clear_of_red = k not in STATIC_GRAPH.neighbors(path[0]) and k != path[0]
                    if (
                        k in land
                        and k not in path
                        and clear_of_red
                        and not set(STATIC_GRAPH.neighbors(k)) & set(path)
                    ):
                        return path, m, k
    raise AssertionError("no geometry found")


@deviation("D-04", "after a cut, Longest Route goes to max length even below 5")
def test_d04_longest_route_returns_to_supply_when_no_route_of_five() -> None:
    board = new_game(5).state.board
    path, m, k = _route_cut_geometry(set(board.map.land_nodes))
    board.build_settlement(Color.RED, path[0], initial_build_phase=True)
    for a, b in pairwise(path):
        board.build_road(Color.RED, tuple(sorted((a, b))))
    assert (board.road_color, board.road_length) == (Color.RED, 6)
    board.build_settlement(Color.BLUE, k, initial_build_phase=True)
    board.build_road(Color.BLUE, tuple(sorted((k, m))))
    board.build_road(Color.BLUE, tuple(sorted((m, path[3]))))
    board.build_settlement(Color.BLUE, path[3], initial_build_phase=False)
    assert board.road_color is None


@deviation("D-06", "response loop does not skip the offering player")
def test_d06_offerer_is_not_asked_to_respond_to_own_offer() -> None:
    game = past_setup(2)
    force_turn(game, 2)
    set_hand(game, 2, (1, 0, 0, 0, 0))
    state = game.state
    offerer = state.colors[2]
    game.execute(Action(offerer, ActionType.OFFER_TRADE, (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)))
    asked = []
    while state.current_prompt.name == "DECIDE_TRADE":
        asked.append(state.current_color())
        game.execute(Action(state.current_color(), ActionType.REJECT_TRADE, state.current_trade))
    assert offerer not in asked


@deviation("D-07", "offerer affordability is checked neither at offer nor at confirm")
def test_d07_offerer_cannot_trade_cards_it_does_not_hold() -> None:
    game = past_setup(3)
    force_turn(game, 0)
    set_hand(game, 0, (0, 0, 0, 0, 0))
    set_hand(game, 1, (0, 0, 0, 0, 1))
    state = game.state
    offerer, acceptor = state.colors[0], state.colors[1]
    value = (3, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    try:
        game.execute(Action(offerer, ActionType.OFFER_TRADE, value))
        while state.current_prompt.name == "DECIDE_TRADE":
            who = state.current_color()
            kind = ActionType.ACCEPT_TRADE if who == acceptor else ActionType.REJECT_TRADE
            game.execute(Action(who, kind, state.current_trade))
        game.execute(Action(offerer, ActionType.CONFIRM_TRADE, (*value, acceptor)))
    except ValueError:
        return  # rejected somewhere in the lifecycle: correct behavior
    assert min(get_player_freqdeck(state, offerer)) >= 0
