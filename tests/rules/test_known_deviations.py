"""Deviations of upstream Catanatron from the 2025 (6th edition) rules.

Each test asserts the *correct* rule. Deviations fixed in our fork
(imnotdeepak/catanatron, branch catan-ai-fixes) are plain regression tests.
Open deviations are marked strict xfail: once patched, the test passes, strict
xfail turns that into a failure, and the marker and docs/rules-matrix.md row
must be updated together. Row IDs (D-xx) refer to docs/rules-matrix.md.
"""

from itertools import pairwise
from typing import Any

import pytest
from catanatron import Action, ActionType, Color
from catanatron.apply_action import yield_resources
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import WOOD
from catanatron.state_functions import get_player_freqdeck

from catan_ai.engine import EngineAdapter
from tests.rules.helpers import force_turn, new_game, past_setup, set_hand


def deviation(row: str, reason: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, reason=f"{row}: {reason}")


def test_d01_branches_do_not_perturb_original_rng() -> None:
    """Fixed in the adapter (EngineAdapter.branch), not in the engine; see test_branching."""
    adapter = EngineAdapter.new_game(7)
    before = adapter.referee_game.random.getstate()
    branch = adapter.branch(1)
    while not branch.is_over:
        branch.execute(branch.legal_actions()[0])
    assert adapter.referee_game.random.getstate() == before


def test_d02_only_active_player_can_win() -> None:
    game = past_setup(8)
    state = game.state
    inactive_seat = (state.current_turn_index + 1) % 4
    state.player_state[f"P{inactive_seat}_ACTUAL_VICTORY_POINTS"] = 10
    assert game.winning_color() is None


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


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ({Color.RED: 4, Color.BLUE: 6, Color.WHITE: 6}, (None, 6)),  # tie among others: supply
        ({Color.RED: 6, Color.BLUE: 6}, (Color.RED, 6)),  # holder tied: keeps it (Q-01)
        ({Color.RED: 4, Color.BLUE: 6}, (Color.BLUE, 6)),  # single longest takes it
    ],
)
def test_d05_longest_route_ties_after_cut(
    lengths: dict[Any, int], expected: tuple[Any, int]
) -> None:
    board = new_game(5).state.board
    board.road_color, board.road_length = Color.RED, 6
    board.road_lengths.update(lengths)
    assert board._holder_after_cut() == expected
