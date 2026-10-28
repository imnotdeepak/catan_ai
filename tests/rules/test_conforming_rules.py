"""Rules the pinned engine implements as written in the 2025 (6th edition) rulebook.

Row IDs (R-xx) refer to docs/rules-matrix.md.
"""

from collections import Counter

import pytest
from catanatron import Action, ActionType
from catanatron.apply_action import apply_action
from catanatron.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    starting_devcard_bank,
    starting_resource_bank,
)
from catanatron.models.enums import ActionRecord
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.state_functions import (
    buy_dev_card,
    get_player_freqdeck,
    player_can_play_dev,
    player_clean_turn,
)

from tests.rules.helpers import force_turn, new_game, past_setup, set_hand


def test_r01_component_supply() -> None:
    assert starting_resource_bank() == [19, 19, 19, 19, 19]
    assert Counter(starting_devcard_bank()) == {
        "KNIGHT": 14,
        "VICTORY_POINT": 5,
        "ROAD_BUILDING": 2,
        "YEAR_OF_PLENTY": 2,
        "MONOPOLY": 2,
    }
    assert PLAYER_INITIAL_STATE["ROADS_AVAILABLE"] == 15
    assert PLAYER_INITIAL_STATE["SETTLEMENTS_AVAILABLE"] == 5
    assert PLAYER_INITIAL_STATE["CITIES_AVAILABLE"] == 4


def test_r02_building_costs() -> None:
    assert ROAD_COST_FREQDECK == [1, 1, 0, 0, 0]
    assert SETTLEMENT_COST_FREQDECK == [1, 1, 1, 1, 0]
    assert CITY_COST_FREQDECK == [0, 0, 0, 2, 3]
    assert DEVELOPMENT_CARD_COST_FREQDECK == [0, 0, 1, 1, 1]


def test_r03_base_map_composition() -> None:
    tiles = new_game(3).state.board.map.land_tiles.values()
    assert Counter(t.resource for t in tiles) == {
        "WOOD": 4,
        "BRICK": 3,
        "SHEEP": 4,
        "WHEAT": 4,
        "ORE": 3,
        None: 1,
    }
    numbers = sorted(t.number for t in tiles if t.resource is not None)
    assert numbers == [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_r04_setup_snake_order_and_first_player(seed: int) -> None:
    game = past_setup(seed)
    state = game.state
    placements = [
        record.action.color
        for record in state.action_records
        if record.action.action_type == ActionType.BUILD_SETTLEMENT
    ]
    seats = [state.colors.index(color) for color in placements]
    assert seats == [0, 1, 2, 3, 3, 2, 1, 0]
    assert state.current_color() == state.colors[0]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_r05_starting_resources_come_from_second_settlement(seed: int) -> None:
    game = past_setup(seed)
    state = game.state
    for color in state.colors:
        second = state.buildings_by_color[color]["SETTLEMENT"][1]
        expected = Counter(
            tile.resource
            for tile in state.board.map.adjacent_tiles[second]
            if tile.resource is not None
        )
        hand = dict(
            zip(
                ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
                get_player_freqdeck(state, color),
                strict=True,
            )
        )
        assert {r: n for r, n in hand.items() if n} == dict(expected)
    total_held = sum(sum(get_player_freqdeck(state, c)) for c in state.colors)
    assert total_held + sum(state.resource_freqdeck) == 95


def test_r07_seven_discards_half_rounded_down_above_seven() -> None:
    game = past_setup(4)
    force_turn(game, 0, has_rolled=False)
    set_hand(game, 1, (3, 3, 3, 0, 0))  # 9 cards -> discard 4
    set_hand(game, 2, (2, 2, 2, 1, 0))  # 7 cards -> no discard
    state = game.state
    roll = Action(state.colors[0], ActionType.ROLL, None)
    apply_action(state, roll, ActionRecord(roll, (3, 4)))
    assert state.discard_counts[1] == 4
    assert state.discard_counts[2] == 0


def test_r10_dev_card_not_playable_on_turn_bought() -> None:
    game = past_setup(5)
    state = game.state
    color = state.colors[0]
    set_hand(game, 0, (0, 0, 1, 1, 1))
    buy_dev_card(state, color, "KNIGHT")
    assert not player_can_play_dev(state, color, "KNIGHT")
    player_clean_turn(state, color)  # end of the buying turn
    assert player_can_play_dev(state, color, "KNIGHT")


def test_r11_dev_card_playable_before_rolling() -> None:
    game = past_setup(6)
    state = game.state
    color = state.colors[0]
    state.player_state["P0_KNIGHT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    force_turn(game, 0, has_rolled=False)
    types = {a.action_type for a in game.playable_actions}
    assert ActionType.PLAY_KNIGHT_CARD in types
    assert ActionType.ROLL in types
    assert color == state.current_color()


def test_r15_trades_require_roll_and_no_inactive_player_trades() -> None:
    game = past_setup(7)
    force_turn(game, 0, has_rolled=False)
    set_hand(game, 0, (4, 0, 0, 0, 0))
    types = {a.action_type for a in game.playable_actions}
    assert ActionType.MARITIME_TRADE not in types
    force_turn(game, 0, has_rolled=True)
    types = {a.action_type for a in game.playable_actions}
    assert ActionType.MARITIME_TRADE in types
    # Only the active player's seat ever generates trade/build actions.
    assert {a.color for a in game.playable_actions} == {game.state.colors[0]}
