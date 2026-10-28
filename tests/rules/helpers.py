"""Shared fixtures for rules tests: game construction and state forcing."""

from typing import Any

from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def new_game(seed: int = 1) -> Any:
    return Game([RandomPlayer(color) for color in COLORS], seed=seed)


def past_setup(seed: int = 1) -> Any:
    """A game advanced through the opening snake draft by random players."""
    game = new_game(seed)
    while game.state.is_initial_build_phase:
        game.play_tick()
    return game


def force_turn(game: Any, seat: int, has_rolled: bool = True) -> None:
    """Make `seat` the active player in its Action phase."""
    state = game.state
    state.current_turn_index = state.current_player_index = seat
    state.player_state[f"P{seat}_HAS_ROLLED"] = has_rolled
    game.playable_actions = generate_playable_actions(state)


def set_hand(game: Any, seat: int, hand: tuple[int, int, int, int, int]) -> None:
    for resource, amount in zip(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"), hand, strict=True):
        game.state.player_state[f"P{seat}_{resource}_IN_HAND"] = amount
