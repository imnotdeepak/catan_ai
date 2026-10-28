"""Environment checks: the package and the pinned engine import and run."""

from catanatron import Color, Game, RandomPlayer

import catan_ai


def test_package_imports() -> None:
    assert catan_ai.__version__


def test_pinned_engine_completes_a_four_player_game() -> None:
    players = [RandomPlayer(color) for color in (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)]
    game = Game(players, seed=42)
    game.play()
    assert len(game.state.colors) == 4
