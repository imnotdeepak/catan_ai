"""The single interface between our code and the Catanatron engine (ENG-004).

Everything above this module works in seats and canonical actions. The engine
`Game` held here is the privileged referee state: it must never be handed to an
agent at a real decision (master plan §2.1). Agents receive observations from
the observation firewall instead (ENG-005).
"""

from collections.abc import Sequence
from typing import Any

from catanatron import Color, Game
from catanatron.game import TURNS_LIMIT, is_valid_action
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import SimplePlayer

from catan_ai.engine.actions import (
    ActionKind,
    CanonicalAction,
    Seat,
    to_canonical,
    to_engine,
)

SEAT_COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


class IllegalActionError(ValueError):
    """Raised when an action is not legal in the current state. State is unchanged."""


class EngineAdapter:
    def __init__(self, game: Any) -> None:
        self._game = game

    @classmethod
    def new_game(cls, seed: int, engine_players: Sequence[Any] | None = None) -> "EngineAdapter":
        """Start a four-player standard game with seats in the given order.

        Seat i is `engine_players[i]` (or a placeholder that never decides, when
        our own agents drive the game). Catanatron shuffles seating with its
        RNG; we override that so the arena controls seat rotation. The shuffle
        still consumes the same RNG draws, so a seed fixes the board regardless
        of who sits where.
        """
        players = (
            list(engine_players)
            if engine_players is not None
            else [SimplePlayer(color) for color in SEAT_COLORS]
        )
        if len(players) != 4:
            raise ValueError("Standard Catan here is exactly four players")

        game = Game(players, seed=seed)
        state = game.state
        state.players = players
        state.colors = tuple(player.color for player in players)
        state.color_to_index = {color: seat for seat, color in enumerate(state.colors)}
        game.playable_actions = generate_playable_actions(state)
        return cls(game)

    # ----- Privileged access (referee, arena, evaluation only)

    @property
    def referee_game(self) -> Any:
        """The full engine game, including hidden hands and deck order."""
        return self._game

    # ----- Status

    @property
    def seed(self) -> int:
        return int(self._game.seed)

    @property
    def current_seat(self) -> Seat:
        """Seat that must decide now (may differ from the turn owner, e.g. discards)."""
        return int(self._game.state.current_player_index)

    @property
    def turn_seat(self) -> Seat:
        return int(self._game.state.current_turn_index)

    @property
    def prompt(self) -> str:
        return str(self._game.state.current_prompt.name)

    @property
    def is_initial_build_phase(self) -> bool:
        return bool(self._game.state.is_initial_build_phase)

    @property
    def num_turns(self) -> int:
        return int(self._game.state.num_turns)

    @property
    def winner_seat(self) -> Seat | None:
        color = self._game.winning_color()
        return None if color is None else int(self._game.state.colors.index(color))

    @property
    def is_truncated(self) -> bool:
        return self.winner_seat is None and self.num_turns >= TURNS_LIMIT

    @property
    def is_over(self) -> bool:
        return self.winner_seat is not None or self.is_truncated

    # ----- Actions

    def legal_actions(self) -> tuple[CanonicalAction, ...]:
        """Every legal action for the deciding seat, except domestic trade offers.

        Offers form an effectively unbounded space; propose them with
        `catan_ai.engine.actions.offer_trade` and check them with `is_legal`.
        """
        colors = self._game.state.colors
        actions = (to_canonical(a, colors) for a in self._game.playable_actions)
        return tuple(sorted(actions, key=lambda a: a.key))

    def is_legal(self, action: CanonicalAction) -> bool:
        if self.is_over or action.seat != self.current_seat:
            return False
        try:
            engine_action = self._to_engine(action)
        except (ValueError, IndexError, TypeError):
            return False
        if action.kind == ActionKind.OFFER_TRADE and not _well_formed_offer(action):
            return False
        return bool(is_valid_action(self._game.playable_actions, self._game.state, engine_action))

    def execute(self, action: CanonicalAction) -> Any:
        """Apply a legal action, or raise IllegalActionError without changing state.

        Returns the engine's random outcome, if any (privileged): dice for ROLL,
        the card drawn for BUY_DEV_CARD, the resource stolen for MOVE_ROBBER.
        """
        if not self.is_legal(action):
            raise IllegalActionError(f"Illegal action: {action.key}")
        return self._game.execute(self._to_engine(action)).result

    def engine_player_decision(self) -> CanonicalAction:
        """Ask the seated Catanatron bot to decide (privileged: it sees the full game)."""
        game = self._game
        player = game.state.current_player()
        engine_action = player.decide(game, game.playable_actions)
        return to_canonical(engine_action, game.state.colors)

    def _to_engine(self, action: CanonicalAction) -> Any:
        state = self._game.state
        return to_engine(action, state.colors, state.current_trade)


def _well_formed_offer(action: CanonicalAction) -> bool:
    give, receive = action.params
    vectors = (give, receive)
    return all(
        len(v) == 5 and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in v)
        for v in vectors
    )
