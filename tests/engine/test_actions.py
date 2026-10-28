"""Exhaustive translation tests: every legal engine action round-trips (ENG-004)."""

import random
from collections import Counter

import pytest
from catanatron import Action, ActionType, Color

from catan_ai.engine import ActionKind, EngineAdapter
from catan_ai.engine.actions import to_canonical, to_engine

GAMES = 12
MAX_DECISIONS = 3000

# Random play never proposes domestic trades; those kinds are covered in test_adapter.
TRADE_KINDS = {
    ActionKind.OFFER_TRADE,
    ActionKind.ACCEPT_TRADE,
    ActionKind.REJECT_TRADE,
    ActionKind.CONFIRM_TRADE,
    ActionKind.CANCEL_TRADE,
}


@pytest.fixture(scope="module")
def kind_coverage() -> Counter[ActionKind]:
    """Play seeded random games, checking every legal action at every decision."""
    covered: Counter[ActionKind] = Counter()
    for seed in range(GAMES):
        adapter = EngineAdapter.new_game(seed)
        policy = random.Random(seed)
        game = adapter.referee_game
        for _ in range(MAX_DECISIONS):
            if adapter.is_over:
                break
            state = game.state
            legal = adapter.legal_actions()
            assert legal, "every non-terminal decision has a legal action"
            assert len({a.key for a in legal}) == len(legal), "canonical keys are unique"

            for engine_action in game.playable_actions:
                canonical = to_canonical(engine_action, state.colors)
                assert to_engine(canonical, state.colors, state.current_trade) == engine_action
                assert canonical.seat == adapter.current_seat
                assert adapter.is_legal(canonical)
                covered[canonical.kind] += 1

            adapter.execute(policy.choice(legal))
    return covered


def test_every_non_trade_kind_is_exercised(kind_coverage: Counter[ActionKind]) -> None:
    missing = set(ActionKind) - TRADE_KINDS - set(kind_coverage)
    assert not missing, f"kinds never seen as legal: {sorted(missing)}"


def test_canonical_actions_hide_outcomes() -> None:
    colors = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)
    # Engine records carry the drawn card / stolen resource only in their results;
    # the requested actions translate without them.
    buy = to_canonical(Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None), colors)
    assert buy.params == ()
    rob = to_canonical(Action(Color.RED, ActionType.MOVE_ROBBER, ((0, 0, 0), Color.BLUE)), colors)
    assert rob.params == ((0, 0, 0), 1)
