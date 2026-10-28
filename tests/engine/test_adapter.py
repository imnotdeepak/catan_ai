"""Adapter behavior: seating, safe execution, offer validation, trade lifecycle (ENG-004)."""

import copy
import random
from typing import Any

import pytest
from catanatron import Color, RandomPlayer

from catan_ai.engine import (
    ActionKind,
    CanonicalAction,
    EngineAdapter,
    IllegalActionError,
    offer_trade,
)


def past_setup(seed: int = 1) -> EngineAdapter:
    adapter = EngineAdapter.new_game(seed)
    policy = random.Random(seed)
    while adapter.is_initial_build_phase:
        adapter.execute(policy.choice(adapter.legal_actions()))
    return adapter


def action_phase(seed: int = 1) -> EngineAdapter:
    """A game where the turn owner has rolled a non-7 and may trade and build."""
    for attempt in range(seed, seed + 50):
        adapter = past_setup(attempt)
        adapter.execute(CanonicalAction(ActionKind.ROLL, adapter.current_seat))
        if adapter.prompt == "PLAY_TURN":
            return adapter
    raise AssertionError("no non-7 roll found")


def set_hand(adapter: EngineAdapter, seat: int, hand: tuple[int, int, int, int, int]) -> None:
    for resource, amount in zip(("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"), hand, strict=True):
        adapter.referee_game.state.player_state[f"P{seat}_{resource}_IN_HAND"] = amount


def snapshot(adapter: EngineAdapter) -> Any:
    state = adapter.referee_game.state
    return (
        copy.deepcopy(state.player_state),
        len(state.action_records),
        state.current_player_index,
        state.current_prompt,
        tuple(state.resource_freqdeck),
    )


# ----- Seating


def test_seats_follow_the_given_order() -> None:
    order = [Color.ORANGE, Color.RED, Color.WHITE, Color.BLUE]
    adapter = EngineAdapter.new_game(3, [RandomPlayer(c) for c in order])
    assert list(adapter.referee_game.state.colors) == order
    assert adapter.current_seat == 0


def test_seat_rotation_keeps_the_same_board() -> None:
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

    def board(order: list[Any]) -> Any:
        game = EngineAdapter.new_game(11, [RandomPlayer(c) for c in order]).referee_game
        tiles = game.state.board.map.land_tiles
        return sorted((coord, t.resource, t.number) for coord, t in tiles.items())

    assert board(colors) == board(colors[1:] + colors[:1])


def test_same_seed_and_policy_replay_identically() -> None:
    def trajectory(seed: int) -> list[str]:
        adapter = EngineAdapter.new_game(seed)
        policy = random.Random(99)
        keys = []
        for _ in range(400):
            action = policy.choice(adapter.legal_actions())
            keys.append(action.key)
            adapter.execute(action)
        return keys

    assert trajectory(5) == trajectory(5)


# ----- Safe execution


def test_illegal_actions_raise_and_leave_state_unchanged() -> None:
    adapter = action_phase(2)
    seat = adapter.current_seat
    before = snapshot(adapter)
    illegal = [
        CanonicalAction(ActionKind.ROLL, seat),  # already rolled
        CanonicalAction(ActionKind.END_TURN, (seat + 1) % 4),  # not their decision
        CanonicalAction(ActionKind.BUILD_CITY, seat, (999,)),
        CanonicalAction(ActionKind.CONFIRM_TRADE, seat, (1,)),  # no pending trade
    ]
    for action in illegal:
        assert not adapter.is_legal(action)
        with pytest.raises(IllegalActionError):
            adapter.execute(action)
    assert snapshot(adapter) == before


def test_no_actions_are_legal_after_the_game_ends() -> None:
    adapter = action_phase(3)
    seat = adapter.current_seat
    adapter.referee_game.state.player_state[f"P{seat}_ACTUAL_VICTORY_POINTS"] = 10
    assert adapter.winner_seat == seat
    assert not adapter.is_legal(CanonicalAction(ActionKind.END_TURN, seat))


# ----- Domestic trade offers


@pytest.mark.parametrize(
    ("give", "receive", "reason"),
    [
        ((2, 0, 0, 0, 0), (0, 0, 0, 0, 1), "offerer holds only 1 wood"),
        ((0, 0, 0, 0, 0), (0, 0, 0, 0, 1), "gives nothing away"),
        ((1, 0, 0, 0, 0), (0, 0, 0, 0, 0), "asks for nothing"),
        ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0), "same resource both sides"),
        ((1, 0, 0, 0, -1), (0, 0, 0, 0, 1), "negative count"),
        ((1, 0, 0, 0), (0, 0, 0, 0, 1), "wrong vector length"),
    ],
)
def test_invalid_offers_are_rejected(
    give: tuple[int, ...], receive: tuple[int, ...], reason: str
) -> None:
    adapter = action_phase(4)
    seat = adapter.current_seat
    set_hand(adapter, seat, (1, 1, 0, 0, 0))
    before = snapshot(adapter)
    offer = CanonicalAction(ActionKind.OFFER_TRADE, seat, (give, receive))
    assert not adapter.is_legal(offer), reason
    with pytest.raises(IllegalActionError):
        adapter.execute(offer)
    assert snapshot(adapter) == before


def test_offer_before_rolling_is_rejected() -> None:
    adapter = past_setup(5)
    seat = adapter.current_seat
    set_hand(adapter, seat, (1, 0, 0, 0, 0))
    assert not adapter.is_legal(offer_trade(seat, (1, 0, 0, 0, 0), (0, 0, 0, 0, 1)))


def test_offers_are_not_listed_among_legal_actions() -> None:
    adapter = action_phase(6)
    assert all(a.kind != ActionKind.OFFER_TRADE for a in adapter.legal_actions())


def test_full_trade_lifecycle_through_the_adapter() -> None:
    adapter = action_phase(7)
    offerer = adapter.current_seat
    partner = (offerer + 1) % 4
    for seat in range(4):
        set_hand(adapter, seat, (0, 1, 0, 0, 0))  # nobody holds ore...
    set_hand(adapter, offerer, (2, 0, 0, 0, 0))
    set_hand(adapter, partner, (0, 0, 0, 0, 1))  # ...except the partner
    adapter.execute(offer_trade(offerer, (2, 0, 0, 0, 0), (0, 0, 0, 0, 1)))

    responders = []
    while adapter.prompt == "DECIDE_TRADE":
        seat = adapter.current_seat
        responders.append(seat)
        legal = {a.kind for a in adapter.legal_actions()}
        if seat == partner:
            assert ActionKind.ACCEPT_TRADE in legal
            adapter.execute(CanonicalAction(ActionKind.ACCEPT_TRADE, seat))
        else:
            assert ActionKind.ACCEPT_TRADE not in legal  # cannot pay 1 ore
            adapter.execute(CanonicalAction(ActionKind.REJECT_TRADE, seat))

    assert sorted(responders) == sorted(s for s in range(4) if s != offerer)
    assert adapter.prompt == "DECIDE_ACCEPTEES"
    assert set(adapter.legal_actions()) == {
        CanonicalAction(ActionKind.CANCEL_TRADE, offerer),
        CanonicalAction(ActionKind.CONFIRM_TRADE, offerer, (partner,)),
    }
    adapter.execute(CanonicalAction(ActionKind.CONFIRM_TRADE, offerer, (partner,)))

    state = adapter.referee_game.state
    assert state.player_state[f"P{offerer}_ORE_IN_HAND"] == 1
    assert state.player_state[f"P{offerer}_WOOD_IN_HAND"] == 0
    assert state.player_state[f"P{partner}_WOOD_IN_HAND"] == 2
    assert adapter.prompt == "PLAY_TURN"
    assert adapter.current_seat == offerer
