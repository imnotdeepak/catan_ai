"""Hidden-information leak tests for the observation firewall (ENG-006, plan §2.1)."""

import random
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from catan_ai.engine import RESOURCES, ActionKind, EngineAdapter
from catan_ai.observations import (
    DEV_CARDS,
    EventType,
    PlayerObservation,
    Referee,
)

SEEDS = range(6)
CHECK_EVERY = 25
MAX_DECISIONS = 3000


def play(seed: int) -> Iterator[Referee]:
    """Play a random game through observations only, yielding at every decision."""
    referee = Referee.new_game(seed)
    policy = random.Random(1000 + seed)
    for _ in range(MAX_DECISIONS):
        if referee.adapter.is_over:
            break
        yield referee
        obs = referee.observe(referee.adapter.current_seat)
        referee.execute(policy.choice(obs.legal_actions))
    yield referee


@pytest.fixture(scope="module")
def finished_games() -> list[Referee]:
    games = []
    for seed in SEEDS:
        *_, last = play(seed)
        games.append(last)
    return games


# ----- Indistinguishability


def scramble_hidden_state(referee: Referee, viewer: int, rng: random.Random) -> Referee:
    """Copy the game and change every fact `viewer` cannot observe."""
    game = referee.adapter.referee_game.copy()
    state = game.state
    game.random = state.random = random.Random(rng.random())  # future dice and draws
    rng.shuffle(state.development_listdeck)  # deck order

    ps = state.player_state
    for seat in range(4):
        if seat == viewer:
            continue
        key = f"P{seat}"
        for prefix, kinds in (("", RESOURCES), ("", DEV_CARDS)):
            total = sum(ps[f"{key}_{prefix}{k}_IN_HAND"] for k in kinds)
            split = [0] * len(kinds)
            for _ in range(total):
                split[rng.randrange(len(kinds))] += 1
            for kind, amount in zip(kinds, split, strict=True):
                ps[f"{key}_{kind}_IN_HAND"] = amount
        for card in DEV_CARDS[:4]:
            ps[f"{key}_{card}_OWNED_AT_START"] = rng.random() < 0.5

    clone = Referee(EngineAdapter(game))
    clone._visible = referee._visible  # identical player-visible history
    return clone


@pytest.mark.parametrize("seed", SEEDS)
def test_observations_do_not_depend_on_hidden_state(seed: int) -> None:
    rng = random.Random(seed)
    checked = 0
    for step, referee in enumerate(play(seed)):
        if step % CHECK_EVERY:
            continue
        for viewer in range(4):
            original = referee.observe(viewer)
            scrambled = scramble_hidden_state(referee, viewer, rng).observe(viewer)
            assert scrambled == original, f"seat {viewer} at step {step} sees hidden state"
            checked += 1
    assert checked > 20


# ----- Event visibility


def test_private_payloads_reach_only_entitled_seats(finished_games: list[Referee]) -> None:
    seen = {"dev_card": 0, "stolen_resource": 0}
    for referee in finished_games:
        for event in referee.events:
            assert "dev_card" not in event.public
            assert "stolen_resource" not in event.public
            entitled: set[int] = set()
            if event.event_type == ActionKind.BUY_DEV_CARD:
                entitled = {event.actor} if event.actor is not None else set()
            elif (
                event.event_type == ActionKind.MOVE_ROBBER and "resource_delta" not in event.public
            ):
                victim = event.public["action"]["params"][1]
                entitled = {event.actor, victim} if event.actor is not None else set()
            assert set(event.private) == entitled, event
            for payload in event.private.values():
                for name in payload:
                    seen[name] += 1
        for seat in range(4):
            for visible, event in zip(referee.visible_history(seat), referee.events, strict=True):
                expected = event.private.get(seat)
                assert visible.private == expected
    assert seen["dev_card"] > 0 and seen["stolen_resource"] > 0


def test_payloads_are_read_only(finished_games: list[Referee]) -> None:
    event = finished_games[0].visible_history(0)[0]
    with pytest.raises(TypeError):
        event.public["action"] = None  # type: ignore[index]


# ----- Completeness: history alone reconstructs what a player may know


def fold_history(referee: Referee, viewer: int) -> tuple[list[int], list[int], list[int]]:
    """Card counts (all seats), dev card counts (all seats) and viewer's own hand."""
    counts, dev_counts, own = [0] * 4, [0] * 4, [0] * 5
    for event in referee.visible_history(viewer):
        public = event.public
        if "card_count_delta" not in public:
            continue
        for seat in range(4):
            counts[seat] += public["card_count_delta"][seat]
            dev_counts[seat] += public["dev_card_count_delta"][seat]
        if "resource_delta" in public:
            own = [a + b for a, b in zip(own, public["resource_delta"][viewer], strict=True)]
        elif event.private is not None and "stolen_resource" in event.private:
            index = RESOURCES.index(event.private["stolen_resource"])
            own[index] += 1 if event.actor == viewer else -1
    return counts, dev_counts, own


@pytest.mark.parametrize("seed", SEEDS)
def test_visible_history_reconstructs_counts_and_own_hand(seed: int) -> None:
    for step, referee in enumerate(play(seed)):
        if step % CHECK_EVERY:
            continue
        for viewer in range(4):
            obs = referee.observe(viewer)
            counts, dev_counts, own = fold_history(referee, viewer)
            assert counts == [p.resource_card_count for p in obs.players]
            assert dev_counts == [p.dev_card_count for p in obs.players]
            assert tuple(own) == obs.own_resources


def test_game_end_is_recorded(finished_games: list[Referee]) -> None:
    for referee in finished_games:
        last = referee.events[-1]
        assert last.event_type == EventType.GAME_END
        assert last.public["winner_seat"] == referee.adapter.winner_seat


# ----- Shape


ALLOWED_SCALARS = (int, str, bool, type(None))


def assert_plain(value: Any, path: str) -> None:
    if isinstance(value, ALLOWED_SCALARS):
        return
    if isinstance(value, tuple):
        for i, item in enumerate(value):
            assert_plain(item, f"{path}[{i}]")
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            assert isinstance(k, str | int), f"{path}: key {k!r}"
            assert_plain(v, f"{path}.{k}")
        return
    if is_dataclass(value) and type(value).__module__.startswith("catan_ai."):
        for f in fields(value):
            assert_plain(getattr(value, f.name), f"{path}.{f.name}")
        return
    raise AssertionError(f"{path} holds a non-plain value: {type(value).__name__}")


def test_observations_hold_only_plain_data(finished_games: list[Referee]) -> None:
    for referee in finished_games:
        for seat in range(4):
            assert_plain(referee.observe(seat), "obs")


def test_observation_has_no_privileged_fields() -> None:
    names = {f.name for f in fields(PlayerObservation)}
    for forbidden in ("seed", "random", "rng", "development_listdeck", "deck", "players_hands"):
        assert forbidden not in names


def test_legal_actions_only_for_the_deciding_seat() -> None:
    referee = Referee.new_game(3)
    current = referee.adapter.current_seat
    for seat in range(4):
        obs = referee.observe(seat)
        assert bool(obs.legal_actions) == (seat == current)
