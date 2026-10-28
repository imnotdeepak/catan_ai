"""Branch isolation and RNG independence (ENG-007, plan §10.2)."""

import copy
import random
from typing import Any

from catan_ai.engine import ActionKind, CanonicalAction, EngineAdapter


def advance(adapter: EngineAdapter, decisions: int, policy_seed: int) -> list[str]:
    policy = random.Random(policy_seed)
    keys = []
    for _ in range(decisions):
        if adapter.is_over:
            break
        action = policy.choice(adapter.legal_actions())
        keys.append(action.key)
        adapter.execute(action)
    return keys


def fingerprint(adapter: EngineAdapter) -> Any:
    game = adapter.referee_game
    state = game.state
    return (
        copy.deepcopy(state.player_state),
        dict(state.board.buildings),
        dict(state.board.roads),
        list(state.development_listdeck),
        tuple(state.resource_freqdeck),
        len(state.action_records),
        game.random.getstate(),
    )


def midgame(seed: int) -> EngineAdapter:
    adapter = EngineAdapter.new_game(seed)
    advance(adapter, 300, policy_seed=seed)
    return adapter


def test_branches_with_the_same_seed_are_identical() -> None:
    adapter = midgame(1)
    a, b = adapter.branch(7), adapter.branch(7)
    assert fingerprint(a) == fingerprint(b)
    assert advance(a, 500, policy_seed=3) == advance(b, 500, policy_seed=3)


def test_advancing_a_branch_leaves_the_original_and_siblings_unchanged() -> None:
    adapter = midgame(2)
    original = fingerprint(adapter)
    sibling = adapter.branch(8)
    sibling_before = fingerprint(sibling)
    advance(adapter.branch(9), 2000, policy_seed=4)
    assert fingerprint(adapter) == original
    assert fingerprint(sibling) == sibling_before


def test_searching_does_not_perturb_the_real_game() -> None:
    searched, untouched = midgame(3), midgame(3)
    for branch_seed in range(20):
        advance(searched.branch(branch_seed), 200, policy_seed=branch_seed)
    assert advance(searched, 400, policy_seed=5) == advance(untouched, 400, policy_seed=5)
    assert fingerprint(searched) == fingerprint(untouched)


def test_branches_do_not_know_the_real_future() -> None:
    differing_decks = 0
    for seed in range(5):
        adapter = midgame(seed)
        real_deck = list(adapter.referee_game.state.development_listdeck)
        branch = adapter.branch(100 + seed)
        branch_deck = list(branch.referee_game.state.development_listdeck)
        assert sorted(branch_deck) == sorted(real_deck)  # same cards remain
        differing_decks += branch_deck != real_deck
        assert branch.referee_game.random is not adapter.referee_game.random
    assert differing_decks >= 4


def test_forced_outcomes_replay_rolls() -> None:
    adapter = EngineAdapter.new_game(4)
    advance(adapter, 30, policy_seed=4)  # past setup into turns
    while (
        adapter.prompt != "PLAY_TURN"
        or adapter.referee_game.state.player_state[f"P{adapter.current_seat}_HAS_ROLLED"]
    ):
        advance(adapter, 1, policy_seed=adapter.num_turns)
    roll = CanonicalAction(ActionKind.ROLL, adapter.current_seat)
    assert adapter.execute(roll, forced_outcome=(6, 6)) == (6, 6)


def test_engine_copy_still_shares_rng() -> None:
    """Upstream hazard behind rules matrix D-01; branch() is the safe path."""
    game = EngineAdapter.new_game(5).referee_game
    assert game.copy().random is game.random
