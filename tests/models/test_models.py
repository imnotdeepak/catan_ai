"""Encoder, policy/value interface and network (ML-013)."""

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_ai.arena import Contestant, play_match
from catan_ai.engine import ActionKind, CanonicalAction, EngineAdapter
from catan_ai.models import (
    ACTION_DIM,
    STATE_DIM,
    HeuristicPolicyValue,
    PolicyAgent,
    PolicyValue,
    encode_actions,
    encode_state,
)
from catan_ai.models.features import (
    _GLOBAL_AT,
    _PLAYERS_AT,
    EDGES,
    PLAYER_WIDTH,
    TILE_COORDS,
    TILE_NODES,
)
from catan_ai.models.interface import Evaluation
from catan_ai.models.network import (
    LearnedPolicyValue,
    PolicyValueNet,
    load_checkpoint,
    save_checkpoint,
)
from catan_ai.observations import PlayerObservation
from tests.observations.test_firewall import play, scramble_hidden_state


@pytest.mark.parametrize("seed", [1, 77, 4242])
def test_static_topology_matches_every_board(seed: int) -> None:
    board = EngineAdapter.new_game(seed).referee_game.state.board.map
    assert sorted(tuple(c) for c in board.land_tiles) == list(TILE_COORDS)
    for coord, tile in board.land_tiles.items():
        assert tuple(sorted(tile.nodes.values())) == TILE_NODES[tuple(coord)]
    edges = {tuple(sorted(e)) for t in board.land_tiles.values() for e in t.edges.values()}
    assert sorted(edges) == list(EDGES)


def sample_observations(seed: int, every: int = 7) -> list[PlayerObservation]:
    observations = []
    for step, referee in enumerate(play(seed)):
        if step % every == 0 and not referee.adapter.is_over:
            observations.append(referee.observe(referee.adapter.current_seat))
    return observations


def test_encodings_have_fixed_shapes_and_are_finite() -> None:
    for obs in sample_observations(3):
        state, actions = encode_state(obs), encode_actions(obs)
        assert state.shape == (STATE_DIM,)
        assert actions.shape == (len(obs.legal_actions), ACTION_DIM)
        assert np.isfinite(state).all() and np.isfinite(actions).all()


def test_distinct_legal_actions_get_distinct_encodings() -> None:
    for obs in sample_observations(5):
        rows = {encode_actions(obs)[i].tobytes() for i in range(len(obs.legal_actions))}
        assert len(rows) == len(obs.legal_actions), obs.prompt


def test_state_encoding_is_seat_relative() -> None:
    """Player row 0 is always the viewer, row 1 the next seat, and so on."""
    for obs in sample_observations(9):
        rows = encode_state(obs)[_PLAYERS_AT:_GLOBAL_AT].reshape(4, PLAYER_WIDTH)
        for r in range(4):
            player = obs.players[(obs.seat + r) % 4]
            assert rows[r, 1] == pytest.approx(player.resource_card_count / 10)
            assert rows[r, 0] == pytest.approx(player.visible_victory_points / 10)


@pytest.mark.parametrize("seed", [11, 12])
def test_policies_do_not_depend_on_hidden_state(seed: int) -> None:
    """Policy-level indistinguishability (observation contract, known limits)."""
    torch.manual_seed(0)
    policies: list[PolicyValue] = [HeuristicPolicyValue(), LearnedPolicyValue(PolicyValueNet())]
    rng = random.Random(seed)
    checked = 0
    for step, referee in enumerate(play(seed)):
        if step % 9 or referee.adapter.is_over:
            continue
        seat = referee.adapter.current_seat
        original = referee.observe(seat)
        scrambled = scramble_hidden_state(referee, seat, rng).observe(seat)
        for policy in policies:
            assert policy.evaluate(original) == policy.evaluate(scrambled)
        checked += 1
    assert checked > 10


def _check(evaluation: Evaluation, obs: PlayerObservation) -> None:
    assert len(evaluation.priors) == len(obs.legal_actions)
    assert sum(evaluation.priors) == pytest.approx(1.0, abs=1e-5)
    assert len(evaluation.win_probabilities) == 4
    assert sum(evaluation.win_probabilities) == pytest.approx(1.0, abs=1e-5)


def test_interface_outputs_are_distributions() -> None:
    net = LearnedPolicyValue(PolicyValueNet())
    for obs in sample_observations(21):
        _check(HeuristicPolicyValue().evaluate(obs), obs)
        _check(net.evaluate(obs), obs)


def test_heuristic_is_deterministic_and_prefers_cities() -> None:
    obs = next(o for o in sample_observations(2, every=1) if len(o.legal_actions) > 1)
    assert HeuristicPolicyValue().evaluate(obs) == HeuristicPolicyValue().evaluate(obs)
    city = CanonicalAction(ActionKind.BUILD_CITY, obs.seat, (0,))
    end = CanonicalAction(ActionKind.END_TURN, obs.seat)
    staged = replace(obs, legal_actions=(city, end))
    priors = HeuristicPolicyValue().evaluate(staged).priors
    assert priors[0] > priors[1]


class _Broken:
    def evaluate(self, observation: PlayerObservation) -> Evaluation:
        raise RuntimeError("model failure")


def test_agent_falls_back_to_heuristic_and_stays_legal() -> None:
    agent = PolicyAgent(_Broken())
    for obs in sample_observations(4):
        assert agent.decide(obs) in obs.legal_actions
    assert agent.fallbacks > 0


def test_checkpoint_round_trip_and_model_contestant(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = PolicyValueNet()
    path = tmp_path / "net.pt"
    save_checkpoint(path, net, {"note": "test"})
    loaded, metadata = load_checkpoint(path)
    assert metadata == {"note": "test"}
    obs = sample_observations(6)[2]
    assert LearnedPolicyValue(net).evaluate(obs) == LearnedPolicyValue(loaded).evaluate(obs)

    lineup = [
        Contestant(f"model:{path}"),
        Contestant("heuristic"),
        Contestant("random"),
        Contestant("random"),
    ]
    result, _ = play_match("m", 5, lineup)
    assert result.error is None
    assert result.winner_seat is not None or result.truncated


def test_policy_and_outcome_towers_share_no_weights() -> None:
    from catan_ai.models.network import is_outcome_parameter

    net = PolicyValueNet()
    obs = sample_observations(8)[3]
    states = torch.from_numpy(encode_state(obs)).unsqueeze(0)
    actions = torch.from_numpy(encode_actions(obs)).unsqueeze(0)
    out = net(states, actions, torch.ones(actions.shape[:2], dtype=torch.bool))
    out["value_logits"].sum().backward()
    for name, parameter in net.named_parameters():
        touched = parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        if not is_outcome_parameter(name):
            assert not touched, f"value loss reaches policy parameter {name}"
