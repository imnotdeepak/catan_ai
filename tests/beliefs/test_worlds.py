"""Sampled engine worlds consistent with a seat's observations (plan §6.2, §10.1)."""

import random

import pytest

from catan_ai.beliefs.worlds import WorldSampler, board_map
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations.firewall import _board_view, _public_player
from tests.observations.test_firewall import play, scramble_hidden_state

CHECK_EVERY = 7


@pytest.mark.parametrize("seed", [1, 3])
def test_every_world_matches_what_the_viewer_sees(seed: int) -> None:
    samplers = [WorldSampler(s, count=3, seed=seed) for s in range(4)]
    checked = 0
    for step, referee in enumerate(play(seed)):
        for viewer, sampler in enumerate(samplers):
            obs = referee.observe(viewer)
            worlds = sampler.update(obs)
            if step % CHECK_EVERY:
                continue
            assert len(worlds) == 3
            for world in worlds:
                state = world.referee_game.state
                assert _board_view(state) == obs.board
                assert tuple(_public_player(state, s) for s in range(4)) == obs.players
                ps = state.player_state
                own = tuple(ps[f"P{viewer}_{r}_IN_HAND"] for r in RESOURCES)
                assert own == obs.own_resources
                assert world.prompt == obs.prompt and world.current_seat == obs.current_seat
                if obs.legal_actions:
                    assert world.legal_actions() == obs.legal_actions
                checked += 1
    assert checked > 300
    assert sum(s.rebuilds for s in samplers) >= 4  # one start per seat, plus recoveries


def test_worlds_depend_only_on_the_observation() -> None:
    """Leakage: games the viewer cannot tell apart give identical worlds."""
    rng = random.Random(0)
    for step, referee in enumerate(play(2)):
        if step < 150 or step % 40:
            continue
        viewer = (referee.adapter.current_seat + 1) % 4
        twin = scramble_hidden_state(referee, viewer, rng)
        worlds = [
            WorldSampler(viewer, count=2, seed=9).update(r.observe(viewer)) for r in (referee, twin)
        ]
        for a, b in zip(*worlds, strict=True):
            sa, sb = a.referee_game.state, b.referee_game.state
            assert sa.player_state == sb.player_state
            assert sa.development_listdeck == sb.development_listdeck


def test_board_map_reproduces_the_observed_board() -> None:
    referee = next(iter(play(4)))
    obs = referee.observe(0)
    real = referee.adapter.referee_game.state.board.map
    rebuilt = board_map(obs)
    assert {c: (t.resource, t.number) for c, t in rebuilt.land_tiles.items()} == {
        c: (t.resource, t.number) for c, t in real.land_tiles.items()
    }
    assert {k: set(v) for k, v in rebuilt.port_nodes.items()} == {
        k: set(v) for k, v in real.port_nodes.items()
    }
