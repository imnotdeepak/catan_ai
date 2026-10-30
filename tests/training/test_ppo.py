"""Masked PPO pieces (ML-029): returns, buffers, league seating and one update."""

import random

import numpy as np
import torch

from catan_ai.models.network import load_checkpoint
from catan_ai.provenance import REPO_ROOT
from catan_ai.training.ppo import (
    LEARNER,
    Job,
    League,
    PPOConfig,
    Trajectory,
    advantages,
    build_buffer,
    rollout,
    update,
)

CHAMPION = REPO_ROOT / "champions" / "imitation-features-v3.pt"


def test_advantages_carry_the_terminal_reward_back() -> None:
    config = PPOConfig(gamma=1.0, lam=1.0)
    won = Trajectory(value=[0.25, 0.5, 0.75], winner=0)
    adv, returns = advantages(won, config)
    # With gamma = lambda = 1 every return is the final reward.
    assert np.allclose(returns, 1.0)
    assert np.allclose(adv, [0.75, 0.5, 0.25])
    lost = Trajectory(value=[0.25, 0.5], winner=2)
    assert np.allclose(advantages(lost, config)[1], 0.0)


def test_truncated_games_are_dropped_not_scored_as_losses() -> None:
    def trajectory(winner: int) -> Trajectory:
        return Trajectory(
            states=[np.zeros(3, dtype=np.float32)],
            actions=[np.zeros((2, 4), dtype=np.float16)],
            chosen=[1],
            logp=[-0.7],
            value=[0.3],
            winner=winner,
        )

    buffer = build_buffer([trajectory(0), trajectory(-1), trajectory(3)], PPOConfig())
    assert len(buffer) == 2
    assert list(buffer.winner) == [0, 3]


def test_league_seats_one_or_two_learners() -> None:
    league = League({"catanatron:F": 1.0, "heuristic": 1.0}, snapshot_weight=1.0)
    league.add_snapshot("model:a.pt")
    rng = random.Random(3)
    counts = set()
    seen = set()
    for _ in range(200):
        seating = league.seating(rng)
        counts.add(seating.count(LEARNER))
        seen.update(seating)
    assert counts == {1, 2}
    assert {"catanatron:F", "heuristic", "model:a.pt"} <= seen
    for i in range(20):
        league.add_snapshot(f"model:{i}.pt")
    assert len(league.snapshots) == league.max_snapshots


def test_rollout_and_update_run_end_to_end() -> None:
    job = Job(11, (LEARNER, "catanatron:W", LEARNER, "random"), str(CHAMPION), 0)
    game = rollout(job)
    assert game["error"] is None and not game["truncated"]
    trajectories = game["trajectories"]
    assert len(trajectories) == 2 and all(t.chosen for t in trajectories)
    for t in trajectories:
        assert all(0 <= c < len(a) for c, a in zip(t.chosen, t.actions, strict=True))
        assert all(lp <= 0 for lp in t.logp)
    net, _ = load_checkpoint(CHAMPION)
    reference, _ = load_checkpoint(CHAMPION)
    before = [p.detach().clone() for p in net.parameters()]
    config = PPOConfig(epochs=1, minibatch=64, anchor=0.05)
    stats = update(
        net,
        torch.optim.Adam(net.parameters(), lr=1e-4),
        build_buffer(trajectories, config),
        config,
        reference,
        np.random.default_rng(0),
    )
    assert all(np.isfinite(v) for v in stats.values())
    # First minibatch: the policy is the one that acted, so nothing is clipped yet.
    assert stats["clipped"] < 0.5
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters(), strict=True))
