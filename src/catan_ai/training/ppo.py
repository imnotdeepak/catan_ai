"""Masked PPO from the imitation champion, against a league (ML-029, plan §9.2 phase D, §9.3).

Rollouts: worker processes play full four-player games. The learner holds
one or two random seats and samples from its masked policy over the legal
candidates; the other seats are drawn from a league of opponents (the frozen
champion, built-in bots, and snapshots of the learner taken during training).
Each learner decision with a real choice is recorded with the player-view
encoding, the chosen candidate, its log-probability and the value estimate.

Returns: the reward is 1 for the winner and 0 for everyone else, at the end
of the game; truncated games are dropped, never scored as losses (plan
§9.2). The value is the value head's P(this seat wins), so advantages come
from GAE over the seat's own decision sequence. The value head keeps its
imitation meaning: it is trained with cross-entropy on the actual winner.

Update: clipped surrogate, entropy bonus and an optional anchor, KL(policy ||
initial policy), which keeps the policy close to the champion it started
from while the value is still learning about the new policy.
"""

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from catan_ai.arena import Contestant, play_match
from catan_ai.engine import CanonicalAction
from catan_ai.models.encoding import encode
from catan_ai.models.network import PolicyValueNet, load_checkpoint
from catan_ai.observations import PlayerObservation

LEARNER = "learner"


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.999  # per decision of the seat; a seat makes ~150-300 real choices a game
    lam: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_weight: float = 0.5
    anchor: float = 0.0  # weight of KL(policy || initial policy)
    epochs: int = 4
    minibatch: int = 256
    max_grad_norm: float = 1.0


@dataclass
class Trajectory:
    """One learner seat's decisions in one game."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)  # [K, A] float16 per decision
    chosen: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    winner: int = -1  # relative to the seat; -1 truncated


class LearnerAgent:
    """Samples from the network's masked policy and records every real choice."""

    def __init__(self, net: PolicyValueNet, seed: int) -> None:
        self.net = net
        self.version = net.config.feature_version
        self.rng = np.random.default_rng(seed)
        self.trajectory = Trajectory()

    @torch.inference_mode()
    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        legal = observation.legal_actions
        if len(legal) == 1:
            return legal[0]
        state, rows = encode(observation, self.version)
        out = self.net(
            torch.from_numpy(state).unsqueeze(0),
            torch.from_numpy(rows).unsqueeze(0),
            torch.ones((1, len(rows)), dtype=torch.bool),
        )
        logp = torch.log_softmax(out["policy_logits"][0], dim=0).numpy().astype(np.float64)
        p = np.exp(logp)
        index = int(self.rng.choice(len(p), p=p / p.sum()))
        t = self.trajectory
        t.states.append(state.astype(np.float32))
        t.actions.append(rows.astype(np.float16))
        t.chosen.append(index)
        t.logp.append(float(logp[index]))
        t.value.append(float(torch.softmax(out["value_logits"][0], dim=0)[0]))
        return legal[index]


# ----- Rollout workers

_worker_net: dict[str, Any] = {"key": None, "net": None}


def _net_for(path: str, version: int) -> PolicyValueNet:
    if _worker_net["key"] != (path, version):
        torch.set_num_threads(1)
        net, _ = load_checkpoint(Path(path))
        _worker_net.update(key=(path, version), net=net)
    return _worker_net["net"]  # type: ignore[no-any-return]


@dataclass(frozen=True)
class Job:
    seed: int
    seating: tuple[str, ...]  # LEARNER or a contestant spec, by seat
    weights: str  # learner checkpoint path
    version: int  # learner checkpoint version, so workers reload it


def rollout(job: Job) -> dict[str, Any]:
    net = _net_for(job.weights, job.version)
    learners = {
        s: LearnerAgent(net, job.seed * 4 + s) for s, n in enumerate(job.seating) if n == LEARNER
    }
    seating = [Contestant("heuristic" if n == LEARNER else n) for n in job.seating]
    result, _ = play_match(f"ppo-{job.seed}", job.seed, seating, overrides=dict(learners))
    trajectories = []
    for seat, agent in learners.items():
        t = agent.trajectory
        if result.winner_seat is not None and result.error is None:
            t.winner = (result.winner_seat - seat) % 4
        trajectories.append(t)
    return {
        "seating": job.seating,
        "winner_seat": result.winner_seat,
        "error": result.error,
        "truncated": result.truncated,
        "trajectories": trajectories,
        "seconds": result.wall_seconds,
    }


# ----- League


@dataclass
class League:
    """Opponent specs with sampling weights; snapshots share `snapshot_weight`."""

    fixed: dict[str, float]
    snapshot_weight: float
    snapshots: list[str] = field(default_factory=list)
    max_snapshots: int = 8

    def add_snapshot(self, spec: str) -> None:
        self.snapshots.append(spec)
        del self.snapshots[: -self.max_snapshots]

    def sample(self, rng: random.Random) -> str:
        specs = list(self.fixed)
        weights = list(self.fixed.values())
        if self.snapshots:
            specs += self.snapshots
            weights += [self.snapshot_weight / len(self.snapshots)] * len(self.snapshots)
        return rng.choices(specs, weights=weights)[0]

    def seating(self, rng: random.Random) -> tuple[str, ...]:
        """One or two learner seats, at random positions; the rest from the league."""
        seats = [self.sample(rng) for _ in range(4)]
        for seat in rng.sample(range(4), rng.choice((1, 2))):
            seats[seat] = LEARNER
        return tuple(seats)


# ----- Learner


def advantages(t: Trajectory, config: PPOConfig) -> tuple[np.ndarray, np.ndarray]:
    """(GAE advantages, returns) for one seat; the only reward is the win at the end."""
    values = np.array([*t.value, 0.0])
    rewards = np.zeros(len(t.value))
    rewards[-1] = float(t.winner == 0)
    adv = np.zeros(len(t.value))
    running = 0.0
    for i in reversed(range(len(t.value))):
        nxt = values[i + 1] if i + 1 < len(t.value) else 0.0
        delta = rewards[i] + config.gamma * nxt - values[i]
        running = delta + config.gamma * config.lam * running
        adv[i] = running
    return adv, adv + values[:-1]


@dataclass
class Buffer:
    states: np.ndarray
    actions: list[np.ndarray]
    chosen: np.ndarray
    logp: np.ndarray
    adv: np.ndarray
    winner: np.ndarray

    def __len__(self) -> int:
        return len(self.chosen)


def build_buffer(trajectories: Sequence[Trajectory], config: PPOConfig) -> Buffer:
    kept = [t for t in trajectories if t.winner >= 0 and t.chosen]
    parts = [advantages(t, config)[0] for t in kept]
    return Buffer(
        states=np.concatenate([np.stack(t.states) for t in kept]),
        actions=[a for t in kept for a in t.actions],
        chosen=np.concatenate([np.array(t.chosen) for t in kept]),
        logp=np.concatenate([np.array(t.logp) for t in kept]),
        adv=np.concatenate(parts),
        winner=np.concatenate([np.full(len(t.chosen), t.winner) for t in kept]),
    )


def _pad(actions: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(a) for a in actions)
    out = np.zeros((len(actions), width, actions[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(actions), width), dtype=bool)
    for i, a in enumerate(actions):
        out[i, : len(a)] = a
        mask[i, : len(a)] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: Buffer,
    config: PPOConfig,
    reference: PolicyValueNet | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    net.train()
    adv = (buffer.adv - buffer.adv.mean()) / (buffer.adv.std() + 1e-8)
    stats: dict[str, list[float]] = {
        k: [] for k in ("policy", "value", "entropy", "kl", "anchor", "clipped")
    }
    for _ in range(config.epochs):
        order = rng.permutation(len(buffer))
        for at in range(0, len(order), config.minibatch):
            idx = order[at : at + config.minibatch]
            states = torch.from_numpy(buffer.states[idx])
            actions, mask = _pad([buffer.actions[i] for i in idx])
            out = net(states, actions, mask)
            logp_all = torch.log_softmax(out["policy_logits"], dim=1)
            chosen = torch.from_numpy(buffer.chosen[idx])
            logp = logp_all.gather(1, chosen[:, None]).squeeze(1)
            old = torch.from_numpy(buffer.logp[idx]).float()
            a = torch.from_numpy(adv[idx]).float()
            ratio = torch.exp(logp - old)
            surrogate = torch.min(ratio * a, ratio.clamp(1 - config.clip, 1 + config.clip) * a)
            p_all = logp_all.exp()
            safe = logp_all.masked_fill(~mask, 0.0)
            entropy = -(p_all * safe).sum(1).mean()
            value = F.cross_entropy(out["value_logits"], torch.from_numpy(buffer.winner[idx]))
            loss: torch.Tensor = (
                -surrogate.mean() - config.entropy * entropy + config.value_weight * value
            )
            anchor = torch.zeros(())
            if reference is not None and config.anchor > 0:
                with torch.no_grad():
                    ref = torch.log_softmax(
                        reference(states, actions, mask)["policy_logits"], dim=1
                    )
                anchor = (p_all * (safe - ref.masked_fill(~mask, 0.0))).sum(1).mean()
                loss = loss + config.anchor * anchor
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()
            stats["policy"].append(-surrogate.mean().item())
            stats["value"].append(value.item())
            stats["entropy"].append(entropy.item())
            stats["kl"].append((old - logp).mean().item())
            stats["anchor"].append(anchor.item())
            stats["clipped"].append(float(((ratio - 1).abs() > config.clip).float().mean()))
    net.eval()
    return {k: float(np.mean(v)) for k, v in stats.items()}


def elapsed(started: float) -> str:
    return f"{(time.perf_counter() - started) / 60:.1f} min"
