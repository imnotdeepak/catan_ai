"""Information-set root search over sampled belief worlds (SRCH-030, plan §10.1-10.3).

At a decision worth searching, the agent keeps the policy's top candidates
and scores each by short rollouts in worlds sampled from its own beliefs
(`catan_ai.beliefs.worlds`):

- **Root sampling, fixed root actions.** Every candidate is played in the
  same sampled worlds, and the choice aggregates over them. The root never
  conditions on which world is the true one, so there is no strategy fusion
  at the root.
- **Observation-respecting rollouts.** Inside a world, each seat (the viewer
  included) acts from its own `Referee` observation of that world: its own
  sampled hand and the public state, never another seat's cards.
- **Safe branching.** Each rollout runs on `EngineAdapter.branch`, with its
  own RNG and reshuffled deck (plan §10.2). Worlds are separate engine games
  built from the observation alone, so search never touches the real game.
- **Common random numbers.** A world's rollouts share one branch seed across
  candidates, so they see the same dice and the comparison is paired.

A rollout runs `turns` turns past the root, then stops at the viewer's next
decision and reads its value head (P(viewer wins)); a rollout that ends the
game scores 1 or 0. Rollout players are the policy, greedy, in every seat.
"""

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Final

from catan_ai.beliefs.worlds import WorldSampler
from catan_ai.engine import CanonicalAction, EngineAdapter, IllegalActionError
from catan_ai.models import PolicyAgent, PolicyValue
from catan_ai.observations import PlayerObservation, Referee

PLAIN, SEARCHED, SWITCHED = 0, 1, 2

# Prompts the policy answers well enough alone, or whose rollouts would be mostly noise.
QUIET_PROMPTS: Final = frozenset({"DECIDE_ACCEPTEES"})


@dataclass(frozen=True)
class SearchConfig:
    candidates: int = 3  # top-prior legal actions scored by rollouts
    worlds: int = 8  # sampled worlds; one rollout per candidate per world
    turns: int = 4  # rollout horizon past the root turn
    max_decisions: int = 400  # per-rollout safety cap
    min_prior: float = 0.02  # candidates the policy all but rules out are not searched
    close: float = 0.5  # search when the top two priors are closer than this
    near_win: int = 8  # or when any seat shows this many points
    switch: float = 0.02  # leave the policy's choice only for a larger mean gain


def candidates(
    observation: PlayerObservation, priors: tuple[float, ...], config: SearchConfig
) -> list[int]:
    """Indexes of the legal actions worth searching, best prior first (maybe < 2)."""
    order = sorted(range(len(priors)), key=lambda i: (-priors[i], i))
    return [i for i in order[: config.candidates] if priors[i] >= config.min_prior]


def should_search(
    observation: PlayerObservation, priors: tuple[float, ...], config: SearchConfig
) -> list[int]:
    """The candidates to search at this decision, or [] (plan §10.3).

    Searched: initial placements, robber moves, any seat near winning, and
    decisions where the policy is unsure between its top two choices. Forced
    and clear-cut decisions are left to the policy.
    """
    if len(observation.legal_actions) < 2 or observation.prompt in QUIET_PROMPTS:
        return []
    picks = candidates(observation, priors, config)
    if len(picks) < 2:
        return []
    near = max(p.visible_victory_points for p in observation.players) >= config.near_win
    priority = observation.is_initial_build_phase or observation.prompt == "MOVE_ROBBER" or near
    close = priors[picks[0]] - priors[picks[1]] < config.close
    return picks if priority or close else []


def rollout(
    world: EngineAdapter,
    action: CanonicalAction,
    viewer: int,
    policy: PolicyValue,
    config: SearchConfig,
    seed: int,
) -> tuple[float, int]:
    """(viewer's value after playing `action` in a branch of `world`, decisions played).

    Raises IllegalActionError if `action` is illegal in this world.
    """
    branch = world.branch(seed)
    referee = Referee(branch)
    referee.execute(action)
    agents = [PolicyAgent(policy, seed=f"{seed}:{s}") for s in range(4)]
    stop = branch.num_turns + config.turns
    played = 0
    while not branch.is_over and played < config.max_decisions:
        seat = branch.current_seat
        if seat == viewer and branch.num_turns >= stop:
            break
        referee.execute(agents[seat].decide(referee.observe(seat)))
        played += 1
    winner = branch.winner_seat
    if winner is not None:
        return float(winner == viewer), played
    return policy.evaluate(referee.observe(viewer)).win_probabilities[viewer], played


class SearchAgent:
    """Greedy policy play, with root search at the decisions `should_search` picks.

    `stats` counts decisions, searches, switches away from the policy's choice,
    rollouts, rollout decisions and failures (a world sampler or rollout error
    falls back to the policy's choice). `last` says what happened at the most
    recent decision: PLAIN, SEARCHED (kept the policy's choice) or SWITCHED;
    distillation data records it (SRCH-031).
    """

    def __init__(
        self, policy: PolicyValue, config: SearchConfig | None = None, seed: str = ""
    ) -> None:
        self.policy = policy
        self.config = config or SearchConfig()
        self.stats: Counter[str] = Counter()
        self.search_seconds = 0.0
        self._plain = PolicyAgent(policy, seed=seed)
        self._rng = random.Random(seed)
        self._seed = seed
        self._sampler: WorldSampler | None = None
        self.last = PLAIN

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        self.stats["decisions"] += 1
        self.last = PLAIN
        legal = observation.legal_actions
        if len(legal) == 1:
            return legal[0]
        try:
            priors = self.policy.evaluate(observation).priors
            if len(priors) != len(legal) or not all(math.isfinite(p) for p in priors):
                raise ValueError("unusable priors")
        except Exception:
            return self._plain.decide(observation)  # its heuristic fallback
        picks = should_search(observation, priors, self.config)
        if not picks:
            return legal[_argmax(priors)]
        started = time.perf_counter()
        try:
            choice = self._search(observation, picks)
        except Exception:
            self.stats["failures"] += 1
            choice = picks[0]
        else:
            self.last = SWITCHED if choice != picks[0] else SEARCHED
        self.search_seconds += time.perf_counter() - started
        return legal[choice]

    def scores(self, observation: PlayerObservation, picks: list[int]) -> list[float]:
        """Mean rollout value of each candidate over the sampled worlds (nan if never legal)."""
        if self._sampler is None or self._sampler.seat != observation.seat:
            self._sampler = WorldSampler(
                observation.seat, count=self.config.worlds, seed=self._rng.randrange(2**31)
            )
        worlds = self._sampler.update(observation)
        seeds = [self._rng.randrange(2**31) for _ in worlds]
        totals = [0.0] * len(picks)
        counts = [0] * len(picks)
        for world, seed in zip(worlds, seeds, strict=True):
            for j, index in enumerate(picks):
                try:
                    value, played = rollout(
                        world,
                        observation.legal_actions[index],
                        observation.seat,
                        self.policy,
                        self.config,
                        seed,
                    )
                except IllegalActionError:
                    continue
                totals[j] += value
                counts[j] += 1
                self.stats["rollouts"] += 1
                self.stats["rollout_decisions"] += played
        return [t / c if c else math.nan for t, c in zip(totals, counts, strict=True)]

    def _search(self, observation: PlayerObservation, picks: list[int]) -> int:
        self.stats["searches"] += 1
        means = self.scores(observation, picks)
        base = means[0]
        best = max(range(len(picks)), key=lambda j: (_finite(means[j]), -j))
        if math.isnan(base) or _finite(means[best]) - base > self.config.switch:
            if best != 0:
                self.stats["switches"] += 1
            return picks[best]
        return picks[0]


def _argmax(values: tuple[float, ...]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def _finite(value: float) -> float:
    return -math.inf if math.isnan(value) else value
