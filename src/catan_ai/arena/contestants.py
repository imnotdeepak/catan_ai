"""Lineup specs: our agents, or Catanatron's built-in bots.

- `random`: our seeded RandomAgent, deciding from observations.
- `random-trader`: RandomAgent that also proposes one random 1-for-1 trade per turn.
- `heuristic`: greedy play from the deterministic HeuristicPolicyValue.
- `model:<checkpoint>`: greedy play from a trained policy/value checkpoint
  (path relative to the repo root, or absolute).
- `opening:<rule>:<spec>`: initial settlements by an opening rule
  (`catan_ai.opening.priors.RULES`), every other decision by the agent `<spec>`,
  which must be one of ours, e.g. `opening:ows:model:champions/imitation-v1.pt`.
- `draft:<rule>:<spec>`: initial settlements and roads by pair search over the
  predicted snake draft (`catan_ai.opening.draft`), scored by `<rule>`;
  `draft-settle:<rule>:<spec>` searches settlements only and leaves roads to `<spec>`.
- `catanatron:<spec>`: a built-in bot by registry spec, e.g. `catanatron:F`,
  `catanatron:AB:2`, `catanatron:W`. These receive the full engine `Game` and
  are therefore privileged (rules matrix H-01); results label them so.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import catanatron.players
import catanatron.players.playouts
from catanatron.models.player import SimplePlayer
from catanatron.registry import REGISTRY

from catan_ai.arena.agents import Agent, RandomAgent, RandomTraderAgent
from catan_ai.models import HeuristicPolicyValue, PolicyAgent, PolicyValue
from catan_ai.provenance import REPO_ROOT

# GreedyPlayouts starts a fresh process pool for every candidate action. Process
# start-up is slow on Windows (spawn), turning one decision into minutes, so run
# its playouts in-process instead (rules matrix H-09).
catanatron.players.playouts.USE_MULTIPROCESSING = False

ENGINE_PREFIX = "catanatron:"
MODEL_PREFIX = "model:"
# Wrappers that take over initial placement: "<prefix><rule>:<inner agent spec>".
OPENING_PREFIXES = ("opening:", "draft:", "draft-settle:")
AGENTS: dict[str, Callable[[str], Agent]] = {
    "random": RandomAgent,
    "random-trader": RandomTraderAgent,
    "heuristic": lambda seed: PolicyAgent(HeuristicPolicyValue(), seed=seed),
}


@cache
def load_policy(checkpoint: str) -> PolicyValue:
    """Load once per process. Torch is imported only when a model plays."""
    import torch

    from catan_ai.models.network import LearnedPolicyValue, load_checkpoint

    torch.set_num_threads(1)  # one game per process; arena parallelism is across games
    path = Path(checkpoint)
    net, _ = load_checkpoint(path if path.is_absolute() else REPO_ROOT / path)
    return LearnedPolicyValue(net)


@dataclass(frozen=True)
class Contestant:
    name: str

    def __post_init__(self) -> None:
        if self.is_engine_bot:
            REGISTRY.lookup(self.engine_spec.split(":")[0])  # fail fast on unknown codes
        elif self.is_model:
            load_policy(self.name.removeprefix(MODEL_PREFIX))  # fail fast on bad checkpoints
        elif self.is_opening:
            from catan_ai.opening.priors import RULES

            rule, inner = self.opening_parts
            if rule not in RULES:
                raise ValueError(f"Unknown opening rule {rule!r}; known: {', '.join(RULES)}")
            if Contestant(inner).is_engine_bot:
                raise ValueError(f"{self.name!r}: engine bots act engine-side, cannot be wrapped")
        elif self.name not in AGENTS:
            known = ", ".join(
                [
                    *AGENTS,
                    f"{MODEL_PREFIX}<checkpoint>",
                    *(f"{p}<rule>:<agent>" for p in OPENING_PREFIXES),
                    f"{ENGINE_PREFIX}<code>",
                ]
            )
            raise ValueError(f"Unknown contestant {self.name!r}. Known: {known}")

    @property
    def is_engine_bot(self) -> bool:
        return self.name.startswith(ENGINE_PREFIX)

    @property
    def is_model(self) -> bool:
        return self.name.startswith(MODEL_PREFIX)

    @property
    def is_opening(self) -> bool:
        return self.name.startswith(OPENING_PREFIXES)

    @property
    def opening_parts(self) -> tuple[str, str]:
        """(rule, inner agent spec) of an opening wrapper."""
        rest = self.name.split(":", 1)[1]
        rule, _, inner = rest.partition(":")
        return rule, inner

    @property
    def privileged(self) -> bool:
        return self.is_engine_bot

    @property
    def engine_spec(self) -> str:
        return self.name.removeprefix(ENGINE_PREFIX)

    def engine_player(self, color: Any) -> Any:
        """The engine-side player for this seat (a placeholder for our agents)."""
        if self.is_engine_bot:
            return REGISTRY.build(self.engine_spec, color)
        return SimplePlayer(color)

    def make_agent(self, seed: str) -> Agent:
        if self.is_opening:
            from catan_ai.opening.agent import OpeningRuleAgent
            from catan_ai.opening.draft import DraftSearchAgent

            rule, inner = self.opening_parts
            agent = Contestant(inner).make_agent(seed)
            if self.name.startswith("opening:"):
                return OpeningRuleAgent(rule, agent)
            return DraftSearchAgent(rule, agent, roads=self.name.startswith("draft:"))
        if self.is_model:
            return PolicyAgent(load_policy(self.name.removeprefix(MODEL_PREFIX)), seed=seed)
        return AGENTS[self.name](seed)
