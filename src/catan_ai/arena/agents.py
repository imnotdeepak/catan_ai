"""Agents decide from their own observation only (plan §2.1)."""

import random
from typing import Protocol

from catan_ai.engine import CanonicalAction
from catan_ai.observations import PlayerObservation


class Agent(Protocol):
    def decide(self, observation: PlayerObservation) -> CanonicalAction: ...


class RandomAgent:
    """Uniformly random over the legal actions. Seeded, so matches are reproducible."""

    def __init__(self, seed: str) -> None:
        self._rng = random.Random(seed)

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        return self._rng.choice(observation.legal_actions)
