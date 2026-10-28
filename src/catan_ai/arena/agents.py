"""Agents decide from their own observation only (plan §2.1)."""

import random
from typing import Protocol

from catan_ai.engine import ActionKind, CanonicalAction, offer_trade
from catan_ai.observations import PlayerObservation


class Agent(Protocol):
    def decide(self, observation: PlayerObservation) -> CanonicalAction: ...


class RandomAgent:
    """Uniformly random over the legal actions. Seeded, so matches are reproducible."""

    def __init__(self, seed: str) -> None:
        self._rng = random.Random(seed)

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        return self._rng.choice(observation.legal_actions)


class RandomTraderAgent(RandomAgent):
    """RandomAgent that also proposes one random 1-for-1 domestic trade per turn.

    Built-in Catanatron bots never propose trades (rules matrix H-02); this
    agent exercises the trade lifecycle against them (ENG-010).
    """

    def __init__(self, seed: str) -> None:
        super().__init__(seed)
        self._last_offer_turn = -1

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        kinds = {a.kind for a in observation.legal_actions}
        if observation.prompt == "DECIDE_ACCEPTEES":
            confirms = [a for a in observation.legal_actions if a.kind == ActionKind.CONFIRM_TRADE]
            return self._rng.choice(confirms)
        held = [i for i, n in enumerate(observation.own_resources) if n > 0]
        can_offer = (
            observation.prompt == "PLAY_TURN"
            and ActionKind.END_TURN in kinds  # rolled, not mid Road Building
            and held
            and observation.num_turns != self._last_offer_turn
        )
        if can_offer:
            self._last_offer_turn = observation.num_turns
            give_index = self._rng.choice(held)
            receive_index = self._rng.choice([i for i in range(5) if i != give_index])
            give = tuple(int(i == give_index) for i in range(5))
            receive = tuple(int(i == receive_index) for i in range(5))
            return offer_trade(observation.seat, give, receive)  # type: ignore[arg-type]
        return super().decide(observation)
