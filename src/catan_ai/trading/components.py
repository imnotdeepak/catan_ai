"""Negotiator styles as opponent-model components (OPP-025 with §8.5).

A style component answers trade prompts as that negotiator would (its
accept or reject, smoothed) and defers every other decision to a base
component, so the opponent model can tell trading styles apart from trade
responses alone.
"""

import numpy as np

from catan_ai.arena.agents import RandomAgent
from catan_ai.engine import ActionKind, EngineAdapter
from catan_ai.models import HeuristicPolicyValue
from catan_ai.opponents.components import Component, ObserveFn, Policy, Uniform, smooth
from catan_ai.trading.negotiators import STYLES, NegotiatorAgent


class NegotiatorStyle:
    def __init__(self, style: str, base: Component) -> None:
        self.name = f"negotiator:{style}"
        self.agent = NegotiatorAgent(style, RandomAgent("unused"))
        self.base = base

    def distribution(self, world: EngineAdapter, observe: ObserveFn) -> np.ndarray:
        if world.prompt != "DECIDE_TRADE":
            return self.base.distribution(world, observe)
        legal = world.legal_actions()
        choice = self.agent.respond(observe(world, world.current_seat))
        pi = np.array([a.kind == choice.kind for a in legal], dtype=np.float64)
        if not pi.any():
            pi = np.array([a.kind == ActionKind.REJECT_TRADE for a in legal], dtype=np.float64)
        return smooth(pi)


def trading_components() -> list[Component]:
    """For games among our agents: a random policy, the heuristic, and the five styles."""
    base = Policy("heuristic", HeuristicPolicyValue())
    return [Uniform(), base, *(NegotiatorStyle(style, base) for style in STYLES)]
