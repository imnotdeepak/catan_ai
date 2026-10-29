"""An agent whose initial settlements follow an opening rule (STR-017, plan §4.5).

Every other decision, initial roads included, goes to the wrapped agent, so
comparing `opening:<rule>:<agent>` with `<agent>` measures the opening rule
alone under the same downstream policy.
"""

from catan_ai.arena.agents import Agent
from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import analyze
from catan_ai.opening.priors import RULES, score


class OpeningRuleAgent:
    def __init__(self, rule: str, inner: Agent) -> None:
        if rule not in RULES:
            raise ValueError(f"Unknown opening rule {rule!r}; known: {', '.join(RULES)}")
        self.rule = rule
        self.inner = inner

    def decide(self, observation: PlayerObservation) -> CanonicalAction:
        if observation.prompt != "BUILD_INITIAL_SETTLEMENT":
            return self.inner.decide(observation)
        analysis = analyze(observation)
        candidates = [a for a in observation.legal_actions if a.kind == ActionKind.BUILD_SETTLEMENT]
        # Ties go to the higher total production, then to the lower node id.
        return max(
            candidates,
            key=lambda a: (
                score(self.rule, analysis, observation.seat, a.params[0]),
                float(analysis.pips[a.params[0]]),
                -a.params[0],
            ),
        )
