"""A learned trade-response predictor (OPP-025, plan §7.3).

P(responder accepts) = P(it can pay) * sigmoid(w . x). P(can pay) comes from
the resource belief, as in the rule; the logistic part is fitted to the
responses the trading agent logged in self-play and ladder games
(`catan_ai.cli.trade_response`). The features are what the proposer can see
when it offers:

- the trade's value to each side, by the planner (rounds saved);
- whether the proposer leads the race to 10, and both sides' visible points;
- the responder's record this game: how often it has accepted, and how many
  offers it has answered. This is what separates a trade-resistant player
  (or a bot that never trades) from a cooperative one after a few offers.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from catan_ai.engine import ActionKind
from catan_ai.observations import PlayerObservation
from catan_ai.trading.negotiators import Judgement
from catan_ai.trading.proposals import Vector

FEATURES: Final = (
    "bias",
    "responder saving",
    "responder saving > 0",
    "proposer saving",
    "proposer leads",
    "proposer points",
    "responder points",
    "cards given",
    "cards asked",
    "responder acceptance rate",
    "responder responses (log)",
    "responder accepted before",
)
CLIP: Final = 5.0
DEFAULT_PATH: Final = Path("champions") / "trade-response.json"


def response_record(observation: PlayerObservation, seat: int) -> tuple[int, int]:
    """(accepts, responses) by `seat` to offers so far this game."""
    accepts = rejects = 0
    for e in observation.history:
        if e.actor != seat:
            continue
        if e.event_type == ActionKind.ACCEPT_TRADE:
            accepts += 1
        elif e.event_type == ActionKind.REJECT_TRADE:
            rejects += 1
    return accepts, accepts + rejects


def features(
    observation: PlayerObservation,
    seat: int,
    verdict: Judgement,
    give: Vector,
    receive: Vector,
    proposer_leads: bool,
) -> np.ndarray:
    """Features of offering `give` for `receive` to `seat`; `verdict` is from the proposer's
    view (`mine` the proposer's saving, `theirs` the responder's)."""
    accepts, responses = response_record(observation, seat)
    players = observation.players
    return np.array(
        [
            1.0,
            float(np.clip(verdict.theirs, -CLIP, CLIP)),
            float(verdict.theirs > 0),
            float(np.clip(verdict.mine, -CLIP, CLIP)),
            float(proposer_leads),
            players[observation.seat].visible_victory_points / 10,
            players[seat].visible_victory_points / 10,
            sum(give) / 2,
            sum(receive) / 2,
            (accepts + 1) / (responses + 2),
            math.log1p(responses),
            float(accepts > 0),
        ],
        dtype=np.float64,
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-np.clip(z, -30, 30))))


@dataclass(frozen=True)
class ResponseModel:
    weights: np.ndarray

    def yes(self, x: np.ndarray) -> float:
        """P(says yes | can pay)."""
        return float(_sigmoid(np.asarray(x @ self.weights)))

    def accept(self, x: np.ndarray, pay: np.ndarray) -> np.ndarray:
        return np.asarray(pay * _sigmoid(x @ self.weights))

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        pay: np.ndarray,
        accepted: np.ndarray,
        l2: float = 1e-3,
        steps: int = 3000,
        rate: float = 0.5,
    ) -> "ResponseModel":
        """Maximum likelihood of accepted ~ Bernoulli(pay * sigmoid(x . w)), by gradient
        ascent with a small L2 penalty (the bias is not penalized)."""
        w = np.zeros(x.shape[1])
        y = accepted.astype(np.float64)
        pay = np.clip(pay, 1e-6, 1.0)
        penalty = np.full(x.shape[1], l2)
        penalty[0] = 0.0
        for _ in range(steps):
            s = _sigmoid(x @ w)
            p = np.clip(pay * s, 1e-9, 1 - 1e-9)
            # d log L / d z = (y/p - (1-y)/(1-p)) * pay * s * (1-s)
            dz = (y / p - (1 - y) / (1 - p)) * pay * s * (1 - s)
            w += rate * (x.T @ dz / len(y) - penalty * w)
        return cls(w)

    def to_dict(self) -> dict[str, float]:
        return dict(zip(FEATURES, (float(v) for v in self.weights), strict=True))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ResponseModel":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.array([raw[name] for name in FEATURES], dtype=np.float64))


def brier(p: Sequence[float] | np.ndarray, y: Sequence[bool] | np.ndarray) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss(p: Sequence[float] | np.ndarray, y: Sequence[bool] | np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
