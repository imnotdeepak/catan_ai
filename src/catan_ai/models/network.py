"""Action-conditioned policy/value network (ML-013, plan §9.1).

The policy scores each legal candidate with a shared f(state, action) and takes
a masked softmax over the candidates, so the action space is whatever the
engine offers, never a fixed index. Heads, all over seats relative to the
viewer:

- value: eventual winner distribution (4 classes);
- aux_hands: opponents' current hidden hand composition (privileged training label);
- aux_points: final victory points including hidden VP cards (training label).

Two towers. The policy tower (state trunk, action encoder, scorer, aux_hands)
and the outcome tower (value trunk, value head, aux_points) share no weights.
Outcome labels are constant within a game, so an outcome tower memorizes
training games' winners within an epoch or two; kept apart, that cannot damage
the policy, and each tower is early-stopped on its own dev loss
(reports/imitation_training.md).
"""

from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from catan_ai.models.features import (
    ACTION_DIM,
    FEATURE_VERSION,
    NUM_SEATS,
    STATE_DIM,
    BoardContext,
    encode_actions,
    encode_state,
)
from catan_ai.models.interface import Evaluation
from catan_ai.observations import PlayerObservation

NUM_RESOURCES = 5


@dataclass(frozen=True)
class NetConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    hidden: int = 256
    layers: int = 3
    action_hidden: int = 128
    dropout: float = 0.0
    value_hidden: int = 128
    value_layers: int = 2
    value_dropout: float = 0.0


OUTCOME_MODULES = ("value_trunk", "value_head", "aux_points")


def is_outcome_parameter(name: str) -> bool:
    return name.split(".")[0] in OUTCOME_MODULES


def _mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in pairwise(sizes):
        layers += [nn.Linear(a, b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class PolicyValueNet(nn.Module):
    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        config = config or NetConfig()
        self.config = config
        h, ah = config.hidden, config.action_hidden
        self.trunk = _mlp([config.state_dim] + [h] * config.layers, config.dropout)
        self.action_encoder = _mlp([config.action_dim, ah, ah])
        self.scorer = nn.Sequential(nn.Linear(h + ah, ah), nn.ReLU(), nn.Linear(ah, 1))
        self.aux_hands = nn.Linear(h, (NUM_SEATS - 1) * NUM_RESOURCES)
        vh = config.value_hidden
        self.value_trunk = _mlp(
            [config.state_dim] + [vh] * config.value_layers, config.value_dropout
        )
        self.value_head = nn.Linear(vh, NUM_SEATS)
        self.aux_points = nn.Linear(vh, NUM_SEATS)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """states [B, S], actions [B, K, A] padded, mask [B, K] true for real candidates."""
        h = self.trunk(states)
        a = self.action_encoder(actions)
        pair = torch.cat([h.unsqueeze(1).expand(-1, a.shape[1], -1), a], dim=-1)
        logits = self.scorer(pair).squeeze(-1).masked_fill(~mask, float("-inf"))
        outcome = self.value_trunk(states)
        return {
            "policy_logits": logits,
            "value_logits": self.value_head(outcome),
            "aux_hands": self.aux_hands(h).view(-1, NUM_SEATS - 1, NUM_RESOURCES),
            "aux_points": self.aux_points(outcome),
        }


def save_checkpoint(path: Path, net: PolicyValueNet, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_version": FEATURE_VERSION,
            "config": asdict(net.config),
            "state_dict": net.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[PolicyValueNet, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["feature_version"] != FEATURE_VERSION:
        raise ValueError(
            f"{path} was trained on features v{blob['feature_version']}, "
            f"this code encodes v{FEATURE_VERSION}"
        )
    net = PolicyValueNet(NetConfig(**blob["config"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob["metadata"]


class LearnedPolicyValue:
    """`PolicyValue` backed by a trained network, one observation at a time."""

    def __init__(self, net: PolicyValueNet) -> None:
        self.net = net.eval()

    @torch.inference_mode()
    def evaluate(self, observation: PlayerObservation) -> Evaluation:
        ctx = BoardContext(observation)
        states = torch.from_numpy(encode_state(observation, ctx)).unsqueeze(0)
        actions = torch.from_numpy(encode_actions(observation, context=ctx)).unsqueeze(0)
        mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        out = self.net(states, actions, mask)
        priors = torch.softmax(out["policy_logits"][0], dim=0).tolist()
        relative = torch.softmax(out["value_logits"][0], dim=0).numpy()
        absolute = np.roll(relative, observation.seat)  # relative r -> seat (viewer + r) % 4
        return Evaluation(tuple(priors), tuple(float(p) for p in absolute))
