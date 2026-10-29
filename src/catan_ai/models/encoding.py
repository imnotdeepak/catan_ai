"""Versioned observation encoders.

Each feature version appends blocks to the one before, so a checkpoint or
dataset keeps the exact encoding it was built with, and each version is a
clean ablation step over the last:

- v1: base player-view encoding (`features.py`, Release 0.2);
- v2: + board analysis (`board_features.py`, STR-016).
"""

from typing import Final

import numpy as np

from catan_ai.engine import CanonicalAction
from catan_ai.models.board_features import (
    BOARD_ACTION_DIM,
    BOARD_STATE_DIM,
    board_actions,
    board_state,
)
from catan_ai.models.features import (
    ACTION_DIM,
    STATE_DIM,
    BoardContext,
    encode_actions,
    encode_state,
)
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import analyze

FEATURE_VERSIONS: Final = (1, 2)
LATEST_FEATURE_VERSION: Final = FEATURE_VERSIONS[-1]


def dims(version: int) -> tuple[int, int]:
    """(state width, per-action width) of a feature version."""
    check(version)
    if version == 1:
        return STATE_DIM, ACTION_DIM
    return STATE_DIM + BOARD_STATE_DIM, ACTION_DIM + BOARD_ACTION_DIM


def check(version: int) -> None:
    if version not in FEATURE_VERSIONS:
        raise ValueError(f"Unknown feature version {version}; known: {FEATURE_VERSIONS}")


def encode(
    observation: PlayerObservation,
    version: int = LATEST_FEATURE_VERSION,
    actions: tuple[CanonicalAction, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(state [S], candidate actions [K, A]); candidates default to the legal actions."""
    check(version)
    candidates = observation.legal_actions if actions is None else actions
    ctx = BoardContext(observation)
    state = encode_state(observation, ctx)
    rows = encode_actions(observation, candidates, ctx)
    if version >= 2:
        analysis = analyze(observation)
        state = np.concatenate([state, board_state(observation, analysis)])
        rows = np.hstack([rows, board_actions(observation, analysis, candidates)])
    return state, rows
