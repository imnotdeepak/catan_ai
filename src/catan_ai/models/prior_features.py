"""Opening-prior features (features v3, STR-017), appended to the v2 encoding.

Each settlement candidate gets the six prior scores of `catan_ai.opening.priors`
(in PRIORS order), so the model can learn when each style fits instead of
following one fixed rule (`reports/openings.md`). Other actions get zeros.
"""

from typing import Final

import numpy as np

from catan_ai.engine import ActionKind, CanonicalAction
from catan_ai.observations import PlayerObservation
from catan_ai.opening.board import BoardAnalysis
from catan_ai.opening.priors import PRIORS, prior_scores

PRIOR_ACTION_DIM: Final = len(PRIORS)
PRIOR_SCALE: Final = 20.0  # typical scores are 0-20


def prior_actions(
    observation: PlayerObservation,
    analysis: BoardAnalysis,
    actions: tuple[CanonicalAction, ...],
) -> np.ndarray:
    out = np.zeros((len(actions), PRIOR_ACTION_DIM), dtype=np.float32)
    for row, action in zip(out, actions, strict=True):
        if action.kind == ActionKind.BUILD_SETTLEMENT:
            row[:] = prior_scores(analysis, observation.seat, action.params[0]) / PRIOR_SCALE
    return out
