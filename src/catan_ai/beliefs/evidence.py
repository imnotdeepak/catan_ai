"""Choice evidence: reweighting resource hypotheses by what a player did not do (plan §6.2).

The ledger weights an unseen steal only by the victim's cards: the card is
drawn at random. But a player's choices also say something about its hand.
A player that ends its turn holding a city's cost, with a settlement to
upgrade, usually builds the city; one holding four of a kind usually trades
them. So when an opponent ends its turn, every hypothesis under which it could
have made such a move is multiplied by that move's miss rate: how often a
player who could, did not.

The miss rates are fitted on the `dev` split by `catan_ai.cli.beliefs_report
--fit-evidence` and shared by every viewer. Legality is approximated from
public counts (pieces left, a settlement to upgrade, cards left in the deck),
not from the board, so the fitted rates absorb the cases where a site was not
reachable. A rate of 1 is no evidence, so `NO_EVIDENCE` reproduces the plain
ledger; no hypothesis is ever eliminated, only reweighted.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from catan_ai.observations import PlayerObservation

# RESOURCES order: WOOD, BRICK, SHEEP, WHEAT, ORE
CITY: Final = np.array([0, 0, 0, 2, 3])
SETTLEMENT: Final = np.array([1, 1, 1, 1, 0])
DEV_CARD: Final = np.array([0, 0, 1, 1, 1])
BANK_RATE: Final = 4
SETTLEMENTS: Final = 5


@dataclass(frozen=True)
class ChoiceEvidence:
    """P(a player that could make the move ended its turn without it)."""

    city: float = 1.0
    settlement: float = 1.0
    dev_card: float = 1.0
    maritime: float = 1.0

    @property
    def active(self) -> bool:
        return min(self.city, self.settlement, self.dev_card, self.maritime) < 1.0

    def factors(self, hands: np.ndarray, observation: PlayerObservation, seat: int) -> np.ndarray:
        """Likelihood of ending the turn, per hypothesis; `hands` is [K, resources] for `seat`."""
        player = observation.players[seat]
        factor = np.ones(len(hands))
        on_board = SETTLEMENTS - player.settlements_available
        if player.cities_available > 0 and on_board > 0:
            factor *= np.where((hands >= CITY).all(axis=1), self.city, 1.0)
        if player.settlements_available > 0:
            factor *= np.where((hands >= SETTLEMENT).all(axis=1), self.settlement, 1.0)
        if observation.dev_deck_remaining > 0:
            factor *= np.where((hands >= DEV_CARD).all(axis=1), self.dev_card, 1.0)
        factor *= np.where((hands >= BANK_RATE).any(axis=1), self.maritime, 1.0)
        return factor


NO_EVIDENCE: Final = ChoiceEvidence()
# Fitted on `dev` by `beliefs_report --fit-evidence` (reports/beliefs.md).
DEFAULT_EVIDENCE: Final = NO_EVIDENCE
