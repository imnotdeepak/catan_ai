"""Opponents' face-down development cards (plan §6, development-card belief).

The viewer knows the deck's composition, its own cards and every card anyone
has played. The rest, the unseen pool, is split between the deck and the
opponents' face-down cards. Cards are drawn at random, so before any play
each opponent's hidden cards are an exchangeable sample from the pool:
hypergeometric.

Holding is evidence. VP cards are never played, while other cards tend to be
played once they can be, so cards held unplayed for many turns are more
likely VP cards. And an opponent still playing holds fewer VP cards than
would bring it to 10 points: it would already have won. A non-VP card is
either hoarded (held on purpose, with probability `hoard`) or played at rate
`play_rate` per turn its owner could have played it and did not; so after
`e` such turns it is still held with probability hoard + (1 - hoard)(1 -
play_rate)^e. Exposure is counted per card (played cards leave first-in,
first-out) and averaged over the hand. Both parameters are fitted on
recorded games (`reports/beliefs.md`); other opponents may play faster or
slower, which the opponent model (OPP-025) can refine.
"""

from dataclasses import dataclass
from math import comb
from typing import Final

import numpy as np

from catan_ai.engine import ActionKind
from catan_ai.observations import DEV_CARDS, PlayerObservation
from catan_ai.observations.events import PLAY_KINDS

DECK: Final = {"KNIGHT": 14, "YEAR_OF_PLENTY": 2, "MONOPOLY": 2, "ROAD_BUILDING": 2,
               "VICTORY_POINT": 5}  # fmt: skip
DECK_COUNTS: Final = np.array([DECK[c] for c in DEV_CARDS])
VP: Final = DEV_CARDS.index("VICTORY_POINT")
# Fitted on the imitation-v3 dev split (ValueFunction and AlphaBeta games), reports/beliefs.md.
# play_rate = 0 and hoard = 0 give the plain hypergeometric.
PLAY_RATE: Final = 0.9
HOARD: Final = 0.1


@dataclass(frozen=True)
class DevBelief:
    unseen: np.ndarray  # [DEV_CARDS] cards the viewer has not seen
    hidden: np.ndarray  # [seats] face-down cards per seat (0 for the viewer)
    exposure: np.ndarray  # [seats] card-turns held, playable, with no card played
    viewer: int
    play_rate: float = PLAY_RATE
    hoard: float = HOARD
    # Most VP cards each seat can hold: one more would have won the game (default: no cap).
    max_hidden_vp: np.ndarray | None = None

    @property
    def pool_size(self) -> int:
        return int(self.unseen.sum())

    def hidden_vp_distribution(self, seat: int) -> np.ndarray:
        """P(`seat` holds exactly k VP cards face down), k = 0..its face-down count."""
        n, total, good = int(self.hidden[seat]), self.pool_size, int(self.unseen[VP])
        if total == 0 or n == 0:
            return np.ones(1)
        prior = np.array(
            [comb(good, k) * comb(total - good, n - k) for k in range(n + 1)], dtype=np.float64
        )
        per_card = float(self.exposure[seat]) / n
        survive = self.hoard + (1 - self.hoard) * (1.0 - self.play_rate) ** per_card
        likelihood = np.array([survive ** (n - k) for k in range(n + 1)])
        posterior = prior * likelihood
        if self.max_hidden_vp is not None:
            capped = posterior.copy()
            capped[int(self.max_hidden_vp[seat]) + 1 :] = 0.0
            if capped.sum() > 0:  # at the game's end the winner may hold more
                posterior = capped
        return np.asarray(posterior / posterior.sum())

    def expected(self, seat: int) -> np.ndarray:
        """Expected count of each card type among `seat`'s face-down cards."""
        out = np.zeros(len(DEV_CARDS))
        n = int(self.hidden[seat])
        if n == 0 or self.pool_size == 0:
            return out
        vp = self.hidden_vp_distribution(seat)
        expected_vp = float((np.arange(len(vp)) * vp).sum())
        others = self.unseen.astype(np.float64).copy()
        others[VP] = 0
        if others.sum() > 0:
            out += (n - expected_vp) * others / others.sum()
        out[VP] = expected_vp
        return out

    def prob_holds(self, seat: int, card: str, at_least: int = 1) -> float:
        """P(`seat` holds at least `at_least` of `card` face down)."""
        n = int(self.hidden[seat])
        vp = self.hidden_vp_distribution(seat)
        if card == "VICTORY_POINT":
            return float(vp[at_least:].sum())
        good = int(self.unseen[DEV_CARDS.index(card)])
        rest = int(self.unseen.sum() - self.unseen[VP])
        total = 0.0
        for k, p_k in enumerate(vp):
            draws = n - k  # non-VP cards, a uniform sample of the non-VP pool
            if draws <= 0 or rest == 0:
                total += p_k * (at_least <= 0)
                continue
            below = sum(
                comb(good, j) * comb(rest - good, draws - j)
                for j in range(min(at_least, draws + 1))
            )
            total += p_k * (1.0 - below / comb(rest, draws))
        return total


def hold_exposure(observation: PlayerObservation, seat: int) -> float:
    """Card-turns `seat`'s face-down cards were held, playable, on its completed turns
    without a development card being played. Played cards leave first-in, first-out."""
    bought: dict[int, int] = {}  # turn -> cards bought
    played_turns: set[int] = set()
    own_turns: set[int] = set()
    for event in observation.history:
        if event.actor != seat:
            continue
        own_turns.add(event.turn_index)
        if event.event_type == ActionKind.BUY_DEV_CARD:
            bought[event.turn_index] = bought.get(event.turn_index, 0) + 1
        elif event.event_type in PLAY_KINDS:
            played_turns.add(event.turn_index)
    held: list[list[float]] = []  # [purchase turn, exposure] per card, oldest first
    for turn in sorted(own_turns | set(bought)):
        playable = [card for card in held if card[0] < turn]
        completed = turn < observation.num_turns or observation.turn_seat != seat
        if turn in played_turns:
            if playable:
                held.remove(playable[0])
        elif completed:
            for card in playable:
                card[1] += 1
        held.extend([float(turn), 0.0] for _ in range(bought.get(turn, 0)))
    return float(sum(card[1] for card in held))


def dev_belief(
    observation: PlayerObservation, play_rate: float = PLAY_RATE, hoard: float = HOARD
) -> DevBelief:
    played = np.sum([p.played_dev_cards for p in observation.players], axis=0)
    own = np.asarray(observation.own_dev_cards)
    unseen = DECK_COUNTS - played - own
    if (unseen < 0).any():
        raise AssertionError("more cards seen than the deck holds")
    hidden = np.array(
        [0 if p.seat == observation.seat else p.dev_card_count for p in observation.players]
    )
    if int(unseen.sum()) != observation.dev_deck_remaining + int(hidden.sum()):
        raise AssertionError("unseen cards do not match the deck plus opponents' hands")
    exposure = np.array(
        [
            0.0 if s == observation.seat else hold_exposure(observation, s)
            for s in range(len(hidden))
        ]
    )
    # At 10 points a seat wins at once on its own turn, so no opponent holds that many.
    cap = np.array([max(0, 9 - p.visible_victory_points) for p in observation.players])
    return DevBelief(unseen, hidden, exposure, observation.seat, play_rate, hoard, cap)
