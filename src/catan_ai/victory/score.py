"""Exact victory accounting from a player's observation (STR-019, plan §5).

The planner (STR-020) builds on these facts, so each is checked against the
engine in `tests/victory/test_score.py`:

- score sources: settlement 1, city 2, Longest Road 2, Largest Army 2, each
  Victory Point card 1 (hidden until the game ends, but counting at once);
- winning: only on your own turn, at 10 points or more;
- pieces: 15 roads, 5 settlements, 4 cities; a city returns its settlement;
- awards: Largest Army needs 3 played knights and Longest Road 5 roads, and a
  challenger must have strictly more than the holder to take either;
- development cards: one per turn, none on the turn bought; VP cards are never played.

Opponents' VP cards are hidden, so for them only bounds are known.
"""

from dataclasses import dataclass
from typing import Final

from catan_ai.observations import DEV_CARDS, PlayerObservation
from catan_ai.opening.board import analyze

POINTS_TO_WIN: Final = 10
SETTLEMENT_POINTS: Final = 1
CITY_POINTS: Final = 2
AWARD_POINTS: Final = 2  # Longest Road, Largest Army
PIECES: Final = {"roads": 15, "settlements": 5, "cities": 4}
ARMY_MIN_KNIGHTS: Final = 3
ROAD_MIN_LENGTH: Final = 5
VP_CARDS_IN_DECK: Final = 5
KNIGHT: Final = DEV_CARDS.index("KNIGHT")
VICTORY_POINT: Final = DEV_CARDS.index("VICTORY_POINT")


@dataclass(frozen=True)
class SeatScore:
    seat: int
    settlements: int
    cities: int
    longest_road: bool
    largest_army: bool
    knights_played: int
    road_length: int  # recounted from the board; the engine's field can be stale (H-13)

    @property
    def public_points(self) -> int:
        """Points everyone can see: buildings and awards."""
        return (
            SETTLEMENT_POINTS * self.settlements
            + CITY_POINTS * self.cities
            + AWARD_POINTS * (self.longest_road + self.largest_army)
        )


def seat_scores(observation: PlayerObservation) -> tuple[SeatScore, ...]:
    buildings = observation.board.buildings
    lengths = analyze(observation).longest_road
    return tuple(
        SeatScore(
            seat=p.seat,
            settlements=sum(1 for _, s, kind in buildings if s == p.seat and kind == "SETTLEMENT"),
            cities=sum(1 for _, s, kind in buildings if s == p.seat and kind == "CITY"),
            longest_road=p.has_longest_road,
            largest_army=p.has_largest_army,
            knights_played=p.played_dev_cards[KNIGHT],
            road_length=lengths[p.seat],
        )
        for p in observation.players
    )


def own_points(observation: PlayerObservation) -> int:
    """The viewer's true score: public points plus its own hidden VP cards."""
    me = seat_scores(observation)[observation.seat]
    return me.public_points + observation.own_dev_cards[VICTORY_POINT]


def hidden_points_bound(observation: PlayerObservation, seat: int) -> int:
    """Most VP cards `seat` could hold, from its face-down cards and the deck's five.

    Only the viewer's own VP cards are excluded: other seats' cards are unknown.
    """
    if seat == observation.seat:
        return observation.own_dev_cards[VICTORY_POINT]
    unseen = VP_CARDS_IN_DECK - observation.own_dev_cards[VICTORY_POINT]
    return min(observation.players[seat].dev_card_count, unseen)


def points_range(observation: PlayerObservation, seat: int) -> tuple[int, int]:
    """(least, most) points `seat` can have, given what the viewer knows."""
    public = seat_scores(observation)[seat].public_points
    if seat == observation.seat:
        exact = own_points(observation)
        return exact, exact
    return public, public + hidden_points_bound(observation, seat)


def wins(points: int, seat: int, turn_seat: int) -> bool:
    """A seat wins only on its own turn, with at least 10 points."""
    return seat == turn_seat and points >= POINTS_TO_WIN


def pieces_left(observation: PlayerObservation, seat: int) -> dict[str, int]:
    """Pieces in `seat`'s supply, recounted from the board."""
    score = seat_scores(observation)[seat]
    roads = sum(1 for *_, s in observation.board.roads if s == seat)
    return {
        "roads": PIECES["roads"] - roads,
        "settlements": PIECES["settlements"] - score.settlements,
        "cities": PIECES["cities"] - score.cities,
    }


def takes_award(count: int, minimum: int, holder_count: int | None) -> bool:
    """Whether `count` earns an award with threshold `minimum` from its holder.

    `holder_count` is the current holder's count, or None if nobody holds it;
    a tie leaves the award with the holder.
    """
    return count >= minimum and (holder_count is None or count > holder_count)


def takes_largest_army(observation: PlayerObservation, seat: int, knights: int) -> bool:
    """Whether `seat` holds Largest Army after reaching `knights` played knights."""
    scores = seat_scores(observation)
    holder = next((s for s in scores if s.largest_army), None)
    if holder is not None and holder.seat == seat:
        return True
    return takes_award(knights, ARMY_MIN_KNIGHTS, None if holder is None else holder.knights_played)


def takes_longest_road(observation: PlayerObservation, seat: int, length: int) -> bool:
    """Whether `seat` holds Longest Road after its road grows to `length`."""
    scores = seat_scores(observation)
    holder = next((s for s in scores if s.longest_road), None)
    if holder is not None and holder.seat == seat:
        return True
    return takes_award(length, ROAD_MIN_LENGTH, None if holder is None else holder.road_length)


def bought_this_turn(observation: PlayerObservation) -> tuple[int, ...]:
    """Development cards the viewer bought this turn, by DEV_CARDS order."""
    counts = [0] * len(DEV_CARDS)
    for event in observation.history:
        if (
            event.event_type == "BUY_DEV_CARD"
            and event.actor == observation.seat
            and event.turn_index == observation.num_turns
            and event.private is not None
        ):
            counts[DEV_CARDS.index(event.private["dev_card"])] += 1
    return tuple(counts)


def playable_dev_cards(observation: PlayerObservation) -> tuple[int, ...]:
    """Cards the viewer may play now, if it is its turn, ignoring card-specific limits
    (e.g. Road Building with no legal road). VP cards are never played."""
    me = observation.players[observation.seat]
    if observation.turn_seat != observation.seat or me.has_played_dev_card_this_turn:
        return (0,) * len(DEV_CARDS)
    new = bought_this_turn(observation)
    return tuple(
        0 if i == VICTORY_POINT else held - fresh
        for i, (held, fresh) in enumerate(zip(observation.own_dev_cards, new, strict=True))
    )
