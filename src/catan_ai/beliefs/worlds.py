"""Complete engine games consistent with one seat's observations (plan §6.2, §10.1).

A world is a fresh engine game on the observed board, driven through the
public history with every outcome forced: public dice, the viewer's own
draws and steals, and sampled values for what the viewer did not see.

- Unseen steals (between two other seats) come from one hypothesis of the
  resource belief, which already satisfies every later public resource
  change (`catan_ai.beliefs.resources`).
- Opponents' development-card draws are assigned so that every card an
  opponent later played was bought on an earlier turn; the cards still face
  down are drawn from the unseen pool with the hold-time VP posterior
  (`catan_ai.beliefs.dev_cards`).

So a world is consistent with the record by construction. `WorldSampler`
keeps a set of worlds and advances them incrementally as events arrive,
sampling new unseen outcomes in each world. A world that a later event
contradicts is replaced by a copy of a survivor; if none survives, all are
rebuilt from the whole record.

Worlds hold sampled hidden cards, never the referee's. They exist only inside
the viewer's own computation: the opponent model (OPP-025), trade evaluation
and search use them to see the game from another seat's information state.
"""

import random
from collections.abc import Callable
from typing import Any, Final

from catanatron import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.map import BASE_MAP_TEMPLATE, PORT_DIRECTION_TO_NODEREFS, CatanMap
from catanatron.models.player import SimplePlayer
from catanatron.models.tiles import LandTile, Port

from catan_ai.beliefs.dev_cards import dev_belief
from catan_ai.beliefs.particles import BeliefTracker
from catan_ai.beliefs.resources import ResourceBelief
from catan_ai.engine import SEAT_COLORS, ActionKind, CanonicalAction, EngineAdapter
from catan_ai.engine.actions import RESOURCES
from catan_ai.observations import DEV_CARDS, PlayerObservation
from catan_ai.observations.events import PLAY_KINDS, EventType, VisibleEvent

DEFAULT_WORLDS: Final = 8
MAX_ASSIGNMENTS: Final = 200


class Contradiction(Exception):
    """A world's sampled past contradicts the public record."""


def board_map(observation: PlayerObservation) -> CatanMap:
    """The engine map of the observed board."""
    template = CatanMap.from_template(BASE_MAP_TEMPLATE, rng=random.Random(0))
    tiles = dict(template.tiles)
    land = {tuple(c): (resource, number) for c, resource, number in observation.board.tiles}
    # Observed ports are grouped by resource (the four 3:1 ports share one entry);
    # match each template port by a land node it serves.
    port_of = {node: resource for resource, nodes in observation.board.ports for node in nodes}
    for coord, tile in tiles.items():
        if isinstance(tile, LandTile):
            tile.resource, tile.number = land[tuple(coord)]
        elif isinstance(tile, Port):
            served, _ = PORT_DIRECTION_TO_NODEREFS[tile.direction]
            tile.resource = port_of[tile.nodes[served]]
    return CatanMap.from_tiles(tiles)


def new_world(observation: PlayerObservation, seed: int) -> EngineAdapter:
    players = [SimplePlayer(color) for color in SEAT_COLORS]
    game = Game(players, seed=seed, catan_map=board_map(observation))
    state = game.state
    state.players = players
    state.colors = tuple(player.color for player in players)
    state.color_to_index = {color: seat for seat, color in enumerate(state.colors)}
    game.playable_actions = generate_playable_actions(state)
    return EngineAdapter(game)


def advance(
    world: EngineAdapter,
    event: VisibleEvent,
    rng: random.Random,
    forced: Any = None,
) -> None:
    """Apply one recorded event to a world. Outcomes the viewer did not see are
    `forced`, or else sampled in this world."""
    if event.actor is None:  # awards and game end follow from the actions
        return
    payload = event.public["action"]
    action = CanonicalAction(ActionKind(payload["kind"]), payload["seat"], tuple(payload["params"]))
    outcome: Any = None
    if action.kind == ActionKind.ROLL:
        outcome = tuple(event.public["dice"])
    elif action.kind == ActionKind.BUY_DEV_CARD:
        if event.private is not None:
            outcome = event.private["dev_card"]
        elif forced is not None:
            outcome = forced
        else:
            deck = world.referee_game.state.development_listdeck
            if not deck:
                raise Contradiction("bought from an empty deck")
            outcome = rng.choice(deck)
    elif action.kind == ActionKind.MOVE_ROBBER and "resource_delta" not in event.public:
        if event.private is not None:
            outcome = event.private["stolen_resource"]
        elif forced is not None:
            outcome = forced
        else:
            outcome = _random_card(world, action.params[1], rng)
    before = _hands(world)
    # A known or forced outcome this world's sampled past cannot produce is a contradiction.
    stolen = action.kind == ActionKind.MOVE_ROBBER and outcome is not None
    if stolen and before[action.params[1]][RESOURCES.index(outcome)] == 0:
        raise Contradiction("the victim lacks the stolen card in this world")
    deck = world.referee_game.state.development_listdeck
    if action.kind == ActionKind.BUY_DEV_CARD and outcome not in deck:
        raise Contradiction("the drawn card is not in this world's deck")
    if not world.is_legal(action):
        raise Contradiction(f"illegal in this world: {action.key}")
    world.execute(action, outcome)
    if "resource_delta" in event.public:
        after = _hands(world)
        expected = event.public["resource_delta"]
        for seat in range(4):
            change = tuple(a - b for a, b in zip(after[seat], before[seat], strict=True))
            if change != tuple(expected[seat]):
                raise Contradiction("a public resource change differs in this world")


def _hands(world: EngineAdapter) -> list[tuple[int, ...]]:
    ps = world.referee_game.state.player_state
    return [tuple(int(ps[f"P{s}_{r}_IN_HAND"]) for r in RESOURCES) for s in range(4)]


def _random_card(world: EngineAdapter, seat: int, rng: random.Random) -> str:
    hand = _hands(world)[seat]
    cards = [r for r, n in zip(RESOURCES, hand, strict=True) for _ in range(n)]
    if not cards:
        raise Contradiction("a steal from an empty hand")
    return rng.choice(cards)


def _unseen_steal(event: VisibleEvent) -> bool:
    return (
        event.event_type == ActionKind.MOVE_ROBBER
        and "resource_delta" not in event.public
        and event.private is None
    )


def dev_assignment(observation: PlayerObservation, rng: random.Random) -> dict[int, str]:
    """A card for every development card an opponent bought, keyed by event sequence id.

    Every card an opponent played is assigned to one of its purchases from an
    earlier turn; cards still face down get a VP count from the hold-time
    posterior, the rest drawn from the unseen non-VP pool. An assignment that
    would have put an opponent at 10 points at any moment of the game so far
    (it would have won) is drawn again.
    """
    timeline = _visible_points(observation)
    over = any(e.event_type == EventType.GAME_END for e in observation.history)
    assigned: dict[int, str] = {}
    for _ in range(MAX_ASSIGNMENTS):
        try:
            assigned = _draw_assignment(observation, rng)
        except Contradiction:
            continue
        if over or _never_won(observation, assigned, timeline):
            return assigned
    return assigned


def _draw_assignment(observation: PlayerObservation, rng: random.Random) -> dict[int, str]:
    belief = dev_belief(observation)
    unseen = {card: int(n) for card, n in zip(DEV_CARDS, belief.unseen, strict=True)}
    assigned: dict[int, str] = {}
    pending: dict[int, list[tuple[int, int]]] = {}  # seat -> [(sequence id, turn bought)]
    for event in observation.history:
        seat = event.actor
        if seat is None or seat == observation.seat:
            continue
        if event.event_type == ActionKind.BUY_DEV_CARD:
            pending.setdefault(seat, []).append((event.sequence_id, event.turn_index))
        elif event.event_type in PLAY_KINDS:
            card = event.public["dev_card_played"]
            eligible = [p for p in pending.get(seat, []) if p[1] < event.turn_index]
            if not eligible:
                raise Contradiction(f"seat {seat} played {card} with no earlier purchase")
            pick = rng.choice(eligible)
            pending[seat].remove(pick)
            assigned[pick[0]] = card
    for seat, cards in sorted(pending.items()):
        vp = belief.hidden_vp_distribution(seat)
        others = [c for c, n in unseen.items() if c != "VICTORY_POINT" for _ in range(n)]
        low = max(0, len(cards) - len(others))  # not enough other cards left: some are VP
        high = min(len(cards), unseen["VICTORY_POINT"])
        if low > high:
            raise Contradiction("the unseen cards cannot fill this seat's hand")
        k = min(max(rng.choices(range(len(vp)), weights=list(vp))[0], low), high)
        unseen["VICTORY_POINT"] -= k
        drawn = rng.sample(others, len(cards) - k)
        for card in drawn:
            unseen[card] -= 1
        types = ["VICTORY_POINT"] * k + drawn
        rng.shuffle(types)
        for (sequence, _), card in zip(cards, types, strict=True):
            assigned[sequence] = card
    return assigned


def _visible_points(observation: PlayerObservation) -> list[tuple[int, ...]]:
    """Each seat's visible points after each event (buildings and awards)."""
    points = [0] * 4
    timeline = []
    for event in observation.history:
        if event.event_type in (ActionKind.BUILD_SETTLEMENT, ActionKind.BUILD_CITY):
            points[event.actor] += 1  # type: ignore[index]  # a city replaces 1 point with 2
        elif event.event_type == EventType.BONUS_TRANSFER:
            if event.public["from_seat"] is not None:
                points[event.public["from_seat"]] -= 2
            if event.public["to_seat"] is not None:
                points[event.public["to_seat"]] += 2
        timeline.append(tuple(points))
    return timeline


def _never_won(
    observation: PlayerObservation, assigned: dict[int, str], timeline: list[tuple[int, ...]]
) -> bool:
    held = [0] * 4
    for index, event in enumerate(observation.history):
        if assigned.get(event.sequence_id) == "VICTORY_POINT" and event.actor is not None:
            held[event.actor] += 1
        if any(timeline[index][s] + held[s] >= 10 for s in range(4) if s != observation.seat):
            return False
    return True


def build_world(
    observation: PlayerObservation, belief: ResourceBelief, rng: random.Random
) -> EngineAdapter:
    """A world consistent with the whole record (see the module docstring)."""
    pick = rng.choices(range(belief.size), weights=list(belief.weights))[0]
    steals = [RESOURCES[r] for r in belief.steals[pick]]
    draws = dev_assignment(observation, rng)
    world = new_world(observation, rng.randrange(2**31))
    unseen = 0
    for event in observation.history:
        forced: Any = None
        if event.event_type == ActionKind.BUY_DEV_CARD and event.private is None:
            forced = draws.get(event.sequence_id)
        elif _unseen_steal(event):
            forced = steals[unseen]
            unseen += 1
        advance(world, event, rng, forced)
    return world


class WorldSampler:
    """`count` worlds for `seat`, kept consistent with its observations."""

    def __init__(self, seat: int, count: int = DEFAULT_WORLDS, seed: int = 0) -> None:
        self.seat = seat
        self.count = count
        self.rng = random.Random(seed)
        self.rebuilds = 0
        self.replacements = 0
        self.tracker = BeliefTracker(seat, seed=seed)
        self._worlds: list[EngineAdapter] = []
        self._seen = 0
        self._last: VisibleEvent | None = None

    @property
    def worlds(self) -> list[EngineAdapter]:
        return self._worlds

    def update(
        self,
        observation: PlayerObservation,
        before: Callable[[VisibleEvent, list[EngineAdapter]], None] | None = None,
    ) -> list[EngineAdapter]:
        """Advance to the observation. `before(event, worlds)` sees each new event with
        the worlds as they were just before it (not for events skipped by a rebuild)."""
        if observation.seat != self.seat:
            raise ValueError(f"sampler for seat {self.seat} got seat {observation.seat}")
        belief = self.tracker.update(observation)
        history = observation.history
        restart = not self._worlds or self._seen > len(history)
        if not restart and self._seen and history[self._seen - 1] is not self._last:
            restart = True
        if restart:
            self._rebuild(observation, belief)
            return self._worlds
        for event in history[self._seen :]:
            if before is not None:
                before(event, self._worlds)
            survivors = []
            for world in self._worlds:
                try:
                    advance(world, event, self.rng)
                    survivors.append(world)
                except Contradiction:
                    continue
            if not survivors:
                self._rebuild(observation, belief)
                return self._worlds
            while len(survivors) < self.count:
                self.replacements += 1
                source = survivors[self.rng.randrange(len(survivors))]
                survivors.append(source.branch(self.rng.randrange(2**31)))
            self._worlds = survivors
            self._seen += 1
            self._last = event
        return self._worlds

    def _rebuild(self, observation: PlayerObservation, belief: ResourceBelief) -> None:
        self.rebuilds += 1
        self._worlds = [build_world(observation, belief, self.rng) for _ in range(self.count)]
        self._seen = len(observation.history)
        self._last = observation.history[-1] if observation.history else None
