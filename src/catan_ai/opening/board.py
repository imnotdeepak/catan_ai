"""Public board analysis (STR-016, plan §4.3).

Everything here is computed from a `PlayerObservation`, so it is public
information plus the board's static geometry:

- production: dice-weighted yield of every node, per resource, with and
  without the robber's tile;
- numbers: which dice totals pay each node (same-number sites are correlated);
- ports and settlement legality under the distance rule;
- road network: for each seat, how many new roads it needs to reach each node,
  respecting other players' roads and buildings;
- longest road: each seat's current longest road, and what a new road would make it.

Production is in pips: the number of the 36 dice outcomes that pay out (a 6 or
8 tile is 5 pips). A settlement collects one card per pip, a city two.
"""

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

from catan_ai.engine.actions import RESOURCES
from catan_ai.engine.topology import EDGES, NUM_NODES, NUM_SEATS, PORT_KINDS, TILE_NODES, pips
from catan_ai.observations import PlayerObservation

UNREACHABLE: Final = 99
Edge = tuple[int, int]

NODE_NEIGHBORS: Final[dict[int, tuple[int, ...]]] = {
    node: tuple(sorted({b for a, b in EDGES if a == node} | {a for a, b in EDGES if b == node}))
    for node in range(NUM_NODES)
}
NODE_TILES: Final[dict[int, tuple[tuple[int, int, int], ...]]] = {
    node: tuple(sorted(c for c, nodes in TILE_NODES.items() if node in nodes))
    for node in range(NUM_NODES)
}


@dataclass(frozen=True)
class BoardAnalysis:
    viewer: int
    production: np.ndarray  # [nodes, resources] pips, ignoring the robber
    blocked_production: np.ndarray  # [nodes, resources] pips on the robber's tile
    numbers: tuple[frozenset[int], ...]  # dice totals that pay each node
    port: np.ndarray  # [nodes] index into PORT_KINDS, -1 for none
    owner: np.ndarray  # [nodes] building owner seat, -1 for empty
    is_city: np.ndarray  # [nodes] bool
    buildable: np.ndarray  # [nodes] empty and no adjacent building (distance rule)
    road_owner: dict[Edge, int]
    distance: np.ndarray  # [seats, nodes] new roads needed to reach a node
    longest_road: tuple[int, ...]  # by seat

    @property
    def pips(self) -> np.ndarray:
        """Total production per node, all resources."""
        return self.production.sum(axis=1)

    def reachable_sites(self, seat: int, within: int) -> np.ndarray:
        """Buildable nodes `seat` can reach with at most `within` new roads."""
        return np.flatnonzero(self.buildable & (self.distance[seat] <= within))

    def distance_after_road(self, seat: int, edge: Edge) -> np.ndarray:
        roads = {**self.road_owner, edge: seat}
        return _distances(seat, roads, self.owner)

    def longest_road_after(self, seat: int, edge: Edge) -> int:
        return _longest_road(seat, {**self.road_owner, edge: seat}, self.owner)


def analyze(observation: PlayerObservation) -> BoardAnalysis:
    board = observation.board
    production = np.zeros((NUM_NODES, len(RESOURCES)), dtype=np.float32)
    blocked = np.zeros_like(production)
    numbers: list[set[int]] = [set() for _ in range(NUM_NODES)]
    for coord, resource, number in board.tiles:
        if resource is None or number is None:
            continue
        for node in TILE_NODES[coord]:
            production[node, RESOURCES.index(resource)] += pips(number)
            numbers[node].add(number)
            if coord == board.robber:
                blocked[node, RESOURCES.index(resource)] += pips(number)

    port = np.full(NUM_NODES, -1, dtype=np.int8)
    for resource, nodes in board.ports:
        port[list(nodes)] = PORT_KINDS.index(resource)

    owner = np.full(NUM_NODES, -1, dtype=np.int8)
    is_city = np.zeros(NUM_NODES, dtype=bool)
    for node, seat, kind in board.buildings:
        owner[node] = seat
        is_city[node] = kind == "CITY"
    occupied = owner >= 0
    near_building = np.array([occupied[list(NODE_NEIGHBORS[n])].any() for n in range(NUM_NODES)])
    buildable = ~occupied & ~near_building

    roads = {(a, b): seat for a, b, seat in board.roads}
    return BoardAnalysis(
        viewer=observation.seat,
        production=production,
        blocked_production=blocked,
        numbers=tuple(frozenset(n) for n in numbers),
        port=port,
        owner=owner,
        is_city=is_city,
        buildable=buildable,
        road_owner=roads,
        distance=np.stack([_distances(s, roads, owner) for s in range(NUM_SEATS)]),
        longest_road=tuple(_longest_road(s, roads, owner) for s in range(NUM_SEATS)),
    )


def _edge(a: int, b: int) -> Edge:
    return (a, b) if a < b else (b, a)


def _distances(seat: int, roads: dict[Edge, int], owner: np.ndarray) -> np.ndarray:
    """0-1 BFS: own roads cost nothing, a new road costs one. A road network cannot
    continue through another player's building or along another player's road."""
    distance = np.full(NUM_NODES, UNREACHABLE, dtype=np.int16)
    start = {n for n in range(NUM_NODES) if owner[n] == seat}
    start |= {n for edge, s in roads.items() if s == seat for n in edge}
    queue: deque[int] = deque()
    for node in start:
        distance[node] = 0
        queue.append(node)
    while queue:
        node = queue.popleft()
        if owner[node] >= 0 and owner[node] != seat:
            continue  # blocked: can arrive here but not pass through
        for neighbor in NODE_NEIGHBORS[node]:
            holder = roads.get(_edge(node, neighbor))
            if holder is not None and holder != seat:
                continue
            cost = 0 if holder == seat else 1
            if distance[node] + cost < distance[neighbor]:
                distance[neighbor] = distance[node] + cost
                if cost == 0:
                    queue.appendleft(neighbor)
                else:
                    queue.append(neighbor)
    return distance


def _longest_road(seat: int, roads: dict[Edge, int], owner: np.ndarray) -> int:
    """Longest trail of `seat`'s roads. Another player's building interrupts it:
    pieces touching the building count, but the trail cannot pass through it.

    The engine sometimes leaves out a piece that touches another player's
    building, depending on build order (rules matrix Q-05).
    """
    blocked = {n for n in range(NUM_NODES) if owner[n] >= 0 and owner[n] != seat}
    adjacent: dict[int, list[Edge]] = {}
    for edge, s in roads.items():
        if s == seat:
            for node in edge:
                adjacent.setdefault(node, []).append(edge)

    def walk(node: int, used: set[Edge]) -> int:
        if node in blocked and used:
            return 0  # arrived at another player's building: the trail ends here
        best = 0
        for edge in adjacent.get(node, ()):
            if edge not in used:
                used.add(edge)
                other = edge[1] if edge[0] == node else edge[0]
                best = max(best, 1 + walk(other, used))
                used.remove(edge)
        return best

    return max((walk(node, set()) for node in adjacent), default=0)
