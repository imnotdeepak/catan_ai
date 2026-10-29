"""Static base-map topology: tile coordinates, node ids, edges and pip counts.

It is Catanatron's base map, public and identical on every board, and is
shared by the feature encoders (`catan_ai.models`) and the board analysis
(`catan_ai.opening`) without either importing the other.
"""

from typing import Final

from catanatron.models.map import TOURNAMENT_MAP

from catan_ai.engine.actions import RESOURCES

NUM_SEATS: Final = 4
TILE_RESOURCES: Final = (*RESOURCES, None)  # None = desert
PORT_KINDS: Final = (None, *RESOURCES)  # None = 3:1

TILE_COORDS: Final = tuple(sorted(tuple(c) for c in TOURNAMENT_MAP.land_tiles))
TILE_NODES: Final = {
    tuple(c): tuple(sorted(t.nodes.values())) for c, t in TOURNAMENT_MAP.land_tiles.items()
}
EDGES: Final = tuple(
    sorted({tuple(sorted(e)) for t in TOURNAMENT_MAP.land_tiles.values() for e in t.edges.values()})
)
NUM_TILES: Final = len(TILE_COORDS)
NUM_NODES: Final = 54
NUM_EDGES: Final = len(EDGES)
TILE_INDEX: Final = {c: i for i, c in enumerate(TILE_COORDS)}
EDGE_INDEX: Final = {e: i for i, e in enumerate(EDGES)}


def pips(number: int | None) -> int:
    """Ways to roll `number` with two dice (0 for the desert)."""
    return 0 if number is None else 6 - abs(7 - number)
