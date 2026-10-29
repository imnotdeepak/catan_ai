"""Six opening strategy priors and the pip baseline (STR-017, plan §4.2).

Each prior scores a candidate settlement by the production profile the seat
would have after building it (its existing buildings plus the candidate).
They are priors, not policies: the model receives them as features, and the
`opening:` contestants use them to measure each one against the pip baseline
under the same downstream policy (plan §4.5).

Production is in pips per resource (WOOD, BRICK, SHEEP, WHEAT, ORE); see
`board.py`.
"""

from collections.abc import Callable
from typing import Final

import numpy as np

from catan_ai.engine.actions import RESOURCES
from catan_ai.engine.topology import PORT_KINDS
from catan_ai.opening.board import BoardAnalysis

WOOD, BRICK, SHEEP, WHEAT, ORE = range(len(RESOURCES))
ANY_PORT: Final = PORT_KINDS.index(None)


def production_after(analysis: BoardAnalysis, seat: int, node: int) -> np.ndarray:
    """Pips per resource from `seat`'s buildings plus a settlement at `node`."""
    owned = analysis.owner == seat
    weights = np.where(analysis.is_city, 2.0, 1.0) * owned
    total = (analysis.production * weights[:, None]).sum(axis=0)
    return np.asarray(total + analysis.production[node], dtype=np.float32)


def ports_after(analysis: BoardAnalysis, seat: int, node: int) -> set[int]:
    """Port kinds (indices into PORT_KINDS) the seat would have."""
    nodes = [*np.flatnonzero(analysis.owner == seat), node]
    return {int(analysis.port[n]) for n in nodes if analysis.port[n] >= 0}


def ore_wheat_sheep(p: np.ndarray, **_: object) -> float:
    """Cities, development cards, Largest Army."""
    return float(p[ORE] + p[WHEAT] + 0.5 * p[SHEEP] + 0.5 * min(p[ORE], p[WHEAT]))


def road_builder(p: np.ndarray, **_: object) -> float:
    """Wood and brick for fast expansion, with some sheep and wheat to settle."""
    return float(p[WOOD] + p[BRICK] + 0.5 * min(p[WOOD], p[BRICK]) + 0.25 * (p[SHEEP] + p[WHEAT]))


def city_road_hybrid(p: np.ndarray, **_: object) -> float:
    """Both economies: limited by the weaker of road-building and city-building."""
    return float(min(p[WOOD] + p[BRICK], p[ORE] + p[WHEAT]) + 0.25 * p.sum())


def specialized_port(p: np.ndarray, ports: set[int], **_: object) -> float:
    """High output of a resource the seat can trade 2:1, or a 3:1 port for its best one."""
    best = 0.0
    for kind in ports:
        if kind == ANY_PORT:
            best = max(best, 0.5 * float(p.max()))
        else:
            best = max(best, 2.0 * float(p[RESOURCES.index(PORT_KINDS[kind])]))
    return best + 0.25 * float(p.sum())


def balanced(p: np.ndarray, **_: object) -> float:
    """Production diversity: every resource covered."""
    return float(3 * np.count_nonzero(p) + 0.25 * p.sum())


def cartel(p: np.ndarray, board_total: np.ndarray, **_: object) -> float:
    """Control of a scarce resource: the largest share of the board's output of one."""
    share = np.divide(p, board_total, out=np.zeros_like(p), where=board_total > 0)
    return float(10 * share.max() + 0.25 * p.sum())


def pips(p: np.ndarray, **_: object) -> float:
    """Baseline: total production."""
    return float(p.sum())


Prior = Callable[..., float]
PRIORS: Final[dict[str, Prior]] = {
    "ows": ore_wheat_sheep,
    "road": road_builder,
    "hybrid": city_road_hybrid,
    "port": specialized_port,
    "balanced": balanced,
    "cartel": cartel,
}
RULES: Final[dict[str, Prior]] = {"pips": pips, **PRIORS}


def score(rule: str, analysis: BoardAnalysis, seat: int, node: int) -> float:
    return RULES[rule](
        production_after(analysis, seat, node),
        ports=ports_after(analysis, seat, node),
        board_total=analysis.production.sum(axis=0),
    )


def prior_scores(analysis: BoardAnalysis, seat: int, node: int) -> np.ndarray:
    """The six prior scores of a settlement at `node`, in PRIORS order."""
    p = production_after(analysis, seat, node)
    context = {
        "ports": ports_after(analysis, seat, node),
        "board_total": analysis.production.sum(axis=0),
    }
    return np.array([prior(p, **context) for prior in PRIORS.values()], dtype=np.float32)
