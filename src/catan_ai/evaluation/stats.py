"""Small statistics helpers for arena reports.

These intervals treat every seat-game as independent. Games on the same board
are correlated, so these understate uncertainty; `cluster_bootstrap_share`
resamples whole boards instead (plan §11.1).
"""

import math
import random
from collections.abc import Sequence


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_p_value(k: int, n: int) -> float:
    """Exact two-sided binomial test of `k` successes in `n` trials against p = 1/2."""
    if n == 0:
        return 1.0
    total = 1 << n  # exact integer arithmetic, so large n does not overflow
    probabilities = [math.comb(n, i) / total for i in range(n + 1)]
    observed = probabilities[k]
    return min(1.0, sum(p for p in probabilities if p <= observed * (1 + 1e-9)))


def cluster_bootstrap_share(
    clusters: Sequence[tuple[int, int]], resamples: int = 4000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval for sum(a) / sum(a + b), resampling whole clusters.

    Each cluster is (a, b) counts from correlated games, e.g. all games on one board.
    """
    rng = random.Random(seed)
    shares = []
    for _ in range(resamples):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        a = sum(x for x, _ in sample)
        total = a + sum(y for _, y in sample)
        if total:
            shares.append(a / total)
    shares.sort()
    if not shares:
        return (0.0, 1.0)
    return (shares[int(0.025 * len(shares))], shares[int(0.975 * len(shares)) - 1])
