"""Small statistics helpers for arena reports.

These intervals treat every seat-game as independent. Games on the same board
are correlated, so these understate uncertainty; clustered/bootstrap intervals
are EVAL-032's job (plan §11.1).
"""

import math


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))
