"""Wilson interval sanity checks."""

import pytest

from catan_ai.evaluation import wilson_interval


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(25, 100)
    assert low == pytest.approx(0.1754, abs=1e-3)
    assert high == pytest.approx(0.3430, abs=1e-3)


def test_wilson_interval_edges() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(0, 50)
    assert low == 0.0 and 0 < high < 0.1
    low, high = wilson_interval(50, 50)
    assert 0.9 < low < 1 and high == 1.0
