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


def test_binomial_p_value() -> None:
    from catan_ai.evaluation import binomial_p_value

    assert binomial_p_value(5, 10) == pytest.approx(1.0)
    assert binomial_p_value(0, 10) == pytest.approx(2 / 1024)
    assert binomial_p_value(60, 100) == pytest.approx(0.0569, abs=1e-3)
    assert binomial_p_value(40, 100) == binomial_p_value(60, 100)


def test_cluster_bootstrap_share_widens_with_correlation() -> None:
    from catan_ai.evaluation import cluster_bootstrap_share

    independent = [(1, 0)] * 30 + [(0, 1)] * 30
    clustered = [(10, 0)] * 3 + [(0, 10)] * 3
    low_i, high_i = cluster_bootstrap_share(independent)
    low_c, high_c = cluster_bootstrap_share(clustered)
    assert low_i < 0.5 < high_i
    assert high_c - low_c > high_i - low_i
