"""Permutation p-values, BH-FDR, max-stat threshold."""

import numpy as np

from wpli_pipeline.stats import (
    edgewise_p_values, benjamini_hochberg, max_stat_threshold,
)


def test_pvalue_bounds():
    T_obs = np.array([0.5, 0.1])
    T_null = np.zeros((100, 2))
    T_null[:, 0] = np.linspace(0, 1, 100)
    T_null[:, 1] = np.linspace(0, 1, 100)
    p = edgewise_p_values(T_obs, T_null)
    assert np.all(p > 0) and np.all(p <= 1)
    assert 0.4 < p[0] < 0.6
    assert p[1] > 0.8


def test_pvalue_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        edgewise_p_values(np.array([0.5, 0.1]), np.zeros((100, 3)))


def test_bh_fdr_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200)
    p[:10] = 0.0001
    reject, p_adj = benjamini_hochberg(p, q=0.05)
    assert np.all((p_adj >= 0) & (p_adj <= 1))
    assert reject[:10].sum() >= 5


def test_max_stat_threshold():
    rng = np.random.default_rng(0)
    T_null = rng.uniform(size=(1000, 50))
    T_crit, max_null = max_stat_threshold(T_null, alpha=0.05)
    assert 0.9 < T_crit <= 1.0
    assert max_null.shape == (1000,)
