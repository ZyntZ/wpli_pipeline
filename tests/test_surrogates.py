"""Tests for surrogate / null generators."""

import numpy as np

from wpli_pipeline.surrogates import (
    trial_shuffle, circular_time_shift, phase_randomization, make_surrogate,
)


def test_trial_shuffle_preserves_marginals():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4, 200))
    Xs = trial_shuffle(X, rng=1)
    for c in range(4):
        assert np.allclose(np.sort(X[:, c].ravel()), np.sort(Xs[:, c].ravel()))


def test_phase_randomization_preserves_amplitude_spectrum():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 3, 256))
    Xs = phase_randomization(X, rng=1)
    A = np.abs(np.fft.rfft(X, axis=-1))
    As = np.abs(np.fft.rfft(Xs, axis=-1))
    np.testing.assert_allclose(A, As, atol=1e-8)


def test_phase_randomization_real_output():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 2, 128))
    Xs = phase_randomization(X, rng=2)
    assert np.isrealobj(Xs) or np.max(np.abs(np.imag(Xs))) < 1e-9


def test_circular_shift_preserves_value_set_per_channel():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3, 100))
    Xs = circular_time_shift(X, rng=1)
    for e in range(5):
        for c in range(3):
            assert np.allclose(np.sort(X[e, c]), np.sort(Xs[e, c]))


def test_make_surrogate_dispatch():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2, 64))
    for name in ("trial_shuffle", "circular_time_shift", "phase_randomization"):
        Xs = make_surrogate(name, X, rng=0)
        assert Xs.shape == X.shape


def test_unknown_surrogate_raises():
    import pytest
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2, 64))
    with pytest.raises(ValueError):
        make_surrogate("not_a_surrogate", X)
