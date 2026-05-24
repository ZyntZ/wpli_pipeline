"""Validate connectivity estimators against ground-truth simulations.

wPLI must suppress zero-lag coupling; PLV must not. This is the basic
correctness check from Vinck et al. (2011).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from wpli_pipeline.connectivity import (
    plv_from_csd, pli_from_csd, wpli_from_csd, dwpli_from_csd,
)
from wpli_pipeline.simulate import simulate_coupled_pair


def _narrow_csd(X, sfreq=250.0, freq=10.0, bw=2.0):
    sos = butter(4, [freq - bw, freq + bw], btype="bandpass", fs=sfreq, output="sos")
    Xf = sosfiltfilt(sos, X, axis=-1)
    Z = hilbert(Xf, axis=-1)
    return np.mean(Z[:, 0, :] * np.conj(Z[:, 1, :]), axis=-1)


def test_independent_oscillators_low():
    X = simulate_coupled_pair(n_epochs=100, coupling=0.0, noise=0.5, rng=0)
    csd = _narrow_csd(X)
    assert plv_from_csd(csd) < 0.3
    assert wpli_from_csd(csd) < 0.3


def test_phase_lagged_detected_by_all():
    X = simulate_coupled_pair(n_epochs=100, coupling=1.0, phase_lag=np.pi / 4,
                              noise=0.5, rng=0)
    csd = _narrow_csd(X)
    assert plv_from_csd(csd) > 0.9
    assert wpli_from_csd(csd) > 0.9
    assert pli_from_csd(csd) > 0.9


def test_zero_lag_suppressed_by_wpli():
    X = simulate_coupled_pair(n_epochs=100, coupling=1.0, phase_lag=0.0,
                              noise=0.5, rng=0)
    csd = _narrow_csd(X)
    assert plv_from_csd(csd) > 0.9
    assert pli_from_csd(csd) < 0.2
    assert wpli_from_csd(csd) < 0.2
    assert abs(dwpli_from_csd(csd)) < 0.2
