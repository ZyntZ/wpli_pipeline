"""Reference estimators on hand-built cross-spectra."""

import numpy as np

from wpli_pipeline.connectivity import (
    plv_from_csd, pli_from_csd, wpli_from_csd, dwpli_from_csd,
)


def _csd_with_phase(phi, n=200, amp_jitter=0.0, rng=None):
    """Synthesize a complex CSD with a fixed phase difference."""
    rng = rng or np.random.default_rng(0)
    amps = 1.0 + amp_jitter * rng.standard_normal(n)
    return amps * np.exp(1j * (phi + 0.0 * rng.standard_normal(n)))


def test_plv_full_locking():
    csd = _csd_with_phase(np.pi / 4)
    assert plv_from_csd(csd) > 0.99


def test_wpli_zero_lag_suppressed():
    csd = _csd_with_phase(0.0, amp_jitter=0.0)
    # zero-lag: Im part is ~0 -> wPLI ~ 0
    assert wpli_from_csd(csd) < 0.1


def test_wpli_phase_lagged_high():
    csd = _csd_with_phase(np.pi / 3)
    assert wpli_from_csd(csd) > 0.99


def test_pli_phase_lagged_high_zero_lag_low():
    assert pli_from_csd(_csd_with_phase(np.pi / 3)) > 0.99
    # Im sign is undefined at exactly 0; use a tiny near-zero CSD
    rng = np.random.default_rng(0)
    csd_zero = np.exp(1j * (0.0 + 0.01 * rng.standard_normal(500)))
    assert pli_from_csd(csd_zero) < 0.2


def test_dwpli_bounded():
    csd = _csd_with_phase(np.pi / 4)
    val = dwpli_from_csd(csd)
    assert -1.0 <= val <= 1.0
