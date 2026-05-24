"""
Core connectivity estimators: PLV, PLI, wPLI, dwPLI.

All estimators take cross-spectral information per (channel-pair, frequency, time)
and reduce across the chosen averaging axis (typically across epochs or across
time within an epoch).

Definitions follow:
- PLV: Lachaux et al., 1999.
- PLI: Stam et al., 2007.
- wPLI / dwPLI: Vinck et al., 2011 ("An improved index of phase-synchronization
  for electrophysiological data in the presence of volume-conduction, noise and
  sample-size bias", NeuroImage).

The implementation here is intentionally simple and dependency-light so it can
be unit-tested against ground-truth simulations. For full pipeline runs we
use `mne_connectivity.spectral_connectivity_epochs` (see `pipeline.py`),
which we cross-validate against these reference implementations.
"""

from __future__ import annotations

import numpy as np


def _check_csd(csd: np.ndarray) -> None:
    if csd.ndim < 1:
        raise ValueError(f"csd must have >=1 dim (got shape {csd.shape})")


def plv_from_csd(csd: np.ndarray, axis: int = 0) -> np.ndarray:
    """Phase Locking Value from complex cross-spectrum.

    PLV = | E[ exp(i * phi_diff) ] | = | E[ csd / |csd| ] |
    """
    _check_csd(csd)
    phase = csd / (np.abs(csd) + 1e-20)
    return np.abs(np.mean(phase, axis=axis))


def pli_from_csd(csd: np.ndarray, axis: int = 0) -> np.ndarray:
    """Phase Lag Index. PLI = | E[ sign(Im(csd)) ] |"""
    _check_csd(csd)
    return np.abs(np.mean(np.sign(np.imag(csd)), axis=axis))


def wpli_from_csd(csd: np.ndarray, axis: int = 0) -> np.ndarray:
    """Weighted Phase Lag Index (Vinck et al. 2011, eq. for wPLI).

    wPLI = | E[ Im(csd) ] | / E[ |Im(csd)| ]

    Range [0, 1]. Insensitive to zero-lag (volume-conducted) coupling.
    """
    _check_csd(csd)
    num = np.abs(np.mean(np.imag(csd), axis=axis))
    den = np.mean(np.abs(np.imag(csd)), axis=axis) + 1e-20
    return num / den


def dwpli_from_csd(csd: np.ndarray, axis: int = 0) -> np.ndarray:
    """Debiased squared wPLI (Vinck et al. 2011).

    Unbiased estimator of squared wPLI; reduces small-N positive bias.
    Can be slightly negative for true zero coupling; we return the raw
    debiased value (caller may clip).
    """
    _check_csd(csd)
    imag = np.imag(csd)
    sum_im = np.sum(imag, axis=axis)
    sum_abs = np.sum(np.abs(imag), axis=axis)
    sum_sq = np.sum(imag ** 2, axis=axis)
    num = sum_im ** 2 - sum_sq
    den = sum_abs ** 2 - sum_sq + 1e-20
    return num / den
