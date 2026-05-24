"""
Ground-truth simulators for testing the pipeline.

We need controlled data where we *know* which channel-pairs are
phase-coupled at lag != 0, and which are coupled only by zero-lag
volume conduction (which wPLI must ignore).

Three generators are provided:

- ``simulate_coupled_pair``: oscillator with controlled phase lag and
  amplitude/phase noise per epoch. Returns (n_epochs, 2, n_times).

- ``simulate_eeg_like``: 8-channel synthetic dataset with a designed
  ground-truth connectivity matrix. Each "truly connected" edge is a
  phase-shifted copy of a shared oscillator (lag = pi/4) plus channel-
  specific pink noise. One pair is connected with zero lag (the
  volume-conduction trap).

- ``add_volume_conduction``: linearly mix channels to mimic spread of a
  source over multiple sensors (creates zero-lag spurious coupling).
"""

from __future__ import annotations

import numpy as np


def simulate_coupled_pair(
    n_epochs: int = 60,
    n_times: int = 500,
    sfreq: float = 250.0,
    freq: float = 10.0,
    phase_lag: float = np.pi / 4,
    coupling: float = 1.0,
    noise: float = 0.5,
    rng: int | np.random.Generator | None = 0,
):
    """Two-channel coupled oscillator with fixed phase lag.

    coupling=1 -> deterministic phase lag (modulo amplitude/noise).
    coupling=0 -> independent oscillators (no phase coupling).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    t = np.arange(n_times) / sfreq
    X = np.zeros((n_epochs, 2, n_times))
    for e in range(n_epochs):
        phi0 = rng.uniform(-np.pi, np.pi)
        c1 = np.cos(2 * np.pi * freq * t + phi0)
        if coupling >= 1:
            c2 = np.cos(2 * np.pi * freq * t + phi0 + phase_lag)
        else:
            phi1 = rng.uniform(-np.pi, np.pi) if coupling == 0 else (
                phi0 + phase_lag
                + (1 - coupling) * rng.uniform(-np.pi, np.pi)
            )
            c2 = np.cos(2 * np.pi * freq * t + phi1)
        X[e, 0] = c1 + noise * rng.standard_normal(n_times)
        X[e, 1] = c2 + noise * rng.standard_normal(n_times)
    return X


def _pink_noise(n_times: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f noise via colored FFT."""
    white = rng.standard_normal(n_times)
    f = np.fft.rfftfreq(n_times, d=1.0)
    spec = np.fft.rfft(white)
    scale = 1.0 / np.sqrt(np.maximum(f, f[1]))
    spec = spec * scale
    return np.fft.irfft(spec, n=n_times)


def simulate_eeg_like(
    n_epochs: int = 80,
    n_channels: int = 8,
    n_times: int = 500,
    sfreq: float = 250.0,
    freq: float = 10.0,
    snr: float = 1.5,
    rng: int | np.random.Generator | None = 0,
    true_edges=((0, 1, np.pi / 4), (2, 3, np.pi / 3)),
    zero_lag_edges=((4, 5),),
):
    """Synthetic n-channel EEG-like dataset with a known ground truth.

    Parameters
    ----------
    true_edges : iterable of (i, j, phase_lag)
        Phase-lagged coupled pairs.
    zero_lag_edges : iterable of (i, j)
        Zero-lag coupled pairs (these should be detected by PLV but
        suppressed by wPLI -- they are the volume-conduction control).

    Returns
    -------
    X : (n_epochs, n_channels, n_times) ndarray
    truth : dict with keys "phase_lagged" and "zero_lag" -- sets of
        edges (i<j) representing the true positives for wPLI.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    t = np.arange(n_times) / sfreq
    X = np.zeros((n_epochs, n_channels, n_times))

    for e in range(n_epochs):
        # Independent pink noise per channel
        for c in range(n_channels):
            X[e, c] = _pink_noise(n_times, rng)

        # Phase-lagged coupled pairs share a common oscillator phase
        for (i, j, lag) in true_edges:
            phi0 = rng.uniform(-np.pi, np.pi)
            osc_i = np.cos(2 * np.pi * freq * t + phi0)
            osc_j = np.cos(2 * np.pi * freq * t + phi0 + lag)
            X[e, i] += snr * osc_i
            X[e, j] += snr * osc_j

        # Zero-lag pairs: same oscillator (mimics volume conduction)
        for (i, j) in zero_lag_edges:
            phi0 = rng.uniform(-np.pi, np.pi)
            osc = np.cos(2 * np.pi * freq * t + phi0)
            X[e, i] += snr * osc
            X[e, j] += snr * osc

    truth = dict(
        phase_lagged={tuple(sorted((i, j))) for (i, j, _) in true_edges},
        zero_lag={tuple(sorted((i, j))) for (i, j) in zero_lag_edges},
    )
    return X, truth


def add_volume_conduction(X: np.ndarray, leakage: float = 0.3) -> np.ndarray:
    """Linearly mix neighboring channels to simulate volume conduction.

    out[c] = (1 - leakage) * X[c] + (leakage/2) * (X[c-1] + X[c+1])
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D; got {X.shape}")
    n_ch = X.shape[1]
    out = X.copy()
    for c in range(n_ch):
        left = X[:, (c - 1) % n_ch, :]
        right = X[:, (c + 1) % n_ch, :]
        out[:, c, :] = (1 - leakage) * X[:, c, :] + (leakage / 2) * (left + right)
    return out
