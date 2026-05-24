"""
Surrogate / null models for phase-connectivity validation.

We implement three standard families and one targeted control.
All take an epochs array X with shape (n_epochs, n_channels, n_times)
and return an array of the same shape (sampling frequency `sfreq` is
needed only by the spectral-domain surrogate).

Surrogates implemented
----------------------

1. ``trial_shuffle`` — break across-epoch phase consistency by permuting
   epoch indices independently per channel. Preserves single-channel
   spectra exactly; destroys true inter-channel phase coupling. This is
   the workhorse null for wPLI/PLV across-epoch estimators (Vinck 2011).

2. ``circular_time_shift`` — independent circular shift per channel
   (per epoch). Preserves within-channel autocorrelation/spectrum and
   approximately preserves single-channel statistics; destroys phase
   alignment across channels. Lemm et al., 2011; Theiler et al., 1992.

3. ``phase_randomization`` (a.k.a. Fourier-transform / IAAFT-lite
   surrogate) — randomize Fourier phases of each channel independently.
   Preserves the amplitude spectrum exactly; destroys cross-channel
   phase relationships. Theiler et al., 1992.

4. ``block_resample`` — non-overlapping block bootstrap of epochs
   (optional, used for variance estimation of the *observed* statistic,
   not as an H0 null).

Notes
-----
- The right null depends on the question. For "is the observed wPLI
  larger than chance given the marginal spectra?", phase_randomization
  or circular_time_shift are the appropriate choices. For "is the
  observed across-epoch phase coupling larger than what would arise from
  the same epochs paired at random?", trial_shuffle is the right null.
- All RNG seeded via numpy.random.Generator for reproducibility.
"""

from __future__ import annotations

import numpy as np


def _as_generator(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def trial_shuffle(
    X: np.ndarray, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """Independent permutation of epoch order, per channel.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_times) ndarray
    rng : seed or numpy Generator

    Returns
    -------
    X_surr : same shape as X
    """
    if X.ndim != 3:
        raise ValueError(f"X must be (n_epochs, n_channels, n_times); got {X.shape}")
    rng = _as_generator(rng)
    n_ep, n_ch, _ = X.shape
    out = np.empty_like(X)
    for c in range(n_ch):
        order = rng.permutation(n_ep)
        out[:, c, :] = X[order, c, :]
    return out


def circular_time_shift(
    X: np.ndarray,
    rng: int | np.random.Generator | None = None,
    min_shift_frac: float = 0.05,
) -> np.ndarray:
    """Independent circular time shift per (epoch, channel).

    Shifts are drawn uniformly in [min_shift_frac * n_times, n_times - 1]
    to avoid near-identity shifts.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D; got {X.shape}")
    rng = _as_generator(rng)
    n_ep, n_ch, n_t = X.shape
    low = max(1, int(min_shift_frac * n_t))
    shifts = rng.integers(low=low, high=n_t, size=(n_ep, n_ch))
    out = np.empty_like(X)
    for e in range(n_ep):
        for c in range(n_ch):
            out[e, c] = np.roll(X[e, c], shifts[e, c])
    return out


def phase_randomization(
    X: np.ndarray, rng: int | np.random.Generator | None = None
) -> np.ndarray:
    """Fourier phase-randomization per (epoch, channel).

    Preserves the amplitude spectrum of each channel exactly; replaces
    phases with iid uniform draws (with the Hermitian-symmetry constraint
    required to keep the inverse FFT real).
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D; got {X.shape}")
    rng = _as_generator(rng)
    n_ep, n_ch, n_t = X.shape

    Xf = np.fft.rfft(X, axis=-1)
    amp = np.abs(Xf)
    # Draw random phases for each freq bin; keep DC (and Nyquist if n_t even)
    # phases at 0 to preserve real-valued mean.
    rand_phase = rng.uniform(-np.pi, np.pi, size=Xf.shape)
    rand_phase[..., 0] = 0.0
    if n_t % 2 == 0:
        # Nyquist bin must be real for the inverse FFT to stay real.
        rand_phase[..., -1] = 0.0
    Xf_surr = amp * np.exp(1j * rand_phase)
    return np.fft.irfft(Xf_surr, n=n_t, axis=-1)


def block_resample(
    X: np.ndarray,
    block_size: int = 1,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Non-overlapping block bootstrap over the epoch axis.

    Used for variance / confidence-interval estimation, not for H0.
    """
    if X.ndim != 3:
        raise ValueError(f"X must be 3D; got {X.shape}")
    rng = _as_generator(rng)
    n_ep = X.shape[0]
    n_blocks = int(np.ceil(n_ep / block_size))
    starts = rng.integers(0, n_ep - block_size + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_ep]
    return X[idx]


SURROGATE_REGISTRY = {
    "trial_shuffle": trial_shuffle,
    "circular_time_shift": circular_time_shift,
    "phase_randomization": phase_randomization,
}


def make_surrogate(name: str, X: np.ndarray, rng=None, **kwargs) -> np.ndarray:
    """Dispatch a surrogate by name."""
    if name not in SURROGATE_REGISTRY:
        raise ValueError(
            f"Unknown surrogate '{name}'. Options: {sorted(SURROGATE_REGISTRY)}"
        )
    return SURROGATE_REGISTRY[name](X, rng=rng, **kwargs)
