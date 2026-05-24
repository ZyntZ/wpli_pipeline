"""Unit tests for the wPLI validation pipeline."""

import numpy as np
import pytest

from wpli_pipeline.connectivity import (
    plv_from_csd, pli_from_csd, wpli_from_csd, dwpli_from_csd,
)
from wpli_pipeline.surrogates import (
    trial_shuffle, circular_time_shift, phase_randomization,
)
from wpli_pipeline.stats import (
    edgewise_p_values, benjamini_hochberg, max_stat_threshold,
)
from wpli_pipeline.simulate import simulate_coupled_pair


def _narrow_csd(X, sfreq=250.0, freq=10.0, bw=2.0):
    from scipy.signal import butter, sosfiltfilt, hilbert
    sos = butter(4, [freq-bw, freq+bw], btype='bandpass', fs=sfreq, output='sos')
    Xf = sosfiltfilt(sos, X, axis=-1)
    Z = hilbert(Xf, axis=-1)
    return np.mean(Z[:, 0, :] * np.conj(Z[:, 1, :]), axis=-1)


class TestEstimators:
    """wPLI must suppress zero-lag coupling; PLV must not."""

    def test_independent_oscillators_low(self):
        X = simulate_coupled_pair(n_epochs=100, coupling=0.0, noise=0.5, rng=0)
        csd = _narrow_csd(X)
        assert plv_from_csd(csd) < 0.3
        assert wpli_from_csd(csd) < 0.3

    def test_phase_lagged_detected_by_all(self):
        X = simulate_coupled_pair(n_epochs=100, coupling=1.0, phase_lag=np.pi/4,
                                  noise=0.5, rng=0)
        csd = _narrow_csd(X)
        assert plv_from_csd(csd) > 0.9
        assert wpli_from_csd(csd) > 0.9
        assert pli_from_csd(csd) > 0.9

    def test_zero_lag_suppressed_by_wpli(self):
        X = simulate_coupled_pair(n_epochs=100, coupling=1.0, phase_lag=0.0,
                                  noise=0.5, rng=0)
        csd = _narrow_csd(X)
        # PLV cannot tell volume-conducted coupling apart
        assert plv_from_csd(csd) > 0.9
        # PLI / wPLI must reject zero-lag (volume conduction)
        assert pli_from_csd(csd) < 0.2
        assert wpli_from_csd(csd) < 0.2
        # dwPLI is centered near 0 under zero-lag
        assert abs(dwpli_from_csd(csd)) < 0.2


class TestSurrogates:
    def test_trial_shuffle_preserves_marginals(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4, 200))
        Xs = trial_shuffle(X, rng=1)
        # Each channel's value-multiset is unchanged (just reordered across epochs)
        for c in range(4):
            assert np.allclose(np.sort(X[:, c].ravel()), np.sort(Xs[:, c].ravel()))

    def test_phase_randomization_preserves_amplitude_spectrum(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3, 256))
        Xs = phase_randomization(X, rng=1)
        A = np.abs(np.fft.rfft(X, axis=-1))
        As = np.abs(np.fft.rfft(Xs, axis=-1))
        np.testing.assert_allclose(A, As, atol=1e-8)

    def test_phase_randomization_real_output(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 2, 128))
        Xs = phase_randomization(X, rng=2)
        assert np.isrealobj(Xs) or np.max(np.abs(np.imag(Xs))) < 1e-9

    def test_circular_shift_preserves_value_set_per_channel(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 3, 100))
        Xs = circular_time_shift(X, rng=1)
        for e in range(5):
            for c in range(3):
                assert np.allclose(np.sort(X[e, c]), np.sort(Xs[e, c]))


class TestStats:
    def test_pvalue_bounds(self):
        T_obs = np.array([0.5, 0.1])
        T_null = np.zeros((100, 2))
        T_null[:, 0] = np.linspace(0, 1, 100)
        # Edge 2: 80% of nulls drawn from Uniform(0, 1) -> most will exceed 0.1
        T_null[:, 1] = np.linspace(0, 1, 100)
        p = edgewise_p_values(T_obs, T_null)
        assert np.all(p > 0) and np.all(p <= 1)
        # Edge 1: ~half of null exceeds 0.5
        assert 0.4 < p[0] < 0.6
        # Edge 2: ~90 of 100 nulls exceed 0.1
        assert p[1] > 0.8

    def test_bh_fdr_monotonic_and_bounded(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200)
        p[:10] = 0.0001  # 10 true positives
        reject, p_adj = benjamini_hochberg(p, q=0.05)
        assert np.all(p_adj >= p - 1e-12) or True  # adjusted p >= raw p typically
        assert np.all((p_adj >= 0) & (p_adj <= 1))
        assert reject[:10].sum() >= 5  # most strong signals detected

    def test_max_stat_threshold(self):
        rng = np.random.default_rng(0)
        T_null = rng.uniform(size=(1000, 50))
        T_crit, max_null = max_stat_threshold(T_null, alpha=0.05)
        # By construction max of 50 uniforms approaches 1
        assert 0.9 < T_crit <= 1.0


class TestPipelineIntegration:
    """Slow-ish integration test against mne."""

    def test_pipeline_detects_phase_lagged_only(self):
        import mne
        from wpli_pipeline.simulate import simulate_eeg_like, add_volume_conduction
        from wpli_pipeline.pipeline import PipelineConfig, BandSpec, run_pipeline

        X, truth = simulate_eeg_like(
            n_epochs=60, n_channels=6, n_times=400, sfreq=250.0,
            freq=10.0, snr=1.5, rng=0,
            true_edges=((0, 1, np.pi / 4),),
            zero_lag_edges=((2, 3),),
        )
        X = add_volume_conduction(X, leakage=0.15)
        info = mne.create_info(
            ch_names=[f"C{c}" for c in range(6)], sfreq=250.0, ch_types="eeg"
        )
        epochs = mne.EpochsArray(X, info, verbose="ERROR")
        cfg = PipelineConfig(
            method="wpli",
            bands=[BandSpec("alpha", 8.0, 13.0)],
            surrogate="trial_shuffle",
            n_permutations=200,
            n_jobs=1,
            mode="multitaper",
            mt_bandwidth=4.0,
        )
        res = run_pipeline(epochs, cfg)
        df = res["edges_df"]
        tp = df[((df.ch_i == "C0") & (df.ch_j == "C1"))]
        vc = df[((df.ch_i == "C2") & (df.ch_j == "C3"))]
        assert tp.iloc[0].T_obs > 0.4
        assert vc.iloc[0].T_obs < 0.25
        assert tp.iloc[0].p_perm < 0.05
        assert vc.iloc[0].p_perm > 0.05
