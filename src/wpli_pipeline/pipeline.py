"""
End-to-end wPLI validation pipeline.

Workflow
--------

1. Load epoched EEG (`mne.Epochs`) or pass an Epochs object directly.
2. Compute observed connectivity (PLV / PLI / wPLI / dwPLI) per band
   via `mne_connectivity.spectral_connectivity_epochs`.
3. For each surrogate type and each of N permutations:
     a. Generate surrogate epochs.
     b. Recompute connectivity.
     c. Store per-edge statistic in the null distribution.
4. Compute permutation p-values, BH-FDR, and max-stat FWER thresholds.
5. Return a structured result object and write CSV / NPZ / JSON outputs.

The whole thing is config-driven so a paper's results can be regenerated
from the YAML alone.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .surrogates import make_surrogate
from .stats import benjamini_hochberg, edgewise_p_values, max_stat_threshold

logger = logging.getLogger(__name__)


# --- We import mne lazily so the rest of the package is testable without mne ---
def _import_mne():
    import mne
    import mne_connectivity as mc
    return mne, mc


@dataclass
class BandSpec:
    name: str
    fmin: float
    fmax: float


@dataclass
class PipelineConfig:
    method: str = "wpli"  # "plv", "pli", "wpli", "wpli2_debiased", "dwpli" aliases
    bands: list[BandSpec] = field(
        default_factory=lambda: [
            BandSpec("theta", 4.0, 8.0),
            BandSpec("alpha", 8.0, 13.0),
            BandSpec("beta", 13.0, 30.0),
        ]
    )
    surrogate: str = "trial_shuffle"
    n_permutations: int = 200
    alpha: float = 0.05
    fdr_q: float = 0.05
    n_jobs: int = 1
    random_seed: int = 0
    mode: str = "multitaper"  # "multitaper" | "fourier" | "cwt_morlet"
    mt_bandwidth: float | None = None
    # If method == "dwpli", we translate to mne-connectivity's "wpli2_debiased"
    _mne_method_map = {
        "plv": "plv",
        "pli": "pli",
        "wpli": "wpli",
        "dwpli": "wpli2_debiased",
        "wpli2_debiased": "wpli2_debiased",
    }

    def mne_method(self) -> str:
        return self._mne_method_map.get(self.method, self.method)


@dataclass
class EdgeResult:
    band: str
    ch_i: str
    ch_j: str
    T_obs: float
    p_perm: float
    p_fdr: float
    reject_fdr: bool
    reject_fwer: bool


def _compute_connectivity(epochs, cfg: PipelineConfig, mc, verbose: bool = False):
    """Return (n_bands, n_edges) array of connectivity values and edge list.

    Uses indices=all-pairs upper triangle for an n_ch x n_ch connectivity
    matrix, so n_edges = n_ch * (n_ch - 1) / 2.
    """
    n_ch = epochs.info["nchan"]
    iu = np.triu_indices(n_ch, k=1)
    indices = (iu[0].astype(int), iu[1].astype(int))

    fmin = [b.fmin for b in cfg.bands]
    fmax = [b.fmax for b in cfg.bands]

    con = mc.spectral_connectivity_epochs(
        epochs,
        method=cfg.mne_method(),
        mode=cfg.mode,
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        indices=indices,
        mt_bandwidth=cfg.mt_bandwidth,
        verbose="ERROR" if not verbose else None,
    )
    data = con.get_data()  # (n_edges, n_bands)
    # Transpose to (n_bands, n_edges)
    return np.asarray(data).T, indices


def _stat_for_data(X: np.ndarray, info, cfg: PipelineConfig, mc, mne):
    """Wrap an ndarray (n_ep, n_ch, n_times) as Epochs and compute connectivity."""
    epochs = mne.EpochsArray(X, info, verbose="ERROR")
    T, _ = _compute_connectivity(epochs, cfg, mc, verbose=False)
    # For dwpli the statistic can go negative; we work on its absolute value
    # for null testing of "any phase coupling", matching wPLI convention.
    if cfg.mne_method() == "wpli2_debiased":
        T = np.abs(T)
    return T


def _one_permutation(X: np.ndarray, info, cfg: PipelineConfig, seed: int):
    mne, mc = _import_mne()
    rng = np.random.default_rng(seed)
    Xs = make_surrogate(cfg.surrogate, X, rng=rng)
    return _stat_for_data(Xs, info, cfg, mc, mne)


def run_pipeline(epochs, cfg: PipelineConfig, out_dir: str | None = None):
    """Run the full wPLI validation pipeline on an `mne.Epochs` object.

    Parameters
    ----------
    epochs : mne.Epochs
    cfg : PipelineConfig
    out_dir : optional directory to write artifacts

    Returns
    -------
    dict with keys:
        T_obs           (n_bands, n_edges)
        T_null          (n_perm, n_bands, n_edges)
        p_perm          (n_bands, n_edges)
        p_fdr           (n_bands, n_edges)
        reject_fdr      bool array, same shape
        reject_fwer     bool array, same shape
        T_crit_fwer     list[float], per band
        edges           list[(ch_i, ch_j)]
        bands           list[str]
        edges_df        long-form pandas DataFrame
        config          dict snapshot
    """
    mne, mc = _import_mne()

    info = epochs.info
    X = epochs.get_data(copy=True)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs data of shape (n_ep, n_ch, n_t); got {X.shape}")

    # 1. Observed
    T_obs, indices = _compute_connectivity(epochs, cfg, mc, verbose=False)
    if cfg.mne_method() == "wpli2_debiased":
        T_obs = np.abs(T_obs)

    n_bands, n_edges = T_obs.shape
    logger.info("Observed connectivity computed: %d bands x %d edges", n_bands, n_edges)

    # 2. Null
    seeds = np.random.default_rng(cfg.random_seed).integers(
        0, 2**31 - 1, size=cfg.n_permutations
    )
    if cfg.n_jobs == 1:
        nulls = [_one_permutation(X, info, cfg, int(s)) for s in seeds]
    else:
        nulls = Parallel(n_jobs=cfg.n_jobs, verbose=0)(
            delayed(_one_permutation)(X, info, cfg, int(s)) for s in seeds
        )
    T_null = np.stack(nulls, axis=0)  # (n_perm, n_bands, n_edges)

    # 3. Edgewise p + FDR (per band, then combined across bands)
    p_perm = np.empty_like(T_obs)
    for b in range(n_bands):
        p_perm[b] = edgewise_p_values(T_obs[b], T_null[:, b, :])

    reject_fdr, p_fdr = benjamini_hochberg(p_perm, q=cfg.fdr_q)

    # 4. Max-stat FWER per band
    T_crit = []
    reject_fwer = np.zeros_like(T_obs, dtype=bool)
    for b in range(n_bands):
        crit, _ = max_stat_threshold(T_null[:, b, :], alpha=cfg.alpha)
        T_crit.append(crit)
        reject_fwer[b] = T_obs[b] > crit

    # 5. Tidy long-form table
    ch_names = epochs.ch_names
    i_idx, j_idx = indices
    rows = []
    for b, band in enumerate(cfg.bands):
        for e in range(n_edges):
            rows.append(
                dict(
                    band=band.name,
                    ch_i=ch_names[i_idx[e]],
                    ch_j=ch_names[j_idx[e]],
                    T_obs=float(T_obs[b, e]),
                    p_perm=float(p_perm[b, e]),
                    p_fdr=float(p_fdr[b, e]),
                    reject_fdr=bool(reject_fdr[b, e]),
                    reject_fwer=bool(reject_fwer[b, e]),
                    T_crit_fwer=float(T_crit[b]),
                )
            )
    edges_df = pd.DataFrame(rows)

    result = dict(
        T_obs=T_obs,
        T_null=T_null,
        p_perm=p_perm,
        p_fdr=p_fdr,
        reject_fdr=reject_fdr,
        reject_fwer=reject_fwer,
        T_crit_fwer=T_crit,
        edges=list(zip([ch_names[i] for i in i_idx], [ch_names[j] for j in j_idx])),
        bands=[b.name for b in cfg.bands],
        edges_df=edges_df,
        config=asdict(cfg),
    )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, "wpli_arrays.npz"),
            T_obs=T_obs, T_null=T_null,
            p_perm=p_perm, p_fdr=p_fdr,
            reject_fdr=reject_fdr, reject_fwer=reject_fwer,
            T_crit_fwer=np.array(T_crit),
        )
        edges_df.to_csv(os.path.join(out_dir, "wpli_edges.csv"), index=False)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)

    return result
