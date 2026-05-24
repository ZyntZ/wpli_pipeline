#!/usr/bin/env python
"""
Demo: end-to-end run of the wPLI validation pipeline on simulated EEG data
with a designed ground truth (two phase-lagged pairs and one volume-conduction
trap pair). Writes figures and a results table.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import mne

from wpli_pipeline.simulate import simulate_eeg_like, add_volume_conduction
from wpli_pipeline.pipeline import BandSpec, PipelineConfig, run_pipeline
from wpli_pipeline.plotting import plot_connectivity_matrix, plot_null_distribution


def main(out_dir: str = "results/demo"):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Simulate
    X, truth = simulate_eeg_like(
        n_epochs=80, n_channels=8, n_times=500, sfreq=250.0,
        freq=10.0, snr=1.5, rng=0,
        true_edges=((0, 1, np.pi / 4), (2, 3, np.pi / 3)),
        zero_lag_edges=((4, 5),),
    )
    X = add_volume_conduction(X, leakage=0.2)
    ch_names = [f"Ch{c+1}" for c in range(X.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    epochs = mne.EpochsArray(X, info, verbose="ERROR")

    # 2. Run pipeline (wPLI + trial_shuffle null)
    cfg = PipelineConfig(
        method="wpli",
        bands=[BandSpec("alpha", 8.0, 13.0)],
        surrogate="trial_shuffle",
        n_permutations=500,
        alpha=0.05,
        fdr_q=0.05,
        n_jobs=4,
        random_seed=0,
        mode="multitaper",
        mt_bandwidth=4.0,
    )
    res = run_pipeline(epochs, cfg, out_dir=out_dir)
    df = res["edges_df"]

    # 3. Plot observed and FDR-significant connectivity
    iu = np.triu_indices(len(ch_names), k=1)
    indices = (iu[0], iu[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_connectivity_matrix(
        res["T_obs"][0], ch_names, indices, title="Observed wPLI",
        ax=axes[0], vmin=0, vmax=1,
    )
    plot_connectivity_matrix(
        res["T_obs"][0], ch_names, indices,
        title="wPLI surviving FDR (q=0.05)", ax=axes[1],
        vmin=0, vmax=1, mask=res["reject_fdr"][0],
    )
    plot_connectivity_matrix(
        res["T_obs"][0], ch_names, indices,
        title="wPLI surviving FWER (max-stat alpha=0.05)", ax=axes[2],
        vmin=0, vmax=1, mask=res["reject_fwer"][0],
    )
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "connectivity_matrices.png"), dpi=150)
    plt.close(fig)

    # 4. Null distribution for an example true-positive vs the volume-conduction pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    tp_idx = df[(df.ch_i == "Ch1") & (df.ch_j == "Ch2")].index[0]
    vc_idx = df[(df.ch_i == "Ch5") & (df.ch_j == "Ch6")].index[0]
    plot_null_distribution(
        res["T_obs"][0, tp_idx], res["T_null"][:, 0, tp_idx],
        ax=axes[0], label="Ch1-Ch2 (true phase-lagged)",
    )
    plot_null_distribution(
        res["T_obs"][0, vc_idx], res["T_null"][:, 0, vc_idx],
        ax=axes[1], label="Ch5-Ch6 (zero-lag / volume conduction)",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "null_distributions.png"), dpi=150)
    plt.close(fig)

    # 5. Print summary
    tp = df[((df.ch_i == "Ch1") & (df.ch_j == "Ch2"))
            | ((df.ch_i == "Ch3") & (df.ch_j == "Ch4"))]
    vc = df[(df.ch_i == "Ch5") & (df.ch_j == "Ch6")]
    print("\nTrue phase-lagged edges (should be detected):")
    print(tp[["ch_i", "ch_j", "T_obs", "p_perm", "p_fdr", "reject_fdr", "reject_fwer"]])
    print("\nVolume-conduction edge (should NOT be detected):")
    print(vc[["ch_i", "ch_j", "T_obs", "p_perm", "p_fdr", "reject_fdr", "reject_fwer"]])
    print(f"\nTotal edges flagged at FDR q=0.05: {df.reject_fdr.sum()}/{len(df)}")
    print(f"Total edges flagged at FWER alpha=0.05: {df.reject_fwer.sum()}/{len(df)}")
    print(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
