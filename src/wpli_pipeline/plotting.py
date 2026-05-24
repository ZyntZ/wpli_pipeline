"""Lightweight plotting helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_connectivity_matrix(values, ch_names, indices, title="",
                              ax=None, vmin=0, vmax=None, mask=None):
    """Render a square connectivity matrix from a vector of edge values."""
    n = len(ch_names)
    M = np.full((n, n), np.nan)
    i_idx, j_idx = indices
    for e, (i, j) in enumerate(zip(i_idx, j_idx)):
        v = values[e]
        if mask is not None and not mask[e]:
            v = np.nan
        M[i, j] = v
        M[j, i] = v
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_null_distribution(T_obs_edge, T_null_edge, ax=None, label="edge"):
    """Histogram of null + observed value for a single edge."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(T_null_edge, bins=40, alpha=0.7, label="null")
    ax.axvline(T_obs_edge, color="red", linewidth=2, label="observed")
    ax.set_xlabel("statistic")
    ax.set_ylabel("count")
    ax.set_title(f"Null distribution: {label}", fontsize=10)
    ax.legend()
    return ax
