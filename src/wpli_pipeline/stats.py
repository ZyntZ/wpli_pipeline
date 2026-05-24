"""
Permutation testing and multiple-comparison correction across edges.

Two complementary procedures
----------------------------

1. Edgewise permutation p-values with FDR (Benjamini-Hochberg) correction.
   For each connectivity edge we compute the observed statistic and the
   distribution of null statistics from many surrogate datasets. The
   one-sided p-value is

       p_edge = (1 + #{T_null >= T_obs}) / (1 + N_null)

   (the +1s give an exact, valid p-value under exchangeability;
   Phipson & Smyth, 2010). We then apply BH-FDR across all edges
   (and frequency bands, if multiple) at the user-chosen q.

2. Maximum-statistic permutation test (Nichols & Holmes, 2002).
   For each surrogate dataset we record the maximum of the null
   statistic across all edges. The corrected threshold T_crit is the
   1-alpha quantile of that distribution; any observed edge above
   T_crit is significant family-wise. This controls FWER strongly and
   makes no assumption about correlation structure across edges.

Both are computed and reported in the pipeline output so the user can
pick a control strategy appropriate to the application.
"""

from __future__ import annotations

import numpy as np


def edgewise_p_values(T_obs: np.ndarray, T_null: np.ndarray) -> np.ndarray:
    """One-sided permutation p-values per edge.

    Parameters
    ----------
    T_obs : (n_edges,) array of observed statistics.
    T_null : (n_perm, n_edges) array of null statistics.

    Returns
    -------
    p : (n_edges,) array of permutation p-values.
    """
    T_obs = np.asarray(T_obs)
    T_null = np.asarray(T_null)
    if T_null.ndim != 2 or T_null.shape[1] != T_obs.shape[0]:
        raise ValueError(
            f"Shape mismatch: T_obs {T_obs.shape}, T_null {T_null.shape}"
        )
    n_perm = T_null.shape[0]
    # Count >= to be conservative.
    ge = np.sum(T_null >= T_obs[None, :], axis=0)
    return (1.0 + ge) / (1.0 + n_perm)


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05):
    """Benjamini-Hochberg FDR correction.

    Returns
    -------
    reject : boolean array of same shape as pvals
    p_adj  : BH-adjusted q-values
    """
    p = np.asarray(pvals, dtype=float)
    shape = p.shape
    p_flat = p.ravel()
    n = p_flat.size
    order = np.argsort(p_flat)
    ranked = p_flat[order]
    # BH adjusted p-values
    adj = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    p_adj = np.empty_like(p_flat)
    p_adj[order] = adj
    reject = p_adj <= q
    return reject.reshape(shape), p_adj.reshape(shape)


def max_stat_threshold(
    T_null: np.ndarray, alpha: float = 0.05
) -> tuple[float, np.ndarray]:
    """Family-wise threshold via the max-statistic permutation test.

    Parameters
    ----------
    T_null : (n_perm, n_edges) array of null statistics.
    alpha : family-wise error rate.

    Returns
    -------
    T_crit : critical value (1-alpha quantile of max-null distribution).
    max_null : (n_perm,) array of max per-permutation statistics.
    """
    T_null = np.asarray(T_null)
    if T_null.ndim != 2:
        raise ValueError(f"T_null must be 2D; got {T_null.shape}")
    max_null = np.max(T_null, axis=1)
    # Conservative quantile: use (n_perm+1) denominator
    T_crit = np.quantile(max_null, 1.0 - alpha, method="higher")
    return float(T_crit), max_null
