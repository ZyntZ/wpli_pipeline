# wPLI Validation Pipeline

A small, reproducible MNE-Python pipeline for **weighted Phase Lag Index (wPLI) /
debiased wPLI / PLV / PLI** with **permutation and surrogate-based statistical
controls**, BH-FDR and max-statistic FWER correction across edges and bands.

Designed as a drop-in validation layer for EEG connectivity analyses where
you need to demonstrate that observed connectivity is more than:
- bias from a finite-sample estimator,
- volume conduction (zero-lag mixing),
- or random phase drift across epochs.

## Features

- Connectivity methods: `plv`, `pli`, `wpli`, `dwpli` (debiased wPLI).
  Backed by `mne_connectivity.spectral_connectivity_epochs` and cross-checked
  against in-package reference implementations on simulated data.
- Surrogates: `trial_shuffle`, `circular_time_shift`, `phase_randomization`
  (Fourier amplitude-preserving). Each implements a different null
  hypothesis — see `surrogates.py` docstring.
- Multiple-comparison correction across edges (and bands):
  Benjamini-Hochberg FDR (Benjamini & Hochberg, 1995) and max-statistic
  permutation FWER (Nichols & Holmes, 2002).
- Config-driven runs (YAML) and a CLI entry point (`wpli-run`).
- Ground-truth simulators for unit testing and method validation.
- Dockerfile + Makefile for one-command reproducibility.

## Install

```bash
pip install -e .
# or for development
pip install -e ".[dev]"
```

## Quick start

```python
import mne
from wpli_pipeline.pipeline import PipelineConfig, BandSpec, run_pipeline

epochs = mne.read_epochs("subject-epo.fif", preload=True)
cfg = PipelineConfig(
    method="wpli",
    bands=[BandSpec("alpha", 8.0, 13.0), BandSpec("beta", 13.0, 30.0)],
    surrogate="trial_shuffle",
    n_permutations=500,
    fdr_q=0.05,
    alpha=0.05,
    n_jobs=4,
    random_seed=42,
)
res = run_pipeline(epochs, cfg, out_dir="results/subject01")
res["edges_df"].head()
```

### CLI

```bash
wpli-run --config configs/example.yaml --epochs subject-epo.fif --out results/subject01
```

### Outputs

The pipeline writes to `out_dir/`:

- `wpli_arrays.npz` — `T_obs`, `T_null`, `p_perm`, `p_fdr`, `reject_fdr`,
  `reject_fwer`, `T_crit_fwer`.
- `wpli_edges.csv` — long-form table, one row per (band, edge).
- `config.json` — full config snapshot.

## What is the right null?

| Surrogate              | Preserves                          | Destroys                       | Use when…                                                          |
|------------------------|------------------------------------|--------------------------------|---------------------------------------------------------------------|
| `trial_shuffle`        | Single-channel epoch contents      | Cross-channel epoch pairing    | Testing across-epoch wPLI/PLV ("is observed coupling > chance?")    |
| `circular_time_shift`  | Within-channel autocorrelation     | Phase alignment across channels| Testing local phase relationships while preserving spectra          |
| `phase_randomization`  | Amplitude spectrum (per channel)   | Cross-channel phase coupling   | Testing that coupling is not an artifact of marginal spectra        |

We default to `trial_shuffle` because wPLI is computed by averaging
imaginary cross-spectra across epochs; shuffling epoch order per channel
is the natural exchangeability null. The other two are appropriate when
within-epoch / within-channel temporal structure matters (e.g. evoked
responses).

## Demo on simulated EEG

`scripts/run_demo.py` simulates 8-channel EEG with:
- two phase-lagged coupled pairs (`Ch1-Ch2`, `Ch3-Ch4`), the true positives,
- one zero-lag (volume conduction) pair (`Ch5-Ch6`), the trap,
- 1/f noise and weak inter-channel leakage on top.

Result (default settings, 500 permutations):

| Edge        | T_obs (wPLI) | p_perm | BH-FDR q | FDR sig | FWER sig |
|-------------|--------------|--------|----------|---------|----------|
| Ch1-Ch2     | 0.717        | 0.002  | 0.019    | yes     | yes      |
| Ch3-Ch4     | 0.789        | 0.002  | 0.019    | yes     | yes      |
| **Ch5-Ch6** | **0.115**    | 0.40   | 0.86     | **no**  | **no**   |

The volume-conduction trap is correctly rejected by wPLI; only the
phase-lagged true edges survive both FDR and FWER correction. PLV
(which does not discount zero-lag coupling) detects all three —
including the spurious one — confirming the qualitative behavior on
Vinck et al. (2011) simulations.

See `results/demo/figures/connectivity_matrices.png` and
`results/demo/figures/null_distributions.png`.

## Reproducibility

```bash
make install      # editable install
make test         # 11 unit / integration tests
make demo         # run scripts/run_demo.py end-to-end
make docker-build # build the image
make docker-test  # run tests inside the container
```

All RNG-driven steps are seeded via `PipelineConfig.random_seed`. Per-permutation
seeds are derived deterministically from that one seed so runs are bit-exact
reproducible.

## References

- Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
  *Measuring phase synchrony in brain signals.* Human Brain Mapping, 8(4).
- Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). *Phase lag index:
  Assessment of functional connectivity from multi channel EEG and MEG
  with diminished bias from common sources.* Human Brain Mapping, 28(11).
- Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., & Pennartz,
  C. M. (2011). *An improved index of phase-synchronization for
  electrophysiological data in the presence of volume-conduction, noise
  and sample-size bias.* NeuroImage, 55(4).
- Nichols, T. E., & Holmes, A. P. (2002). *Nonparametric permutation tests
  for functional neuroimaging.* Human Brain Mapping, 15(1).
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the false discovery
  rate.* JRSS-B, 57(1).
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D.
  (1992). *Testing for nonlinearity in time series: the method of
  surrogate data.* Physica D, 58.

## Limitations

- The pipeline currently treats every channel as independent of source
  space; SNR/leakage in source-reconstructed data should be addressed
  before applying it.
- Per-permutation cost scales with the spectral-connectivity computation;
  for >64 channels and >1000 permutations you'll want `n_jobs > 1`
  and probably a cluster.
- The default `trial_shuffle` null preserves single-channel statistics
  exactly. If your data has very short epochs or strong evoked components,
  consider `circular_time_shift` or `phase_randomization` instead.
