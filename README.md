# wPLI Validation Pipeline

Weighted Phase Lag Index (wPLI) connectivity pipeline with permutation
and surrogate controls.

## Implemented

- Reference connectivity estimators (`plv`, `pli`, `wpli`, `dwpli`).
- Surrogate / null generators:
  - `trial_shuffle` — permute epochs per channel (across-epoch null).
  - `circular_time_shift` — circular shift per (epoch, channel).
  - `phase_randomization` — Fourier amplitude-preserving surrogate
    (Theiler et al., 1992).

## Install (dev)

```bash
pip install -e ".[dev]"
pytest -q
```
