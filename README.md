# wPLI Validation Pipeline

Weighted Phase Lag Index (wPLI) connectivity pipeline with permutation
and surrogate controls.

## Implemented

- Reference connectivity estimators (`plv`, `pli`, `wpli`, `dwpli`).
- Surrogate / null generators (`trial_shuffle`, `circular_time_shift`,
  `phase_randomization`).
- Multiple-comparison correction:
  - One-sided permutation p-values with the +1 / +1 correction
    (Phipson & Smyth, 2010).
  - Benjamini-Hochberg FDR (Benjamini & Hochberg, 1995).
  - Max-statistic permutation FWER (Nichols & Holmes, 2002).

## Install (dev)

```bash
pip install -e ".[dev]"
pytest -q
```
