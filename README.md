# wPLI Validation Pipeline

Weighted Phase Lag Index (wPLI) connectivity pipeline with permutation
and surrogate controls.

## Implemented

- Reference connectivity estimators (`plv`, `pli`, `wpli`, `dwpli`).
- Surrogate / null generators (`trial_shuffle`, `circular_time_shift`,
  `phase_randomization`).
- Multiple-comparison correction: permutation p-values, BH-FDR,
  max-statistic FWER.
- Ground-truth EEG-like simulators (`simulate_coupled_pair`,
  `simulate_eeg_like`, `add_volume_conduction`) for unit testing and
  method validation.

## Install (dev)

```bash
pip install -e ".[dev]"
pytest -q
```
