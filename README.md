# wPLI Validation Pipeline

Weighted Phase Lag Index (wPLI) connectivity pipeline with permutation
and surrogate controls.

## What's implemented so far

- Reference estimators (`plv`, `pli`, `wpli`, `dwpli`) from a complex
  cross-spectrum. Definitions follow Lachaux et al. (1999), Stam et al.
  (2007), and Vinck et al. (2011).

## Install (dev)

```bash
pip install -e ".[dev]"
pytest -q
```
