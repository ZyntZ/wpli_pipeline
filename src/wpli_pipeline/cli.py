"""
Command-line entry point.

Usage:
    wpli-run --config configs/example.yaml --epochs path/to/epo.fif --out results/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

from .pipeline import BandSpec, PipelineConfig, run_pipeline


def _load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    bands = [BandSpec(**b) for b in raw.pop("bands", [])]
    cfg = PipelineConfig(**raw)
    if bands:
        cfg.bands = bands
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="wPLI validation pipeline.")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--epochs", required=True, help="MNE Epochs file (.fif).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import mne
    cfg = _load_config(args.config)
    epochs = mne.read_epochs(args.epochs, preload=True, verbose="ERROR")
    run_pipeline(epochs, cfg, out_dir=args.out)
    print(f"Wrote results to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
