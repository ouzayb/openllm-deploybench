#!/usr/bin/env python3
"""Generate plots (direct script entry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.plotting import run_plot
from deploybench.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument("--results-dir", "-r", default="results")
    parser.add_argument("--output-dir", "-o", default="reports/figures")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    created = run_plot(Path(args.results_dir), Path(args.output_dir))
    for p in created:
        print(p)


if __name__ == "__main__":
    main()
