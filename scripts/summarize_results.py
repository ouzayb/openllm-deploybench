#!/usr/bin/env python3
"""Summarize benchmark results (direct script entry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.analysis import run_summarize
from deploybench.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize results")
    parser.add_argument("--results-dir", "-r", default="results")
    parser.add_argument("--output", default=None, help="Legacy: single CSV path")
    parser.add_argument("--output-dir", "-o", default="reports")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    out_dir = Path(args.output_dir)
    outputs = run_summarize(Path(args.results_dir), out_dir)
    if args.output:
        import shutil
        shutil.copy(outputs["serving"], Path(args.output))
    for p in outputs.values():
        print(p)


if __name__ == "__main__":
    main()
