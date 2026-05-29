#!/usr/bin/env python3
"""Run serving benchmark (direct script entry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.benchmark_runner import run_serving_benchmark
from deploybench.config import load_hardware_config
from deploybench.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving benchmark")
    parser.add_argument("--config", "-c", default="configs/benchmark_matrix.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--hardware-config", default=None)
    parser.add_argument("--output-dir", "-o", default="results/serving")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    hw_path = Path(args.hardware_config) if args.hardware_config else None
    hw = load_hardware_config(hw_path) if hw_path else None
    run_serving_benchmark(
        matrix_path=Path(args.config),
        models_path=Path(args.models_config),
        output_dir=Path(args.output_dir),
        hardware_path=hw_path,
        hardware_config=hw,
        cli_args=["python", "scripts/run_serving_benchmark.py"],
    )


if __name__ == "__main__":
    main()
