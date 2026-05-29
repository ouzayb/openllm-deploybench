#!/usr/bin/env python3
"""Run quantization benchmark (direct script entry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.quantization import run_quantization_benchmark
from deploybench.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization benchmark")
    parser.add_argument("--config", "-c", default="configs/benchmark_matrix.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--hardware-config", default=None)
    parser.add_argument("--output-dir", "-o", default="results/quantization")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    hw = Path(args.hardware_config) if args.hardware_config else None
    run_quantization_benchmark(
        matrix_path=Path(args.config),
        models_path=Path(args.models_config),
        output_dir=Path(args.output_dir),
        hardware_path=hw,
        cli_args=["python", "scripts/run_quantization_benchmark.py"],
    )


if __name__ == "__main__":
    main()
