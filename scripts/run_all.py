#!/usr/bin/env python3
"""Run full benchmark pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, extra_args: list[str]) -> int:
    script = ROOT / "scripts" / name
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--hardware-config", default="configs/hardware.local.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--config", "-c", default="configs/benchmark_matrix.yaml")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-serving", action="store_true")
    parser.add_argument("--skip-long-context", action="store_true")
    parser.add_argument("--skip-summarize", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    common = [
        "--hardware-config", args.hardware_config,
        "--models-config", args.models_config,
        "--config", args.config,
    ]

    if not args.skip_probe:
        rc = run_script("run_hardware_probe.py", [
            "--output", "results/hardware.json",
            "--hardware-config", args.hardware_config,
        ])
        if rc != 0:
            sys.exit(rc)

    if not args.skip_serving:
        rc = run_script("run_serving_benchmark.py", common + ["--output-dir", "results/serving"])
        if rc != 0:
            print("Serving benchmark exited with errors (continuing)")

    if not args.skip_long_context:
        rc = run_script("run_long_context_benchmark.py", common + ["--output-dir", "results/long_context"])
        if rc != 0:
            print("Long context benchmark exited with errors (continuing)")

    if not args.skip_summarize:
        run_script("summarize_results.py", ["--results-dir", "results", "--output-dir", "reports"])

    if not args.skip_plot:
        run_script("plot_results.py", ["--results-dir", "results", "--output-dir", "reports/figures"])


if __name__ == "__main__":
    main()
