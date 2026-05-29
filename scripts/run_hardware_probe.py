#!/usr/bin/env python3
"""Run hardware probe (direct script entry)."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.hardware_probe import run_probe
from deploybench.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware probe")
    parser.add_argument("--output", "-o", default="results/hardware.json")
    parser.add_argument("--hardware-config", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    hw = Path(args.hardware_config) if args.hardware_config else None
    run_probe(Path(args.output), hw)


if __name__ == "__main__":
    main()
