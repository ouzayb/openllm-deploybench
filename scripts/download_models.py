#!/usr/bin/env python3
"""Pre-download Hugging Face models from models.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from deploybench.config import ModelsConfig
from deploybench.utils import setup_logging

logger = __import__("logging").getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed")
        return

    models = ModelsConfig.from_yaml(Path(args.models_config))
    for m in models.models:
        print(f"Downloading {m.hf_id} ...")
        try:
            snapshot_download(
                repo_id=m.hf_id,
                trust_remote_code=m.trust_remote_code,
            )
            print(f"  OK: {m.hf_id}")
        except Exception as e:
            print(f"  FAILED: {m.hf_id}: {e}")


if __name__ == "__main__":
    main()
