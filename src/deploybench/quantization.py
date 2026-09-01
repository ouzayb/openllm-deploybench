"""Quantization benchmark helpers."""

from __future__ import annotations

import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from deploybench.benchmark_runner import run_serving_benchmark
from deploybench.config import load_hardware_config
from deploybench.utils import load_yaml

logger = logging.getLogger(__name__)

SUPPORTED_QUANTS: list[str | None] = [None, "fp8", "awq", "gptq", "bitsandbytes"]


def run_quantization_benchmark(
    matrix_path: Path,
    models_path: Path,
    output_dir: Path,
    hardware_path: Path | None = None,
    quantizations: list[str | None] | None = None,
    cli_args: list[str] | None = None,
) -> Path:
    """Run serving benchmark across quantization settings."""
    data = deepcopy(load_yaml(matrix_path))
    quants = quantizations or SUPPORTED_QUANTS
    hardware = load_hardware_config(hardware_path) if hardware_path else None
    output_dir = Path(output_dir)
    last_output: Path | None = None

    base_models = data.get("models", [])

    for quant in quants:
        matrix_data = deepcopy(data)
        matrix_data["models"] = []
        for entry in base_models:
            e = deepcopy(entry)
            e["quantization"] = quant
            base_id = e.get("model_id", "")

            # If applying quantization to a base model, remap to quantized catalog ID if available
            if quant and not base_id.endswith(f"_{quant}"):
                quant_id = f"{base_id}_{quant}"
                e["model_id"] = quant_id
                e["dtype"] = "auto"
            elif not quant:
                # If testing baseline (None), strip any quantization suffix if present
                for known_q in ["_fp8", "_awq", "_gptq"]:
                    if base_id.endswith(known_q):
                        e["model_id"] = base_id[: -len(known_q)]
                        e["dtype"] = "bfloat16"
                        break

            matrix_data["models"].append(e)

        quant_label = quant or "none"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as tmp:
            yaml.dump(matrix_data, tmp, default_flow_style=False)
            tmp_path = Path(tmp.name)

        try:
            last_output = run_serving_benchmark(
                matrix_path=tmp_path,
                models_path=models_path,
                output_dir=output_dir / quant_label,
                hardware_path=hardware_path,
                hardware_config=hardware,
                cli_args=cli_args,
            )
        except Exception as e:
            logger.error("Quantization %s failed: %s", quant_label, e)
        finally:
            tmp_path.unlink(missing_ok=True)

    return last_output or output_dir
