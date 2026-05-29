"""Needle-in-a-haystack long context benchmark."""

from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path
from typing import Any

from deploybench.config import BenchmarkMatrix, HardwareConfig, ModelsConfig, resolve_tensor_parallel
from deploybench.hardware_probe import probe_hardware
from deploybench.metrics import classify_error
from deploybench.result_schema import LongContextResult, ReproducibilityMeta
from deploybench.utils import (
    append_jsonl,
    collect_relevant_env,
    config_hash_from_paths,
    get_git_commit,
    utc_now_iso,
)
from deploybench.workload_generator import generate_needle_prompt

logger = logging.getLogger(__name__)


def _normalize_answer(text: str) -> str:
    return text.strip().strip('"').strip("'")


def run_long_context_benchmark(
    matrix_path: Path,
    models_path: Path,
    output_dir: Path,
    hardware_path: Path | None = None,
    hardware_config: HardwareConfig | None = None,
    cli_args: list[str] | None = None,
) -> Path:
    matrix = BenchmarkMatrix.from_yaml(matrix_path)
    models = ModelsConfig.from_yaml(models_path)
    if not matrix.long_context.enabled:
        logger.info("Long context benchmark disabled in config")
        return output_dir

    hardware = hardware_config
    if hardware is None and hardware_path:
        from deploybench.config import load_hardware_config

        hardware = load_hardware_config(hardware_path)

    probe = probe_hardware(hardware)
    gpu_count = max(probe.gpu_count, 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_id = hardware.machine_id if hardware else "unknown"
    output_path = output_dir / f"{machine_id}_longctx_{utc_now_iso().replace(':', '')}.jsonl"

    repro = ReproducibilityMeta(
        git_commit=get_git_commit(),
        cli_args=cli_args or [],
        config_hash=config_hash_from_paths(matrix_path, models_path, hardware_path),
        config_paths={
            "benchmark_matrix": str(matrix_path),
            "models": str(models_path),
        },
        env_vars=collect_relevant_env(),
    )

    lc = matrix.long_context
    rt = matrix.runtime

    for model_entry in matrix.models:
        try:
            model_spec = models.get(model_entry.model_id)
        except KeyError as e:
            logger.error("Unknown model: %s", e)
            continue

        for max_model_len in model_entry.max_model_len:
            if max_model_len < min(lc.context_lengths, default=8192):
                continue

            tp = resolve_tensor_parallel(
                rt.tensor_parallel_size, model_spec.size_class, gpu_count,
            )

            llm = None
            try:
                from vllm import LLM, SamplingParams

                kwargs: dict[str, Any] = {
                    "model": model_spec.hf_id,
                    "dtype": model_entry.dtype or model_spec.default_dtype,
                    "max_model_len": max_model_len,
                    "tensor_parallel_size": tp,
                    "gpu_memory_utilization": rt.gpu_memory_utilization,
                    "trust_remote_code": model_spec.trust_remote_code,
                    "seed": rt.seed,
                }
                if model_entry.quantization:
                    kwargs["quantization"] = model_entry.quantization
                if rt.enforce_eager:
                    kwargs["enforce_eager"] = True

                llm = LLM(**kwargs)
                sampling = SamplingParams(
                    temperature=0,
                    max_tokens=64,
                    seed=rt.seed,
                )

                rng = random.Random(rt.seed)
                from deploybench.workload_generator import _get_tokenizer

                tokenizer = _get_tokenizer(model_spec.hf_id)

                for ctx_len in lc.context_lengths:
                    if ctx_len > max_model_len:
                        continue
                    for pos in lc.needle_positions:
                        for trial in range(lc.num_trials_per_setting):
                            run_id = str(uuid.uuid4())
                            try:
                                prompt, expected, _ = generate_needle_prompt(
                                    ctx_len, pos, trial, rng, tokenizer,
                                )
                                t0 = time.perf_counter()
                                outputs = llm.generate([prompt], sampling)
                                latency_ms = (time.perf_counter() - t0) * 1000
                                model_answer = ""
                                if outputs and outputs[0].outputs:
                                    model_answer = outputs[0].outputs[0].text
                                exact = _normalize_answer(model_answer) == _normalize_answer(expected)

                                result = LongContextResult(
                                    run_id=run_id,
                                    timestamp_utc=utc_now_iso(),
                                    machine_id=hardware.machine_id if hardware else "unknown",
                                    model_id=model_entry.model_id,
                                    hf_id=model_spec.hf_id,
                                    max_model_len=max_model_len,
                                    context_length=ctx_len,
                                    needle_position=pos,
                                    trial=trial,
                                    expected_answer=expected,
                                    model_answer=model_answer.strip(),
                                    exact_match=exact,
                                    latency_ms=latency_ms,
                                    success=True,
                                    reproducibility=repro,
                                )
                                append_jsonl(output_path, result)

                            except Exception as e:
                                et, em = classify_error(e)
                                append_jsonl(
                                    output_path,
                                    LongContextResult(
                                        run_id=run_id,
                                        timestamp_utc=utc_now_iso(),
                                        machine_id=hardware.machine_id if hardware else "unknown",
                                        model_id=model_entry.model_id,
                                        hf_id=model_spec.hf_id,
                                        max_model_len=max_model_len,
                                        context_length=ctx_len,
                                        needle_position=pos,
                                        trial=trial,
                                        success=False,
                                        error_type=et,
                                        error_message=em,
                                        reproducibility=repro,
                                    ),
                                )

            except Exception as e:
                et, em = classify_error(e)
                logger.exception("Failed to load model %s: %s", model_entry.model_id, e)
                for ctx_len in lc.context_lengths:
                    for pos in lc.needle_positions:
                        for trial in range(lc.num_trials_per_setting):
                            append_jsonl(
                                output_path,
                                LongContextResult(
                                    run_id=str(uuid.uuid4()),
                                    timestamp_utc=utc_now_iso(),
                                    machine_id=hardware.machine_id if hardware else "unknown",
                                    model_id=model_entry.model_id,
                                    hf_id=model_spec.hf_id,
                                    max_model_len=max_model_len,
                                    context_length=ctx_len,
                                    needle_position=pos,
                                    trial=trial,
                                    success=False,
                                    error_type=et,
                                    error_message=em,
                                    reproducibility=repro,
                                ),
                            )
            finally:
                if llm is not None:
                    del llm
                    try:
                        import gc
                        import torch

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

    logger.info("Long context results: %s", output_path)
    return output_path
