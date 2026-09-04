"""Orchestrates serving benchmark matrix execution."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from deploybench.config import (
    BenchmarkMatrix,
    HardwareConfig,
    ModelsConfig,
    merge_model_runtime,
    resolve_tensor_parallel,
)
from deploybench.gpu_monitor import GPUMonitor
from deploybench.hardware_probe import probe_hardware
from deploybench.metrics import classify_error, gpu_summary_to_metrics
from deploybench.result_schema import BenchmarkMetrics, ReproducibilityMeta, ServingBenchmarkResult
from deploybench.utils import (
    append_jsonl,
    collect_relevant_env,
    config_hash_from_paths,
    get_git_commit,
    get_package_versions,
    utc_now_iso,
)
from deploybench.vllm_runner import (
    build_serve_command,
    run_bench_throughput_offline,
    stop_server,
)
from deploybench.workload_generator import generate_synthetic_dataset

logger = logging.getLogger(__name__)


def _generate_dynamic_concurrency(max_cap: int = 2048) -> Iterator[int]:
    """Yields powers of two starting from 1 up to max_cap."""
    current_concurrency = 1
    while current_concurrency <= max_cap:
        yield current_concurrency
        current_concurrency *= 2

def resolve_dynamic_num_prompts(
    model_id: str,
    workload_id: str,
    hardware: HardwareConfig | None,
    base_num_prompts: int,
) -> int:
    """
    Dynamically resolve optimal num_prompts based on workload type, model
    architecture, and hardware capacity.
    """
    workload_name = workload_id.lower()
    model_name = model_id.lower()
    machine_label = (hardware.machine_label if hardware else "").lower()
    machine_id = (hardware.machine_id if hardware else "").lower()
    is_h200 = "h200" in machine_label or "h200" in machine_id

    # 1. RAG and Long-Context Workloads (Compute-bound prefill phase)
    # Each request contains 7.6k - 32k tokens, causing high prefill overhead and VRAM usage.
    if "rag" in workload_name or "long" in workload_name:
        if is_h200:
            # H200 has 282GB HBM3e; it can comfortably process deeper batches
            return max(base_num_prompts, 256)
        # Dual 4090/5090 saturate at lower concurrencies (16-32); keep prompt volume bounded
        return min(base_num_prompts, 128)

    # 2. Coding Workloads (2048 prompt / 1024 output)
    if "coding" in workload_name or "code" in workload_name:
        if is_h200:
            return 1024
        return max(base_num_prompts, 256)

    # 3. Standard Chat Workloads - High-end Datacenter Tier (H200)
    if is_h200:
        if "72b" in model_name and "fp8" in model_name:
            return 4096
        return 2048

    # 4. Standard Chat Workloads - Consumer Dual 4090 / Dual 5090 Tier
    # Small parameter models leave massive KV cache headroom
    if any(size in model_name for size in ["7b", "9b", "12b"]):
        return 1024

    # Quantized models (AWQ, FP8) with large KV cache margin
    if any(q in model_name for q in ["awq", "27b_fp8"]):
        return 1024

    # 5. Dense 14B / 35B models (saturate early around 128-256 concurrency)
    return base_num_prompts


def _base_result(
    hardware: HardwareConfig | None,
    repro: ReproducibilityMeta,
    versions: dict[str, str | None],
    probe: Any,
) -> dict[str, Any]:
    return {
        "machine_id": hardware.machine_id if hardware else "unknown",
        "machine_label": hardware.machine_label if hardware else "",
        "provider": hardware.provider if hardware else "local",
        "location_type": hardware.location_type if hardware else "owned",
        "hourly_price_usd": hardware.hourly_price_usd if hardware else None,
        "engine": "vllm",
        "engine_version": versions.get("vllm_version"),
        "python_version": versions.get("python_version"),
        "cuda_version": getattr(probe, "cuda_version", None),
        "driver_version": getattr(probe, "driver_version", None),
        "reproducibility": repro,
    }


def _write_failure(
    output_path: Path,
    hardware: HardwareConfig | None,
    repro: ReproducibilityMeta,
    versions: dict[str, str | None],
    probe: Any,
    error_type: str,
    error_message: str,
    **kwargs: Any,
) -> None:
    base = _base_result(hardware, repro, versions, probe)
    result = ServingBenchmarkResult(
        run_id=str(uuid.uuid4()),
        timestamp_utc=utc_now_iso(),
        success=False,
        error_type=error_type,
        error_message=error_message,
        **base,
        **kwargs,
    )
    append_jsonl(output_path, result)


def should_stop_early(
    current_metrics: Any,
    prev_metrics: Any | None,
    early_stop_cfg: Any | None,
    step_index: int,
) -> tuple[bool, str]:
    """Evaluates saturation metrics to determine whether to stop sweeping concurrency."""
    if not early_stop_cfg or not getattr(early_stop_cfg, "enabled", False):
        return False, ""

    min_steps = getattr(early_stop_cfg, "min_concurrency_steps", 3)
    if step_index < min_steps:
        return False, ""

    max_ttft = getattr(early_stop_cfg, "max_ttft_p99_ms", 15000.0)
    curr_ttft_p99 = getattr(current_metrics, "ttft_p99_ms", None) or getattr(current_metrics, "ttft_mean_ms", 0.0)
    if curr_ttft_p99 and curr_ttft_p99 > max_ttft:
        return True, f"P99 TTFT ({curr_ttft_p99:.1f}ms) exceeded threshold ({max_ttft:.1f}ms)"

    if prev_metrics:
        prev_tps = getattr(prev_metrics, "generation_tokens_per_second", 0.0) or getattr(prev_metrics, "tokens_per_second", 0.0)
        curr_tps = getattr(current_metrics, "generation_tokens_per_second", 0.0) or getattr(current_metrics, "tokens_per_second", 0.0)
        tps_threshold = getattr(early_stop_cfg, "tps_improvement_threshold", 0.03)

        if prev_tps > 0.0:
            improvement = (curr_tps - prev_tps) / prev_tps
            if improvement < tps_threshold:
                return True, f"TPS saturated: improvement was {improvement * 100:.2f}% (threshold {tps_threshold * 100:.1f}%)"

    return False, ""


def run_serving_benchmark(
    matrix_path: Path,
    models_path: Path,
    output_dir: Path,
    hardware_path: Path | None = None,
    hardware_config: HardwareConfig | None = None,
    cli_args: list[str] | None = None,
) -> Path:
    matrix = BenchmarkMatrix.from_yaml(matrix_path)
    models = ModelsConfig.from_yaml(models_path)
    hardware = hardware_config
    if hardware is None and hardware_path:
        from deploybench.config import load_hardware_config

        hardware = load_hardware_config(hardware_path)

    probe = probe_hardware(hardware)
    gpu_count = max(probe.gpu_count, 1)

    if probe.gpu_count == 0:
        logger.error("No NVIDIA GPUs detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    machine_id = hardware.machine_id if hardware else "unknown"
    output_path = output_dir / f"{machine_id}_{utc_now_iso().replace(':', '')}.jsonl"

    repro = ReproducibilityMeta(
        git_commit=get_git_commit(),
        cli_args=cli_args or [],
        config_hash=config_hash_from_paths(matrix_path, models_path, hardware_path),
        config_paths={
            "benchmark_matrix": str(matrix_path),
            "models": str(models_path),
            "hardware": str(hardware_path) if hardware_path else "",
        },
        env_vars=collect_relevant_env(),
    )
    versions = get_package_versions()
    rt = matrix.runtime
    early_stop_cfg = getattr(matrix, "early_stopping", None)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    for model_entry in matrix.models:
        try:
            model_spec = models.get(model_entry.model_id)
        except KeyError as e:
            _write_failure(
                output_path, hardware, repro, versions, probe,
                "config_error", str(e), model_id=model_entry.model_id,
            )
            continue

        merged = merge_model_runtime(model_entry, model_spec)

        if model_spec.requires_hf_token and not (
            repro.env_vars.get("HF_TOKEN") or repro.env_vars.get("HUGGING_FACE_HUB_TOKEN")
        ):
            for max_len in model_entry.max_model_len:
                _write_failure(
                    output_path, hardware, repro, versions, probe,
                    "hf_auth",
                    "Model requires Hugging Face token (HF_TOKEN)",
                    model_id=model_entry.model_id,
                    hf_id=model_spec.hf_id,
                    max_model_len=max_len,
                )
            continue

        for max_model_len in model_entry.max_model_len:
            tp_override = getattr(model_entry, "tensor_parallel_size", None)
            tp = tp_override if tp_override is not None else resolve_tensor_parallel(
                rt.tensor_parallel_size,
                model_spec.size_class,
                gpu_count,
            )

            serve_kwargs = dict(
                hf_id=model_spec.hf_id,
                dtype=merged["dtype"],
                max_model_len=max_model_len,
                tensor_parallel_size=tp,
                gpu_memory_utilization=rt.gpu_memory_utilization,
                quantization=merged["quantization"],
                trust_remote_code=merged["trust_remote_code"],
                port=rt.port,
                host=rt.host,
                enforce_eager=rt.enforce_eager,
                use_v1_engine=rt.use_v1_engine,
                reproducible=rt.reproducible,
                use_flashinfer_sampler=rt.use_flashinfer_sampler,
            )

            # Load server once per model+max_model_len for online mode
            server_loaded = False
            skip_model_len = False
            server_config: dict[str, Any] = {}
            server_log = logs_dir / f"serve_{model_entry.model_id}_{max_model_len}.log"

            for workload in matrix.workloads:
                if skip_model_len:
                    break

                # Resolve dynamic prompt count based on model, workload type, and hardware
                effective_num_prompts = resolve_dynamic_num_prompts(
                    model_id=model_entry.model_id,
                    workload_id=workload.id,
                    hardware=hardware,
                    base_num_prompts=workload.num_prompts,
                )

                try:
                    # Pass effective_num_prompts so the generated file has enough prompts
                    dataset_path = generate_synthetic_dataset(
                        workload, model_spec.hf_id, seed=rt.seed, num_prompts=effective_num_prompts
                    )
                except TypeError:
                    # Fallback if generate_synthetic_dataset does not take num_prompts argument
                    dataset_path = generate_synthetic_dataset(
                        workload, model_spec.hf_id, seed=rt.seed
                )
                except Exception as e:
                    et, em = classify_error(e)
                    fallback_concurrencies = getattr(workload, "concurrency", None) or [1]
                    for conc in fallback_concurrencies:
                        _write_failure(
                            output_path, hardware, repro, versions, probe,
                            et, em,
                            model_id=model_entry.model_id,
                            hf_id=model_spec.hf_id,
                            workload_id=workload.id,
                            concurrency=conc,
                            max_model_len=max_model_len,
                        )
                    continue

                prev_metrics: BenchmarkMetrics | None = None
                patience_counter = 0
                max_patience = getattr(early_stop_cfg, "patience", 1) if early_stop_cfg else 1

                # Select defined concurrency or dynamically sweep powers of two
                concurrency_sequence = (
                    workload.concurrency
                    if getattr(workload, "concurrency", None)
                    else _generate_dynamic_concurrency()
                )

                for step_idx, concurrency in enumerate(concurrency_sequence, start=1):
                    
                    if concurrency > effective_num_prompts:
                        logger.info(
                            "Skipping concurrency %d exceeding effective_num_prompts (%d)",
                            concurrency, effective_num_prompts,
                        )
                        break 
                    run_id = str(uuid.uuid4())
                    monitor = GPUMonitor(matrix.monitoring.sample_interval_seconds)
                    try:
                        if rt.mode == "offline":
                            out = run_bench_throughput_offline(
                                hf_id=model_spec.hf_id,
                                prompt_tokens=workload.prompt_tokens,
                                output_tokens=workload.output_tokens,
                                num_prompts=effective_num_prompts,
                                dtype=merged["dtype"],
                                max_model_len=max_model_len,
                                tensor_parallel_size=tp,
                                gpu_memory_utilization=rt.gpu_memory_utilization,
                                quantization=merged["quantization"],
                                trust_remote_code=merged["trust_remote_code"],
                                seed=rt.seed,
                                enforce_eager=rt.enforce_eager,
                                monitor=monitor,
                            )
                            success = out.get("success", False)
                            metrics = out.get("metrics", BenchmarkMetrics())
                            raw = out.get("raw", {})
                            et = em = None
                            if not success:
                                et, em = classify_error(
                                    raw.get("stderr", "") or "offline benchmark failed"
                                )
                        else:
                            from deploybench.vllm_runner import start_vllm_server

                            if not server_loaded:
                                ok, err, serve_cmd, server_config = start_vllm_server(
                                    log_path=server_log,
                                    startup_timeout_sec=rt.server_startup_timeout_sec,
                                    **serve_kwargs,
                                )
                                if not ok:
                                    et, em = classify_error(err)
                                    result = ServingBenchmarkResult(
                                        run_id=run_id,
                                        timestamp_utc=utc_now_iso(),
                                        success=False,
                                        error_type=et,
                                        error_message=em,
                                        **_base_result(hardware, repro, versions, probe),
                                        model_id=model_entry.model_id,
                                        hf_id=model_spec.hf_id,
                                        model_size_class=model_spec.size_class,
                                        dtype=merged["dtype"],
                                        quantization=merged["quantization"],
                                        tensor_parallel_size=tp,
                                        max_model_len=max_model_len,
                                        workload_id=workload.id,
                                        prompt_tokens_target=workload.prompt_tokens,
                                        output_tokens_target=workload.output_tokens,
                                        num_prompts=effective_num_prompts,
                                        concurrency=concurrency,
                                        metrics=BenchmarkMetrics(),
                                        server_config=server_config,
                                        raw={"server_log": str(server_log)},
                                    )
                                    append_jsonl(output_path, result)
                                    stop_server()
                                    server_loaded = False
                                    skip_model_len = True
                                    break
                                server_loaded = True
                                # env_vars is a parent-shell snapshot and can
                                # disagree with what the server subprocess
                                # actually launched with; make the recorded env
                                # reflect the real (server_config) values.
                                if server_config.get("flashinfer_sampler") is not None:
                                    repro.env_vars["VLLM_USE_FLASHINFER_SAMPLER"] = (
                                        server_config["flashinfer_sampler"]
                                    )
                                if server_config.get("vllm_use_v1") is not None:
                                    repro.env_vars["VLLM_USE_V1"] = server_config["vllm_use_v1"]

                            monitor.start()
                            from deploybench.vllm_runner import run_bench_serve

                            bench = run_bench_serve(
                                hf_id=model_spec.hf_id,
                                dataset_path=dataset_path,
                                num_prompts=effective_num_prompts,
                                max_concurrency=concurrency,
                                host=rt.host,
                                port=rt.port,
                                output_tokens=workload.output_tokens,
                                seed=rt.seed,
                                reproducible=rt.reproducible,
                                num_warmups=rt.num_warmups,
                            )
                            samples = monitor.stop()
                            summary = monitor.summarize(samples)
                            metrics = bench.get("metrics", BenchmarkMetrics())
                            for k, v in gpu_summary_to_metrics(summary).items():
                                setattr(metrics, k, v)
                            raw = bench.get("raw", {})
                            success = bench.get(
                                "success", raw.get("returncode") == 0
                            )
                            et = em = None
                            if not success:
                                err_text = (
                                    raw.get("stderr", "")
                                    or raw.get("stdout", "")
                                    or raw.get("http_fallback", {}).get("error", "")
                                    or "bench serve failed"
                                )
                                et, em = classify_error(err_text)
                            if isinstance(raw, dict) and raw.get("bench_profile"):
                                server_config["bench_profile"] = raw["bench_profile"]

                        result = ServingBenchmarkResult(
                            run_id=run_id,
                            timestamp_utc=utc_now_iso(),
                            success=success,
                            error_type=et,
                            error_message=em,
                            **_base_result(hardware, repro, versions, probe),
                            model_id=model_entry.model_id,
                            hf_id=model_spec.hf_id,
                            model_size_class=model_spec.size_class,
                            dtype=merged["dtype"],
                            quantization=merged["quantization"],
                            tensor_parallel_size=tp,
                            max_model_len=max_model_len,
                            workload_id=workload.id,
                            prompt_tokens_target=workload.prompt_tokens,
                            output_tokens_target=workload.output_tokens,
                            num_prompts=workload.num_prompts,
                            concurrency=concurrency,
                            metrics=metrics if isinstance(metrics, BenchmarkMetrics) else metrics,
                            server_config=server_config,
                            raw=raw if isinstance(raw, dict) else {"output": raw},
                        )
                        append_jsonl(output_path, result)
                        logger.info(
                            "Completed %s / %s / conc=%s success=%s",
                            model_entry.model_id, workload.id, concurrency, success,
                        )

                        # Terminate dynamic concurrency sweep if the run failed
                        if not success:
                            logger.warning(
                                "Run failed at concurrency=%d (%s: %s). Halting sweep.",
                                concurrency, et, em
                            )
                            break

                        if success:
                            should_stop, reason = should_stop_early(
                                metrics, prev_metrics, early_stop_cfg, step_idx
                            )
                            if should_stop:
                                patience_counter += 1
                                logger.info(
                                    "Early stopping condition triggered (%s). Patience: %d/%d",
                                    reason, patience_counter, max_patience,
                                )
                                if patience_counter >= max_patience:
                                    logger.info(
                                        "Early stopping sweep for %s on %s at concurrency=%d",
                                        model_entry.model_id, workload.id, concurrency,
                                    )
                                    break
                            else:
                                patience_counter = 0

                            prev_metrics = metrics

                    except Exception as e:
                        et, em = classify_error(e)
                        logger.exception("Benchmark failed: %s", e)
                        _write_failure(
                            output_path, hardware, repro, versions, probe,
                            et, em,
                            model_id=model_entry.model_id,
                            hf_id=model_spec.hf_id,
                            model_size_class=model_spec.size_class,
                            dtype=merged["dtype"],
                            quantization=merged["quantization"],
                            tensor_parallel_size=tp,
                            max_model_len=max_model_len,
                            workload_id=workload.id,
                            prompt_tokens_target=workload.prompt_tokens,
                            output_tokens_target=workload.output_tokens,
                            num_prompts=effective_num_prompts,
                            concurrency=concurrency,
                        )
                        # Abort higher concurrencies on unexpected execution failure
                        break
                    finally:
                        if rt.mode == "offline":
                            monitor.stop()

            if rt.mode == "online":
                stop_server()

    logger.info("Serving benchmark results: %s", output_path)
    return output_path