"""Orchestrates serving benchmark matrix execution."""

from __future__ import annotations

import logging
import uuid
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
            tp = resolve_tensor_parallel(
                rt.tensor_parallel_size,
                model_spec.size_class,
                gpu_count,
            )

            serve_cmd = build_serve_command(
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
            )

            # Load server once per model+max_model_len for online mode
            server_loaded = False
            skip_model_len = False
            server_log = logs_dir / f"serve_{model_entry.model_id}_{max_model_len}.log"

            for workload in matrix.workloads:
                if skip_model_len:
                    break
                try:
                    dataset_path = generate_synthetic_dataset(
                        workload, model_spec.hf_id, seed=rt.seed,
                    )
                except Exception as e:
                    et, em = classify_error(e)
                    for conc in workload.concurrency:
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

                for concurrency in workload.concurrency:
                    run_id = str(uuid.uuid4())
                    monitor = GPUMonitor(matrix.monitoring.sample_interval_seconds)

                    try:
                        if rt.mode == "offline":
                            out = run_bench_throughput_offline(
                                hf_id=model_spec.hf_id,
                                prompt_tokens=workload.prompt_tokens,
                                output_tokens=workload.output_tokens,
                                num_prompts=workload.num_prompts,
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
                            from deploybench.vllm_runner import start_server

                            if not server_loaded:
                                ok, err = start_server(
                                    serve_cmd,
                                    server_log,
                                    rt.server_startup_timeout_sec,
                                    rt.host,
                                    rt.port,
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
                                        num_prompts=workload.num_prompts,
                                        concurrency=concurrency,
                                        metrics=BenchmarkMetrics(),
                                        raw={"server_log": str(server_log)},
                                    )
                                    append_jsonl(output_path, result)
                                    stop_server()
                                    server_loaded = False
                                    skip_model_len = True
                                    break
                                server_loaded = True

                            monitor.start()
                            from deploybench.vllm_runner import run_bench_serve

                            bench = run_bench_serve(
                                hf_id=model_spec.hf_id,
                                dataset_path=dataset_path,
                                num_prompts=workload.num_prompts,
                                max_concurrency=concurrency,
                                host=rt.host,
                                port=rt.port,
                                output_tokens=workload.output_tokens,
                                seed=rt.seed,
                            )
                            samples = monitor.stop()
                            summary = monitor.summarize(samples)
                            metrics = bench.get("metrics", BenchmarkMetrics())
                            for k, v in gpu_summary_to_metrics(summary).items():
                                setattr(metrics, k, v)
                            raw = bench.get("raw", {})
                            success = raw.get("returncode") == 0
                            et = em = None
                            if not success:
                                et, em = classify_error(
                                    raw.get("stderr", "") or "bench serve failed"
                                )

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
                            raw=raw if isinstance(raw, dict) else {"output": raw},
                        )
                        append_jsonl(output_path, result)
                        logger.info(
                            "Completed %s / %s / conc=%s success=%s",
                            model_entry.model_id, workload.id, concurrency, success,
                        )

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
                            num_prompts=workload.num_prompts,
                            concurrency=concurrency,
                        )
                    finally:
                        if rt.mode == "offline":
                            monitor.stop()

            if rt.mode == "online":
                stop_server()

    logger.info("Serving benchmark results: %s", output_path)
    return output_path
