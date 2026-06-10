"""Pydantic schemas for benchmark results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkMetrics(BaseModel):
    # None means the metric was not reported by the benchmark (e.g. a percentile
    # vLLM did not emit). It is serialized as JSON null rather than a misleading 0.0.
    requests_per_second: float | None = None
    output_tokens_per_second: float | None = None
    total_tokens_per_second: float | None = None
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    ttft_ms_p99: float | None = None
    tpot_ms_p50: float | None = None
    tpot_ms_p95: float | None = None
    tpot_ms_p99: float | None = None
    e2e_latency_ms_p50: float | None = None
    e2e_latency_ms_p95: float | None = None
    e2e_latency_ms_p99: float | None = None
    peak_vram_gb: float | None = None
    avg_power_watts: float | None = None
    peak_power_watts: float | None = None
    energy_wh: float | None = None
    avg_gpu_utilization: float | None = None
    max_temperature_c: float | None = None


class GPUSample(BaseModel):
    timestamp: str
    gpu_index: int
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    utilization_gpu_percent: float = 0.0
    utilization_memory_percent: float = 0.0
    power_draw_watts: float | None = None
    temperature_c: float | None = None
    sm_clock_mhz: float | None = None
    memory_clock_mhz: float | None = None


class GPUSampleSummary(BaseModel):
    peak_vram_gb: float | None = None
    average_power_watts: float | None = None
    peak_power_watts: float | None = None
    average_gpu_utilization: float | None = None
    max_temperature_c: float | None = None
    energy_wh: float | None = None


class ReproducibilityMeta(BaseModel):
    git_commit: str | None = None
    cli_args: list[str] = Field(default_factory=list)
    config_hash: str | None = None
    config_paths: dict[str, str] = Field(default_factory=dict)
    env_vars: dict[str, str] = Field(default_factory=dict)


class ServingBenchmarkResult(BaseModel):
    run_id: str
    timestamp_utc: str
    machine_id: str = "unknown"
    machine_label: str = ""
    provider: str = "local"
    location_type: str = "owned"
    hourly_price_usd: float | None = None

    engine: str = "vllm"
    engine_version: str | None = None
    python_version: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None

    model_id: str = ""
    hf_id: str = ""
    model_size_class: str = ""
    dtype: str = "bfloat16"
    quantization: str | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 8192

    workload_id: str = ""
    prompt_tokens_target: int = 0
    output_tokens_target: int = 0
    num_prompts: int = 0
    concurrency: int = 1

    success: bool = True
    error_type: str | None = None
    error_message: str | None = None

    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
    reproducibility: ReproducibilityMeta = Field(default_factory=ReproducibilityMeta)
    raw: dict[str, Any] = Field(default_factory=dict)


class LongContextResult(BaseModel):
    benchmark_type: str = "long_context_needle"
    run_id: str
    timestamp_utc: str
    machine_id: str = "unknown"
    model_id: str = ""
    hf_id: str = ""
    max_model_len: int = 0
    context_length: int = 0
    needle_position: float = 0.5
    trial: int = 0
    expected_answer: str = ""
    model_answer: str = ""
    exact_match: bool = False
    latency_ms: float = 0.0
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
    reproducibility: ReproducibilityMeta = Field(default_factory=ReproducibilityMeta)
    raw: dict[str, Any] = Field(default_factory=dict)


class HardwareProbeResult(BaseModel):
    timestamp_utc: str
    hostname: str = ""
    os: str = ""
    kernel_version: str = ""
    python_version: str = ""
    cpu_model: str = ""
    cpu_core_count: int = 0
    ram_total_gb: float = 0.0
    disk_info: list[dict[str, Any]] = Field(default_factory=list)

    machine_id: str | None = None
    machine_label: str | None = None
    location_type: str | None = None
    provider: str | None = None
    hourly_price_usd: float | None = None
    tags: list[str] = Field(default_factory=list)
    expected_gpus: list[dict[str, Any]] = Field(default_factory=list)

    gpu_count: int = 0
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None
    raw_outputs: dict[str, Any] = Field(default_factory=dict)
