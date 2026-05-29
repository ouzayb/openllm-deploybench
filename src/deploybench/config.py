"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from deploybench.utils import load_yaml


class ExpectedGPU(BaseModel):
    name_contains: str
    count: int = 1


class HardwareConfig(BaseModel):
    machine_id: str
    machine_label: str = ""
    location_type: Literal["owned", "rented_cloud", "consumer_local"] = "owned"
    provider: str = "local"
    hourly_price_usd: float | None = None
    notes: str = ""
    expected_gpus: list[ExpectedGPU] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ModelSpec(BaseModel):
    id: str
    hf_id: str
    size_class: str = ""
    default_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    requires_hf_token: bool = False
    notes: str = ""


class ModelsConfig(BaseModel):
    models: list[ModelSpec]

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Unknown model_id: {model_id}")

    @classmethod
    def from_yaml(cls, path: Path | str) -> ModelsConfig:
        data = load_yaml(path)
        return cls(**data)


class BenchmarkMeta(BaseModel):
    name: str = "openllm_deploybench"
    output_format: str = "jsonl"


class RuntimeConfig(BaseModel):
    engine: str = "vllm"
    mode: Literal["online", "offline"] = "online"
    tensor_parallel_size: int | str = "auto"
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    seed: int = 42
    port: int = 8000
    host: str = "127.0.0.1"
    server_startup_timeout_sec: int = 600


class MatrixModelEntry(BaseModel):
    model_id: str
    quantization: str | None = None
    dtype: str = "bfloat16"
    max_model_len: list[int]

    @field_validator("quantization", mode="before")
    @classmethod
    def empty_quant(cls, v: Any) -> str | None:
        if v in ("", "none", "null"):
            return None
        return v


class WorkloadSpec(BaseModel):
    id: str
    type: str = "synthetic"
    template: str = "chat"
    prompt_tokens: int
    output_tokens: int
    num_prompts: int = 64
    concurrency: list[int] = Field(default_factory=lambda: [1])


class LongContextConfig(BaseModel):
    enabled: bool = True
    context_lengths: list[int] = Field(default_factory=list)
    needle_positions: list[float] = Field(default_factory=list)
    num_trials_per_setting: int = 5


class MonitoringConfig(BaseModel):
    sample_interval_seconds: float = 0.5
    log_gpu_power: bool = True
    log_gpu_memory: bool = True
    log_gpu_utilization: bool = True


class BenchmarkMatrix(BaseModel):
    benchmark: BenchmarkMeta = Field(default_factory=BenchmarkMeta)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    models: list[MatrixModelEntry]
    workloads: list[WorkloadSpec]
    long_context: LongContextConfig = Field(default_factory=LongContextConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> BenchmarkMatrix:
        data = load_yaml(path)
        return cls(**data)


def load_hardware_config(path: Path | str | None) -> HardwareConfig | None:
    if path is None:
        return None
    data = load_yaml(path)
    return HardwareConfig(**data)


def resolve_tensor_parallel(
    tp_setting: int | str,
    model_size_class: str,
    gpu_count: int,
) -> int:
    if isinstance(tp_setting, int):
        return min(tp_setting, max(gpu_count, 1))
    if tp_setting != "auto":
        try:
            return min(int(tp_setting), max(gpu_count, 1))
        except ValueError:
            pass

    size = model_size_class.upper().replace("B", "")
    try:
        billions = int("".join(c for c in size if c.isdigit()) or "7")
    except ValueError:
        billions = 7

    if gpu_count >= 2 and billions >= 32:
        return min(2, gpu_count)
    if gpu_count >= 2 and billions >= 70:
        return min(2, gpu_count)
    return 1


def merge_model_runtime(
    matrix_entry: MatrixModelEntry,
    model_spec: ModelSpec,
) -> dict[str, Any]:
    return {
        "model_id": matrix_entry.model_id,
        "hf_id": model_spec.hf_id,
        "model_size_class": model_spec.size_class,
        "dtype": matrix_entry.dtype or model_spec.default_dtype,
        "quantization": matrix_entry.quantization,
        "trust_remote_code": model_spec.trust_remote_code,
        "requires_hf_token": model_spec.requires_hf_token,
        "max_model_len": matrix_entry.max_model_len,
    }
