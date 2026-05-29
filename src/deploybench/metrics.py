"""Metric parsing and computation."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from deploybench.result_schema import BenchmarkMetrics, GPUSample, GPUSampleSummary


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def compute_energy_wh(samples: list[GPUSample], interval_sec: float) -> float:
    """Approximate energy from average power across GPUs per interval."""
    if not samples or interval_sec <= 0:
        return 0.0
    by_ts: dict[str, list[float]] = {}
    for s in samples:
        if s.power_draw_watts is not None:
            by_ts.setdefault(s.timestamp, []).append(s.power_draw_watts)
    if not by_ts:
        return 0.0
    total_j = 0.0
    for powers in by_ts.values():
        avg_w = sum(powers) / len(powers)
        total_j += avg_w * interval_sec
    return total_j / 3600.0


def summarize_gpu_samples(
    samples: list[GPUSample],
    interval_sec: float = 0.5,
) -> GPUSampleSummary:
    if not samples:
        return GPUSampleSummary()

    mem_gb = [s.memory_used_mb / 1024.0 for s in samples]
    powers = [s.power_draw_watts for s in samples if s.power_draw_watts is not None]
    utils = [s.utilization_gpu_percent for s in samples]
    temps = [s.temperature_c for s in samples if s.temperature_c is not None]

    return GPUSampleSummary(
        peak_vram_gb=max(mem_gb) if mem_gb else 0.0,
        average_power_watts=float(np.mean(powers)) if powers else 0.0,
        peak_power_watts=max(powers) if powers else 0.0,
        average_gpu_utilization=float(np.mean(utils)) if utils else 0.0,
        max_temperature_c=max(temps) if temps else 0.0,
        energy_wh=compute_energy_wh(samples, interval_sec),
    )


def gpu_summary_to_metrics(summary: GPUSampleSummary) -> dict[str, float]:
    return {
        "peak_vram_gb": summary.peak_vram_gb,
        "avg_power_watts": summary.average_power_watts,
        "peak_power_watts": summary.peak_power_watts,
        "energy_wh": summary.energy_wh,
        "avg_gpu_utilization": summary.average_gpu_utilization,
        "max_temperature_c": summary.max_temperature_c,
    }


def parse_vllm_bench_output(stdout: str, stderr: str = "") -> BenchmarkMetrics:
    """Parse vLLM bench serve/throughput text or JSON output."""
    text = stdout + "\n" + stderr
    metrics = BenchmarkMetrics()

    # Try JSON blob in output
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                return _metrics_from_dict(data, metrics)
            except json.JSONDecodeError:
                pass

    # Regex patterns for vLLM bench serve result block
    patterns = {
        "requests_per_second": r"Request throughput \(req/s\):\s*([\d.]+)",
        "output_tokens_per_second": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "total_tokens_per_second": r"Total token throughput \(tok/s\):\s*([\d.]+)",
        "ttft_ms_p50": r"Mean TTFT \(ms\):\s*([\d.]+)|TTFT \(ms\).*?p50[:\s]+([\d.]+)",
        "ttft_ms_p95": r"P95 TTFT \(ms\):\s*([\d.]+)",
        "ttft_ms_p99": r"P99 TTFT \(ms\):\s*([\d.]+)",
        "tpot_ms_p50": r"Mean TPOT \(ms\):\s*([\d.]+)",
        "tpot_ms_p95": r"P95 TPOT \(ms\):\s*([\d.]+)",
        "tpot_ms_p99": r"P99 TPOT \(ms\):\s*([\d.]+)",
        "e2e_latency_ms_p50": r"Mean E2EL \(ms\):\s*([\d.]+)",
        "e2e_latency_ms_p95": r"P95 E2EL \(ms\):\s*([\d.]+)",
        "e2e_latency_ms_p99": r"P99 E2EL \(ms\):\s*([\d.]+)",
    }

    for field, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                setattr(metrics, field, float(groups[0]))

    # Throughput offline format
    m = re.search(
        r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,\s*([\d.]+)\s*output tokens/s",
        text,
    )
    if m:
        metrics.requests_per_second = float(m.group(1))
        metrics.total_tokens_per_second = float(m.group(2))
        metrics.output_tokens_per_second = float(m.group(3))

    return metrics


def _metrics_from_dict(data: dict[str, Any], base: BenchmarkMetrics) -> BenchmarkMetrics:
    mapping = {
        "request_throughput": "requests_per_second",
        "output_throughput": "output_tokens_per_second",
        "total_token_throughput": "total_tokens_per_second",
        "output_tokens_per_second": "output_tokens_per_second",
        "requests_per_second": "requests_per_second",
    }
    flat: dict[str, Any] = {}
    if "result" in data and isinstance(data["result"], dict):
        flat.update(data["result"])
    flat.update(data)

    for src, dst in mapping.items():
        if src in flat and flat[src] is not None:
            setattr(base, dst, float(flat[src]))

    for key in base.model_fields:
        if key in flat and flat[key] is not None:
            try:
                setattr(base, key, float(flat[key]))
            except (TypeError, ValueError):
                pass
    return base


def classify_error(exc: BaseException | str) -> tuple[str, str]:
    msg = str(exc).lower()
    if "out of memory" in msg or "oom" in msg or "cuda" in msg and "memory" in msg:
        return "oom", str(exc)
    if "401" in msg or "gated" in msg or "unauthorized" in msg or "hf_token" in msg:
        return "hf_auth", str(exc)
    if "quantization" in msg or "unsupported" in msg and "quant" in msg:
        return "unsupported_quantization", str(exc)
    if "no module named" in msg or "not found" in msg and "vllm" in msg:
        return "vllm_missing", str(exc)
    if "connection" in msg or "health" in msg or "startup" in msg:
        return "server_startup_failure", str(exc)
    if "config" in msg or "yaml" in msg:
        return "config_error", str(exc)
    return "runtime_error", str(exc)
