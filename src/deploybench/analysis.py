"""Summarize benchmark results into CSV reports.

Filters duplicate/historical sweeps to retain only the latest runs,
and calculates saturation, peak throughput, and efficiency metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from deploybench.utils import find_jsonl_files, read_jsonl

logger = logging.getLogger(__name__)


def _load_hardware(results_dir: Path) -> dict[str, Any] | None:
    hw_path = results_dir / "hardware.json"
    if hw_path.exists():
        with hw_path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _flatten_serving(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        m = row.get("metrics", {}) or {}
        # Shallow copy without bulky console command traces
        rec = {k: v for k, v in row.items() if k not in ("raw", "metrics")}
        for k, v in m.items():
            rec[f"metric_{k}"] = v
            # Also provide clean keys for downstream processing
            rec[k] = v
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _flatten_long_context(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _filter_latest_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """Retains only the latest benchmark attempt per concurrency step for each sweep."""
    if df.empty or "timestamp_utc" not in df.columns:
        return df

    df["parsed_timestamp"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
    group_keys = [
        "machine_id",
        "model_id",
        "workload_id",
        "tensor_parallel_size",
        "quantization",
    ]
    # Filter group keys present in the DataFrame
    valid_group_keys = [k for k in group_keys if k in df.columns]

    latest_records = []
    for _, group in df.groupby(valid_group_keys):
        # Pick the latest row per concurrency step within this workload sweep
        latest_step_records = (
            group.sort_values("parsed_timestamp")
            .groupby("concurrency", as_index=False)
            .last()
        )
        latest_records.append(latest_step_records)

    filtered_df = pd.concat(latest_records, ignore_index=True)
    filtered_df = filtered_df.drop(columns=["parsed_timestamp"])
    return filtered_df


def _compute_saturation_and_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Computes peak throughput flags, saturation boundaries, and energy metrics."""
    if df.empty or "metric_output_tokens_per_second" not in df.columns:
        return df

    group_keys = [
        "machine_id",
        "model_id",
        "workload_id",
        "tensor_parallel_size",
        "quantization",
    ]
    valid_group_keys = [k for k in group_keys if k in df.columns]
    df = df.sort_values(by=valid_group_keys + ["concurrency"]).reset_index(drop=True)

    df["saturation_reached"] = False
    df["is_peak_throughput"] = False
    df["tps_improvement_pct"] = 0.0
    df["peak_sweep_tps"] = 0.0
    df["peak_concurrency"] = 0
    df["joules_per_output_token"] = None

    processed_groups = []
    for _, group in df.groupby(valid_group_keys):
        group = group.copy()
        valid_mask = (group["success"] == True) & (group["metric_output_tokens_per_second"] > 0)  # noqa: E712

        if not valid_mask.any():
            processed_groups.append(group)
            continue

        # 1. Identify global peak throughput in the sweep
        max_tps = group.loc[valid_mask, "metric_output_tokens_per_second"].max()
        peak_idx = group.loc[valid_mask & (group["metric_output_tokens_per_second"] == max_tps)].index[0]
        peak_concurrency = group.loc[peak_idx, "concurrency"]

        group["peak_sweep_tps"] = round(max_tps, 2)
        group["peak_concurrency"] = peak_concurrency
        group.loc[peak_idx, "is_peak_throughput"] = True

        # 2. Concurrency step improvement
        prev_tps = group["metric_output_tokens_per_second"].shift(1)
        group["tps_improvement_pct"] = round(
            ((group["metric_output_tokens_per_second"] - prev_tps) / prev_tps) * 100, 2
        ).fillna(0.0)

        # 3. Saturation: Mark True for all steps starting from peak concurrency
        group["saturation_reached"] = group["concurrency"] >= peak_concurrency

        # 4. Energy Efficiency: Joules per token = (Wh * 3600) / Total Output Tokens
        if "num_prompts" in group.columns and "output_tokens_target" in group.columns and "metric_energy_wh" in group.columns:
            total_tokens = group["num_prompts"] * group["output_tokens_target"]
            has_energy = group["metric_energy_wh"].notna() & (total_tokens > 0)
            group.loc[has_energy, "joules_per_output_token"] = round(
                (group.loc[has_energy, "metric_energy_wh"] * 3600.0) / total_tokens.loc[has_energy],
                4,
            )

        processed_groups.append(group)

    return pd.concat(processed_groups, ignore_index=True)


def summarize_serving(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    cols = [
        "machine_id", "machine_label", "provider", "location_type",
        "hourly_price_usd", "model_id", "workload_id", "concurrency",
        "max_model_len", "quantization", "success", "early_stopped",
        "saturation_reached", "is_peak_throughput", "tps_improvement_pct",
        "peak_sweep_tps", "peak_concurrency", "joules_per_output_token",
        "metric_output_tokens_per_second", "metric_requests_per_second",
        "metric_ttft_ms_p50", "metric_ttft_ms_p95", "metric_ttft_ms_p99",
        "metric_tpot_ms_p50", "metric_tpot_ms_p95", "metric_tpot_ms_p99",
        "metric_e2e_latency_ms_p50", "metric_e2e_latency_ms_p95", "metric_e2e_latency_ms_p99",
        "metric_peak_vram_gb", "metric_avg_power_watts", "metric_peak_power_watts",
        "metric_energy_wh", "metric_avg_gpu_utilization", "metric_max_temperature_c",
    ]
    existing = [c for c in cols if c in df.columns]
    summary = df[existing].copy()
    summary.to_csv(output_path, index=False)
    logger.info("Wrote clean serving summary to %s", output_path)


def summarize_long_context(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    agg = df.groupby(
        ["machine_id", "model_id", "context_length", "needle_position"],
        dropna=False,
    ).agg(
        accuracy=("exact_match", "mean"),
        trials=("exact_match", "count"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()

    # Context retention: accuracy at max length vs min length per model
    retention_rows = []
    for (mid, model), grp in df.groupby(["machine_id", "model_id"]):
        by_len = grp.groupby("context_length")["exact_match"].mean()
        if len(by_len) >= 2:
            min_len, max_len = by_len.index.min(), by_len.index.max()
            retention = by_len.get(max_len, 0) / by_len.get(min_len, 1e-9)
            retention_rows.append({
                "machine_id": mid,
                "model_id": model,
                "context_retention": min(retention, 1.0),
                "max_stable_context_length": max_len if by_len.get(max_len, 0) >= 0.8 else min_len,
            })
    retention_df = pd.DataFrame(retention_rows)

    # Lost-in-the-middle: middle positions vs edges
    lim_rows = []
    for (mid, model), grp in df.groupby(["machine_id", "model_id"]):
        edge = grp[grp["needle_position"].isin([0.05, 0.95])]["exact_match"].mean()
        middle = grp[grp["needle_position"].isin([0.25, 0.50, 0.75])]["exact_match"].mean()
        if pd.notna(edge) and pd.notna(middle):
            lim_rows.append({
                "machine_id": mid,
                "model_id": model,
                "lost_middle_drop": edge - middle,
            })
    lim_df = pd.DataFrame(lim_rows)

    out = agg.merge(retention_df, on=["machine_id", "model_id"], how="left")
    out = out.merge(lim_df, on=["machine_id", "model_id"], how="left")
    out.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


def summarize_hardware(hw: dict[str, Any] | None, output_path: Path) -> None:
    if not hw:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    flat = {
        "timestamp_utc": hw.get("timestamp_utc"),
        "machine_id": hw.get("machine_id"),
        "machine_label": hw.get("machine_label"),
        "location_type": hw.get("location_type"),
        "provider": hw.get("provider"),
        "gpu_count": hw.get("gpu_count"),
        "driver_version": hw.get("driver_version"),
        "cuda_version": hw.get("cuda_version"),
        "cpu_model": hw.get("cpu_model"),
        "ram_total_gb": hw.get("ram_total_gb"),
    }
    gpus = hw.get("gpus", [])
    for i, g in enumerate(gpus):
        flat[f"gpu_{i}_name"] = g.get("name")
        flat[f"gpu_{i}_memory_mb"] = g.get("memory_total_mb")
    pd.DataFrame([flat]).to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


def summarize_price_performance(
    serving_df: pd.DataFrame,
    hw: dict[str, Any] | None,
    output_path: Path,
) -> None:
    if serving_df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = serving_df[serving_df["success"] == True].copy()  # noqa: E712
    if "metric_output_tokens_per_second" not in df.columns:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df["output_tokens_per_hour"] = df["metric_output_tokens_per_second"] * 3600
    df["tokens_per_dollar"] = None
    mask = df["hourly_price_usd"].notna() & (df["hourly_price_usd"] > 0)
    df.loc[mask, "tokens_per_dollar"] = (
        df.loc[mask, "output_tokens_per_hour"] / df.loc[mask, "hourly_price_usd"]
    )

    if "metric_energy_wh" in df.columns and df["metric_energy_wh"].notna().any():
        energy = df["metric_energy_wh"].replace(0, float("nan"))
        tok_per_wh = df["metric_output_tokens_per_second"] * 3600 / energy
        df["energy_per_1m_tokens_wh"] = 1e6 / tok_per_wh

    # Relative to owned H200 baseline (from hardware.json tags)
    hw_tags = [t.lower() for t in (hw or {}).get("tags", [])]
    hw_is_owned_h200 = (
        (hw or {}).get("location_type") == "owned"
        and "h200" in hw_tags
    )
    owned = df[df["location_type"] == "owned"]
    if hw_is_owned_h200:
        owned = df
    elif "h200" in hw_tags:
        owned = df[df["location_type"] == "owned"]

    if not owned.empty:
        baseline_tps = owned["metric_output_tokens_per_second"].max()
        if baseline_tps and baseline_tps > 0:
            df["relative_to_owned_h200"] = df["metric_output_tokens_per_second"] / baseline_tps

    cols = [
        "machine_id", "machine_label", "provider", "location_type",
        "hourly_price_usd", "model_id", "workload_id",
        "metric_output_tokens_per_second", "output_tokens_per_hour",
        "tokens_per_dollar", "energy_per_1m_tokens_wh", "relative_to_owned_h200",
    ]
    existing = [c for c in cols if c in df.columns]
    df[existing].drop_duplicates().to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


def run_summarize(results_dir: Path, output_dir: Path) -> dict[str, Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hw = _load_hardware(results_dir)
    serving_rows: list[dict] = []
    longctx_rows: list[dict] = []

    for path in find_jsonl_files(results_dir):
        # Avoid reading previously aggregated JSONL files
        if "aggregated" in path.name:
            continue
        rows = read_jsonl(path)
        if "long_context" in str(path) or (rows and rows[0].get("benchmark_type") == "long_context_needle"):
            longctx_rows.extend(rows)
        else:
            serving_rows.extend(rows)

    raw_serving_df = _flatten_serving(serving_rows)
    longctx_df = _flatten_long_context(longctx_rows)

    # 1. Filter out historical duplicate sweeps, retaining only the latest run per step
    filtered_serving_df = _filter_latest_sweeps(raw_serving_df)

    # 2. Compute accurate saturation points, peak throughput, and energy metrics
    serving_df = _compute_saturation_and_derivatives(filtered_serving_df)

    outputs = {
        "serving": output_dir / "summary_serving.csv",
        "long_context": output_dir / "summary_long_context.csv",
        "hardware": output_dir / "summary_hardware.csv",
        "price_performance": output_dir / "summary_price_performance.csv",
    }

    summarize_serving(serving_df, outputs["serving"])
    summarize_long_context(longctx_df, outputs["long_context"])
    summarize_hardware(hw, outputs["hardware"])
    summarize_price_performance(serving_df, hw, outputs["price_performance"])

    return outputs