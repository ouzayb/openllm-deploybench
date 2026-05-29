"""Summarize benchmark results into CSV reports."""

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
        rec = {**row}
        for k, v in m.items():
            rec[f"metric_{k}"] = v
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _flatten_long_context(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def summarize_serving(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    cols = [
        "machine_id", "machine_label", "provider", "location_type",
        "hourly_price_usd", "model_id", "workload_id", "concurrency",
        "max_model_len", "quantization", "success",
        "metric_output_tokens_per_second", "metric_requests_per_second",
        "metric_ttft_ms_p95", "metric_tpot_ms_p95",
        "metric_peak_vram_gb", "metric_avg_power_watts", "metric_energy_wh",
    ]
    existing = [c for c in cols if c in df.columns]
    summary = df[existing].copy()
    summary.to_csv(output_path, index=False)
    logger.info("Wrote %s", output_path)


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
        owned = df  # baseline machine results in this run
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
        rows = read_jsonl(path)
        if "long_context" in str(path) or rows and rows[0].get("benchmark_type") == "long_context_needle":
            longctx_rows.extend(rows)
        else:
            serving_rows.extend(rows)

    serving_df = _flatten_serving(serving_rows)
    longctx_df = _flatten_long_context(longctx_rows)

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
