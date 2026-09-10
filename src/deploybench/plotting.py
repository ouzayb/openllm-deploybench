"""Generate standardized, annotated matplotlib plots and summary tables across hardware and models."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deploybench.analysis import _flatten_long_context, _flatten_serving
from deploybench.utils import find_jsonl_files, read_jsonl

logger = logging.getLogger(__name__)


def _load_serving_df(results_dir: Path, exclude_active_last_file: bool = False) -> pd.DataFrame:
    """Load and aggregate all serving benchmark jsonl files."""
    all_files = sorted(find_jsonl_files(results_dir), key=lambda p: p.stat().st_mtime)
    if not all_files:
        return pd.DataFrame()

    target_files = all_files[:-1] if (exclude_active_last_file and len(all_files) > 1) else all_files

    rows: list[dict] = []
    for path in target_files:
        if "long_context" in str(path):
            continue
        data = read_jsonl(path)
        if data and data[0].get("benchmark_type") == "long_context_needle":
            continue
        rows.extend(data)

    if not rows:
        return pd.DataFrame()
    return _flatten_serving(rows)


def _load_longctx_df(results_dir: Path) -> pd.DataFrame:
    """Load long-context needle-in-a-haystack benchmark files."""
    rows: list[dict] = []
    for path in find_jsonl_files(results_dir):
        data = read_jsonl(path)
        if data and (data[0].get("benchmark_type") == "long_context_needle" or "long_context" in str(path)):
            continue
        rows.extend(data)
    return _flatten_long_context(rows)


def _prepare_cleaned_df(
    df: pd.DataFrame, 
    machine_id: str | None = None,
    keep_latest_only: bool = True,
) -> pd.DataFrame:
    """Clean numeric types, filter successful runs, and retain latest benchmark data."""
    if df.empty or "success" not in df.columns:
        return pd.DataFrame()

    ok = df[df["success"] == True].copy()  # noqa: E712
    if ok.empty:
        return pd.DataFrame()

    if "machine_label" not in ok.columns or ok["machine_label"].isna().all():
        ok["hw_label"] = ok["machine_id"]
    else:
        ok["hw_label"] = ok["machine_label"].fillna(ok["machine_id"])

    if machine_id:
        ok = ok[ok["machine_id"] == machine_id]

    numeric_cols = [
        "concurrency",
        "metric_output_tokens_per_second",
        "metric_total_tokens_per_second",
        "metric_ttft_ms_p50",
        "metric_ttft_ms_p95",
        "metric_tpot_ms_p50",
        "metric_tpot_ms_p95",
        "metric_peak_vram_gb",
        "metric_avg_power_watts",
        "quality_retention",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    if keep_latest_only and "timestamp_utc" in ok.columns:
        ok["timestamp_utc"] = pd.to_datetime(ok["timestamp_utc"], errors="coerce")
        ok = (
            ok.sort_values("timestamp_utc")
            .groupby(["machine_id", "model_id", "workload_id", "concurrency"], as_index=False)
            .last()
        )
        logger.info("Filtered out older runs; retained %d latest benchmark points.", len(ok))

    return ok


def _plot_concurrency_curves(
    df: pd.DataFrame,
    metric_col: str,
    title_metric: str,
    y_label: str,
    output_path: Path,
    include_workload_in_label: bool = True,
) -> None:
    """Plot concurrency scaling curves with distinct hardware and workload labels."""
    if df.empty or metric_col not in df.columns or "concurrency" not in df.columns:
        return

    group_keys = ["hw_label", "model_id"]
    if include_workload_in_label and "workload_id" in df.columns:
        group_keys.append("workload_id")

    agg = (
        df.groupby(group_keys + ["concurrency"], as_index=False)[metric_col]
        .mean()
        .dropna(subset=[metric_col, "concurrency"])
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    groups = agg.groupby(group_keys)

    for keys, grp in groups:
        if isinstance(keys, str):
            keys = (keys,)
        hw, model = keys[0], keys[1]
        workload = keys[2] if len(keys) > 2 else None

        trace_label = f"[{hw}] {model}"
        if workload:
            trace_label += f" ({workload})"

        grp_sorted = grp.sort_values("concurrency")
        ax.plot(
            grp_sorted["concurrency"],
            grp_sorted[metric_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=trace_label,
        )
        for _, row in grp_sorted.iterrows():
            val = row[metric_col]
            text = f"{int(val)}" if val >= 10 else f"{val:.1f}"
            ax.annotate(
                text,
                (row["concurrency"], val),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                fontweight="bold",
            )

    ax.set_title(f"{title_metric} vs Concurrency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency (Concurrent Streams)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(agg["concurrency"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def _plot_concurrency_curves_zoomed(
    df: pd.DataFrame,
    metric_col: str,
    title_metric: str,
    y_label: str,
    output_path: Path,
    max_y_limit: float = 45000.0,
    include_workload_in_label: bool = True,
) -> None:
    """Plot zoomed-in concurrency curves filtering out extreme tail latencies for clarity."""
    if df.empty or metric_col not in df.columns or "concurrency" not in df.columns:
        return

    group_keys = ["hw_label", "model_id"]
    if include_workload_in_label and "workload_id" in df.columns:
        group_keys.append("workload_id")

    agg = (
        df.groupby(group_keys + ["concurrency"], as_index=False)[metric_col]
        .mean()
        .dropna(subset=[metric_col, "concurrency"])
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    groups = agg.groupby(group_keys)

    for keys, grp in groups:
        if isinstance(keys, str):
            keys = (keys,)
        hw, model = keys[0], keys[1]
        workload = keys[2] if len(keys) > 2 else None

        trace_label = f"[{hw}] {model}"
        if workload:
            trace_label += f" ({workload})"

        grp_sorted = grp.sort_values("concurrency")
        ax.plot(
            grp_sorted["concurrency"],
            grp_sorted[metric_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=trace_label,
        )
        for _, row in grp_sorted.iterrows():
            val = row[metric_col]
            if val <= max_y_limit:
                text = f"{int(val)}" if val >= 10 else f"{val:.1f}"
                ax.annotate(
                    text,
                    (row["concurrency"], val),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_title(f"{title_metric} vs Concurrency (Zoomed View)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency (Concurrent Streams)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(agg["concurrency"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, max_y_limit)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Wrote zoomed plot %s", output_path)


def _plot_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Generate a heatmap representation of metrics across configurations and concurrency."""
    if df.empty or metric_col not in df.columns or "concurrency" not in df.columns:
        return

    df["config_label"] = "[" + df["hw_label"].astype(str) + "] " + df["model_id"].astype(str) + " (" + df["workload_id"].astype(str) + ")"
    
    pivot = df.pivot_table(index="config_label", columns="concurrency", values=metric_col, aggfunc="mean")
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)), dpi=150)
    cax = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    plt.xticks(rotation=30, ha="right")
    ax.set_title(f"{title} (Heatmap)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency", fontsize=10)
    ax.set_ylabel("Configuration", fontsize=10)

    # Annotate values inside cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                text_color = "white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black"
                ax.text(j, i, f"{int(val)}" if val >= 10 else f"{val:.1f}",
                        ha="center", va="center", color=text_color, fontsize=7, fontweight="bold")

    fig.colorbar(cax, ax=ax, label=metric_col.replace("metric_", "").replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Wrote heatmap %s", output_path)


def _plot_grouped_bar(
    df: pd.DataFrame,
    category_col: str,
    group_col: str,
    metric_col: str,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Grouped bar plot comparing metrics across hardware instances or workloads."""
    if df.empty or metric_col not in df.columns or category_col not in df.columns:
        return

    valid_df = df.dropna(subset=[metric_col]).copy()
    if valid_df.empty:
        return

    agg = valid_df.groupby([category_col, group_col], as_index=False)[metric_col].max()
    if agg.empty:
        return

    pivot = agg.pivot(index=category_col, columns=group_col, values=metric_col).fillna(0)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    pivot.plot(kind="bar", ax=ax, width=0.75, edgecolor="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xlabel(category_col.replace("_", " ").title(), fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title=group_col.replace("_", " ").title(), frameon=True)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Wrote %s", output_path)


def _export_summary_tables(df: pd.DataFrame, output_dir: Path, prefix: str) -> list[Path]:
    """Generate markdown and CSV summary tables across models, workloads, and concurrency levels."""
    if df.empty:
        return []

    cols_to_keep = [
        "hw_label",
        "model_id",
        "workload_id",
        "concurrency",
        "metric_output_tokens_per_second",
        "metric_ttft_ms_p50",
        "metric_ttft_ms_p95",
        "metric_tpot_ms_p50",
        "metric_tpot_ms_p95",
        "metric_peak_vram_gb",
        "metric_avg_power_watts",
    ]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    table_df = df[available_cols].copy()

    rename_map = {
        "hw_label": "Hardware",
        "model_id": "Model",
        "workload_id": "Workload",
        "concurrency": "Concurrency",
        "metric_output_tokens_per_second": "Output Tok/s",
        "metric_ttft_ms_p50": "TTFT P50 (ms)",
        "metric_ttft_ms_p95": "TTFT P95 (ms)",
        "metric_tpot_ms_p50": "TPOT P50 (ms)",
        "metric_tpot_ms_p95": "TPOT P95 (ms)",
        "metric_peak_vram_gb": "Peak VRAM (GB)",
        "metric_avg_power_watts": "Avg Power (W)",
    }
    table_df = table_df.rename(columns=rename_map)
    table_df = table_df.sort_values(by=["Workload", "Model", "Concurrency"], ascending=[True, True, True])

    csv_path = output_dir / f"{prefix}benchmark_summary_table.csv"
    md_path = output_dir / f"{prefix}benchmark_summary_table.md"

    table_df.round(2).to_csv(csv_path, index=False)
    
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Benchmark Summary Table\n\n")
        f.write(table_df.round(2).to_markdown(index=False))

    logger.info("Exported summary tables: %s and %s", csv_path, md_path)
    return [csv_path, md_path]


def run_plot(
    results_dir: Path,
    output_dir: Path,
    machine_id: str | None = None,
) -> list[Path]:
    """Main plotting entrypoint invoked by the CLI or runner."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_serving = _load_serving_df(results_dir, exclude_active_last_file=False)
    ok = _prepare_cleaned_df(raw_serving, machine_id=machine_id, keep_latest_only=True)
    longctx = _load_longctx_df(results_dir)

    if machine_id and not longctx.empty and "machine_id" in longctx.columns:
        longctx = longctx[longctx["machine_id"] == machine_id]

    created: list[Path] = []

    if not ok.empty:
        prefix = f"{machine_id}_" if machine_id else "cross_hardware_"
        hw_tag = f"[{ok['hw_label'].iloc[0]}]" if machine_id else "[Cross Hardware]"

        # -------------------------------------------------------------
        # 1. Overall Combined Concurrency Curves & Heatmaps
        # -------------------------------------------------------------
        scaling_plots = [
            (f"{prefix}concurrency_throughput_all.png", "metric_output_tokens_per_second", f"{hw_tag} Overall Throughput Scaling", "Output Tokens / sec"),
            (f"{prefix}concurrency_ttft_p95_all.png", "metric_ttft_ms_p95", f"{hw_tag} Overall P95 TTFT Latency", "P95 TTFT (ms)"),
            (f"{prefix}concurrency_tpot_p95_all.png", "metric_tpot_ms_p95", f"{hw_tag} Overall P95 TPOT Latency", "P95 TPOT (ms)"),
        ]
        for fname, metric, title, ylabel in scaling_plots:
            p = output_dir / fname
            _plot_concurrency_curves(ok, metric, title, ylabel, p, include_workload_in_label=True)
            created.append(p)

            # Heatmap eşlikçisi
            p_hm = output_dir / fname.replace(".png", "_heatmap.png")
            _plot_heatmap(ok, metric, title, p_hm)
            created.append(p_hm)

        # Zoomed TTFT plot for lower dense curves
        p_zoomed = output_dir / f"{prefix}concurrency_ttft_p95_all_zoomed.png"
        _plot_concurrency_curves_zoomed(
            ok, 
            "metric_ttft_ms_p95", 
            f"{hw_tag} Overall P95 TTFT Latency (Zoomed)", 
            "P95 TTFT (ms)", 
            p_zoomed, 
            max_y_limit=45000.0,
            include_workload_in_label=True
        )
        created.append(p_zoomed)

        # -------------------------------------------------------------
        # 2. Per-Workload Isolated Curves
        # -------------------------------------------------------------
        if "workload_id" in ok.columns:
            workloads_dir = output_dir / "workloads"
            workloads_dir.mkdir(parents=True, exist_ok=True)

            for w_id, w_df in ok.groupby("workload_id"):
                w_tag = f"{hw_tag} [{w_id.upper()}]"
                w_prefix = f"{prefix}{w_id}_"

                w_plots = [
                    (f"{w_prefix}throughput.png", "metric_output_tokens_per_second", f"{w_tag} Throughput Scaling", "Output Tokens / sec"),
                    (f"{w_prefix}ttft_p95.png", "metric_ttft_ms_p95", f"{w_tag} P95 TTFT Latency", "P95 TTFT (ms)"),
                    (f"{w_prefix}tpot_p95.png", "metric_tpot_ms_p95", f"{w_tag} P95 TPOT Latency", "P95 TPOT (ms)"),
                ]
                for fname, metric, title, ylabel in w_plots:
                    p = workloads_dir / fname
                    _plot_concurrency_curves(w_df, metric, title, ylabel, p, include_workload_in_label=False)
                    created.append(p)

        # -------------------------------------------------------------
        # 3. Workload Breakdown (Grouped Bars)
        # -------------------------------------------------------------
        if "workload_id" in ok.columns:
            ok["model_with_hw"] = ok["hw_label"] + " | " + ok["model_id"]
            p = output_dir / f"{prefix}workload_peak_throughput.png"
            _plot_grouped_bar(
                ok,
                category_col="model_with_hw",
                group_col="workload_id",
                metric_col="metric_output_tokens_per_second",
                title=f"{hw_tag} Peak Throughput by Workload",
                y_label="Peak Tokens / sec",
                output_path=p,
            )
            created.append(p)

        # -------------------------------------------------------------
        # 4. Cross Hardware Comparison
        # -------------------------------------------------------------
        if not machine_id and "hw_label" in ok.columns:
            p = output_dir / "throughput_by_hardware.png"
            _plot_grouped_bar(
                ok,
                category_col="model_id",
                group_col="hw_label",
                metric_col="metric_output_tokens_per_second",
                title="Throughput by Hardware Across Models",
                y_label="Max Tokens / sec",
                output_path=p,
            )
            created.append(p)

            p = output_dir / "ttft_p95_by_hardware.png"
            _plot_grouped_bar(
                ok,
                category_col="model_id",
                group_col="hw_label",
                metric_col="metric_ttft_ms_p95",
                title="P95 TTFT Latency Across Hardware by Model",
                y_label="P95 TTFT (ms)",
                output_path=p,
            )
            created.append(p)

        # -------------------------------------------------------------
        # 5. Peak VRAM Footprint
        # -------------------------------------------------------------
        if "metric_peak_vram_gb" in ok.columns:
            ok["model_with_hw"] = ok["hw_label"] + " | " + ok["model_id"]
            agg_vram = ok.groupby(["model_with_hw", "hw_label"], as_index=False)["metric_peak_vram_gb"].max().dropna()
            if not agg_vram.empty:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                bars = ax.bar(agg_vram["model_with_hw"], agg_vram["metric_peak_vram_gb"], color="#2b5c8f", edgecolor="black", width=0.6)
                ax.set_title(f"{hw_tag} Peak VRAM Consumption (GB)", fontsize=12, fontweight="bold")
                ax.set_ylabel("VRAM (GB)", fontsize=10)
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                plt.xticks(rotation=25, ha="right")
                ax.bar_label(bars, fmt="%.1f GB", padding=3, fontsize=8, fontweight="bold")
                fig.tight_layout()
                p = output_dir / f"{prefix}peak_vram_by_model.png"
                fig.savefig(p)
                plt.close(fig)
                created.append(p)

        # -------------------------------------------------------------
        # 6. Price-Performance & Relative Plots
        # -------------------------------------------------------------
        summary_candidates = [
            results_dir.parent / "reports" / "summary_price_performance.csv",
            results_dir / ".." / "reports" / "summary_price_performance.csv",
            output_dir.parent / "summary_price_performance.csv",
        ]
        price_path = next((path for path in summary_candidates if path.exists()), None)

        if price_path:
            ppdf = pd.read_csv(price_path)
            target_label = "machine_label" if "machine_label" in ppdf.columns else "machine_id"
            if "tokens_per_dollar" in ppdf.columns and ppdf["tokens_per_dollar"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                sub = ppdf[ppdf["tokens_per_dollar"].notna()]
                bars = ax.bar(sub[target_label].astype(str), sub["tokens_per_dollar"], color="#2b5c8f", edgecolor="black", width=0.5)
                ax.set_title("Tokens per Dollar Across Platforms", fontsize=12, fontweight="bold")
                ax.set_ylabel("Tokens / USD", fontsize=10)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=30, ha="right")
                ax.bar_label(bars, fmt="%d", padding=3, fontsize=8, fontweight="bold")
                fig.tight_layout()
                p = output_dir / "tokens_per_dollar.png"
                fig.savefig(p)
                plt.close(fig)
                created.append(p)

            if "relative_to_owned_h200" in ppdf.columns and ppdf["relative_to_owned_h200"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                sub = ppdf[ppdf["relative_to_owned_h200"].notna()]
                bars = ax.bar(sub[target_label].astype(str), sub["relative_to_owned_h200"], color="#3d72a4", edgecolor="black", width=0.5)
                ax.set_title("Rented vs Owned H200 Relative Performance", fontsize=12, fontweight="bold")
                ax.set_ylabel("Relative Throughput (1.0 = Owned H200)", fontsize=10)
                ax.axhline(1.0, linestyle="--", color="red", linewidth=1.5, label="Owned H200 Baseline (1.0)")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.legend(loc="lower right")
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )
                plt.xticks(rotation=30, ha="right")
                fig.tight_layout()
                p = output_dir / "owned_vs_rented_h200_relative_perf.png"
                fig.savefig(p)
                plt.close(fig)
                created.append(p)

        # -------------------------------------------------------------
        # 7. Quality Retention vs Throughput
        # -------------------------------------------------------------
        if "quality_retention" in ok.columns and ok["quality_retention"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.scatter(ok["metric_output_tokens_per_second"], ok["quality_retention"], color="#2b5c8f", s=50)
            for _, r in ok.dropna(subset=["quality_retention", "metric_output_tokens_per_second"]).iterrows():
                ax.annotate(
                    f"[{r['hw_label']}] {r['model_id']}",
                    (r["metric_output_tokens_per_second"], r["quality_retention"]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    fontsize=7,
                )
            ax.set_xlabel("Output Tokens / sec", fontsize=10)
            ax.set_ylabel("Quality Retention", fontsize=10)
            ax.set_title(f"{hw_tag} Quality Retention vs Throughput", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            p = output_dir / f"{prefix}quality_vs_throughput.png"
            fig.savefig(p)
            plt.close(fig)
            created.append(p)

        # -------------------------------------------------------------
        # 8. Summary Table Export (CSV & Markdown)
        # -------------------------------------------------------------
        tables = _export_summary_tables(ok, output_dir, prefix)
        created.extend(tables)

    # -------------------------------------------------------------
    # 9. Long Context Needle-in-a-Haystack Metrics
    # -------------------------------------------------------------
    if not longctx.empty:
        ok_lc = longctx[longctx["success"] == True] if "success" in longctx.columns else longctx  # noqa: E712
        if "context_length" in ok_lc.columns and "exact_match" in ok_lc.columns:
            agg = ok_lc.groupby("context_length")["exact_match"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.plot(agg["context_length"], agg["exact_match"], marker="o", linewidth=2, color="#2ca02c")
            ax.set_title("Long Context Accuracy by Context Length", fontsize=12, fontweight="bold")
            ax.set_xlabel("Context Length (Tokens)", fontsize=10)
            ax.set_ylabel("Accuracy (Exact Match)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            p = output_dir / "long_context_accuracy_by_length.png"
            fig.savefig(p)
            plt.close(fig)
            created.append(p)

        if "needle_position" in ok_lc.columns and "exact_match" in ok_lc.columns:
            agg = ok_lc.groupby("needle_position")["exact_match"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            ax.plot(agg["needle_position"], agg["exact_match"], marker="o", linewidth=2, color="#d62728")
            ax.set_title("Long Context Accuracy by Needle Depth Position", fontsize=12, fontweight="bold")
            ax.set_xlabel("Needle Depth Position (%)", fontsize=10)
            ax.set_ylabel("Accuracy (Exact Match)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            fig.tight_layout()
            p = output_dir / "needle_position_accuracy.png"
            fig.savefig(p)
            plt.close(fig)
            created.append(p)

    return created