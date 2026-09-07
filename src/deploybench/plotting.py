"""Generate standardized, annotated matplotlib plots across hardware and models."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deploybench.analysis import _flatten_long_context, _flatten_serving
from deploybench.utils import find_jsonl_files, read_jsonl

logger = logging.getLogger(__name__)


def _load_serving_df(results_dir: Path, exclude_active_last_file: bool = True) -> pd.DataFrame:
    """Load and aggregate all serving benchmark jsonl files, excluding ongoing runs."""
    all_files = sorted(find_jsonl_files(results_dir), key=lambda p: p.stat().st_mtime)
    if not all_files:
        return pd.DataFrame()

    # Exclude the latest modified file to avoid parsing incomplete active writes
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
            rows.extend(data)
    return _flatten_long_context(rows)


def _prepare_cleaned_df(df: pd.DataFrame, machine_id: str | None = None) -> pd.DataFrame:
    """Clean numeric types, filter successful runs, and resolve hardware display labels."""
    if df.empty or "success" not in df.columns:
        return pd.DataFrame()

    ok = df[df["success"] == True].copy()  # noqa: E712
    if ok.empty:
        return pd.DataFrame()

    # Hardware label fallback
    if "machine_label" not in ok.columns or ok["machine_label"].isna().all():
        ok["hw_label"] = ok["machine_id"]
    else:
        ok["hw_label"] = ok["machine_label"].fillna(ok["machine_id"])

    if machine_id:
        ok = ok[ok["machine_id"] == machine_id]

    numeric_cols = [
        "concurrency",
        "metric_output_tokens_per_second",
        "metric_ttft_ms_p95",
        "metric_tpot_ms_p95",
        "metric_peak_vram_gb",
        "quality_retention",
    ]
    for col in numeric_cols:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")

    return ok


def _plot_concurrency_curves(
    df: pd.DataFrame,
    metric_col: str,
    title_metric: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Plot concurrency scaling curves with distinct hardware and workload labels."""
    if df.empty or metric_col not in df.columns or "concurrency" not in df.columns:
        return

    agg = (
        df.groupby(["hw_label", "model_id", "workload_id", "concurrency"], as_index=False)[metric_col]
        .mean()
        .dropna(subset=[metric_col, "concurrency"])
    )
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    groups = agg.groupby(["hw_label", "model_id", "workload_id"])

    for (hw, model, workload), grp in groups:
        grp_sorted = grp.sort_values("concurrency")
        trace_label = f"[{hw}] {model} ({workload})"
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

    agg = (
        df.groupby([category_col, group_col], as_index=False)[metric_col]
        .max()
        .dropna(subset=[metric_col])
    )
    if agg.empty:
        return

    pivot = agg.pivot(index=category_col, columns=group_col, values=metric_col)
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


def run_plot(
    results_dir: Path,
    output_dir: Path,
    machine_id: str | None = None,
) -> list[Path]:
    """Main plotting entrypoint invoked by the CLI or runner."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_serving = _load_serving_df(results_dir, exclude_active_last_file=True)
    ok = _prepare_cleaned_df(raw_serving, machine_id=machine_id)
    longctx = _load_longctx_df(results_dir)

    if machine_id and not longctx.empty and "machine_id" in longctx.columns:
        longctx = longctx[longctx["machine_id"] == machine_id]

    created: list[Path] = []

    if not ok.empty:
        prefix = f"{machine_id}_" if machine_id else "cross_hardware_"
        hw_tag = f"[{ok['hw_label'].iloc[0]}]" if machine_id else "[Cross Hardware]"

        # -------------------------------------------------------------
        # 1. Concurrency Curves (Throughput, TTFT, TPOT)
        # -------------------------------------------------------------
        scaling_plots = [
            (f"{prefix}concurrency_throughput.png", "metric_output_tokens_per_second", f"{hw_tag} Throughput Scaling", "Output Tokens / sec"),
            (f"{prefix}concurrency_ttft_p95.png", "metric_ttft_ms_p95", f"{hw_tag} P95 TTFT Latency", "P95 TTFT (ms)"),
            (f"{prefix}concurrency_tpot_p95.png", "metric_tpot_ms_p95", f"{hw_tag} P95 TPOT Latency", "P95 TPOT (ms)"),
        ]
        for fname, metric, title, ylabel in scaling_plots:
            p = output_dir / fname
            _plot_concurrency_curves(ok, metric, title, ylabel, p)
            created.append(p)

        # -------------------------------------------------------------
        # 2. Workload Breakdown (Chat vs Coding vs RAG)
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
        # 3. Peak Hardware Comparison (Across Devices)
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
        # 4. Peak VRAM Footprint
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
        # 5. Price-Performance & Owned vs Rented Relative Plots
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
        # 6. Quality Retention vs Throughput
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
    # 7. Long Context Needle-in-a-Haystack Metrics
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