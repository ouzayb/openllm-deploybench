"""Generate matplotlib plots from benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deploybench.analysis import _flatten_long_context, _flatten_serving, _load_hardware
from deploybench.utils import find_jsonl_files, read_jsonl

logger = logging.getLogger(__name__)


def _load_serving_df(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in find_jsonl_files(results_dir):
        if "long_context" in str(path):
            continue
        data = read_jsonl(path)
        if data and data[0].get("benchmark_type") == "long_context_needle":
            continue
        rows.extend(data)
    return _flatten_serving(rows)


def _load_longctx_df(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in find_jsonl_files(results_dir):
        data = read_jsonl(path)
        if data and (data[0].get("benchmark_type") == "long_context_needle" or "long_context" in str(path)):
            rows.extend(data)
    return _flatten_long_context(rows)


def _bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    if df.empty or y not in df.columns or df[y].dropna().empty:
        logger.warning("Skipping plot %s: Column '%s' has no valid numerical data", path.name, y)
        return

    valid_df = df.dropna(subset=[y])
    agg = valid_df.groupby(x, dropna=False)[y].mean().reset_index()
    if agg.empty:
        logger.warning("Skipping plot %s: Aggregated data is empty", path.name)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(agg[x].astype(str), agg[y], color="#2b5c8f", edgecolor="black", width=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)

def _grouped_bar_plot(
    df: pd.DataFrame,
    category_col: str,
    group_col: str,
    metric_col: str,
    title: str,
    path: Path,
) -> None:
    """Plot cross-hardware comparison grouped by model to avoid averaging disparate workloads."""
    if df.empty or metric_col not in df.columns or df[metric_col].dropna().empty:
        return

    valid_df = df.dropna(subset=[metric_col])
    pivot_df = valid_df.pivot_table(
        index=category_col,
        columns=group_col,
        values=metric_col,
        aggfunc="mean",
    )
    if pivot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_df.plot(kind="bar", ax=ax, width=0.75, edgecolor="black")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Model Architecture / Quantization")
    ax.set_ylabel(metric_col.replace("metric_", "").replace("_", " ").title())
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Hardware Instance", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote grouped plot %s", path)

def run_plot(
    results_dir: Path,
    output_dir: Path,
    machine_id: str | None = None,  # Cihaz filtresi eklendi
) -> list[Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serving = _load_serving_df(results_dir)
    longctx = _load_longctx_df(results_dir)

    if machine_id:
        if not serving.empty and "machine_id" in serving.columns:
            serving = serving[serving["machine_id"] == machine_id]
        if not longctx.empty and "machine_id" in longctx.columns:
            longctx = longctx[longctx["machine_id"] == machine_id]
        logger.info("Filtered benchmark results for machine_id='%s'", machine_id)
    
    created: list[Path] = []

    if not serving.empty and serving["success"].any():
        ok = serving[serving["success"] == True]  # noqa: E712
        
        hw_col = "machine_label" if "machine_label" in ok.columns else "machine_id"

        # Case 1: Targeted single-device profiling
        if machine_id:
            plots = [
                (f"{machine_id}_throughput.png", "model_id", "metric_output_tokens_per_second", f"Throughput by Model on {machine_id}"),
                (f"{machine_id}_ttft_p95.png", "model_id", "metric_ttft_ms_p95", f"TTFT p95 by Model on {machine_id}"),
                (f"{machine_id}_tpot_p95.png", "model_id", "metric_tpot_ms_p95", f"TPOT p95 by Model on {machine_id}"),
            ]
            for fname, x, y, title in plots:
                p = output_dir / fname
                _bar_plot(ok, x, y, title, p)
                created.append(p)

            if "model_id" in ok.columns:
                p = output_dir / f"{machine_id}_peak_vram_by_model.png"
                _bar_plot(ok, "model_id", "metric_peak_vram_gb", f"Peak VRAM by Model on {machine_id}", p)
                created.append(p)

        # Case 2: Multi-device cross comparison (Research Question 1 & 2)
        else:
            cross_plots = [
                ("throughput_by_hardware.png", "model_id", hw_col, "metric_output_tokens_per_second", "Throughput by Hardware Across Models"),
                ("ttft_p95_by_hardware.png", "model_id", hw_col, "metric_ttft_ms_p95", "TTFT p95 Across Hardware by Model"),
                ("tpot_p95_by_hardware.png", "model_id", hw_col, "metric_tpot_ms_p95", "TPOT p95 Across Hardware by Model"),
            ]
            for fname, cat_col, grp_col, metric, title in cross_plots:
                p = output_dir / fname
                _grouped_bar_plot(ok, category_col=cat_col, group_col=grp_col, metric_col=metric, title=title, path=p)
                created.append(p)

            if "model_id" in ok.columns:
                p = output_dir / "peak_vram_by_model.png"
                _bar_plot(ok, "model_id", "metric_peak_vram_gb", "Peak VRAM by Model", p)
                created.append(p)

       
        summary_candidates = [
            results_dir.parent / "reports" / "summary_price_performance.csv",
            results_dir / ".." / "reports" / "summary_price_performance.csv",
            output_dir.parent / "summary_price_performance.csv",
        ]
        price_path = next((p for p in summary_candidates if p.exists()), None)

        if price_path:
            ppdf = pd.read_csv(price_path)
            target_label = "machine_label" if "machine_label" in ppdf.columns else "machine_id"
            if "tokens_per_dollar" in ppdf.columns and ppdf["tokens_per_dollar"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6))
                sub = ppdf[ppdf["tokens_per_dollar"].notna()]
                ax.bar(sub[target_label].astype(str), sub["tokens_per_dollar"], color="#2b5c8f")
                ax.set_title("Tokens per Dollar", fontsize=12, fontweight="bold")
                ax.set_ylabel("tokens / USD")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                p = output_dir / "tokens_per_dollar.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                created.append(p)

            if "relative_to_owned_h200" in ppdf.columns and ppdf["relative_to_owned_h200"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6))
                sub = ppdf[ppdf["relative_to_owned_h200"].notna()]
                bars = ax.bar(sub[target_label].astype(str), sub["relative_to_owned_h200"], color="#3d72a4", edgecolor="black", width=0.5)
                ax.set_title("Rented vs Owned H200 Relative Performance", fontsize=12, fontweight="bold")
                ax.set_ylabel("Relative Throughput (1.0 = Owned H200)")
                ax.axhline(1.0, linestyle="--", color="red", linewidth=1.5, label="Owned H200 Baseline (1.0)")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                ax.legend(loc="lower right")

                # Print value label on top of each bar
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

                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                p = output_dir / "owned_vs_rented_h200_relative_perf.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                created.append(p)

        if "quality_retention" in ok.columns and ok["quality_retention"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(ok["metric_output_tokens_per_second"], ok["quality_retention"])
            ax.set_xlabel("output tokens/s")
            ax.set_ylabel("quality retention")
            ax.set_title("Quality vs Throughput")
            fig.tight_layout()
            p = output_dir / "quality_vs_throughput.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(p)

    if not longctx.empty:
        ok = longctx[longctx["success"] == True] if "success" in longctx.columns else longctx  # noqa: E712
        if "context_length" in ok.columns:
            agg = ok.groupby("context_length")["exact_match"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(agg["context_length"], agg["exact_match"], marker="o")
            ax.set_title("Long Context Accuracy by Length")
            ax.set_xlabel("context length")
            ax.set_ylabel("accuracy")
            fig.tight_layout()
            p = output_dir / "long_context_accuracy_by_length.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(p)

        if "needle_position" in ok.columns:
            agg = ok.groupby("needle_position")["exact_match"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(agg["needle_position"], agg["exact_match"], marker="o")
            ax.set_title("Accuracy by Needle Position")
            ax.set_xlabel("needle position")
            ax.set_ylabel("accuracy")
            fig.tight_layout()
            p = output_dir / "needle_position_accuracy.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            created.append(p)

    return created
