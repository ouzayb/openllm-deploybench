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
    if df.empty or y not in df.columns:
        logger.warning("Skipping plot %s (no data)", path.name)
        return
    agg = df.groupby(x, dropna=False)[y].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(agg[x].astype(str), agg[y])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)


def run_plot(results_dir: Path, output_dir: Path) -> list[Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serving = _load_serving_df(results_dir)
    longctx = _load_longctx_df(results_dir)
    created: list[Path] = []

    if not serving.empty and serving["success"].any():
        ok = serving[serving["success"] == True]  # noqa: E712
        label_col = "machine_label" if "machine_label" in ok.columns else "machine_id"

        plots = [
            ("throughput_by_hardware.png", label_col, "metric_output_tokens_per_second", "Output Throughput by Hardware"),
            ("ttft_p95_by_hardware.png", label_col, "metric_ttft_ms_p95", "TTFT p95 by Hardware"),
            ("tpot_p95_by_hardware.png", label_col, "metric_tpot_ms_p95", "TPOT p95 by Hardware"),
        ]
        for fname, x, y, title in plots:
            p = output_dir / fname
            _bar_plot(ok, x, y, title, p)
            created.append(p)

        if "model_id" in ok.columns:
            p = output_dir / "peak_vram_by_model.png"
            _bar_plot(ok, "model_id", "metric_peak_vram_gb", "Peak VRAM by Model", p)
            created.append(p)

        price_path = results_dir.parent / "reports" / "summary_price_performance.csv"
        if not price_path.exists():
            price_path = results_dir / ".." / "reports" / "summary_price_performance.csv"
        pp = Path(results_dir).parent / "reports" / "summary_price_performance.csv"
        if pp.exists():
            ppdf = pd.read_csv(pp)
            if "tokens_per_dollar" in ppdf.columns and ppdf["tokens_per_dollar"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6))
                sub = ppdf[ppdf["tokens_per_dollar"].notna()]
                ax.bar(sub[label_col].astype(str) if label_col in sub.columns else sub["machine_id"].astype(str),
                       sub["tokens_per_dollar"])
                ax.set_title("Tokens per Dollar")
                ax.set_ylabel("tokens / USD")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                p = output_dir / "tokens_per_dollar.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                created.append(p)

            if "relative_to_owned_h200" in ppdf.columns and ppdf["relative_to_owned_h200"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6))
                sub = ppdf[ppdf["relative_to_owned_h200"].notna()]
                ax.bar(sub["machine_id"].astype(str), sub["relative_to_owned_h200"])
                ax.set_title("Rented vs Owned H200 Relative Performance")
                ax.set_ylabel("relative throughput")
                ax.axhline(1.0, linestyle="--", color="gray")
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
