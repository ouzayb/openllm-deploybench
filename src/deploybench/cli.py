"""Typer CLI for OpenLLM DeployBench."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from deploybench.analysis import run_summarize
from deploybench.benchmark_runner import run_serving_benchmark
from deploybench.config import load_hardware_config
from deploybench.hardware_probe import run_probe
from deploybench.long_context import run_long_context_benchmark
from deploybench.plotting import run_plot
from deploybench.quantization import run_quantization_benchmark
from deploybench.utils import PROJECT_ROOT, setup_logging

app = typer.Typer(
    name="deploybench",
    help="OpenLLM DeployBench - LLM deployment benchmark suite",
    no_args_is_help=True,
)
console = Console()


@app.command("probe-hardware")
def probe_hardware_cmd(
    output: Path = typer.Option(
        Path("results/hardware.json"),
        "--output", "-o",
        help="Output JSON path",
    ),
    hardware_config: Optional[Path] = typer.Option(
        None,
        "--hardware-config",
        help="Hardware metadata YAML",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Collect hardware fingerprint."""
    setup_logging(verbose)
    run_probe(output, hardware_config)
    console.print(f"[green]Hardware probe saved to[/green] {output}")


@app.command("run-serving")
def run_serving_cmd(
    config: Path = typer.Option(
        Path("configs/benchmark_matrix.yaml"),
        "--config", "-c",
    ),
    models_config: Path = typer.Option(
        Path("configs/models.yaml"),
        "--models-config",
    ),
    hardware_config: Optional[Path] = typer.Option(
        None,
        "--hardware-config",
    ),
    output_dir: Path = typer.Option(
        Path("results/serving"),
        "--output-dir", "-o",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run vLLM serving benchmark matrix."""
    setup_logging(verbose)
    hw = load_hardware_config(hardware_config) if hardware_config else None
    out = run_serving_benchmark(
        matrix_path=config,
        models_path=models_config,
        output_dir=output_dir,
        hardware_path=hardware_config,
        hardware_config=hw,
        cli_args=["deploybench", "run-serving"],
    )
    console.print(f"[green]Serving results written to[/green] {out}")


@app.command("run-long-context")
def run_long_context_cmd(
    config: Path = typer.Option(Path("configs/benchmark_matrix.yaml"), "--config", "-c"),
    models_config: Path = typer.Option(Path("configs/models.yaml"), "--models-config"),
    hardware_config: Optional[Path] = typer.Option(None, "--hardware-config"),
    output_dir: Path = typer.Option(Path("results/long_context"), "--output-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run needle-in-a-haystack long context benchmark."""
    setup_logging(verbose)
    hw = load_hardware_config(hardware_config) if hardware_config else None
    out = run_long_context_benchmark(
        matrix_path=config,
        models_path=models_config,
        output_dir=output_dir,
        hardware_path=hardware_config,
        hardware_config=hw,
        cli_args=["deploybench", "run-long-context"],
    )
    console.print(f"[green]Long context results in[/green] {out}")


@app.command("run-quantization")
def run_quantization_cmd(
    config: Path = typer.Option(Path("configs/benchmark_matrix.yaml"), "--config", "-c"),
    models_config: Path = typer.Option(Path("configs/models.yaml"), "--models-config"),
    hardware_config: Optional[Path] = typer.Option(None, "--hardware-config"),
    output_dir: Path = typer.Option(Path("results/quantization"), "--output-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run quantization comparison benchmarks."""
    setup_logging(verbose)
    out = run_quantization_benchmark(
        matrix_path=config,
        models_path=models_config,
        output_dir=output_dir,
        hardware_path=hardware_config,
        cli_args=["deploybench", "run-quantization"],
    )
    console.print(f"[green]Quantization results in[/green] {out}")


@app.command("summarize")
def summarize_cmd(
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r"),
    output_dir: Path = typer.Option(Path("reports"), "--output-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate CSV summary reports."""
    setup_logging(verbose)
    outputs = run_summarize(results_dir, output_dir)
    for name, path in outputs.items():
        console.print(f"  [cyan]{name}[/cyan] -> {path}")


@app.command("plot")
def plot_cmd(
    results_dir: Path = typer.Option(Path("results"), "--results-dir", "-r"),
    output_dir: Path = typer.Option(Path("reports/figures"), "--output-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate matplotlib plots."""
    setup_logging(verbose)
    created = run_plot(results_dir, output_dir)
    for p in created:
        console.print(f"  [cyan]plot[/cyan] -> {p}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
