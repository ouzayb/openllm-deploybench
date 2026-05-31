"""vLLM server lifecycle and benchmark subprocess execution."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

from deploybench.gpu_monitor import GPUMonitor
from deploybench.metrics import classify_error, gpu_summary_to_metrics, parse_vllm_bench_output
from deploybench.result_schema import BenchmarkMetrics

logger = logging.getLogger(__name__)

_active_server: subprocess.Popen | None = None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError) as e:
        logger.debug("Kill process: %s", e)
        try:
            proc.kill()
        except OSError:
            pass


def _cleanup_server() -> None:
    global _active_server
    if _active_server is not None:
        _kill_process_tree(_active_server)
        _active_server = None


atexit.register(_cleanup_server)


def _tail_log(log_path: Path, lines: int = 40) -> str:
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except OSError:
        return ""


def build_serve_command(
    hf_id: str,
    dtype: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    quantization: str | None,
    trust_remote_code: bool,
    port: int,
    host: str,
    enforce_eager: bool,
) -> list[str]:
    # vLLM 0.22+: prefer `vllm serve <model>` (openai.api_server module is deprecated)
    cmd = [
        sys.executable, "-m", "vllm", "serve", hf_id,
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--host", host,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if enforce_eager:
        cmd.append("--enforce-eager")
    if quantization:
        cmd.extend(["--quantization", quantization])
    return cmd


def build_serve_command_legacy(
    hf_id: str,
    dtype: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    quantization: str | None,
    trust_remote_code: bool,
    port: int,
    host: str,
    enforce_eager: bool,
) -> list[str]:
    """Fallback for older vLLM (<0.22)."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_id,
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--host", host,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if enforce_eager:
        cmd.append("--enforce-eager")
    if quantization:
        cmd.extend(["--quantization", quantization])
    return cmd


def start_vllm_server(
    hf_id: str,
    dtype: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    quantization: str | None,
    trust_remote_code: bool,
    port: int,
    host: str,
    enforce_eager: bool,
    log_path: Path,
    startup_timeout_sec: int = 600,
) -> tuple[bool, str, list[str]]:
    """Try vllm serve (0.22+), then legacy api_server."""
    attempts = [
        build_serve_command(
            hf_id, dtype, max_model_len, tensor_parallel_size,
            gpu_memory_utilization, quantization, trust_remote_code,
            port, host, enforce_eager,
        ),
        build_serve_command_legacy(
            hf_id, dtype, max_model_len, tensor_parallel_size,
            gpu_memory_utilization, quantization, trust_remote_code,
            port, host, enforce_eager,
        ),
    ]
    last_err = ""
    for cmd in attempts:
        ok, err = start_server(cmd, log_path, startup_timeout_sec, host, port)
        if ok:
            return True, "", cmd
        last_err = err
        logger.warning("Server start failed (%s), trying next command variant...", cmd[2:4])
    return False, last_err, attempts[-1]


def _vllm_subprocess_env() -> dict[str, str]:
    """Environment for vLLM child processes."""
    env = os.environ.copy()
    # FlashInfer JIT needs nvcc; fall back to PyTorch sampler if CUDA toolkit missing
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def start_server(
    cmd: list[str],
    log_path: Path,
    startup_timeout_sec: int = 600,
    health_host: str = "127.0.0.1",
    health_port: int = 8000,
) -> tuple[bool, str]:
    global _active_server
    _cleanup_server()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    vllm_env = _vllm_subprocess_env()
    try:
        if sys.platform != "win32":
            _active_server = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=vllm_env,
                preexec_fn=os.setsid,
            )
        else:
            _active_server = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=vllm_env,
            )
    except Exception as e:
        log_file.close()
        return False, str(e)

    deadline = time.time() + startup_timeout_sec
    health_urls = [
        f"http://{health_host}:{health_port}/health",
        f"http://{health_host}:{health_port}/v1/models",
    ]
    last_err = ""
    while time.time() < deadline:
        if _active_server.poll() is not None:
            log_file.close()
            tail = _tail_log(log_path)
            msg = f"Server exited early with code {_active_server.returncode}"
            if tail:
                msg += f"\n--- last lines of {log_path} ---\n{tail}"
            return False, msg
        for url in health_urls:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code < 500:
                    logger.info("Server healthy at %s", url)
                    return True, ""
            except requests.RequestException as e:
                last_err = str(e)
        time.sleep(2)

    log_file.close()
    return False, f"Health check timeout: {last_err}"


def stop_server() -> None:
    _cleanup_server()


def run_bench_serve(
    hf_id: str,
    dataset_path: Path,
    num_prompts: int,
    max_concurrency: int,
    host: str,
    port: int,
    output_tokens: int,
    seed: int = 42,
    result_dir: Path | None = None,
) -> dict[str, Any]:
    """Run vllm bench serve against a running server."""
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "bench_result.json"
        cmd = [
            sys.executable, "-m", "vllm.benchmarks.bench_serve",
            "--backend", "vllm",
            "--model", hf_id,
            "--endpoint", "/v1/completions",
            "--dataset-name", "custom",
            "--dataset-path", str(dataset_path),
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(max_concurrency),
            "--port", str(port),
            "--host", host,
            "--seed", str(seed),
            "--custom-output-len", str(output_tokens),
            "--custom-skip-chat-template",
        ]
        # Try alternate module path for different vLLM versions
        alt_cmds = [
            ["vllm", "bench", "serve"] + cmd[4:],
            cmd,
        ]
        last_result: dict[str, Any] = {"stdout": "", "stderr": "", "returncode": -1}
        for attempt_cmd in alt_cmds:
            try:
                proc = subprocess.run(
                    attempt_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                    env=_vllm_subprocess_env(),
                )
                last_result = {
                    "command": " ".join(attempt_cmd),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                }
                if proc.returncode == 0:
                    break
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                last_result["error"] = "benchmark timeout"
                break

        metrics = parse_vllm_bench_output(
            last_result.get("stdout", ""),
            last_result.get("stderr", ""),
        )

        # Try loading saved result JSON if vLLM wrote one
        if result_dir:
            for p in sorted(result_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    import json

                    data = json.loads(p.read_text(encoding="utf-8"))
                    metrics = parse_vllm_bench_output(json.dumps(data))
                    last_result["saved_result"] = str(p)
                    break
                except Exception:
                    pass

        return {"metrics": metrics, "raw": last_result}


def run_bench_throughput_offline(
    hf_id: str,
    prompt_tokens: int,
    output_tokens: int,
    num_prompts: int,
    dtype: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    quantization: str | None,
    trust_remote_code: bool,
    seed: int,
    enforce_eager: bool,
    monitor: GPUMonitor | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "vllm.benchmarks.bench_throughput",
        "--model", hf_id,
        "--dataset-name", "random",
        "--random-input-len", str(prompt_tokens),
        "--random-output-len", str(output_tokens),
        "--num-prompts", str(num_prompts),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--seed", str(seed),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if enforce_eager:
        cmd.append("--enforce-eager")
    if quantization:
        cmd.extend(["--quantization", quantization])

    alt_cmds = [
        ["vllm", "bench", "throughput"] + cmd[4:],
        cmd,
    ]
    if monitor:
        monitor.start()

    last_result: dict[str, Any] = {}
    for attempt_cmd in alt_cmds:
        try:
            proc = subprocess.run(
                attempt_cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                env=_vllm_subprocess_env(),
            )
            last_result = {
                "command": " ".join(attempt_cmd),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
            if proc.returncode == 0:
                break
        except FileNotFoundError:
            continue

    samples = monitor.stop() if monitor else []
    summary = monitor.summarize(samples) if monitor else None
    metrics = parse_vllm_bench_output(
        last_result.get("stdout", ""),
        last_result.get("stderr", ""),
    )
    if summary:
        for k, v in gpu_summary_to_metrics(summary).items():
            setattr(metrics, k, v)

    return {
        "metrics": metrics,
        "raw": last_result,
        "gpu_samples": [s.model_dump() for s in samples],
        "success": last_result.get("returncode") == 0,
    }


def run_online_benchmark(
    hf_id: str,
    dataset_path: Path,
    workload_output_tokens: int,
    num_prompts: int,
    concurrency: int,
    serve_cmd: list[str],
    log_path: Path,
    startup_timeout_sec: int,
    host: str,
    port: int,
    seed: int,
    monitor: GPUMonitor,
) -> dict[str, Any]:
    ok, err = start_server(serve_cmd, log_path, startup_timeout_sec, host, port)
    if not ok:
        et, em = classify_error(err)
        return {
            "success": False,
            "error_type": et,
            "error_message": em,
            "metrics": BenchmarkMetrics(),
            "raw": {"server_log": str(log_path)},
        }

    try:
        monitor.start()
        bench = run_bench_serve(
            hf_id=hf_id,
            dataset_path=dataset_path,
            num_prompts=num_prompts,
            max_concurrency=concurrency,
            host=host,
            port=port,
            output_tokens=workload_output_tokens,
            seed=seed,
        )
        samples = monitor.stop()
        summary = monitor.summarize(samples)
        metrics: BenchmarkMetrics = bench["metrics"]
        for k, v in gpu_summary_to_metrics(summary).items():
            setattr(metrics, k, v)

        raw = bench.get("raw", {})
        success = raw.get("returncode") == 0
        error_type = None
        error_message = None
        if not success:
            error_type, error_message = classify_error(
                raw.get("stderr", "") or raw.get("stdout", "benchmark failed")
            )

        return {
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
            "metrics": metrics,
            "raw": {**raw, "server_log": str(log_path), "gpu_warning": monitor.warning},
            "gpu_samples": [s.model_dump() for s in samples],
        }
    finally:
        stop_server()
