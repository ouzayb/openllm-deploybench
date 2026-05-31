"""vLLM server lifecycle and benchmark subprocess execution."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
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


_ROOT_CAUSE_MARKERS = (
    "Could not find nvcc",
    "FlashInfer",
    "flashinfer",
    "EngineCore failed",
    "EngineCore_DP",
    "EngineCore_",
    "CUDA out of memory",
    "CUDA error",
    "OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "No CUDA GPUs",
    "NCCL",
    "Failed to load",
    "ImportError",
    "ModuleNotFoundError",
)


def _extract_vllm_failure_excerpt(log_path: Path, max_lines: int = 80) -> str:
    """Prefer EngineCore / worker errors over the APIServer wrapper traceback."""
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""

    hits: list[str] = []
    for i, line in enumerate(lines):
        if any(m in line for m in _ROOT_CAUSE_MARKERS):
            start = max(0, i - 4)
            end = min(len(lines), i + 15)
            chunk = lines[start:end]
            if chunk and (not hits or chunk != hits[-1]):
                hits.extend(chunk)
                hits.append("---")

    if hits:
        text = "\n".join(hits)
        excerpt_lines = text.splitlines()
        if len(excerpt_lines) > max_lines:
            excerpt_lines = excerpt_lines[-max_lines:]
        return "\n".join(excerpt_lines)

    # Skip repetitive APIServer-only tail if a non-APIServer traceback exists earlier
    non_api = [ln for ln in lines if "(APIServer pid=" not in ln]
    if len(non_api) > 20:
        return "\n".join(non_api[-max_lines:])
    return "\n".join(lines[-40:])


def _tail_log(log_path: Path, lines: int = 40) -> str:
    return _extract_vllm_failure_excerpt(log_path, max_lines=lines)


def _resolve_vllm_argv_prefixes() -> list[list[str]]:
    """vLLM 0.22+ exposes `vllm serve` via console script, not `python -m vllm`."""
    prefixes: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(prefix: list[str]) -> None:
        key = tuple(prefix)
        if key not in seen:
            seen.add(key)
            prefixes.append(prefix)

    venv_bin = Path(sys.executable).resolve().parent / "vllm"
    if venv_bin.is_file():
        add([str(venv_bin)])
    which = shutil.which("vllm")
    if which:
        add([which])

    for module in ("vllm.entrypoints.cli.main", "vllm.entrypoints.cli"):
        try:
            import importlib.util

            if importlib.util.find_spec(module) is not None:
                add([sys.executable, "-m", module])
                break
        except (ImportError, ValueError, ModuleNotFoundError):
            continue

    return prefixes


def _build_vllm_bench_commands(subcommand: str, args: list[str]) -> list[list[str]]:
    """Build `vllm bench <subcommand>` command variants (0.22+ CLI + legacy modules)."""
    bench_tail = ["bench", subcommand] + args
    commands: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add(cmd: list[str]) -> None:
        key = tuple(cmd)
        if key not in seen:
            seen.add(key)
            commands.append(cmd)

    for prefix in _resolve_vllm_argv_prefixes():
        add(prefix + bench_tail)

    legacy_modules = {
        "serve": "vllm.benchmarks.bench_serve",
        "throughput": "vllm.benchmarks.bench_throughput",
    }
    legacy_mod = legacy_modules.get(subcommand)
    if legacy_mod:
        add([sys.executable, "-m", legacy_mod] + args)

    return commands


def _run_command_attempts(
    commands: list[list[str]],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run commands in order; stop on first success."""
    if env is None:
        env = _vllm_subprocess_env()
    last_result: dict[str, Any] = {"stdout": "", "stderr": "", "returncode": -1}
    for attempt_cmd in commands:
        try:
            proc = subprocess.run(
                attempt_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            last_result = {
                "command": " ".join(attempt_cmd),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
            if proc.returncode == 0:
                break
        except FileNotFoundError as e:
            last_result = {
                "command": " ".join(attempt_cmd),
                "stdout": "",
                "stderr": str(e),
                "returncode": 127,
            }
            continue
        except subprocess.TimeoutExpired:
            last_result = {
                "command": " ".join(attempt_cmd),
                "error": "benchmark timeout",
                "returncode": -1,
            }
            break
    return last_result


def _serve_cli_args(
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
    args = [
        "serve",
        hf_id,
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--port",
        str(port),
        "--host",
        host,
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if enforce_eager:
        args.append("--enforce-eager")
    if quantization:
        args.extend(["--quantization", quantization])
    return args


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
    """Primary vLLM 0.22+ serve command (`vllm serve` console script)."""
    prefixes = _resolve_vllm_argv_prefixes()
    prefix = prefixes[0] if prefixes else [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    return prefix + _serve_cli_args(
        hf_id, dtype, max_model_len, tensor_parallel_size,
        gpu_memory_utilization, quantization, trust_remote_code,
        port, host, enforce_eager,
    )


def build_serve_command_variants(
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
) -> list[list[str]]:
    """All modern `vllm serve` invocation variants to try before legacy api_server."""
    args = _serve_cli_args(
        hf_id, dtype, max_model_len, tensor_parallel_size,
        gpu_memory_utilization, quantization, trust_remote_code,
        port, host, enforce_eager,
    )
    prefixes = _resolve_vllm_argv_prefixes()
    if not prefixes:
        prefixes = [[sys.executable, "-m", "vllm.entrypoints.cli.main"]]
    return [prefix + args for prefix in prefixes]


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
    use_v1_engine: bool = False,
) -> tuple[bool, str, list[str]]:
    """Try vllm serve (0.22+), then legacy api_server; retry with stable env fallbacks."""
    attempts = build_serve_command_variants(
        hf_id, dtype, max_model_len, tensor_parallel_size,
        gpu_memory_utilization, quantization, trust_remote_code,
        port, host, enforce_eager,
    )
    legacy = build_serve_command_legacy(
        hf_id, dtype, max_model_len, tensor_parallel_size,
        gpu_memory_utilization, quantization, trust_remote_code,
        port, host, enforce_eager,
    )
    if legacy not in attempts:
        attempts.append(legacy)
    env_attempts: list[tuple[str, dict[str, str] | None]] = [
        ("default", _vllm_subprocess_env(use_v1_engine=use_v1_engine)),
        ("VLLM_USE_V1=0", _vllm_subprocess_env(use_v1_engine=False)),
        (
            "VLLM_USE_V1=0,FlashInfer off",
            _vllm_subprocess_env(use_v1_engine=False, use_flashinfer_sampler=False),
        ),
    ]
    # Deduplicate identical env dicts
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique_env_attempts: list[tuple[str, dict[str, str] | None]] = []
    for label, env in env_attempts:
        key = tuple(sorted((env or {}).items()))
        if key in seen:
            continue
        seen.add(key)
        unique_env_attempts.append((label, env))

    last_err = ""
    for cmd in attempts:
        for attempt_idx, (label, vllm_env) in enumerate(unique_env_attempts):
            ok, err = start_server(
                cmd,
                log_path,
                startup_timeout_sec,
                host,
                port,
                vllm_env=vllm_env,
                append_log=attempt_idx > 0,
            )
            if ok:
                if label != "default":
                    logger.info("Server started with fallback env: %s", label)
                return True, "", cmd
            last_err = err
            logger.warning(
                "Server start failed (cmd=%s env=%s): %s",
                " ".join(cmd[:5]),
                label,
                err.splitlines()[0] if err else "",
            )
    return False, last_err, attempts[-1]


def _resolve_cuda_home() -> str | None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        for candidate in (
            "/usr/local/cuda/bin/nvcc",
            "/usr/lib/nvidia-cuda-toolkit/bin/nvcc",
            "/usr/bin/nvcc",
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                nvcc = candidate
                break
    if not nvcc:
        return os.environ.get("CUDA_HOME")
    return str(Path(nvcc).resolve().parent.parent)


def _vllm_subprocess_env(
    *,
    use_v1_engine: bool | None = None,
    use_flashinfer_sampler: bool | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Environment for vLLM child processes (CUDA toolkit + FlashInfer when nvcc exists)."""
    from deploybench.utils import load_project_cuda_env

    load_project_cuda_env()
    env = os.environ.copy()
    cuda_home = _resolve_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
        lib = f"{cuda_home}/lib64"
        env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"

    if use_flashinfer_sampler is None:
        if "VLLM_USE_FLASHINFER_SAMPLER" not in env:
            if cuda_home and shutil.which("nvcc"):
                env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
            else:
                env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                logger.warning(
                    "nvcc not found; VLLM_USE_FLASHINFER_SAMPLER=0. "
                    "Run: bash scripts/setup_cuda_env.sh"
                )
    else:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer_sampler else "0"

    if use_v1_engine is None:
        if "VLLM_USE_V1" not in env:
            # V1 engine often hides worker errors; legacy engine is more stable for smoke runs
            env["VLLM_USE_V1"] = os.environ.get("DEPLOYBENCH_VLLM_USE_V1", "0")
    else:
        env["VLLM_USE_V1"] = "1" if use_v1_engine else "0"

    if extra:
        env.update(extra)
    return env


def start_server(
    cmd: list[str],
    log_path: Path,
    startup_timeout_sec: int = 600,
    health_host: str = "127.0.0.1",
    health_port: int = 8000,
    vllm_env: dict[str, str] | None = None,
    append_log: bool = False,
) -> tuple[bool, str]:
    global _active_server
    _cleanup_server()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a" if append_log else "w", encoding="utf-8")
    if append_log:
        log_file.write(f"\n\n=== retry: {' '.join(cmd[:6])}... ===\n")
        log_file.flush()
    if vllm_env is None:
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
                msg += f"\n--- server log excerpt ({log_path}) ---\n{tail}"
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
    with tempfile.TemporaryDirectory():
        bench_args = [
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
        last_result = _run_command_attempts(
            _build_vllm_bench_commands("serve", bench_args),
            timeout=3600,
        )

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
    bench_args = [
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
        bench_args.append("--trust-remote-code")
    if enforce_eager:
        bench_args.append("--enforce-eager")
    if quantization:
        bench_args.extend(["--quantization", quantization])

    if monitor:
        monitor.start()

    last_result = _run_command_attempts(
        _build_vllm_bench_commands("throughput", bench_args),
        timeout=7200,
    )

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
