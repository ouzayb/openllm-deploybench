"""vLLM server lifecycle and benchmark subprocess execution."""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from deploybench.gpu_monitor import GPUMonitor
from deploybench.metrics import (
    classify_error,
    gpu_summary_to_metrics,
    parse_vllm_bench_output,
    percentile,
)
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


def _parse_attention_backend(log_path: Path) -> str | None:
    """Best-effort extraction of the attention backend vLLM selected (varies per GPU)."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r"Using (.+?) backend", text)
    if m:
        # Strip a trailing "attention" word, e.g. "FLASH_ATTN attention" -> "FLASH_ATTN".
        return re.sub(r"\s+attention$", "", m.group(1).strip(), flags=re.IGNORECASE)
    m = re.search(r"attn[_ ]backend[=:]\s*([A-Za-z0-9_]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _server_config(
    label: str,
    cmd: list[str],
    env: dict[str, str],
    enforce_eager: bool,
    log_path: Path,
    *,
    reproducible: bool,
) -> dict[str, Any]:
    """Record the configuration a vLLM server actually launched with."""
    return {
        "reproducible": reproducible,
        "env_label": label,
        "vllm_use_v1": env.get("VLLM_USE_V1"),
        "flashinfer_sampler": env.get("VLLM_USE_FLASHINFER_SAMPLER"),
        "enforce_eager": enforce_eager,
        "attention_backend": _parse_attention_backend(log_path),
        "serve_command": " ".join(cmd),
    }


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
    reproducible: bool = False,
    use_flashinfer_sampler: bool | None = None,
) -> tuple[bool, str, list[str], dict[str, Any]]:
    """Start a vLLM server and return (ok, error, command, server_config).

    In reproducible mode a single pinned (command, env) is launched and the
    server must come up or the run fails: no command/env fallback cascade, so
    every device measures the same engine configuration. `server_config`
    records what actually ran (V1 engine, FlashInfer sampler, enforce_eager,
    attention backend) so each result row is self-documenting.
    """
    if reproducible:
        flashinfer_on = bool(use_flashinfer_sampler)  # None -> off (portable)
        cmd = build_serve_command(
            hf_id, dtype, max_model_len, tensor_parallel_size,
            gpu_memory_utilization, quantization, trust_remote_code,
            port, host, enforce_eager,
        )
        env = _vllm_subprocess_env(
            use_v1_engine=use_v1_engine,
            use_flashinfer_sampler=flashinfer_on,
        )
        ok, err = start_server(
            cmd, log_path, startup_timeout_sec, host, port, vllm_env=env,
        )
        server_config = _server_config(
            "pinned", cmd, env, enforce_eager, log_path, reproducible=True
        )
        if not ok:
            logger.error(
                "Reproducible mode: pinned vLLM config failed to start; "
                "NOT falling back (strict). %s",
                err.splitlines()[0] if err else "",
            )
        return ok, err, cmd, server_config

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
    if use_v1_engine:
        env_attempts = [
            ("default", _vllm_subprocess_env(use_v1_engine=True)),
            ("VLLM_USE_V1=0", _vllm_subprocess_env(use_v1_engine=False)),
            (
                "VLLM_USE_V1=0,FlashInfer off",
                _vllm_subprocess_env(use_v1_engine=False, use_flashinfer_sampler=False),
            ),
        ]
    else:
        # Try native vLLM defaults first (matches manual `vllm serve`), then fallbacks
        env_attempts = [
            ("native", _vllm_subprocess_env()),
            (
                "legacy+no-flashinfer",
                _vllm_subprocess_env(use_v1_engine=False, use_flashinfer_sampler=False),
            ),
            ("legacy", _vllm_subprocess_env(use_v1_engine=False)),
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
                server_config = _server_config(
                    label, cmd, vllm_env or {}, enforce_eager, log_path,
                    reproducible=False,
                )
                return True, "", cmd, server_config
            last_err = err
            logger.warning(
                "Server start failed (cmd=%s env=%s): %s",
                " ".join(cmd[:5]),
                label,
                err.splitlines()[0] if err else "",
            )
    return False, last_err, attempts[-1], {"reproducible": False, "env_label": None}


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
        # nvcc from apt (CUDA 12) often breaks FlashInfer JIT on H200 (CUB FlagHeads).
        # Default off even if scripts/env.cuda.sh still has =1; opt in via DEPLOYBENCH_ENABLE_FLASHINFER_SAMPLER=1
        enable_fi = os.environ.get("DEPLOYBENCH_ENABLE_FLASHINFER_SAMPLER", "").strip().lower()
        if enable_fi in ("1", "true", "yes") and cuda_home and shutil.which("nvcc"):
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
        else:
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            if not shutil.which("nvcc"):
                logger.warning(
                    "nvcc not found; VLLM_USE_FLASHINFER_SAMPLER=0. "
                    "Run: bash scripts/setup_cuda_env.sh"
                )
    else:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer_sampler else "0"

    if use_v1_engine is not None and "VLLM_USE_V1" not in env:
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


def _load_prompts_from_dataset(dataset_path: Path, num_prompts: int) -> list[str]:
    prompts: list[str] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(str(row["prompt"]))
            if len(prompts) >= num_prompts:
                break
    return prompts


def _is_instruct_model(hf_id: str) -> bool:
    name = hf_id.lower()
    return "instruct" in name or "chat" in name


def _bench_serve_profiles(hf_id: str) -> list[dict[str, Any]]:
    """CLI profiles to try; chat API first for Instruct models."""
    if _is_instruct_model(hf_id):
        return [
            {
                "label": "openai-chat",
                "backend": "openai-chat",
                "endpoint": "/v1/chat/completions",
                "skip_template": False,
            },
            {
                "label": "vllm-chat",
                "backend": "vllm",
                "endpoint": "/v1/chat/completions",
                "skip_template": False,
            },
            {
                "label": "vllm-completions",
                "backend": "vllm",
                "endpoint": "/v1/completions",
                "skip_template": True,
            },
        ]
    return [
        {
            "label": "vllm-completions",
            "backend": "vllm",
            "endpoint": "/v1/completions",
            "skip_template": True,
        },
    ]


def run_bench_serve_http(
    hf_id: str,
    dataset_path: Path,
    num_prompts: int,
    max_concurrency: int,
    host: str,
    port: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Direct OpenAI HTTP benchmark when `vllm bench serve` is unavailable or fails."""
    prompts = _load_prompts_from_dataset(dataset_path, num_prompts)
    if not prompts:
        return {
            "success": False,
            "metrics": BenchmarkMetrics(),
            "raw": {"error": "empty dataset", "returncode": 1},
        }

    modes = [
        ("chat", f"http://{host}:{port}/v1/chat/completions"),
        ("completion", f"http://{host}:{port}/v1/completions"),
    ]
    if not _is_instruct_model(hf_id):
        modes.reverse()

    last_error = ""
    for mode, url in modes:
        latencies_ms: list[float] = []
        errors: list[str] = []

        def one_request(prompt: str) -> float:
            t0 = time.perf_counter()
            if mode == "chat":
                payload: dict[str, Any] = {
                    "model": hf_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": output_tokens,
                    "temperature": 0,
                }
            else:
                payload = {
                    "model": hf_id,
                    "prompt": prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0,
                }
            resp = requests.post(url, json=payload, timeout=1800)
            resp.raise_for_status()
            return (time.perf_counter() - t0) * 1000.0

        t_batch = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
            futures = {pool.submit(one_request, p): p for p in prompts}
            for fut in as_completed(futures):
                try:
                    latencies_ms.append(fut.result())
                except Exception as e:
                    errors.append(str(e))
        wall_s = time.perf_counter() - t_batch

        if not errors and len(latencies_ms) == len(prompts):
            metrics = BenchmarkMetrics(
                requests_per_second=len(prompts) / wall_s if wall_s > 0 else 0.0,
                e2e_latency_ms_p50=percentile(latencies_ms, 50),
                e2e_latency_ms_p95=percentile(latencies_ms, 95),
                e2e_latency_ms_p99=percentile(latencies_ms, 99),
            )
            return {
                "success": True,
                "metrics": metrics,
                "raw": {
                    "fallback": "http",
                    "endpoint": url,
                    "returncode": 0,
                    "successful_requests": len(prompts),
                },
            }
        last_error = errors[0] if errors else "unknown HTTP error"
        logger.warning("HTTP bench mode=%s failed: %s", mode, last_error)

    return {
        "success": False,
        "metrics": BenchmarkMetrics(),
        "raw": {"fallback": "http", "error": last_error, "returncode": 1},
    }


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
    reproducible: bool = False,
    num_warmups: int = 0,
) -> dict[str, Any]:
    """Run vllm bench serve against a running server.

    In reproducible mode only the first (canonical) bench profile is used and
    there is NO HTTP fallback: if `vllm bench serve` fails the run fails, so a
    paper never mixes CLI numbers with the different client-side HTTP method.
    """
    common_args = [
        "--model", hf_id,
        "--dataset-name", "custom",
        "--dataset-path", str(dataset_path),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(max_concurrency),
        "--port", str(port),
        "--host", host,
        "--seed", str(seed),
        "--custom-output-len", str(output_tokens),
        # Ask vLLM for the percentiles/metrics we report; otherwise it only
        # emits Mean/Median/P99 for TTFT/TPOT/ITL and no E2EL block at all,
        # leaving p95 and e2e_latency_* unparseable.
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,95,99",
    ]
    # Warm up at least as many requests as the concurrency so CUDA-graph capture
    # for the run's batch shape happens during warmup, not in the measured run
    # (otherwise the first batch pays the capture cost and inflates p95/p99).
    effective_warmups = num_warmups
    if reproducible:
        effective_warmups = max(num_warmups, max_concurrency)
    if effective_warmups > 0:
        common_args.extend(["--num-warmups", str(effective_warmups)])

    profiles = _bench_serve_profiles(hf_id)
    if reproducible:
        profiles = profiles[:1]  # pin one profile; do not vary endpoint per device

    last_result: dict[str, Any] = {"stdout": "", "stderr": "", "returncode": -1}
    for profile in profiles:
        bench_args = [
            "--backend", profile["backend"],
            "--endpoint", profile["endpoint"],
            *common_args,
        ]
        if profile["skip_template"]:
            bench_args.append("--custom-skip-chat-template")
        commands = _build_vllm_bench_commands("serve", bench_args)
        last_result = _run_command_attempts(commands, timeout=3600)
        last_result["bench_profile"] = profile["label"]
        if last_result.get("returncode") == 0:
            break
        logger.warning(
            "vllm bench serve profile=%s failed (rc=%s): %s",
            profile["label"],
            last_result.get("returncode"),
            (last_result.get("stderr") or last_result.get("stdout") or "")[:300],
        )

    metrics = parse_vllm_bench_output(
        last_result.get("stdout", ""),
        last_result.get("stderr", ""),
    )

    if result_dir and last_result.get("returncode") == 0:
        for p in sorted(result_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                metrics = parse_vllm_bench_output(json.dumps(data))
                last_result["saved_result"] = str(p)
                break
            except Exception:
                pass

    if last_result.get("returncode") == 0:
        return {"success": True, "metrics": metrics, "raw": last_result}

    if reproducible:
        logger.error(
            "Reproducible mode: `vllm bench serve` failed and HTTP fallback is "
            "disabled (strict). Recording failure."
        )
        return {"success": False, "metrics": metrics, "raw": last_result}

    logger.warning("vllm bench serve failed; using HTTP fallback")
    http_out = run_bench_serve_http(
        hf_id=hf_id,
        dataset_path=dataset_path,
        num_prompts=num_prompts,
        max_concurrency=max_concurrency,
        host=host,
        port=port,
        output_tokens=output_tokens,
    )
    http_out["raw"] = {**last_result, "http_fallback": http_out.get("raw", {})}
    if http_out.get("success"):
        http_out["raw"]["returncode"] = 0
    return http_out


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
