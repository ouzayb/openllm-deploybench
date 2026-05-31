# OpenLLM-DeployBench

Benchmark suite for evaluating open-weight LLM deployment on NVIDIA GPUs. Compare **RTX 4090**, **RTX 5090**, **owned/local H200**, and **cloud-rented GPU** systems with reproducible serving, long-context, and cost metrics.

## Research Questions

1. Do you really need H200s, or can RTX 4090/5090-class systems deliver comparable practical LLM deployment performance for some workloads?
2. When renting an H200, do you receive the same practical inference performance as an owned/local H200?

## What It Measures

- Hardware fingerprinting (CPU, RAM, disk, GPU topology, power)
- LLM serving latency (TTFT, TPOT, E2E)
- Throughput (requests/s, tokens/s)
- VRAM and power usage
- Long-context reliability (needle-in-a-haystack)
- Quantization impact (speed/VRAM ratios)
- Price-performance (tokens per dollar when hourly price is set)
- Owned vs rented H200 relative performance

## Supported Hardware Profiles

Twelve example configs cover **RTX 4090**, **RTX 5090**, and **H200** in single- and dual-GPU layouts, for both **owned/local** and **cloud-rented** machines. See [`configs/HARDWARE_PROFILES.md`](configs/HARDWARE_PROFILES.md) for the full index.

| GPU | GPUs | Owned / local | Cloud / rented |
|-----|------|---------------|----------------|
| RTX 4090 | 1 | `hardware.owned.rtx4090.single.example.yaml` | `hardware.cloud.rtx4090.single.example.yaml` |
| RTX 4090 | 2 | `hardware.owned.rtx4090.dual.example.yaml` | `hardware.cloud.rtx4090.dual.example.yaml` |
| RTX 5090 | 1 | `hardware.owned.rtx5090.single.example.yaml` | `hardware.cloud.rtx5090.single.example.yaml` |
| RTX 5090 | 2 | `hardware.owned.rtx5090.dual.example.yaml` | `hardware.cloud.rtx5090.dual.example.yaml` |
| H200 | 1 | `hardware.owned.h200.single.example.yaml` | `hardware.cloud.h200.single.example.yaml` |
| H200 | 2 | `hardware.owned.h200.dual.example.yaml` | `hardware.cloud.h200.dual.example.yaml` |

All paths are under `configs/`. Cloud examples include placeholder `hourly_price_usd` values — update them to match your provider invoice.

## Requirements

- Ubuntu Desktop or Server (22.04+ recommended)
- NVIDIA GPU + **driver** (`nvidia-smi` works)
- **CUDA toolkit** (`nvcc`) — installed automatically by `scripts/install_ubuntu.sh`
- Python 3.10, 3.11, or 3.12
- ~20 GB free disk for small models; more for 32B/70B
- SSH-friendly (no GUI required)

## Quick Start (download and run)

```bash
git clone https://github.com/ouzayb/openllm-deploybench.git
cd openllm-deploybench

# Installs: build tools, nvidia-cuda-toolkit (nvcc), Python venv, vLLM, deploybench
# Requires sudo for apt packages. NVIDIA driver must already be installed.
bash scripts/install_ubuntu.sh

source .venv/bin/activate   # also loads scripts/env.cuda.sh (CUDA_HOME, FlashInfer)

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If you see `deploybench: command not found`, the venv is not active or `pip install -e .` was not run. Use either:

```bash
python -m deploybench probe-hardware --hardware-config configs/hardware.local.yaml --output results/hardware.json
```

or the script wrapper:

```bash
python scripts/run_hardware_probe.py --hardware-config configs/hardware.local.yaml --output results/hardware.json
```

```bash
# Continue setup (Linux example)
source .venv/bin/activate

# Pick one of 12 hardware profiles (see configs/HARDWARE_PROFILES.md)
cp configs/hardware.owned.rtx4090.single.example.yaml configs/hardware.local.yaml
cp configs/models.example.yaml configs/models.yaml
cp configs/benchmark_matrix.example.yaml configs/benchmark_matrix.yaml

# Edit hardware.local.yaml: machine_id, provider, hourly_price_usd (cloud)
# Set hourly_price_usd for cloud instances

bash scripts/check_environment.sh

deploybench probe-hardware \
  --hardware-config configs/hardware.local.yaml \
  --output results/hardware.json

# Smoke test (small matrix)
deploybench run-serving \
  --hardware-config configs/hardware.local.yaml \
  --models-config configs/models.yaml \
  --config configs/benchmark_matrix.smoke.yaml \
  --output-dir results/serving

deploybench run-long-context \
  --hardware-config configs/hardware.local.yaml \
  --models-config configs/models.yaml \
  --config configs/benchmark_matrix.smoke.yaml \
  --output-dir results/long_context

deploybench summarize --results-dir results --output-dir reports
deploybench plot --results-dir results --output-dir reports/figures
```

## Configure Your Machine

Copy the matching profile from [`configs/HARDWARE_PROFILES.md`](configs/HARDWARE_PROFILES.md):

```bash
# Owned dual RTX 5090
cp configs/hardware.owned.rtx5090.dual.example.yaml configs/hardware.local.yaml

# Cloud single H200
cp configs/hardware.cloud.h200.single.example.yaml configs/hardware.local.yaml
```

Edit `machine_id`, `provider`, and `hourly_price_usd` (required for cloud price-performance). Owned RTX cards use `location_type: consumer_local`; owned H200 uses `owned`; cloud profiles use `rented_cloud`.

## Configure Models

Edit `configs/models.yaml` (from `models.example.yaml`). Add or remove models; set `requires_hf_token: true` for gated models.

```bash
export HF_TOKEN=your_token   # for Llama and other gated models
python scripts/download_models.py --models-config configs/models.yaml
```

## Run Benchmarks

### Hardware probe

```bash
deploybench probe-hardware --hardware-config configs/hardware.local.yaml --output results/hardware.json
```

### Serving benchmark

```bash
deploybench run-serving \
  --hardware-config configs/hardware.local.yaml \
  --models-config configs/models.yaml \
  --config configs/benchmark_matrix.yaml \
  --output-dir results/serving
```

### Long-context (needle-in-a-haystack)

```bash
deploybench run-long-context \
  --hardware-config configs/hardware.local.yaml \
  --models-config configs/models.yaml \
  --config configs/benchmark_matrix.yaml \
  --output-dir results/long_context
```

### Quantization comparison

```bash
deploybench run-quantization \
  --hardware-config configs/hardware.local.yaml \
  --models-config configs/models.yaml \
  --config configs/benchmark_matrix.yaml \
  --output-dir results/quantization
```

### Full pipeline

```bash
python scripts/run_all.py --hardware-config configs/hardware.local.yaml
```

## Troubleshooting vLLM server: `Could not find nvcc`

vLLM 0.22+ uses **FlashInfer** for sampling, which needs **`nvcc`** (CUDA toolkit), not only the NVIDIA driver.

**Fix (recommended):**

```bash
bash scripts/setup_cuda_env.sh   # apt: nvidia-cuda-toolkit + writes scripts/env.cuda.sh
source .venv/bin/activate
which nvcc
deploybench run-serving ...
```

Re-run `bash scripts/install_ubuntu.sh` on a fresh machine — it calls `setup_cuda_env.sh` automatically.

**Emergency fallback** (no nvcc, slower sampling): `export VLLM_USE_FLASHINFER_SAMPLER=0`

## Troubleshooting NVML / GPU detection

### `NVML/RM version mismatch` or `found 0` GPUs

This means the **NVIDIA kernel driver** and **user-space NVML library** are out of sync (common right after a driver upgrade without reboot).

**Check:**
```bash
nvidia-smi
```

- If `nvidia-smi` works but the probe shows 0 GPUs, pull the latest code (we fall back to `nvidia-smi` CSV parsing).
- If `nvidia-smi` also fails, fix the driver first:

```bash
# Typical fix: reboot after driver install
sudo reboot

# Or reload modules (Linux)
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null
sudo modprobe nvidia
nvidia-smi
```

**Python package:** prefer `nvidia-ml-py` over deprecated `pynvml`:
```bash
pip install nvidia-ml-py
pip uninstall pynvml -y 2>/dev/null || true
```

### `hardware.local.yaml` expects H200 but machine has different GPUs

Edit `configs/hardware.local.yaml` to match **your** GPUs, e.g.:
```bash
cp configs/hardware.owned.rtx4090.dual.example.yaml configs/hardware.local.yaml
```

## Troubleshooting `deploybench: command not found`

1. `cd` into the repo root (the folder that contains `pyproject.toml`).
2. Activate the virtualenv: `source .venv/bin/activate` (Linux) or `.\.venv\Scripts\Activate.ps1` (Windows).
3. Install: `pip install -e .`
4. Verify: `deploybench --help` or `python -m deploybench --help`
5. Copy configs before probe: `cp configs/hardware.owned.rtx4090.single.example.yaml configs/hardware.local.yaml`

## Direct Script Execution

Without the `deploybench` command on PATH (still needs `pip install -r requirements.txt` or `pip install -e .` for dependencies):

```bash
source .venv/bin/activate
python scripts/run_hardware_probe.py --output results/hardware.json --hardware-config configs/hardware.local.yaml
python scripts/run_serving_benchmark.py --config configs/benchmark_matrix.yaml --output-dir results/serving
python scripts/summarize_results.py --results-dir results --output-dir reports
python scripts/plot_results.py --results-dir results --output-dir reports/figures
```

## Summarize and Compare

```bash
deploybench summarize --results-dir results --output-dir reports
```

Outputs:

- `reports/summary_serving.csv`
- `reports/summary_long_context.csv`
- `reports/summary_hardware.csv`
- `reports/summary_price_performance.csv`

### Owned vs rented H200

1. Run benchmarks on owned H200 with `location_type: owned` and tag `h200` in `hardware.local.yaml`.
2. Run on rented H200 with `hourly_price_usd` set.
3. Compare `relative_to_owned_h200` and `tokens_per_dollar` in `summary_price_performance.csv`.

## Result Schema (serving)

Each line in `results/serving/*.jsonl`:

```json
{
  "run_id": "uuid",
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "machine_id": "owned_h200_01",
  "model_id": "qwen2_5_14b_instruct",
  "workload_id": "rag_8192_512",
  "concurrency": 4,
  "success": true,
  "metrics": {
    "output_tokens_per_second": 0.0,
    "ttft_ms_p95": 0.0,
    "peak_vram_gb": 0.0,
    "avg_power_watts": 0.0
  }
}
```

Failed runs are recorded with `success: false`, `error_type`, and `error_message` — the suite continues.

## Plots

```bash
deploybench plot --results-dir results --output-dir reports/figures
```

Generates: `throughput_by_hardware.png`, `ttft_p95_by_hardware.png`, `tpot_p95_by_hardware.png`, `peak_vram_by_model.png`, `long_context_accuracy_by_length.png`, `needle_position_accuracy.png`, `tokens_per_dollar.png`, `owned_vs_rented_h200_relative_perf.png`.

## Known Limitations

- **vLLM version drift**: CLI flags may change between vLLM releases; raw stdout/stderr are always saved.
- **CUDA/PyTorch pinning**: You may need to pin `torch` and `vllm` for your driver version.
- **Quantization**: Not all models support AWQ/GPTQ/bitsandbytes; unsupported quants are recorded as failures.
- **No fake metrics**: All numbers come from real runs; empty/failed runs show `success: false`.
- **Single-node only**: No multi-node distributed benchmarks in v1.
- **lm-eval**: Not included in v1.

## Cloud Cost Safety

Running full benchmark matrices on cloud GPUs can be **expensive**. Start with `configs/benchmark_matrix.smoke.yaml`, verify one model, then scale up. Monitor provider billing and stop instances when finished.

## Project Layout

```
openllm-deploybench/
  configs/          # YAML configuration
  scripts/          # install, probe, benchmark, summarize
  src/deploybench/  # Python package
  workloads/        # prompt templates and generated JSONL
  results/          # JSONL benchmark output (gitignored)
  reports/          # CSV summaries and PNG plots
```

## License

MIT
