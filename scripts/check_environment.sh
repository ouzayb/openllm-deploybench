#!/usr/bin/env bash
# Preflight checks for OpenLLM DeployBench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Environment Check ==="
FAIL=0
WARN=0

check() {
  if eval "$2"; then
    echo "[OK] $1"
  else
    echo "[FAIL] $1"
    FAIL=1
  fi
}

warn() {
  echo "[WARN] $1"
  WARN=1
}

check "Python 3.10+" "python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'"
check "nvidia-smi" "command -v nvidia-smi"
check "Virtual env" "[[ -d .venv ]]"
check "deploybench import" "source .venv/bin/activate 2>/dev/null && python3 -c 'import deploybench' 2>/dev/null || python3 -c 'import sys; sys.path.insert(0,\"src\"); import deploybench'"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# nvidia-ml-py (imports as pynvml); avoid standalone deprecated pynvml package
if python3 -c "
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pynvml
" 2>/dev/null; then
  if python3 -c "import importlib.metadata; importlib.metadata.version('nvidia-ml-py')" 2>/dev/null; then
    echo "[OK] nvidia-ml-py (NVML)"
  elif pip show pynvml 2>/dev/null | grep -q Name; then
    warn "legacy pynvml installed — run: pip uninstall pynvml -y && pip install nvidia-ml-py"
  else
    echo "[OK] NVML Python bindings"
  fi
else
  echo "[FAIL] nvidia-ml-py / NVML bindings"
  echo "       Fix: pip install nvidia-ml-py"
  FAIL=1
fi

check "vllm" "python3 -c 'import vllm' 2>/dev/null || command -v vllm"
check "transformers" "python3 -c 'import transformers'"

# NVML init vs nvidia-smi
if command -v nvidia-smi &>/dev/null; then
  SMI_RC=0
  SMI_OUT=$(nvidia-smi 2>&1) || SMI_RC=$?
  if [[ "${SMI_RC}" -ne 0 ]]; then
    warn "nvidia-smi failed — fix NVIDIA driver before benchmarks"
    echo "${SMI_OUT}" | head -5
  else
    GPU_LINE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ -n "${GPU_LINE}" ]]; then
      echo "[OK] nvidia-smi sees GPU(s): ${GPU_LINE}"
    fi
  fi

  NVML_ERR=$(python3 -c "
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    import pynvml
    pynvml.nvmlInit()
    print('OK')
    pynvml.nvmlShutdown()
except Exception as e:
    print(e)
" 2>&1 | tail -1)

  if [[ "${NVML_ERR}" == "OK" ]]; then
    echo "[OK] NVML Python init"
  else
    warn "NVML Python init failed: ${NVML_ERR}"
    echo "       Benchmarks still run; hardware probe uses nvidia-smi fallback."
    echo "       For power/GPU monitoring during runs, fix driver mismatch:"
    echo "         sudo reboot"
    echo "       Or after driver update:"
    echo "         sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null; sudo modprobe nvidia"
  fi
fi

# Disk space (need ~50GB+ for large models)
AVAIL_GB=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
echo "Disk free: ${AVAIL_GB} GB"
if [[ "${AVAIL_GB}" -lt 30 ]]; then
  warn "Low disk space; large models need 30GB+ free"
fi

# HF token
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[OK] Hugging Face token set"
else
  warn "HF_TOKEN not set (required for gated models like Llama)"
fi

if command -v nvidia-smi &>/dev/null; then
  echo ""
  echo "GPU summary:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  if [[ "${WARN}" -eq 1 ]]; then
    echo "Environment check passed with warnings (see above)."
  else
    echo "Environment check passed."
  fi
else
  echo "Environment check found blocking issues. Fix before running benchmarks."
  exit 1
fi
