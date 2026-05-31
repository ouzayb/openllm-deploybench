#!/usr/bin/env bash
# Preflight checks for OpenLLM DeployBench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

PYTHON="$(deploybench_python "${PROJECT_ROOT}")"
echo "=== Environment Check ==="
echo "Using Python: ${PYTHON}"
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

check "Python 3.10+" "\"${PYTHON}\" -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'"
check "nvidia-smi" "command -v nvidia-smi"
check "Virtual env" "[[ -d .venv ]]"
check "deploybench import" "\"${PYTHON}\" -c 'import deploybench' 2>/dev/null || \"${PYTHON}\" -c 'import sys; sys.path.insert(0,\"src\"); import deploybench'"

# nvidia-ml-py (imports as pynvml) — must use venv python, not system python3
if "${PYTHON}" -c "
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pynvml
print(pynvml.__file__)
" 2>/dev/null; then
  echo "[OK] nvidia-ml-py (import pynvml via ${PYTHON})"
else
  echo "[FAIL] cannot import pynvml with ${PYTHON}"
  echo "       pip shows nvidia-ml-py in .venv but 'python3' may be system Python."
  echo "       Fix:"
  echo "         source .venv/bin/activate"
  echo "         bash scripts/fix_nvml.sh"
  FAIL=1
fi

check "vllm" "\"${PYTHON}\" -c 'import vllm' 2>/dev/null || command -v vllm"
check "transformers" "\"${PYTHON}\" -c 'import transformers'"

# vLLM 0.22 + torch 2.11 need setuptools<81 on Python 3.12
if "${PYTHON}" -c "
import setuptools
def parse(s):
    p = [int(x) for x in s.split('.')[:3]]
    while len(p) < 3:
        p.append(0)
    return tuple(p)
ver = parse(setuptools.__version__)
ok = parse('77.0.3') <= ver < parse('81.0.0')
print(f'setuptools {setuptools.__version__}')
exit(0 if ok else 1)
" 2>/dev/null; then
  echo "[OK] setuptools version compatible with vLLM/torch"
else
  echo "[FAIL] setuptools version incompatible (need >=77.0.3,<81.0.0 for Python 3.12 + vLLM 0.22)"
  echo "       Fix: .venv/bin/python -m pip install 'setuptools>=77.0.3,<81.0.0'"
  FAIL=1
fi

if command -v nvcc &>/dev/null; then
  echo "[OK] nvcc ($(nvcc --version | tail -1))"
  if [[ -f "${SCRIPT_DIR}/env.cuda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env.cuda.sh"
    echo "[OK] CUDA_HOME=${CUDA_HOME:-unset}"
  fi
else
  echo "[FAIL] nvcc not found — required for vLLM 0.22+ (FlashInfer). Run:"
  echo "       bash scripts/setup_cuda_env.sh"
  FAIL=1
fi

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

  NVML_ERR=$("${PYTHON}" -c "
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
    if echo "${NVML_ERR}" | grep -qi "No module named"; then
      echo "       You are likely using system python3 instead of .venv/bin/python."
      echo "       Fix: source .venv/bin/activate && bash scripts/fix_nvml.sh"
    elif echo "${NVML_ERR}" | grep -qi "mismatch"; then
      echo "       Fix driver mismatch: sudo reboot"
    fi
  fi
fi

AVAIL_GB=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
echo "Disk free: ${AVAIL_GB} GB"
if [[ "${AVAIL_GB}" -lt 30 ]]; then
  warn "Low disk space; large models need 30GB+ free"
fi

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
