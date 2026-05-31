#!/usr/bin/env bash
# OpenLLM DeployBench - Ubuntu install script (driver + CUDA toolkit + Python venv)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

echo "=== OpenLLM DeployBench Install ==="

# --- System: CUDA toolkit (nvcc) for vLLM / FlashInfer ---
bash "${SCRIPT_DIR}/setup_cuda_env.sh" || {
  echo "WARNING: CUDA toolkit setup incomplete. vLLM may fail until nvcc is installed."
}

# Source generated CUDA env for this install session
if [[ -f "${SCRIPT_DIR}/env.cuda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env.cuda.sh"
fi

# --- Python ---
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYVER}"
case "${PYVER}" in
  3.10|3.11|3.12) echo "Python version OK" ;;
  *) echo "WARNING: Python 3.10–3.12 recommended (found ${PYVER})" ;;
esac

if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Auto-load CUDA env whenever venv is activated
ACTIVATE_D=".venv/bin/activate.d"
mkdir -p "${ACTIVATE_D}"
cat > "${ACTIVATE_D}/deploybench-cuda.sh" << 'ACTIVATE_EOF'
# OpenLLM DeployBench — CUDA / vLLM environment
_DB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -f "${_DB_ROOT}/scripts/env.cuda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${_DB_ROOT}/scripts/env.cuda.sh"
fi
unset _DB_ROOT
ACTIVATE_EOF

PYTHON="$(deploybench_python "${PROJECT_ROOT}")"
echo "Using Python: ${PYTHON}"

echo "Installing pinned pip/setuptools (vLLM 0.22 + torch require setuptools<81)..."
"${PYTHON}" -m pip install --upgrade "pip>=24" "wheel"
"${PYTHON}" -m pip install "setuptools>=77.0.3,<81.0.0"

echo "Installing requirements (this may take a while)..."
"${PYTHON}" -m pip install -r requirements.txt

echo "Installing package in editable mode..."
"${PYTHON}" -m pip install -e .

# Re-pin setuptools if a dependency pulled a newer version
"${PYTHON}" -m pip install "setuptools>=77.0.3,<81.0.0"

# NVML: nvidia-ml-py provides `import pynvml`
"${PYTHON}" -m pip install --upgrade "nvidia-ml-py>=12.535.133"
"${PYTHON}" -c "import pynvml; print('nvidia-ml-py import OK')" || {
  echo "ERROR: nvidia-ml-py install failed. Run: bash scripts/fix_nvml.sh"
  exit 1
}
"${PYTHON}" -c "import pynvml; pynvml.nvmlInit(); pynvml.nvmlShutdown(); print('NVML init OK')" || {
  echo "WARNING: NVML init failed (driver mismatch?). Hardware probe still uses nvidia-smi."
  echo "         Try: sudo reboot"
}

echo ""
echo "=== NVIDIA / CUDA Diagnostics ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
else
  echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first, then re-run this script."
  exit 1
fi

if command -v nvcc &>/dev/null; then
  echo ""
  echo "nvcc: $(command -v nvcc)"
  nvcc --version | tail -1
  echo "CUDA_HOME=${CUDA_HOME:-not set}"
else
  echo ""
  echo "ERROR: nvcc still not found after setup. Try:"
  echo "  sudo apt install -y nvidia-cuda-toolkit build-essential ninja-build"
  echo "  bash scripts/setup_cuda_env.sh"
  exit 1
fi

echo ""
"${PYTHON}" -c "
import sys
print('Python:', sys.version)
print('Executable:', sys.executable)
for pkg in ('torch', 'vllm', 'transformers'):
    try:
        m = __import__(pkg)
        print(f'{pkg}:', getattr(m, '__version__', 'unknown'))
    except ImportError:
        print(f'{pkg}: NOT INSTALLED')
try:
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}:', torch.cuda.get_device_name(i))
except Exception as e:
    print('torch check error:', e)
" || true

echo ""
echo "=== Next steps ==="
echo "  source .venv/bin/activate   # loads scripts/env.cuda.sh automatically"
echo "  cp configs/hardware.owned.h200.dual.example.yaml configs/hardware.local.yaml  # edit for your GPUs"
echo "  cp configs/models.example.yaml configs/models.yaml"
echo "  cp configs/benchmark_matrix.smoke.yaml configs/benchmark_matrix.yaml"
echo "  bash scripts/check_environment.sh"
echo "  deploybench probe-hardware --hardware-config configs/hardware.local.yaml --output results/hardware.json"
echo "  deploybench run-serving --hardware-config configs/hardware.local.yaml --models-config configs/models.yaml --config configs/benchmark_matrix.yaml --output-dir results/serving"
