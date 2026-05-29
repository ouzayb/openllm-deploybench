#!/usr/bin/env bash
# OpenLLM DeployBench - Ubuntu install script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== OpenLLM DeployBench Install ==="

# System packages (optional - user may already have these)
if command -v apt-get &>/dev/null; then
  echo "Suggested system packages (run manually if needed):"
  echo "  sudo apt update"
  echo "  sudo apt install -y git curl wget build-essential python3-venv python3-pip"
fi

# Python version check
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYVER}"
case "${PYVER}" in
  3.10|3.11|3.12) echo "Python version OK" ;;
  *) echo "WARNING: Python 3.10–3.12 recommended (found ${PYVER})" ;;
esac

# Virtual environment
if [[ ! -d .venv ]]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

echo "Installing requirements (this may take a while)..."
pip install -r requirements.txt

echo "Installing package in editable mode..."
pip install -e .

echo ""
echo "=== NVIDIA / CUDA Diagnostics ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found. Install NVIDIA drivers before running benchmarks."
fi

echo ""
python3 -c "
import sys
print('Python:', sys.version)
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
echo "NOTE: CUDA/PyTorch/vLLM compatibility can be fragile."
echo "      You may need to pin versions for your driver/CUDA stack."
echo "      See README.md for guidance."
echo ""
echo "=== Next steps ==="
echo "  source .venv/bin/activate"
echo "  cp configs/hardware.local.example.yaml configs/hardware.local.yaml"
echo "  cp configs/models.example.yaml configs/models.yaml"
echo "  cp configs/benchmark_matrix.example.yaml configs/benchmark_matrix.yaml"
echo "  # Edit hardware.local.yaml for your machine (4090/5090/H200)"
echo "  bash scripts/check_environment.sh"
echo "  deploybench probe-hardware --hardware-config configs/hardware.local.yaml --output results/hardware.json"
