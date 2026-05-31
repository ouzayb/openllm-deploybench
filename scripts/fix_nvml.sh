#!/usr/bin/env bash
# Repair NVML Python bindings (nvidia-ml-py provides `import pynvml`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "=== Fix NVML (nvidia-ml-py) ==="

# Install first — never uninstall until import works
pip install --upgrade "nvidia-ml-py>=12.535.133"

if ! python3 -c "import pynvml; pynvml.nvmlInit(); pynvml.nvmlShutdown(); print('NVML OK')"; then
  echo "ERROR: NVML still broken after install."
  python3 -c "import pynvml" 2>&1 || true
  exit 1
fi

# Remove deprecated standalone PyPI package only if both were present
if pip show pynvml &>/dev/null && pip show nvidia-ml-py &>/dev/null; then
  PYVML_LOC=$(python3 -c "import pynvml; print(pynvml.__file__)" 2>/dev/null || true)
  echo "pynvml loaded from: ${PYVML_LOC}"
fi

echo "Done. Run: bash scripts/check_environment.sh"
