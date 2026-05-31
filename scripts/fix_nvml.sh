#!/usr/bin/env bash
# Repair NVML Python bindings (nvidia-ml-py provides `import pynvml`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=lib_common.sh
source "${SCRIPT_DIR}/lib_common.sh"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON="$(deploybench_python "${PROJECT_ROOT}")"
echo "=== Fix NVML (nvidia-ml-py) ==="
echo "Python: ${PYTHON}"
echo "python3 on PATH: $(command -v python3 2>/dev/null || echo missing)"
if [[ "${PYTHON}" != "$(command -v python3 2>/dev/null || true)" ]]; then
  echo "NOTE: Use .venv Python — system 'python3' is different from where pip installs."
fi

"${PYTHON}" -m pip install --upgrade "nvidia-ml-py>=12.535.133"

if ! "${PYTHON}" -c "import pynvml; pynvml.nvmlInit(); pynvml.nvmlShutdown(); print('NVML OK')"; then
  echo "ERROR: NVML still broken."
  "${PYTHON}" -c "import sys; print('executable:', sys.executable); import pynvml" 2>&1 || true
  exit 1
fi

PYVML_LOC=$("${PYTHON}" -c "import pynvml; print(pynvml.__file__)")
echo "pynvml loaded from: ${PYVML_LOC}"
echo "Done. Run: bash scripts/check_environment.sh"
