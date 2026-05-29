#!/usr/bin/env bash
# Preflight checks for OpenLLM DeployBench
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Environment Check ==="
FAIL=0

check() {
  if eval "$2"; then
    echo "[OK] $1"
  else
    echo "[FAIL] $1"
    FAIL=1
  fi
}

check "Python 3.10+" "python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)'"
check "nvidia-smi" "command -v nvidia-smi"
check "Virtual env" "[[ -d .venv ]]"
check "deploybench import" "source .venv/bin/activate 2>/dev/null && python3 -c 'import deploybench' 2>/dev/null || python3 -c 'import sys; sys.path.insert(0,\"src\"); import deploybench'"

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

check "pynvml" "python3 -c 'import pynvml'"
check "vllm" "python3 -c 'import vllm' 2>/dev/null || command -v vllm"
check "transformers" "python3 -c 'import transformers'"

# Disk space (need ~50GB+ for large models)
AVAIL_GB=$(df -BG . | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
echo "Disk free: ${AVAIL_GB} GB"
if [[ "${AVAIL_GB}" -lt 30 ]]; then
  echo "[WARN] Low disk space; large models need 30GB+ free"
fi

# HF token
if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "[OK] Hugging Face token set"
else
  echo "[WARN] HF_TOKEN not set (required for gated models like Llama)"
fi

if command -v nvidia-smi &>/dev/null; then
  echo ""
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
fi

echo ""
if [[ "${FAIL}" -eq 0 ]]; then
  echo "Environment check passed."
else
  echo "Environment check found issues. Fix before running benchmarks."
  exit 1
fi
