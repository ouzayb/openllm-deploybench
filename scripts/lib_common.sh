#!/usr/bin/env bash
# Shared helpers for DeployBench shell scripts.

# Use venv Python when present (pip installs there; system python3 won't see packages).
deploybench_python() {
  local root="${1:-.}"
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${root}/.venv/bin/python" ]]; then
    echo "${root}/.venv/bin/python"
  else
    command -v python3
  fi
}

resolve_cuda_home_from_nvcc() {
  local nvcc_path="$1"
  nvcc_path="$(readlink -f "${nvcc_path}")"
  local cuda_home
  cuda_home="$(dirname "$(dirname "${nvcc_path}")")"
  # /usr/bin/nvcc from apt yields CUDA_HOME=/usr — wrong
  if [[ "${cuda_home}" == "/usr" ]]; then
    for d in /usr/lib/nvidia-cuda-toolkit /usr/local/cuda; do
      if [[ -x "${d}/bin/nvcc" ]]; then
        echo "${d}"
        return 0
      fi
    done
  fi
  echo "${cuda_home}"
}
