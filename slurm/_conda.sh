#!/usr/bin/env bash

activate_dalc_conda_env() {
  local env_name="${CONDA_ENV_NAME:-dalc}"
  local conda_sh=""

  if [[ -n "${CONDA_BASE:-}" ]]; then
    conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
    if [[ -f "${conda_sh}" ]]; then
      source "${conda_sh}"
    elif command -v conda >/dev/null 2>&1; then
      echo "Warning: CONDA_BASE=${CONDA_BASE} does not contain etc/profile.d/conda.sh; using conda from PATH." >&2
      eval "$(conda shell.bash hook)"
    else
      echo "CONDA_BASE=${CONDA_BASE} does not contain etc/profile.d/conda.sh." >&2
      echo "Set CONDA_BASE to your Conda installation root, unset CONDA_BASE if conda is on PATH, or pass an explicit Python executable to the Slurm script." >&2
      exit 1
    fi
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Could not find conda. Set CONDA_BASE to your Conda installation root or pass an explicit Python executable." >&2
    exit 1
  fi

  conda activate "${env_name}"
}
