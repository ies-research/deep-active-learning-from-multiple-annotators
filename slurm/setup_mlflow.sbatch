#!/usr/bin/env bash
#SBATCH --job-name=setup-mlflow
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/setup_mlflow.sbatch <experiment_name> [results_path] [python_executable]" >&2
  exit 1
fi

REPO_ROOT="${DALC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${REPO_ROOT}"


EXPERIMENT_NAME="$1"
RESULTS_PATH="${2:-}"
PYTHON_BIN="${3:-python}"

if [[ $# -lt 3 ]]; then
  if [[ -n "${CONDA_BASE:-}" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Could not find conda. Set CONDA_BASE or pass an explicit Python executable." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_NAME:-dalc}"
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/setup_mlflow.py"
  --experiment-name "${EXPERIMENT_NAME}"
)

if [[ -n "${RESULTS_PATH}" ]]; then
  CMD+=(--results-path "${RESULTS_PATH}")
fi

"${CMD[@]}"
