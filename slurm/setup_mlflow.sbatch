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

source "${REPO_ROOT}/slurm/_conda.sh"

EXPERIMENT_NAME="$1"
RESULTS_PATH="${2:-}"
PYTHON_BIN="${3:-python}"
RESULTS_ROOT="${DALC_RESULTS_ROOT:-${DALCE_RESULTS_ROOT:-}}"

if [[ $# -lt 3 ]]; then
  activate_dalc_conda_env
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/setup_mlflow.py"
  --experiment-name "${EXPERIMENT_NAME}"
)

if [[ -n "${RESULTS_PATH}" ]]; then
  CMD+=(--results-path "${RESULTS_PATH}")
elif [[ -n "${RESULTS_ROOT}" ]]; then
  CMD+=(--results-path "${RESULTS_ROOT}/mlflow")
fi

"${CMD[@]}"
