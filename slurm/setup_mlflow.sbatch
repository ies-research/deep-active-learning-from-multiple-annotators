#!/usr/bin/env bash
#SBATCH --job-name=setup-mlflow
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/setup_mlflow.sbatch <experiment_name> [results_path] [python_executable]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

source /home/mherde/miniconda3/bin/activate dalc

EXPERIMENT_NAME="$1"
RESULTS_PATH="${2:-}"
PYTHON_BIN="${3:-python}"

CMD=(
  "${PYTHON_BIN}" scripts/setup_mlflow.py
  --experiment-name "${EXPERIMENT_NAME}"
)

if [[ -n "${RESULTS_PATH}" ]]; then
  CMD+=(--results-path "${RESULTS_PATH}")
fi

"${CMD[@]}"
