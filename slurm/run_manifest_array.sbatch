#!/usr/bin/env bash
#SBATCH --job-name=dalc-manifest
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=0-N slurm/run_manifest_array.sbatch <manifest.jsonl> [python_executable]" >&2
  exit 1
fi

REPO_ROOT="${DALC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/slurm/_conda.sh"

mkdir -p slurm/logs

MANIFEST_PATH="$1"
PYTHON_BIN="${2:-python}"
DATA_ROOT="${DALC_DATA_ROOT:-}"
RESULTS_ROOT="${DALC_RESULTS_ROOT:-${DALCE_RESULTS_ROOT:-}}"

if [[ $# -lt 2 ]]; then
  activate_dalc_conda_env
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_manifest_row.py"
  --manifest "${MANIFEST_PATH}"
  --row "${SLURM_ARRAY_TASK_ID}"
)

if [[ -n "${DATA_ROOT}" ]]; then
  CMD+=(--override "paths.master_dir=${DATA_ROOT}")
fi

if [[ -n "${RESULTS_ROOT}" ]]; then
  CMD+=(--override "paths.results_dir=${RESULTS_ROOT}")
fi

"${CMD[@]}"
