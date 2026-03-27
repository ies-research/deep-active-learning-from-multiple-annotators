#!/usr/bin/env bash
#SBATCH --job-name=dalc-manifest
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=0-N slurm/run_manifest_array.sbatch <manifest.jsonl> [python_executable]" >&2
  exit 1
fi

REPO_ROOT="${DALC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

MANIFEST_PATH="$1"
PYTHON_BIN="${2:-python}"

if [[ $# -lt 2 ]]; then
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

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_manifest_row.py" \
  --manifest "${MANIFEST_PATH}" \
  --row "${SLURM_ARRAY_TASK_ID}"
