#!/usr/bin/env bash
#SBATCH --job-name=dalc-manifest
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch --array=0-N slurm/run_manifest_array.sbatch <manifest.jsonl> [python_executable]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

MANIFEST_PATH="$1"
PYTHON_BIN="${2:-python}"

"${PYTHON_BIN}" scripts/run_manifest_row.py \
  --manifest "${MANIFEST_PATH}" \
  --row "${SLURM_ARRAY_TASK_ID}"
