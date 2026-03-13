#!/usr/bin/env bash
#SBATCH --job-name=simulate-noisy-labels
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Usage: sbatch --array=0-6 slurm/simulate_noisy_labels.sh [python_executable]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

PYTHON_BIN="${1:-python}"

DATASETS=(
  "dtd47"
  "dermamnist7"
  "food101"
  "audiomnist10"
  "skits2i14"
  "banking77"
  "trec6"
)

CLASSIFICATION_EMBEDDERS=(
  "clip"
  "dinov2"
  "dinov2"
  "wav2vec"
  "wav2vec"
  "minilmv2"
  "minilmv2"
)

SIMULATION_EMBEDDERS=(
  "dinov2"
  "clip"
  "clip"
  "wavlm"
  "wavlm"
  "mpnetv2"
  "mpnetv2"
)

TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
TASK_COUNT="${#DATASETS[@]}"

if (( TASK_INDEX < 0 || TASK_INDEX >= TASK_COUNT )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_INDEX} is out of range; expected 0-$((TASK_COUNT - 1))." >&2
  exit 1
fi

DATASET="${DATASETS[TASK_INDEX]}"
CLASSIFICATION_EMBEDDER="${CLASSIFICATION_EMBEDDERS[TASK_INDEX]}"
SIMULATION_EMBEDDER="${SIMULATION_EMBEDDERS[TASK_INDEX]}"

echo ">>> Preparing ${DATASET} (classification=${CLASSIFICATION_EMBEDDER}, simulation=${SIMULATION_EMBEDDER})"

"${PYTHON_BIN}" scripts/experiment.py \
  "dataset=${DATASET}" \
  "simulation=${DATASET}" \
  "embedder@classification_embedder=${CLASSIFICATION_EMBEDDER}" \
  "embedder@simulation_embedder=${SIMULATION_EMBEDDER}" \
  "exit_after_simulation=True"

echo "Completed task ${TASK_INDEX}: ${DATASET}"
