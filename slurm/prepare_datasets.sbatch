#!/usr/bin/env bash
#SBATCH --job-name=prepare-dataset-artifacts
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Usage: sbatch --array=0-8 slurm/prepare_datasets.sbatch [python_executable]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

PYTHON_BIN="${1:-python}"

if [[ $# -lt 1 ]]; then
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

DATASETS=(
  "dtd47"
  "dermamnist7"
  "food101"
  "audiomnist10"
  "skits2i14"
  "banking77"
  "trec6"
  "letter26"
  "dopanim"
)

CLASSIFICATION_EMBEDDERS=(
  "clip"
  "dinov2"
  "dinov2"
  "wav2vec"
  "wav2vec"
  "minilmv2"
  "minilmv2"
  "identity_tabular"
  "dinov2"
)

SIMULATION_EMBEDDERS=(
  "dinov2"
  "clip"
  "clip"
  "wavlm"
  "wavlm"
  "mpnetv2"
  "mpnetv2"
  "identity_tabular"
  "null"
)

SIMULATION_CONFIGS=(
  "dtd47"
  "dermamnist7"
  "food101"
  "audiomnist10"
  "skits2i14"
  "banking77"
  "trec6"
  "letter26"
  "null"
)

PREP_STEPS=(
  "none"
  "none"
  "none"
  "none"
  "none"
  "none"
  "none"
  "none"
  "prepare_dopanim"
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
SIMULATION_CONFIG="${SIMULATION_CONFIGS[TASK_INDEX]}"
PREP_STEP="${PREP_STEPS[TASK_INDEX]}"

if [[ "${PREP_STEP}" == "prepare_dopanim" ]]; then
  echo ">>> Preparing raw ${DATASET} dataset"
  "${PYTHON_BIN}" scripts/prepare_dopanim.py --variant full
fi

echo ">>> Preparing ${DATASET} (classification=${CLASSIFICATION_EMBEDDER}, simulation=${SIMULATION_CONFIG})"

CMD=(
  "${PYTHON_BIN}" scripts/experiment.py
  "dataset=${DATASET}"
  "embedder@classification_embedder=${CLASSIFICATION_EMBEDDER}"
  "exit_after_simulation=True"
)

if [[ "${SIMULATION_CONFIG}" == "null" ]]; then
  CMD+=("~simulation")
else
  CMD+=("simulation=${SIMULATION_CONFIG}")
fi

if [[ "${SIMULATION_EMBEDDER}" == "null" ]]; then
  CMD+=("~embedder@simulation_embedder")
else
  CMD+=("embedder@simulation_embedder=${SIMULATION_EMBEDDER}")
fi

"${CMD[@]}"

echo "Completed task ${TASK_INDEX}: ${DATASET}"
