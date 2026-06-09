#!/usr/bin/env bash
#SBATCH --job-name=prepare-dataset-artifacts
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Usage: sbatch --array=0-10 slurm/prepare_datasets.sbatch [python_executable]" >&2
  exit 1
fi

REPO_ROOT="${DALC_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${REPO_ROOT}"

mkdir -p slurm/logs

DATA_ROOT="${DALC_DATA_ROOT:-}"
RESULTS_ROOT="${DALC_RESULTS_ROOT:-${DALCE_RESULTS_ROOT:-}}"
PREP_DEVICE="${DALC_PREP_DEVICE:-cuda}"
FORCE_DATA_DOWNLOAD="${DALC_FORCE_DATA_DOWNLOAD:-0}"
FORCE_DATA_REBUILD="${DALC_FORCE_DATA_REBUILD:-0}"

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
  "audiomnist10"
  "banking77"
  "dermamnist7"
  "dopanim"
  "food101"
  "letter26"
  "skits2i14"
  "trec6"
  "al_rcta_agnews"
  "al_rcta_consumer_complaints"
  "al_rcta_wiki_movie_plots"
)

CLASSIFICATION_EMBEDDERS=(
  "wav2vec"
  "mpnetv2"
  "dinov2"
  "dinov2"
  "dinov2"
  "identity_tabular"
  "wav2vec"
  "mpnetv2"
  "mpnetv2"
  "mpnetv2"
  "mpnetv2"
)

SIMULATION_EMBEDDERS=(
  "wavlm"
  "minilmv2"
  "clip"
  "null"
  "clip"
  "identity_tabular"
  "wavlm"
  "minilmv2"
  "null"
  "null"
  "null"
)

SIMULATION_CONFIGS=(
  "audiomnist10"
  "banking77"
  "dermamnist7"
  "null"
  "food101"
  "letter26"
  "skits2i14"
  "trec6"
  "null"
  "null"
  "null"
)

PREP_STEPS=(
  "none"
  "none"
  "none"
  "prepare_dopanim"
  "none"
  "none"
  "none"
  "none"
  "prepare_al_rcta:agnews"
  "prepare_al_rcta:consumer_complaints"
  "prepare_al_rcta:wiki_movie_plots"
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
  PREP_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_dopanim.py"
    --variant full
  )
  if [[ -n "${DATA_ROOT}" ]]; then
    PREP_CMD+=(--data-root "${DATA_ROOT}")
  fi
  if [[ "${FORCE_DATA_DOWNLOAD}" == "1" ]]; then
    PREP_CMD+=(--force-download)
  fi
  if [[ "${FORCE_DATA_REBUILD}" == "1" ]]; then
    PREP_CMD+=(--force-rebuild)
  fi
  "${PREP_CMD[@]}"
fi

if [[ "${PREP_STEP}" == prepare_al_rcta:* ]]; then
  AL_RCTA_DATASET="${PREP_STEP#prepare_al_rcta:}"
  AL_RCTA_OUTPUT_ROOT="${DATA_ROOT:-/home/datasets}"
  echo ">>> Preparing raw ${DATASET} dataset from al_rcta/${AL_RCTA_DATASET}"
  PREP_CMD=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_al_rcta.py"
    --dataset "${AL_RCTA_DATASET}"
    --max-test-size 3000
    --output-dir "${AL_RCTA_OUTPUT_ROOT}/${DATASET}_test3000"
  )
  if [[ -n "${DATA_ROOT}" ]]; then
    PREP_CMD+=(--data-root "${DATA_ROOT}")
  fi
  if [[ "${FORCE_DATA_DOWNLOAD}" == "1" ]]; then
    PREP_CMD+=(--force-download)
  fi
  if [[ "${FORCE_DATA_REBUILD}" == "1" ]]; then
    PREP_CMD+=(--force-rebuild)
  fi
  "${PREP_CMD[@]}"
fi

echo ">>> Preparing ${DATASET} (classification=${CLASSIFICATION_EMBEDDER}, simulation=${SIMULATION_CONFIG})"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/experiment.py"
  "dataset=${DATASET}"
  "embedder@classification_embedder=${CLASSIFICATION_EMBEDDER}"
  "exit_after_simulation=True"
  "device=${PREP_DEVICE}"
)

if [[ "${CLASSIFICATION_EMBEDDER}" != "identity_tabular" ]]; then
  CMD+=("classification_embedder.device=${PREP_DEVICE}")
fi

if [[ -n "${DATA_ROOT}" ]]; then
  CMD+=("paths.master_dir=${DATA_ROOT}")
fi

if [[ -n "${RESULTS_ROOT}" ]]; then
  CMD+=("paths.results_dir=${RESULTS_ROOT}")
fi

if [[ "${SIMULATION_CONFIG}" == "null" ]]; then
  CMD+=("~simulation")
else
  CMD+=("simulation=${SIMULATION_CONFIG}")
fi

if [[ "${SIMULATION_EMBEDDER}" == "null" ]]; then
  CMD+=("~embedder@simulation_embedder")
else
  CMD+=("embedder@simulation_embedder=${SIMULATION_EMBEDDER}")
  if [[ "${SIMULATION_EMBEDDER}" != "identity_tabular" ]]; then
    CMD+=("simulation_embedder.device=${PREP_DEVICE}")
  fi
fi

"${CMD[@]}"

echo "Completed task ${TASK_INDEX}: ${DATASET}"
