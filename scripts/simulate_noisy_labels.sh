#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

run_prep() {
  local dataset="$1"
  local classification_embedder="$2"
  local simulation_embedder="$3"

  echo ">>> Preparing ${dataset} (classification=${classification_embedder}, simulation=${simulation_embedder})"
  python scripts/experiment.py \
    "dataset=${dataset}" \
    "simulation=${dataset}" \
    "embedder@classification_embedder=${classification_embedder}" \
    "embedder@simulation_embedder=${simulation_embedder}" \
    "exit_after_simulation=True"
}

# Development datasets: classify with CLIP, simulate with DINOv2.
run_prep dtd47 clip dinov2

# Image datasets: classify with DINOv2, simulate with CLIP.
run_prep dermamnist7 dinov2 clip
run_prep food101 dinov2 clip

# Audio datasets: classify with wav2vec2, simulate with WavLM.
run_prep audiomnist10 wav2vec wavlm
run_prep skits2i14 wav2vec wavlm

# Text datasets: classify with MPNet-v2, simulate with MiniLM-v2.
run_prep banking77 minilmv2 mpnetv2
run_prep trec6 minilmv2 mpnetv2

echo "All embedding caches and annotator simulations are prepared."
