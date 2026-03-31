# Deep Active Learning from Multiple Annotators

This repository contains the current experiment code for deep active learning
with multiple annotators. The codebase is centered around a Hydra-configured
training pipeline in `scripts/experiment.py`, dataset and cache preparation
helpers, and a manifest-driven launcher for local and SLURM-backed sweeps.

## Environment

Create one of the Conda environments before running preparation scripts or
experiments:

```bash
conda env create -f environment.yml
conda activate dalc
```

If you need to activate Conda via an explicit path, the equivalent command is:

```bash
source /path/to/miniconda3/bin/activate dalc
```

Additional environment files:

- [environment.yml](environment.yml): lean runtime environment for experiments
  and preparation scripts
- [environment.gpu.yml](environment.gpu.yml): GPU-oriented runtime environment
  with extra accelerator-focused packages
- [environment.dev.yml](environment.dev.yml): development environment with
  testing, linting, notebook, and documentation tooling

GPU-oriented environment:

```bash
conda env create -f environment.gpu.yml
conda activate dalc-gpu
```

Development environment:

```bash
conda env create -f environment.dev.yml
conda activate dalc-dev
```

All three environment files currently target Python 3.12.

## Paths And Overrides

Default paths are defined in
[configs/paths/default.yaml](configs/paths/default.yaml):

- `paths.master_dir=/home/datasets`
- `paths.results_dir=/home/results`
- `paths.dataset_cache_dir=${paths.master_dir}`
- `paths.embedder_cache_dir=${paths.master_dir}/.hf_embedders`
- `paths.pipeline_cache_dir=${paths.master_dir}/.hf_embed_cache`
- `paths.multi_annotator_cache_dir=${paths.master_dir}/.hf_multi_annotator_cache`
- `results_path=${paths.results_dir}/mlflow`

Helper scripts and SLURM wrappers also respect:

- `DALC_DATA_ROOT`: overrides `paths.master_dir`
- `DALC_RESULTS_ROOT`: overrides `paths.results_dir`
- `DALC_REPO_ROOT`: absolute repository root used by SLURM wrappers
- `CONDA_BASE`: optional Conda installation root if `conda` is not already on
  `PATH`
- `CONDA_ENV_NAME`: optional environment name for SLURM wrappers, defaults to
  `dalc`

## Git-Tracked Directory Overview

The following directories are currently represented in `git ls-files`. This
section intentionally lists tracked directories only; local-only directories
such as `tests/`, `figures/`, or `mlflow/` are not included here unless they
are versioned.

| Directory | Purpose |
| --- | --- |
| `.` | Repository root with the main README, license, and Conda environment files |
| `configs` | Hydra configuration root |
| `configs/al` | Active-learning budgets, cycle counts, and annotator-capacity settings |
| `configs/assigner` | Sample-annotator pair assignment strategies |
| `configs/classifier` | Multi-annotator classifier definitions |
| `configs/dataset` | Dataset source specifications |
| `configs/embedder` | Feature extractor and embedding backends |
| `configs/launch/methods` | Reusable manifest method definitions |
| `configs/launch/use_cases` | Manifest use-case grids and study definitions |
| `configs/module` | Neural module definitions such as linear and MLP heads |
| `configs/paths` | Shared path defaults |
| `configs/pipeline` | Feature-pipeline settings |
| `configs/sample` | Initial and active sample-query strategies |
| `configs/scheduler` | Ratio scheduler variants |
| `configs/scorer` | Pair scoring and acquisition utility models |
| `configs/simulation` | Simulated annotator setups |
| `configs/simulation/profile` | Simulation difficulty/profile presets |
| `configs/training` | Optimizer and batch-size presets |
| `manifests` | Generated JSONL manifest output directory placeholder |
| `notebooks` | Exploratory and analysis notebooks |
| `scripts` | CLI entry points for experiments, manifest generation, MLflow setup, evaluation, and dataset preparation |
| `slurm` | Batch wrappers for dataset preparation, MLflow setup, and manifest arrays |
| `src` | Python package root |
| `src/assigner` | Pair assignment implementations |
| `src/classifier` | Classifier implementations for aggregate, EM, RegCrowdNet, DAlC-like, and annotator-mixture models |
| `src/dataset` | Dataset specs, caching, I/O, and feature-pipeline code |
| `src/embedder` | Embedding backends including Hugging Face and tabular identity encoders |
| `src/module` | Torch modules and losses |
| `src/scheduler` | Ratio scheduler implementations |
| `src/scorer` | Acquisition scoring implementations |
| `src/utils` | Seeding, evaluation, printing, and MLflow helpers |

## Running Experiments

The main entry point is [scripts/experiment.py](scripts/experiment.py). It
loads the Hydra defaults from [configs/experiment.yaml](configs/experiment.yaml)
and then composes dataset, embedder, simulation, classifier, scheduler,
sampler, scorer, assigner, and training settings from the config groups.

A minimal example looks like:

```bash
conda activate dalc
python scripts/experiment.py \
  dataset=trec6 \
  simulation=trec6 \
  al=trec6
```

You can override any Hydra value on the command line, for example:

```bash
python scripts/experiment.py \
  dataset=trec6 \
  simulation=trec6 \
  al=trec6 \
  paths.master_dir=/path/to/data/root \
  paths.results_dir=/path/to/results/root \
  experiment_name=local_debug_run
```

## Dataset Preparation

### DOPAnim

The repository can use DOPAnim as a local Hugging Face `DatasetDict` via
`source_kind: from_disk`.

Prepare the default `full` variant with:

```bash
conda activate dalc
python scripts/prepare_dopanim.py --variant full
```

With the current defaults, the script resolves:

- `--data-root`: `DALC_DATA_ROOT` when set, otherwise `paths.master_dir`
- `--raw-dir`: `<data-root>/raw/dopanim`
- `--output-dir`: `<data-root>/dopanim_<variant>`

That means `--variant full` writes to `<data-root>/dopanim_full`, which matches
[configs/dataset/dopanim.yaml](configs/dataset/dopanim.yaml).

Available parameters:

- `--data-root PATH`: base directory for raw and processed dataset artifacts
- `--raw-dir PATH`: exact directory for the downloaded Zenodo archive and
  extracted files
- `--output-dir PATH`: exact directory where the processed `DatasetDict` is
  saved
- `--variant {full,worst-var,rand-var,worst-1,worst-2,worst-3,worst-4,rand-1,rand-2,rand-3,rand-4}`:
  train annotation subset to materialize
- `--force-download`: re-download and re-extract the raw archive
- `--force-rebuild`: overwrite an existing processed dataset directory

Example with an explicit data root:

```bash
python scripts/prepare_dopanim.py \
  --data-root /path/to/data/root \
  --variant full \
  --force-rebuild
```

If you prepare a non-`full` variant, update
[configs/dataset/dopanim.yaml](configs/dataset/dopanim.yaml) or override
`dataset.source` at runtime so it points to the matching output directory.

### Cached Embeddings And Simulated Multi-Annotator Labels

The batch entry point for preparing cached artifacts is
[slurm/prepare_datasets.sbatch](slurm/prepare_datasets.sbatch). The current
array definition prepares exactly these datasets:

- `0`: `trec6` with classification embedder `bert` and simulation `trec6`
- `1`: `letter26` with classification embedder `identity_tabular` and
  simulation `letter26`
- `2`: `dopanim` with classification embedder `dinov2` and no simulation step

For `dopanim`, the script runs `scripts/prepare_dopanim.py --variant full`
before preparing cached embeddings. For `letter26`, the helper uses the
tabular identity embedder for both classification and simulation.

Submit the full SLURM array with:

```bash
export DALC_REPO_ROOT=/absolute/path/to/deep-active-learning-from-multiple-annotators
mkdir -p "${DALC_REPO_ROOT}/slurm/logs"

sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.err" \
  --array=0-2 \
  slurm/prepare_datasets.sbatch
```

You can pass a specific Python executable as the first positional argument:

```bash
sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.err" \
  --array=0-2 \
  slurm/prepare_datasets.sbatch \
  /path/to/python
```

You can also run the same helper locally by setting `SLURM_ARRAY_TASK_ID`
manually:

```bash
SLURM_ARRAY_TASK_ID=2 bash slurm/prepare_datasets.sbatch
```

Run all currently configured preparation tasks locally:

```bash
for i in {0..2}; do
  SLURM_ARRAY_TASK_ID=$i bash slurm/prepare_datasets.sbatch
done
```

## Manifest-Driven Launch Pipeline

The repository contains a manifest workflow for reproducible experiment grids.
The relevant inputs live in:

- [configs/launch/methods](configs/launch/methods): reusable method definitions
- [configs/launch/use_cases](configs/launch/use_cases): study-specific Cartesian
  products over datasets, seeds, methods, and other factors

The basic flow is:

1. Generate a JSONL manifest from a use-case specification.
2. Execute one manifest row locally or submit the full manifest as a SLURM
   array.

Generate a manifest:

```bash
python scripts/generate_manifest.py annotator_selection_main
```

By default, this writes `manifests/<use_case>.jsonl` and prints a short preview
of the generated rows.

Execute one row locally:

```bash
python scripts/run_manifest_row.py \
  --manifest manifests/annotator_selection_main.jsonl \
  --row 0
```

Append extra Hydra overrides after the manifest row:

```bash
python scripts/run_manifest_row.py \
  --manifest manifests/annotator_selection_main.jsonl \
  --row 0 \
  --override paths.master_dir=/path/to/data/root \
  --override paths.results_dir=/path/to/results/root
```

### Launching A Manifest Via SLURM

The batch wrapper is
[slurm/run_manifest_array.sbatch](slurm/run_manifest_array.sbatch). It maps
`SLURM_ARRAY_TASK_ID` to the corresponding manifest row.

Generate the manifest first:

```bash
python scripts/generate_manifest.py annotator_selection_main
```

Determine the number of rows:

```bash
wc -l manifests/annotator_selection_main.jsonl
```

Submit one SLURM task per manifest row:

```bash
ROWS=$(wc -l < manifests/annotator_selection_main.jsonl)
sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.err" \
  --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/annotator_selection_main.jsonl
```

Pass a custom Python executable as the second positional argument if needed:

```bash
sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.err" \
  --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/annotator_selection_main.jsonl \
  /path/to/python
```

### Manifest Schema Summary

Each method file in `configs/launch/methods` is a JSON object with:

- `name`: stable method identifier
- `description`: short human-readable summary
- `tags`: optional metadata copied into manifest rows
- `overrides`: Hydra CLI overrides applied to each run

Each use-case file in `configs/launch/use_cases` is a JSON object with:

- `name`: stable manifest or study identifier
- `description`: short summary
- `common_overrides`: Hydra overrides shared by all rows
- `axes`: ordered expansion axes
- `exclude`: optional list of partial assignments to skip

Supported axis types are:

- `template`: render override templates for each value
- `choices`: pick overrides and tags from a named mapping
- `registry`: load method definitions from `methods/*.json`

## Optional MLflow Setup Step

Runs create the SQLite backend lazily when logging starts, but for larger
sweeps it is often useful to pre-create the backend and experiment once.

The helper script is [scripts/setup_mlflow.py](scripts/setup_mlflow.py). It
uses the resolved `results_path` as both the SQLite backend location and the
artifact root.

Prepare an experiment locally:

```bash
conda activate dalc
python scripts/setup_mlflow.py \
  --experiment-name good_pot_bad_crop
```

The matching SLURM wrapper is
[slurm/setup_mlflow.sbatch](slurm/setup_mlflow.sbatch):

```bash
sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%j.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%j.err" \
  slurm/setup_mlflow.sbatch \
  good_pot_bad_crop
```

You can optionally pass a custom results path and Python executable:

```bash
sbatch \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%j.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%j.err" \
  slurm/setup_mlflow.sbatch \
  good_pot_bad_crop \
  /path/to/mlflow \
  /path/to/python
```

To enforce ordering, submit the run array with an `afterok` dependency on the
setup job:

```bash
SETUP_JOB_ID=$(sbatch --parsable \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%j.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%j.err" \
  slurm/setup_mlflow.sbatch good_pot_bad_crop)
ROWS=$(wc -l < manifests/good_pot_bad_crop.jsonl)
sbatch --dependency=afterok:${SETUP_JOB_ID} \
  --chdir="${DALC_REPO_ROOT}" \
  --output="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.out" \
  --error="${DALC_REPO_ROOT}/slurm/logs/%x_%A_%a.err" \
  --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/good_pot_bad_crop.jsonl
```
