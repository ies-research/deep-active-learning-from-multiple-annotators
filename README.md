# Deep Active Learning from Multiple Annotators

This repository contains the code used for experiments on deep active
learning from multiple annotators. The README is intended to become the
central entry point for reproducing results, preparing datasets, and
launching runs.

## Environment

Activate the project environment before running data preparation or
experiments:

```bash
source /home/mherde/miniconda3/bin/activate dalc
```

Default paths are defined in
[configs/paths/default.yaml](configs/paths/default.yaml):

- `paths.master_dir=/home/datasets`
- `paths.results_dir=/home/results`
- `paths.dataset_cache_dir=${paths.master_dir}`
- `paths.embedder_cache_dir=${paths.master_dir}/.hf_embedders`
- `paths.pipeline_cache_dir=${paths.master_dir}/.hf_embed_cache`
- `paths.multi_annotator_cache_dir=${paths.master_dir}/.hf_multi_annotator_cache`
- `results_path=${paths.results_dir}/mlflow`

## Dataset Preparation

### `dopanim`

The repository can use `dopanim` as a local Hugging Face `DatasetDict` via
`source_kind: from_disk`.

Prepare the default `full` variant with:

```bash
source /home/mherde/miniconda3/bin/activate dalc
python scripts/prepare_dopanim.py --variant full
```

With the current path settings, the script defaults are:

- `--data-root /home/datasets`
- `--raw-dir /home/datasets/raw/dopanim`
- `--output-dir /home/datasets/dopanim_<variant>`

That means `--variant full` writes the processed dataset to
`/home/datasets/dopanim_full`, which matches
[configs/dataset/dopanim.yaml](configs/dataset/dopanim.yaml).

Available parameters:

- `--data-root PATH`: base directory for raw and processed dataset artifacts
- `--raw-dir PATH`: exact directory for the downloaded Zenodo archive and extracted files
- `--output-dir PATH`: exact directory where the processed `DatasetDict` is saved
- `--variant {full,worst-var,rand-var,worst-1,worst-2,worst-3,worst-4,rand-1,rand-2,rand-3,rand-4}`: train annotation subset to materialize
- `--force-download`: re-download and re-extract the raw archive
- `--force-rebuild`: overwrite an existing processed dataset directory

Example with explicit paths:

```bash
source /home/mherde/miniconda3/bin/activate dalc
python scripts/prepare_dopanim.py \
  --data-root /home/datasets \
  --variant full \
  --force-rebuild
```

After preparation, run experiments with `dataset=dopanim`. The legacy config
name `dataset=dopanim15` still exists as an alias. If you build a non-`full`
variant, either update
[configs/dataset/dopanim.yaml](configs/dataset/dopanim.yaml)
or override `dataset.source` at runtime so it points to the matching output
directory.

### Preparing Cached Embeddings And Simulated Multi-Annotator Labels

The batch entry point for preparing dataset artifacts is
[slurm/prepare_datasets.sbatch](slurm/prepare_datasets.sbatch).
It prepares cached embeddings for the configured datasets and, where needed,
also prepares simulated multi-annotator labels.

Current array index mapping:

- `0`: `dtd47`
- `1`: `dermamnist7`
- `2`: `food101`
- `3`: `audiomnist10`
- `4`: `skits2i14`
- `5`: `banking77`
- `6`: `trec6`
- `7`: `letter_recognition`
- `8`: `dopanim`

For `letter_recognition`, the script uses the Hugging Face dataset
`wwydmanski/tabular-letter-recognition` together with the
`identity_tabular` embedder for both classification and simulation. The
tabular embedder fits feature-wise standardization on the training split and
reuses that transform for test and simulation features.

For `dopanim`, the script additionally runs `scripts/prepare_dopanim.py
--variant full` before preparing embeddings. Its classification embedder is
`dinov2`, and no simulation step is run because `dopanim` already contains annotator
labels.

#### Via SLURM

Submit the full array with:

```bash
source /home/mherde/miniconda3/bin/activate dalc
sbatch --array=0-8 slurm/prepare_datasets.sbatch
```

You can also pass a specific Python executable as the first positional
argument:

```bash
sbatch --array=0-8 slurm/prepare_datasets.sbatch /path/to/python
```

#### Without SLURM

The script can also be executed as a plain shell script by setting
`SLURM_ARRAY_TASK_ID` manually.

Run one dataset locally, for example `dopanim` (`SLURM_ARRAY_TASK_ID=8`):

```bash
source /home/mherde/miniconda3/bin/activate dalc
SLURM_ARRAY_TASK_ID=8 bash slurm/prepare_datasets.sbatch
```

Run one dataset with a custom Python executable:

```bash
source /home/mherde/miniconda3/bin/activate dalc
SLURM_ARRAY_TASK_ID=8 bash slurm/prepare_datasets.sbatch /path/to/python
```

Run the full preparation sequence locally:

```bash
source /home/mherde/miniconda3/bin/activate dalc
for i in {0..8}; do
  SLURM_ARRAY_TASK_ID=$i bash slurm/prepare_datasets.sbatch
done
```

## Experiment Launch Pipeline

The repository also contains a manifest-driven launch pipeline for experiment
runs.

### Layout

- `configs/launch/methods/*.json`: frozen method definitions with reusable
  Hydra overrides
- `configs/launch/use_cases/*.json`: study-specific Cartesian products over
  datasets, seeds, methods, and other factors

### Workflow

The launch flow consists of two steps:

1. Generate a JSONL manifest from a use-case specification.
2. Execute one manifest row locally or submit the full manifest as a SLURM
   array.

Example:

```bash
python scripts/generate_manifest.py annotator_selection_main
```

By default, this writes `manifests/<use_case>.jsonl` and prints a short
preview of the generated rows.

To execute a single row locally:

```bash
python scripts/run_manifest_row.py \
  --manifest manifests/annotator_selection_main.jsonl \
  --row 0
```

### Launching A Manifest Via SLURM

The batch entry point is
[slurm/run_manifest_array.sbatch](slurm/run_manifest_array.sbatch).
The script runs
[scripts/run_manifest_row.py](scripts/run_manifest_row.py)
and maps each SLURM array index to one manifest row through
`SLURM_ARRAY_TASK_ID`.

Generate the manifest first:

```bash
python scripts/generate_manifest.py annotator_selection_main
```

Determine the number of rows in the manifest:

```bash
wc -l manifests/annotator_selection_main.jsonl
```

Submit one SLURM task per manifest row:

```bash
ROWS=$(wc -l < manifests/annotator_selection_main.jsonl)
sbatch --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/annotator_selection_main.jsonl
```

Example for a 128-row manifest:

```bash
sbatch --array=0-127 \
  slurm/run_manifest_array.sbatch \
  manifests/annotator_selection_main.jsonl
```

An alternative Python executable can be passed as the second positional
argument:

```bash
sbatch --array=0-127 \
  slurm/run_manifest_array.sbatch \
  manifests/annotator_selection_main.jsonl \
  /path/to/python
```

The default batch script requests one GPU, four CPUs, and 32 GB of memory.
Logs are written to `slurm/logs/%x_%A_%a.out` and
`slurm/logs/%x_%A_%a.err`.

### Optional MLflow Setup Step

Runs already create the SQLite backend and experiment lazily when logging
starts, but for larger SLURM sweeps it can be useful to pre-create the MLflow
database and experiment once before launching the array. This avoids a
"first writer wins" race on a fresh backend and makes the artifact location
explicit up front.

The helper script is [scripts/setup_mlflow.py](scripts/setup_mlflow.py). It
uses the same settings as training runs: `results_path` is used both as the
SQLite backend location and as the experiment artifact root.

Prepare an experiment locally:

```bash
source /home/mherde/miniconda3/bin/activate dalc
python scripts/setup_mlflow.py \
  --experiment-name good_pot_bad_crop
```

The matching SLURM wrapper is
[slurm/setup_mlflow.sbatch](slurm/setup_mlflow.sbatch):

```bash
source /home/mherde/miniconda3/bin/activate dalc
sbatch slurm/setup_mlflow.sbatch good_pot_bad_crop
```

You can optionally pass a custom results path and Python executable:

```bash
sbatch slurm/setup_mlflow.sbatch \
  good_pot_bad_crop \
  /path/to/mlflow \
  /path/to/python
```

To enforce ordering, submit the run array with an `afterok` dependency on the
setup job:

```bash
SETUP_JOB_ID=$(sbatch --parsable slurm/setup_mlflow.sbatch good_pot_bad_crop)
ROWS=$(wc -l < manifests/good_pot_bad_crop.jsonl)
sbatch --dependency=afterok:${SETUP_JOB_ID} \
  --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/good_pot_bad_crop.jsonl
```

### Method Schema

Each method file is a JSON object with:

- `name`: stable method identifier
- `description`: short human-readable summary
- `tags`: optional metadata copied into manifest rows
- `overrides`: list of Hydra CLI overrides

Example:

```json
{
  "name": "ks_big",
  "description": "Kernel-smoothed Bayesian gain with greedy sample assignment",
  "tags": {
    "scorer_family": "ks_big",
    "assigner_family": "greedy_sample"
  },
  "overrides": [
    "scorer@scorer.actual=ks_big",
    "assigner@assigner.actual=greedy_sample"
  ]
}
```

### Use-Case Schema

Each use-case file is a JSON object with:

- `name`: stable manifest or use-case identifier
- `description`: short summary
- `common_overrides`: Hydra overrides shared by all rows
- `axes`: ordered list of expansion axes

Supported axis types:

- `template`: expand `values` and render each listed override template with `{value}`
- `choices`: expand a mapping from option name to `{overrides, tags}`
- `registry`: load one file per named method from `methods/*.json`

The generator forms the Cartesian product across axes and writes one JSON line
per run to `manifests/<use_case>.jsonl`.

### Coupled Parameters

Use-case specs can restrict combinations in two ways:

- Option-level `when`: a choice or registry entry may include a `when` mapping;
  the option is only valid if the selected labels of the referenced axes match
  that mapping
- Top-level `exclude`: a use-case may define a list of partial axis assignments
  to skip entirely

Example:

```json
{
  "name": "example",
  "common_overrides": [],
  "exclude": [
    {
      "channel_variant": "scalar_uniform_confusion",
      "gain_type": ["entropy", "brier"]
    }
  ],
  "axes": []
}
```

This keeps the specification compact while avoiding meaningless full-grid
combinations.

## Planned README Extensions

This README is intended to grow into the main reproducibility document for the
associated paper. In a later stage, it should also document the repository
structure, the central methodological components, and the recommended path for
reproducing the main experimental results end to end.
