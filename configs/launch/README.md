# Launch Pipeline

This directory defines a manifest-driven launch pipeline for experiment runs.

## Layout

- `methods/*.json`
  Frozen method definitions. Each file defines one reusable method and its
  Hydra overrides.
- `use_cases/*.json`
  Study-specific Cartesian products over datasets, seeds, methods, and other
  factors.

## Workflow

The launch flow consists of two steps:

1. Generate a JSONL manifest from a use-case specification.
2. Execute one manifest row locally or submit the full manifest as a SLURM
   array.

Example:

```bash
python scripts/generate_manifest.py worker_selection_main
```

By default, the command writes `manifests/<use_case>.jsonl` and prints a short
preview of the generated rows.

To execute a single row locally:

```bash
python scripts/run_manifest_row.py \
  --manifest manifests/worker_selection_main.jsonl \
  --row 0
```

## Starting An Experiment Via SLURM

The batch entry point is `slurm/run_manifest_array.sbatch`. The script runs
`scripts/run_manifest_row.py` and maps each SLURM array index to one row in the
manifest through `SLURM_ARRAY_TASK_ID`.

Generate the manifest first:

```bash
python scripts/generate_manifest.py worker_selection_main
```

Determine the number of rows in the manifest:

```bash
wc -l manifests/worker_selection_main.jsonl
```

Submit one SLURM task per manifest row:

```bash
ROWS=$(wc -l < manifests/worker_selection_main.jsonl)
sbatch --array=0-$((ROWS-1)) \
  slurm/run_manifest_array.sbatch \
  manifests/worker_selection_main.jsonl
```

Example for a 128-row manifest:

```bash
sbatch --array=0-127 \
  slurm/run_manifest_array.sbatch \
  manifests/worker_selection_main.jsonl
```

An alternative Python executable can be passed as the second positional
argument:

```bash
sbatch --array=0-127 \
  slurm/run_manifest_array.sbatch \
  manifests/worker_selection_main.jsonl \
  /path/to/python
```

The default batch script requests one GPU, four CPUs, and 32 GB of memory.
Logs are written to `slurm/logs/%x_%A_%a.out` and
`slurm/logs/%x_%A_%a.err`.

## Method Schema

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

## Use-Case Schema

Each use-case file is a JSON object with:

- `name`: stable manifest/use-case identifier
- `description`: short summary
- `common_overrides`: Hydra overrides shared by all rows
- `axes`: ordered list of expansion axes

Supported axis types:

- `template`
  Expand `values` and render each listed override template with `{value}`.
- `choices`
  Expand a mapping from option name to `{overrides, tags}`.
- `registry`
  Load one file per named method from `methods/*.json`.

The generator forms the Cartesian product across axes and writes one JSON line
per run to `manifests/<use_case>.jsonl`.

## Coupled Parameters

Use-case specs can restrict combinations in two ways:

- Option-level `when`
  A choice or registry entry may include a `when` mapping. The option is only
  valid if the selected labels of the referenced axes match that mapping.
- Top-level `exclude`
  A use-case may define a list of partial axis assignments to skip entirely.

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
