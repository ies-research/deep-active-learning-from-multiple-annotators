# deep-active-learning-from-crowds

## Preparing DOPAnim

The repository can use DOPAnim as a local Hugging Face `DatasetDict` via
`source_kind: from_disk`.

Activate the project environment and prepare the default `full` variant with:

```bash
source /home/mherde/miniconda3/bin/activate dalc
python scripts/prepare_dopanim.py --variant full
```

With the current path settings in
[configs/paths/default.yaml](/home/mherde/PycharmProjects/deep-active-learning-from-crowds/configs/paths/default.yaml),
the script defaults are:

- `--data-root /home/datasets`
- `--raw-dir /home/datasets/raw/dopanim`
- `--output-dir /home/datasets/dopanim_<variant>`

That means `--variant full` writes the processed dataset to
`/home/datasets/dopanim_full`, which matches
[configs/dataset/dopanim.yaml](/home/mherde/PycharmProjects/deep-active-learning-from-crowds/configs/dataset/dopanim.yaml).

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

After preparation, run experiments with `dataset=dopanim`. If you build a
non-`full` variant, either update
[configs/dataset/dopanim.yaml](/home/mherde/PycharmProjects/deep-active-learning-from-crowds/configs/dataset/dopanim.yaml)
or override `dataset.source` at runtime so it points to the matching output
directory.
