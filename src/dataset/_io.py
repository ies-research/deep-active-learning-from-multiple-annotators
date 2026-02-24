from __future__ import annotations
from datasets import load_dataset, load_from_disk, concatenate_datasets
from ._spec import HFDatasetSpec
from ._cache import file_signature, dir_signature


def load_datasetdict(spec: HFDatasetSpec):
    if spec.source_kind == "from_disk":
        ds = load_from_disk(spec.source)
        if isinstance(ds, dict):
            return ds
        raise TypeError("Expected DatasetDict from load_from_disk.")
    return load_dataset(
        spec.source,
        name=spec.dataset_config,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        cache_dir=spec.cache_dir,
        data_dir=spec.data_dir,
        data_files=spec.data_files,
    )


def merge_train_splits(ds_dict, spec: HFDatasetSpec):
    parts = [ds_dict[s] for s in spec.train_splits]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def spec_fingerprint(spec: HFDatasetSpec) -> dict:
    fpr = spec.__dict__.copy()
    if spec.local_signature:
        # basic invalidation for local setups
        from pathlib import Path

        src_path = Path(spec.source)
        if spec.source_kind == "from_disk" and src_path.exists():
            fpr["source_sig"] = dir_signature(spec.source)
        else:
            if src_path.exists():
                fpr["source_sig"] = file_signature(spec.source)
            if spec.data_dir is not None and Path(spec.data_dir).exists():
                fpr["data_dir_sig"] = dir_signature(spec.data_dir)
    return fpr
