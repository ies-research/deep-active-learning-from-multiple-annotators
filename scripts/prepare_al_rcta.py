from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MISSING_LABEL = -1
UPSTREAM_REPO_URL = "https://github.com/varuntotakura/al_rcta"
UPSTREAM_COMMIT = "13f30c3e5641da0c9c7196d6313ed76288473a8e"
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass(frozen=True)
class ALRCTADatasetSpec:
    name: str
    output_name: str
    root_dir: str
    main_file: str
    annotation_dir: str
    annotation_glob: str
    classes: tuple[str, ...]
    main_index_key: str
    main_text_key: str
    main_label_key: str
    annotation_text_key: str
    expected_annotators: int
    expected_train_size: int = 3000

    @property
    def n_classes(self) -> int:
        return len(self.classes)


DATASET_SPECS: dict[str, ALRCTADatasetSpec] = {
    "agnews": ALRCTADatasetSpec(
        name="agnews",
        output_name="al_rcta_agnews",
        root_dir="Data_AGNewsGroups",
        main_file="Cleaned_AG_News_Dataset_3_columns_ALL.xlsx",
        annotation_dir="Annotations",
        annotation_glob="AG_Upwork_*.*",
        classes=("world", "sports", "business", "science_technology"),
        main_index_key="__col0",
        main_text_key="Description",
        main_label_key="Class Index",
        annotation_text_key="Description",
        expected_annotators=10,
    ),
    "consumer_complaints": ALRCTADatasetSpec(
        name="consumer_complaints",
        output_name="al_rcta_consumer_complaints",
        root_dir="Data_ConsumerComplaints",
        main_file="Cleaned_Dataset_All.xlsx",
        annotation_dir="Annotations",
        annotation_glob="CC_Upwork_*.*",
        classes=(
            "debt_collection",
            "prepaid_card_debit_card",
            "mortgage",
            "checking_savings_account",
            "student_loan",
            "vehicle_loan_lease",
        ),
        main_index_key="__col0",
        main_text_key="Consumer complaint narrative",
        main_label_key="Product",
        annotation_text_key="Consumer complaint narrative",
        expected_annotators=10,
    ),
    "wiki_movie_plots": ALRCTADatasetSpec(
        name="wiki_movie_plots",
        output_name="al_rcta_wiki_movie_plots",
        root_dir="Data_WikipediaMoviePlots",
        main_file="Cleaned_Dataset_All.csv",
        annotation_dir="Annotations",
        annotation_glob="Wiki_Upwork_*.*",
        classes=("drama", "comedy", "horror", "action"),
        main_index_key="Index",
        main_text_key="Plot",
        main_label_key="Genre",
        annotation_text_key="Plot",
        expected_annotators=9,
    ),
}


def _resolve_default_data_root() -> Path:
    env_data_root = os.environ.get("DALC_DATA_ROOT")
    if env_data_root:
        return Path(env_data_root).expanduser().resolve()

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="experiment")

    data_root = Path(str(cfg.paths.master_dir)).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()
    return data_root


def parse_args() -> argparse.Namespace:
    default_data_root = _resolve_default_data_root()
    parser = argparse.ArgumentParser(
        description=(
            "Convert the al_rcta crowd-annotated text datasets into local "
            "Hugging Face DatasetDict directories."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        required=True,
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Base directory for raw and processed dataset artifacts. Defaults "
            "to DALC_DATA_ROOT when set, otherwise to Hydra paths.master_dir."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Path to a local clone or extracted copy of varuntotakura/al_rcta. "
            "Defaults to <data-root>/raw/al_rcta."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination of the saved DatasetDict. Use a distinct directory "
            "for each test cap. Defaults to <data-root>/<dataset>_testN when "
            "--max-test-size N is set, otherwise <data-root>/<dataset>."
        ),
    )
    parser.add_argument(
        "--max-test-size",
        type=int,
        default=None,
        help=(
            "Optional maximum number of test examples. When set below the full "
            "test size, a deterministic stratified sample is retained."
        ),
    )
    parser.add_argument(
        "--test-random-state",
        type=int,
        default=0,
        help="Random seed for deterministic stratified test sampling.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Delete and re-clone the managed raw al_rcta repository before "
            "building the processed DatasetDict."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Overwrite an existing processed DatasetDict.",
    )
    return parser.parse_args()


def default_raw_dir(data_root: Path) -> Path:
    return data_root / "raw" / "al_rcta"


def default_output_dir(
    data_root: Path, spec: ALRCTADatasetSpec, max_test_size: int | None
) -> Path:
    if max_test_size is None:
        return data_root / spec.output_name
    return data_root / f"{spec.output_name}_test{max_test_size}"


def run_git(args: list[str], *, cwd: Path | None = None) -> None:
    cmd = ["git", *args]
    try:
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is required to download AL-RCTA raw data automatically. "
            f"Install git or manually clone {UPSTREAM_REPO_URL} into the raw "
            "directory and pass --raw-dir if needed."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Git command failed with exit code {exc.returncode}: {' '.join(cmd)}"
        ) from exc


def raw_dir_is_empty(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return True
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw al_rcta path is not a directory: {raw_dir}")
    return not any(raw_dir.iterdir())


def raw_dataset_is_valid(raw_dir: Path, spec: ALRCTADatasetSpec) -> bool:
    dataset_root = raw_dir / spec.root_dir
    return (
        (dataset_root / spec.main_file).exists()
        and (dataset_root / spec.annotation_dir).is_dir()
    )


def checkout_pinned_commit(raw_dir: Path) -> None:
    try:
        run_git(["checkout", UPSTREAM_COMMIT], cwd=raw_dir)
    except RuntimeError:
        run_git(["fetch", "origin", UPSTREAM_COMMIT], cwd=raw_dir)
        run_git(["checkout", UPSTREAM_COMMIT], cwd=raw_dir)


def clone_raw_dataset(raw_dir: Path) -> None:
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    if raw_dir.exists():
        raw_dir.rmdir()

    print(f"Cloning {UPSTREAM_REPO_URL} -> {raw_dir}")
    run_git(["clone", UPSTREAM_REPO_URL, str(raw_dir)])
    checkout_pinned_commit(raw_dir)


def prepare_raw_dataset(
    raw_dir: Path,
    spec: ALRCTADatasetSpec,
    *,
    force_download: bool,
) -> Path:
    if force_download:
        print(f"Force download requested; clearing raw data under {raw_dir}")
        if raw_dir.exists():
            if not raw_dir.is_dir():
                raise NotADirectoryError(
                    f"Cannot force-download into non-directory path: {raw_dir}"
                )
            shutil.rmtree(raw_dir)

    if raw_dir_is_empty(raw_dir):
        clone_raw_dataset(raw_dir)
    elif (raw_dir / ".git").is_dir():
        print(f"Reusing raw al_rcta clone at {raw_dir}")
        checkout_pinned_commit(raw_dir)
    else:
        print(f"Reusing existing raw al_rcta directory at {raw_dir}")

    if not raw_dataset_is_valid(raw_dir, spec):
        raise FileNotFoundError(
            "Raw al_rcta directory does not contain the expected files for "
            f"{spec.name}: {raw_dir / spec.root_dir}. If this is a partial or "
            "stale checkout, rerun with --force-download or pass a valid "
            "--raw-dir."
        )

    return raw_dir


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def parse_int(value: object) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        number = float(text)
        if not number.is_integer():
            return None
        return int(number)
    except Exception:
        return None


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _xlsx_column_index(cell_ref: str | None) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref or "A1").group(1)
    out = 0
    for char in letters:
        out = out * 26 + ord(char) - ord("A") + 1
    return out - 1


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", XLSX_NS):
                parts = [text.text or "" for text in item.findall(".//a:t", XLSX_NS)]
                shared_strings.append("".join(parts))

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row_node in sheet_root.findall(".//a:sheetData/a:row", XLSX_NS):
            row: list[str] = []
            for cell in row_node.findall("a:c", XLSX_NS):
                col_idx = _xlsx_column_index(cell.attrib.get("r"))
                while len(row) <= col_idx:
                    row.append("")

                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", XLSX_NS)
                if cell_type == "inlineStr":
                    value = "".join(
                        text.text or "" for text in cell.findall(".//a:t", XLSX_NS)
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text or ""
                row[col_idx] = value
            rows.append(row)

    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def read_csv_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.reader(handle))


def read_table_rows(path: Path) -> list[list[str]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_xlsx_rows(path)
    if path.suffix.lower() == ".csv":
        return read_csv_rows(path)
    raise ValueError(f"Unsupported table file extension for {path}.")


def rows_to_dicts(rows: list[list[str]]) -> tuple[list[str], list[dict[str, str]]]:
    if not rows:
        raise ValueError("Cannot parse an empty table.")

    header = rows[0]
    keys = [name if name else f"__col{idx}" for idx, name in enumerate(header)]
    records = []
    for row in rows[1:]:
        padded = row + [""] * (len(keys) - len(row))
        records.append({key: padded[idx] for idx, key in enumerate(keys)})
    return keys, records


def annotation_label_key(header: list[str]) -> str:
    for key in ("Annotation", "Annotations", "Labels"):
        if key in header:
            return key
    raise ValueError(
        "Could not find an annotation label column. Expected one of "
        "'Annotation', 'Annotations', or 'Labels'."
    )


def read_main_records(
    raw_dir: Path,
    spec: ALRCTADatasetSpec,
) -> list[dict[str, object]]:
    main_path = raw_dir / spec.root_dir / spec.main_file
    if not main_path.exists():
        raise FileNotFoundError(f"Missing main dataset file: {main_path}")

    _, rows = rows_to_dicts(read_table_rows(main_path))
    records: list[dict[str, object]] = []
    seen_indices: set[int] = set()
    for row in rows:
        source_index = parse_int(row.get(spec.main_index_key))
        raw_label = parse_int(row.get(spec.main_label_key))
        text = normalize_text(row.get(spec.main_text_key))
        if source_index is None or raw_label is None or not text:
            continue
        if source_index in seen_indices:
            raise ValueError(
                f"Duplicate source index {source_index} in {main_path}."
            )
        if raw_label < 1 or raw_label > spec.n_classes:
            raise ValueError(
                f"Label {raw_label} for source index {source_index} is outside "
                f"1..{spec.n_classes}."
            )
        seen_indices.add(source_index)
        records.append(
            {
                "source_index": source_index,
                "text": text,
                "label": raw_label - 1,
            }
        )
    records.sort(key=lambda row: int(row["source_index"]))
    return records


def read_annotation_files(
    raw_dir: Path,
    spec: ALRCTADatasetSpec,
) -> tuple[list[str], list[dict[int, int]], dict[str, object]]:
    annotation_root = raw_dir / spec.root_dir / spec.annotation_dir
    annotation_paths = sorted(annotation_root.glob(spec.annotation_glob), key=natural_key)
    if len(annotation_paths) != spec.expected_annotators:
        raise ValueError(
            f"Expected {spec.expected_annotators} annotation files for {spec.name}, "
            f"found {len(annotation_paths)} under {annotation_root}."
        )

    annotator_names: list[str] = []
    labels_by_annotator: list[dict[int, int]] = []
    per_annotator_stats: list[dict[str, object]] = []
    invalid_nonzero_count = 0

    for annotator_idx, path in enumerate(annotation_paths):
        header, rows = rows_to_dicts(read_table_rows(path))
        label_key = annotation_label_key(header)
        annotator_name = path.stem
        annotator_names.append(annotator_name)

        labels: dict[int, int] = {}
        raw_label_counts: dict[str, int] = {}
        invalid_nonzero_for_file = 0
        for row in rows:
            source_index = parse_int(row.get("__col0") or row.get("Index"))
            if source_index is None:
                continue

            raw_label = parse_int(row.get(label_key))
            raw_label_counts[str(raw_label)] = raw_label_counts.get(str(raw_label), 0) + 1
            if raw_label is None or raw_label == 0:
                label = MISSING_LABEL
            elif 1 <= raw_label <= spec.n_classes:
                label = raw_label - 1
            else:
                label = MISSING_LABEL
                invalid_nonzero_for_file += 1

            if source_index in labels:
                raise ValueError(
                    f"Duplicate annotation for source index {source_index} in {path}."
                )
            labels[source_index] = label

        invalid_nonzero_count += invalid_nonzero_for_file
        labels_by_annotator.append(labels)
        per_annotator_stats.append(
            {
                "annotator": annotator_name,
                "file": str(path),
                "resolved_rows": len(labels),
                "invalid_nonzero_count": invalid_nonzero_for_file,
                "raw_label_counts": raw_label_counts,
            }
        )

        if len(labels) == 0:
            raise ValueError(f"Annotation file {path} did not yield any rows.")
        if annotator_idx == 0:
            continue

    metadata = {
        "annotation_files": [str(path) for path in annotation_paths],
        "invalid_nonzero_annotation_count": invalid_nonzero_count,
        "per_annotator": per_annotator_stats,
    }
    return annotator_names, labels_by_annotator, metadata


def class_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(int(record["label"]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def allocate_stratified_counts(
    labels: np.ndarray,
    *,
    max_size: int,
) -> dict[int, int]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if max_size < 0:
        raise ValueError("max_size must be >= 0.")
    if max_size >= labels.size:
        values, counts = np.unique(labels, return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}
    if max_size == 0 or labels.size == 0:
        return {}

    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(values)
    values = values[order]
    counts = counts[order].astype(np.int64)
    total = int(counts.sum())
    n_classes = len(values)

    if max_size < n_classes:
        ranked = sorted(
            zip(values.tolist(), counts.tolist()),
            key=lambda item: (-item[1], item[0]),
        )
        return {int(label): 1 for label, _ in ranked[:max_size]}

    raw = counts.astype(float) * (float(max_size) / float(total))
    allocation = np.ones(n_classes, dtype=np.int64)
    remaining = int(max_size - allocation.sum())
    raw_extra = np.maximum(raw - 1.0, 0.0)
    extra = np.floor(raw_extra).astype(np.int64)
    extra = np.minimum(extra, counts - allocation)

    if int(extra.sum()) > remaining:
        extra = np.zeros(n_classes, dtype=np.int64)

    allocation += extra
    remaining = int(max_size - allocation.sum())

    while remaining > 0:
        priorities = raw - allocation
        priorities = np.where(allocation < counts, priorities, -np.inf)
        best = int(np.argmax(priorities))
        if not np.isfinite(priorities[best]):
            break
        allocation[best] += 1
        remaining -= 1

    return {
        int(label): int(count)
        for label, count in zip(values.tolist(), allocation.tolist())
        if count > 0
    }


def stratified_sample_records(
    records: list[dict[str, object]],
    *,
    max_test_size: int | None,
    random_state: int,
) -> list[dict[str, object]]:
    if max_test_size is None or max_test_size >= len(records):
        return list(records)
    if max_test_size < 0:
        raise ValueError("max_test_size must be >= 0.")
    if max_test_size == 0:
        return []

    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    allocation = allocate_stratified_counts(labels, max_size=max_test_size)
    rng = np.random.RandomState(random_state)

    selected_indices: list[int] = []
    for label in sorted(allocation):
        candidate_indices = np.flatnonzero(labels == label)
        chosen = rng.choice(candidate_indices, size=allocation[label], replace=False)
        selected_indices.extend(int(idx) for idx in chosen)

    return [records[idx] for idx in sorted(selected_indices)]


def build_train_records_and_annotations(
    main_records: list[dict[str, object]],
    labels_by_annotator: list[dict[int, int]],
) -> tuple[list[dict[str, object]], np.ndarray]:
    main_by_index = {int(row["source_index"]): row for row in main_records}
    train_ids: set[int] | None = None
    for labels in labels_by_annotator:
        resolved_ids = set(labels) & set(main_by_index)
        train_ids = resolved_ids if train_ids is None else train_ids & resolved_ids

    if train_ids is None:
        raise ValueError("No annotation dictionaries were provided.")

    ordered_train_ids = sorted(train_ids)
    train_records = [main_by_index[source_index] for source_index in ordered_train_ids]
    annotations = np.full(
        (len(ordered_train_ids), len(labels_by_annotator)),
        fill_value=MISSING_LABEL,
        dtype=np.int64,
    )
    for annotator_idx, labels in enumerate(labels_by_annotator):
        for row_idx, source_index in enumerate(ordered_train_ids):
            annotations[row_idx, annotator_idx] = labels[source_index]

    return train_records, annotations


def build_split_dataset(
    records: list[dict[str, object]],
    *,
    classes: tuple[str, ...],
    annotations: np.ndarray | None = None,
) -> Dataset:
    features_dict = {
        "text": Value("string"),
        "label": ClassLabel(names=list(classes)),
        "source_index": Value("int64"),
    }
    data_dict = {
        "text": [str(record["text"]) for record in records],
        "label": [int(record["label"]) for record in records],
        "source_index": [int(record["source_index"]) for record in records],
    }

    if annotations is not None:
        if len(records) != annotations.shape[0]:
            raise ValueError(
                "records and annotations must agree on the number of rows; "
                f"got {len(records)} and {annotations.shape[0]}."
            )
        features_dict["z"] = Sequence(
            feature=Value("int64"),
            length=int(annotations.shape[1]),
        )
        data_dict["z"] = annotations.astype(np.int64, copy=False).tolist()

    return Dataset.from_dict(data_dict, features=Features(features_dict))


def build_datasetdict_from_records(
    spec: ALRCTADatasetSpec,
    main_records: list[dict[str, object]],
    labels_by_annotator: list[dict[int, int]],
    *,
    max_test_size: int | None = None,
    test_random_state: int = 0,
    expected_train_size: int | None = None,
) -> tuple[DatasetDict, dict[str, object]]:
    train_records, annotations = build_train_records_and_annotations(
        main_records,
        labels_by_annotator,
    )
    if expected_train_size is not None and len(train_records) != expected_train_size:
        raise ValueError(
            f"Expected {expected_train_size} train rows for {spec.name}, "
            f"got {len(train_records)}."
        )

    train_ids = {int(record["source_index"]) for record in train_records}
    full_test_records = [
        record
        for record in main_records
        if int(record["source_index"]) not in train_ids
    ]
    test_records = stratified_sample_records(
        full_test_records,
        max_test_size=max_test_size,
        random_state=test_random_state,
    )

    dataset_dict = DatasetDict(
        {
            "train": build_split_dataset(
                train_records,
                classes=spec.classes,
                annotations=annotations,
            ),
            "test": build_split_dataset(test_records, classes=spec.classes),
        }
    )

    observed_mask = annotations != MISSING_LABEL
    metadata = {
        "name": spec.name,
        "classes": list(spec.classes),
        "missing_label": MISSING_LABEL,
        "train_size": len(train_records),
        "full_test_size": len(full_test_records),
        "retained_test_size": len(test_records),
        "max_test_size": max_test_size,
        "test_random_state": test_random_state,
        "train_class_counts": class_counts(train_records),
        "full_test_class_counts": class_counts(full_test_records),
        "retained_test_class_counts": class_counts(test_records),
        "annotator_count": int(annotations.shape[1]),
        "annotation_slots": int(annotations.size),
        "observed_annotation_count": int(observed_mask.sum()),
        "missing_annotation_count": int((~observed_mask).sum()),
    }
    return dataset_dict, metadata


def build_datasetdict(
    raw_dir: Path,
    spec: ALRCTADatasetSpec,
    *,
    max_test_size: int | None,
    test_random_state: int,
) -> tuple[DatasetDict, dict[str, object]]:
    main_records = read_main_records(raw_dir, spec)
    annotators, labels_by_annotator, annotation_metadata = read_annotation_files(
        raw_dir,
        spec,
    )
    dataset_dict, metadata = build_datasetdict_from_records(
        spec,
        main_records,
        labels_by_annotator,
        max_test_size=max_test_size,
        test_random_state=test_random_state,
        expected_train_size=spec.expected_train_size,
    )
    metadata.update(
        {
            "source_repo": UPSTREAM_REPO_URL,
            "source_commit": UPSTREAM_COMMIT,
            "raw_dir": str(raw_dir),
            "main_file": str(raw_dir / spec.root_dir / spec.main_file),
            "annotators": annotators,
            **annotation_metadata,
        }
    )
    return dataset_dict, metadata


def save_datasetdict(
    dataset_dict: DatasetDict,
    metadata: dict[str, object],
    *,
    output_dir: Path,
    force_rebuild: bool,
) -> None:
    if output_dir.exists():
        if not force_rebuild:
            raise FileExistsError(
                f"{output_dir} already exists. Use --force-rebuild to overwrite it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    metadata_path = output_dir / "al_rcta_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    raw_dir = (args.raw_dir or default_raw_dir(args.data_root)).expanduser().resolve()
    output_dir = (
        args.output_dir or default_output_dir(args.data_root, spec, args.max_test_size)
    ).expanduser().resolve()

    print(f"al_rcta dataset      : {spec.name}")
    print(f"Raw directory        : {raw_dir}")
    print(f"Processed output dir : {output_dir}")
    print(f"Max test size        : {args.max_test_size}")
    print(f"Test random state    : {args.test_random_state}")
    print(f"Force download       : {args.force_download}")
    print(f"Force rebuild        : {args.force_rebuild}")

    if args.force_download:
        raw_dir = prepare_raw_dataset(
            raw_dir,
            spec,
            force_download=True,
        )

    if output_dir.exists() and not args.force_rebuild:
        print(
            f"Processed dataset already exists at {output_dir}; "
            "skipping rebuild. Use --force-rebuild to overwrite it."
        )
        return

    if not args.force_download:
        raw_dir = prepare_raw_dataset(
            raw_dir,
            spec,
            force_download=False,
        )

    dataset_dict, metadata = build_datasetdict(
        raw_dir,
        spec,
        max_test_size=args.max_test_size,
        test_random_state=args.test_random_state,
    )
    save_datasetdict(
        dataset_dict,
        metadata,
        output_dir=output_dir,
        force_rebuild=args.force_rebuild,
    )

    print(
        f"Saved {spec.name}: train={metadata['train_size']} "
        f"test={metadata['retained_test_size']} "
        f"(full_test={metadata['full_test_size']})"
    )
    print("Use it with a dataset config whose source_kind is 'from_disk'.")


if __name__ == "__main__":
    main()
