from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value
from hydra import compose, initialize_config_dir


DOPANIM_URL = "https://zenodo.org/api/records/14016659/files-archive"
DOPANIM_ARCHIVE_NAME = "14016659.zip"

CLASSES = (
    "German Yellowjacket",
    "European Paper Wasp",
    "Yellow-legged Hornet",
    "European Hornet",
    "Brown Hare",
    "Black-tailed Jackrabbit",
    "Marsh Rabbit",
    "Desert Cottontail",
    "European Rabbit",
    "Eurasian Red Squirrel",
    "American Red Squirrel",
    "Douglas' Squirrel",
    "Cheetah",
    "Jaguar",
    "Leopard",
)

ANNOTATORS = (
    "digital-dragon",
    "pixel-pioneer",
    "ocean-oracle",
    "starry-scribe",
    "sunlit-sorcerer",
    "emerald-empath",
    "sapphire-sphinx",
    "echo-eclipse",
    "lunar-lynx",
    "neon-ninja",
    "quantum-quokka",
    "velvet-voyager",
    "radiant-raven",
    "dreamy-drifter",
    "azure-artist",
    "twilight-traveler",
    "galactic-gardener",
    "cosmic-wanderer",
    "frosty-phoenix",
    "mystic-merlin",
)

SUPPORTED_SPLITS = ("train", "valid", "test")
SUPPORTED_VARIANTS = (
    "full",
    "worst-var",
    "rand-var",
    "worst-1",
    "worst-2",
    "worst-3",
    "worst-4",
    "rand-1",
    "rand-2",
    "rand-3",
    "rand-4",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
            "Download the DOPAnim Zenodo archive, convert it into a local "
            "Hugging Face DatasetDict, and save it to disk."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help=(
            "Base directory for raw and processed dataset artifacts. Defaults "
            "to DALC_DATA_ROOT when set, otherwise to the resolved "
            "`paths.master_dir` from the Hydra config."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory used for the downloaded Zenodo archive and extracted files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination of the saved DatasetDict.",
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="full",
        help="Train-split annotation subset to materialize.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download and re-extract the raw Zenodo archive.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Overwrite an existing processed DatasetDict.",
    )
    return parser.parse_args()


def default_raw_dir(data_root: Path) -> Path:
    return data_root / "raw" / "dopanim"


def default_output_dir(data_root: Path, variant: str) -> Path:
    return data_root / f"dopanim_{variant}"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as dst:
        shutil.copyfileobj(response, dst)


def extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {destination}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(destination)


def extract_nested_archives(root: Path) -> None:
    for archive_path in sorted(root.rglob("*.zip")):
        target_dir = archive_path.parent / archive_path.stem
        if target_dir.exists() and any(target_dir.iterdir()):
            continue
        extract_zip(archive_path, target_dir)


def locate_dataset_root(search_root: Path) -> Path:
    candidates = []
    for task_path in search_root.rglob("task_data.json"):
        candidate = task_path.parent
        if (candidate / "annotation_data.json").exists():
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a DOPAnim root under {search_root}."
        )

    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def prepare_raw_dataset(raw_dir: Path, *, force_download: bool) -> Path:
    archive_path = raw_dir / DOPANIM_ARCHIVE_NAME
    extract_root = raw_dir / "extracted"

    if force_download:
        if archive_path.exists():
            archive_path.unlink()
        if extract_root.exists():
            shutil.rmtree(extract_root)

    if not archive_path.exists():
        download_file(DOPANIM_URL, archive_path)

    if not extract_root.exists():
        extract_zip(archive_path, extract_root)
        extract_nested_archives(extract_root)

    return locate_dataset_root(extract_root)


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def build_label_index() -> dict[str, int]:
    return {label_name: idx for idx, label_name in enumerate(CLASSES)}


def resolve_image_path(
    dataset_root: Path,
    *,
    split_name: str,
    label_name: str,
    observation_id: str,
) -> Path:
    candidate = dataset_root / split_name / label_name / f"{observation_id}.jpeg"
    if candidate.exists():
        return candidate

    matches = list(dataset_root.rglob(f"{observation_id}.jpeg"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "Could not uniquely resolve image path for "
        f"split={split_name!r}, label={label_name!r}, observation_id={observation_id!r}."
    )


def collect_split_records(dataset_root: Path) -> dict[str, list[dict[str, object]]]:
    task_data = load_json(dataset_root / "task_data.json")
    label_to_index = build_label_index()

    records_by_split = {split: [] for split in SUPPORTED_SPLITS}
    for observation_id, observation in task_data.items():
        split_name = observation["split"]
        if split_name not in records_by_split:
            continue

        label_name = observation["taxon_name"]
        image_path = resolve_image_path(
            dataset_root,
            split_name=split_name,
            label_name=label_name,
            observation_id=str(observation_id),
        )
        records_by_split[split_name].append(
            {
                "observation_id": int(observation_id),
                "label_name": label_name,
                "label": label_to_index[label_name],
                "image": str(image_path),
            }
        )

    for split_name in SUPPORTED_SPLITS:
        records_by_split[split_name].sort(key=lambda row: row["observation_id"])

    return records_by_split


def rand_argmax(values: np.ndarray, *, axis: int, random_state: int) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    max_values = values.max(axis=axis, keepdims=True)
    ties = values == max_values
    noise = rng.random_sample(values.shape)
    tie_break = np.where(ties, noise, -1.0)
    return tie_break.argmax(axis=axis)


def load_likelihoods(
    dataset_root: Path,
    *,
    observation_ids: np.ndarray,
) -> np.ndarray:
    annotation_data = load_json(dataset_root / "annotation_data.json")
    obs_to_index = {
        int(observation_id): idx for idx, observation_id in enumerate(observation_ids)
    }
    annotator_to_index = {
        annotator: idx for idx, annotator in enumerate(ANNOTATORS)
    }
    class_to_index = build_label_index()

    likelihoods = np.full(
        (len(observation_ids), len(ANNOTATORS), len(CLASSES)),
        fill_value=-1.0,
        dtype=np.float32,
    )

    for annotation in annotation_data.values():
        obs_idx = obs_to_index.get(int(annotation["observation_id"]))
        annot_idx = annotator_to_index.get(annotation["annotator_id"])
        if obs_idx is None or annot_idx is None:
            continue

        for class_name, value in annotation["likelihoods"].items():
            cls_idx = class_to_index[class_name]
            likelihoods[obs_idx, annot_idx, cls_idx] = float(value)

        row = likelihoods[obs_idx, annot_idx]
        if np.all(row >= 0) and row.sum() > 0:
            likelihoods[obs_idx, annot_idx] = row / row.sum()

    return likelihoods


def build_train_annotations(
    dataset_root: Path,
    train_records: list[dict[str, object]],
    *,
    variant: str,
) -> np.ndarray:
    observation_ids = np.asarray(
        [row["observation_id"] for row in train_records], dtype=np.int64
    )
    y_true = np.asarray([row["label"] for row in train_records], dtype=np.int64)
    likelihoods = load_likelihoods(
        dataset_root,
        observation_ids=observation_ids,
    )

    is_not_annotated = np.any(likelihoods == -1, axis=-1)
    class_labels = rand_argmax(likelihoods, axis=-1, random_state=0).astype(
        np.int64
    )
    class_labels[is_not_annotated] = -1
    z = class_labels.copy()

    if variant.startswith("worst-") and variant != "worst-var":
        n_annotators_per_sample = int(variant.split("-")[-1])
        is_false = np.zeros_like(class_labels, dtype=np.float32)
        is_false += (y_true[:, None] != class_labels).astype(np.float32)
        is_false -= 2.0 * is_not_annotated.astype(np.float32)
        is_not_selected = np.ones_like(is_false, dtype=bool)
        random_floats = np.random.RandomState(n_annotators_per_sample).rand(
            *is_false.shape
        )
        selected_indices = np.argsort(-(is_false + random_floats), axis=-1)[
            :, :n_annotators_per_sample
        ]
        for col_idx in range(n_annotators_per_sample):
            is_not_selected[np.arange(len(z)), selected_indices[:, col_idx]] = False
        z[is_not_selected] = -1
    elif variant.startswith("rand-") and variant != "rand-var":
        n_annotators_per_sample = int(variant.split("-")[-1])
        is_annotated = (~is_not_annotated).astype(np.float32)
        is_not_selected = np.ones_like(is_annotated, dtype=bool)
        random_floats = np.random.RandomState(n_annotators_per_sample + 4).rand(
            *is_annotated.shape
        )
        selected_indices = np.argsort(-(is_annotated + random_floats), axis=-1)[
            :, :n_annotators_per_sample
        ]
        for col_idx in range(n_annotators_per_sample):
            is_not_selected[np.arange(len(z)), selected_indices[:, col_idx]] = False
        z[is_not_selected] = -1
    elif variant in {"rand-var", "worst-var"}:
        random_state = np.random.RandomState(0)
        mutable_missing = is_not_annotated.copy()
        for row_idx in range(len(mutable_missing)):
            annotated_indices = np.where(~mutable_missing[row_idx])[0]
            if len(annotated_indices) == 0:
                continue

            subset_size = random_state.randint(0, len(annotated_indices))
            if subset_size == 0:
                continue

            if variant == "worst-var":
                is_false = (
                    class_labels[row_idx][annotated_indices] == y_true[row_idx]
                )
                random_floats = random_state.rand(*is_false.shape)
                selected = np.argsort(-(is_false + random_floats), axis=-1)[
                    :subset_size
                ]
                indices_to_mask = annotated_indices[selected]
            else:
                indices_to_mask = random_state.choice(
                    annotated_indices, size=subset_size, replace=False
                )

            mutable_missing[row_idx, indices_to_mask] = True
        z[mutable_missing] = -1
    elif variant != "full":
        raise ValueError(
            f"Unsupported variant {variant!r}. Expected one of {SUPPORTED_VARIANTS}."
        )

    return z


def build_split_dataset(
    records: list[dict[str, object]],
    *,
    include_annotations: bool,
    annotations: np.ndarray | None = None,
) -> Dataset:
    features_dict = {
        "image": Image(),
        "label": ClassLabel(names=list(CLASSES)),
        "label_name": Value("string"),
        "observation_id": Value("int64"),
    }
    data_dict = {
        "image": [row["image"] for row in records],
        "label": [row["label"] for row in records],
        "label_name": [row["label_name"] for row in records],
        "observation_id": [row["observation_id"] for row in records],
    }

    if include_annotations:
        if annotations is None:
            raise ValueError("annotations must be provided for the train split.")
        features_dict["z"] = Sequence(
            feature=Value("int64"),
            length=len(ANNOTATORS),
        )
        data_dict["z"] = annotations.tolist()

    return Dataset.from_dict(data_dict, features=Features(features_dict))


def build_datasetdict(dataset_root: Path, *, variant: str) -> DatasetDict:
    records_by_split = collect_split_records(dataset_root)
    train_annotations = build_train_annotations(
        dataset_root,
        records_by_split["train"],
        variant=variant,
    )

    return DatasetDict(
        {
            "train": build_split_dataset(
                records_by_split["train"],
                include_annotations=True,
                annotations=train_annotations,
            ),
            "valid": build_split_dataset(
                records_by_split["valid"],
                include_annotations=False,
            ),
            "test": build_split_dataset(
                records_by_split["test"],
                include_annotations=False,
            ),
        }
    )


def save_datasetdict(
    dataset_dict: DatasetDict,
    *,
    output_dir: Path,
    variant: str,
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

    metadata = {
        "name": "dopanim",
        "variant": variant,
        "classes": list(CLASSES),
        "annotators": list(ANNOTATORS),
        "splits": list(dataset_dict.keys()),
    }
    metadata_path = output_dir / "dopanim_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir or default_raw_dir(args.data_root)
    output_dir = args.output_dir or default_output_dir(
        args.data_root, args.variant
    )

    dataset_root = prepare_raw_dataset(
        raw_dir,
        force_download=args.force_download,
    )
    dataset_dict = build_datasetdict(dataset_root, variant=args.variant)
    save_datasetdict(
        dataset_dict,
        output_dir=output_dir,
        variant=args.variant,
        force_rebuild=args.force_rebuild,
    )

    print(f"Saved DOPAnim ({args.variant}) to {output_dir}")
    print("Use it with a dataset config whose source_kind is 'from_disk'.")


if __name__ == "__main__":
    main()
