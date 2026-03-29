from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USE_CASE_DIR = REPO_ROOT / "configs" / "launch" / "use_cases"
DEFAULT_METHOD_DIR = REPO_ROOT / "configs" / "launch" / "methods"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "manifests"


def resolve_json(identifier: str | Path, *, base_dir: Path) -> Path:
    path = Path(identifier)
    if path.exists():
        return path.resolve()
    if path.suffix:
        candidate = base_dir / path.name
    else:
        candidate = base_dir / f"{identifier}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not resolve JSON file for {identifier!r}.")
    return candidate.resolve()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def slugify(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def _template_options(axis: Mapping[str, Any]) -> list[Dict[str, Any]]:
    options = []
    for raw_value in axis["values"]:
        overrides = [
            item.format(value=raw_value) for item in axis.get("overrides", [])
        ]
        options.append(
            {
                "label": str(raw_value),
                "overrides": overrides,
                "tags": {axis["name"]: str(raw_value)},
                "when": dict(axis.get("when", {})),
            }
        )
    return options


def _choice_options(axis: Mapping[str, Any]) -> list[Dict[str, Any]]:
    options = []
    for label, payload in axis["values"].items():
        options.append(
            {
                "label": label,
                "overrides": list(payload.get("overrides", [])),
                "tags": {axis["name"]: label, **payload.get("tags", {})},
                "when": dict(payload.get("when", {})),
            }
        )
    return options


def _registry_options(
    axis: Mapping[str, Any], *, method_dir: Path
) -> list[Dict[str, Any]]:
    options = []
    for method_name in axis["values"]:
        method_path = resolve_json(method_name, base_dir=method_dir)
        method_cfg = load_json(method_path)
        options.append(
            {
                "label": str(method_cfg["name"]),
                "overrides": list(method_cfg.get("overrides", [])),
                "tags": {
                    axis["name"]: str(method_cfg["name"]),
                    **method_cfg.get("tags", {}),
                },
                "when": dict(method_cfg.get("when", {})),
            }
        )
    return options


def axis_options(
    axis: Mapping[str, Any], *, method_dir: Path
) -> list[Dict[str, Any]]:
    axis_type = axis["type"]
    if axis_type == "template":
        return _template_options(axis)
    if axis_type == "choices":
        return _choice_options(axis)
    if axis_type == "registry":
        return _registry_options(axis, method_dir=method_dir)
    raise ValueError(f"Unknown axis type: {axis_type!r}.")


def matches_condition(
    values: Mapping[str, str], condition: Mapping[str, Any]
) -> bool:
    for key, expected in condition.items():
        actual = values.get(key)
        if isinstance(expected, list):
            if actual not in {str(item) for item in expected}:
                return False
        else:
            if actual != str(expected):
                return False
    return True


def combo_is_valid(
    axis_names: Iterable[str],
    combo: Iterable[Mapping[str, Any]],
    use_case_cfg: Mapping[str, Any],
) -> bool:
    axis_values = {
        axis_name: option["label"]
        for axis_name, option in zip(axis_names, combo)
    }

    for option in combo:
        when = option.get("when", {})
        if when and not matches_condition(axis_values, when):
            return False

    for condition in use_case_cfg.get("exclude", []):
        if matches_condition(axis_values, condition):
            return False

    return True


def build_rows(
    use_case_cfg: Mapping[str, Any], *, method_dir: Path
) -> Iterator[Dict[str, Any]]:
    axes = use_case_cfg.get("axes", [])
    axis_names = [axis["name"] for axis in axes]
    axis_option_lists = [
        axis_options(axis, method_dir=method_dir) for axis in axes
    ]

    for combo in itertools.product(*axis_option_lists):
        if not combo_is_valid(axis_names, combo, use_case_cfg):
            continue

        axis_values: Dict[str, str] = {}
        tags: Dict[str, str] = {}
        overrides = list(use_case_cfg.get("common_overrides", []))

        for axis_name, option in zip(axis_names, combo):
            axis_values[axis_name] = option["label"]
            tags.update(option["tags"])
            overrides.extend(option["overrides"])

        run_id_parts = [slugify(use_case_cfg["name"])]
        for axis_name in axis_names:
            run_id_parts.append(
                f"{slugify(axis_name)}_{slugify(axis_values[axis_name])}"
            )

        yield {
            "run_id": "__".join(run_id_parts),
            "use_case": use_case_cfg["name"],
            "axis_values": axis_values,
            "tags": tags,
            "hydra_overrides": overrides,
        }
