#!/usr/bin/env python3
import argparse
import itertools
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USE_CASE_DIR = REPO_ROOT / "configs" / "launch" / "use_cases"
DEFAULT_METHOD_DIR = REPO_ROOT / "configs" / "launch" / "methods"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "manifests"


def _resolve_json(identifier, *, base_dir):
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


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _slugify(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "value"


def _template_options(axis):
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


def _choice_options(axis):
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


def _registry_options(axis, *, method_dir):
    options = []
    for method_name in axis["values"]:
        method_path = _resolve_json(method_name, base_dir=method_dir)
        method_cfg = _load_json(method_path)
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


def _axis_options(axis, *, method_dir):
    axis_type = axis["type"]
    if axis_type == "template":
        return _template_options(axis)
    if axis_type == "choices":
        return _choice_options(axis)
    if axis_type == "registry":
        return _registry_options(axis, method_dir=method_dir)
    raise ValueError(f"Unknown axis type: {axis_type!r}.")


def _matches_condition(values, condition):
    for key, expected in condition.items():
        actual = values.get(key)
        if isinstance(expected, list):
            if actual not in {str(item) for item in expected}:
                return False
        else:
            if actual != str(expected):
                return False
    return True


def _combo_is_valid(axis_names, combo, use_case_cfg):
    axis_values = {
        axis_name: option["label"]
        for axis_name, option in zip(axis_names, combo)
    }

    for option in combo:
        when = option.get("when", {})
        if when and not _matches_condition(axis_values, when):
            return False

    for condition in use_case_cfg.get("exclude", []):
        if _matches_condition(axis_values, condition):
            return False

    return True


def _build_rows(use_case_cfg, *, method_dir):
    axes = use_case_cfg.get("axes", [])
    axis_names = [axis["name"] for axis in axes]
    axis_options = [_axis_options(axis, method_dir=method_dir) for axis in axes]

    for combo in itertools.product(*axis_options):
        if not _combo_is_valid(axis_names, combo, use_case_cfg):
            continue

        axis_values = {}
        tags = {}
        overrides = list(use_case_cfg.get("common_overrides", []))

        for axis_name, option in zip(axis_names, combo):
            axis_values[axis_name] = option["label"]
            tags.update(option["tags"])
            overrides.extend(option["overrides"])

        run_id_parts = [_slugify(use_case_cfg["name"])]
        for axis_name in axis_names:
            run_id_parts.append(
                f"{_slugify(axis_name)}_{_slugify(axis_values[axis_name])}"
            )

        yield {
            "run_id": "__".join(run_id_parts),
            "use_case": use_case_cfg["name"],
            "axis_values": axis_values,
            "tags": tags,
            "hydra_overrides": overrides,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a JSONL manifest from a launch use-case config."
    )
    parser.add_argument(
        "use_case",
        help="Use-case name from configs/launch/use_cases or a direct JSON path.",
    )
    parser.add_argument(
        "--output",
        help="Optional manifest output path. Defaults to manifests/<use_case>.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to write.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="How many rows to print after generation.",
    )
    args = parser.parse_args()

    use_case_path = _resolve_json(args.use_case, base_dir=DEFAULT_USE_CASE_DIR)
    use_case_cfg = _load_json(use_case_path)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else (DEFAULT_MANIFEST_DIR / f"{use_case_cfg['name']}.jsonl").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(use_case_cfg, method_dir=DEFAULT_METHOD_DIR)
    if args.limit is not None:
        rows = itertools.islice(rows, args.limit)

    count = 0
    previews = []
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            if len(previews) < max(args.preview, 0):
                previews.append(row)
            count += 1

    print(f"Wrote {count} rows to {output_path}")
    for row in previews:
        print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
