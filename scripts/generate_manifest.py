#!/usr/bin/env python3
import sys
import argparse
import itertools
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src._manifest import (
    DEFAULT_MANIFEST_DIR,
    DEFAULT_METHOD_DIR,
    DEFAULT_USE_CASE_DIR,
    build_rows as _build_rows,
    load_json as _load_json,
    resolve_json as _resolve_json,
)


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
