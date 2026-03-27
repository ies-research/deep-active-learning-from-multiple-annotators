#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_SCRIPT = REPO_ROOT / "scripts" / "experiment.py"


def _load_manifest_row(manifest_path, row_index):
    with Path(manifest_path).open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx == row_index:
                return json.loads(line)
    raise IndexError(f"Manifest row {row_index} not found in {manifest_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Run one experiment row from a JSONL manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="0-based row index. Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run scripts/experiment.py.",
    )
    parser.add_argument(
        "--experiment-script",
        default=str(DEFAULT_EXPERIMENT_SCRIPT),
        help="Path to the experiment driver.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Additional Hydra override appended after the manifest row. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without executing it.",
    )
    args = parser.parse_args()

    row_index = args.row
    if row_index is None:
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is None:
            raise ValueError("Provide --row or set SLURM_ARRAY_TASK_ID.")
        row_index = int(task_id)

    row = _load_manifest_row(args.manifest, row_index)
    cmd = [
        args.python,
        args.experiment_script,
        *row["hydra_overrides"],
        *args.override,
    ]

    print(f"Running manifest row {row_index}: {row['run_id']}")
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
