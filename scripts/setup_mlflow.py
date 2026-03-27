#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils._mlflow import (
    _resolve_under_original_cwd,
    configure_mlflow_sqlite,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create or select an MLflow SQLite backend and pre-create the "
            "target experiment."
        )
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name to create or select.",
    )
    parser.add_argument(
        "--results-path",
        default="mlflow",
        help=(
            "Directory or .db file path for the MLflow SQLite backend. "
            "Defaults to 'mlflow', matching configs/paths/default.yaml."
        ),
    )
    args = parser.parse_args()

    results_path = _resolve_under_original_cwd(args.results_path)
    configure_mlflow_sqlite(
        db_path=results_path,
        experiment_name=args.experiment_name,
        artifact_root=results_path,
    )

    db_file = (
        results_path
        if results_path.suffix.lower() == ".db"
        else results_path / "mlruns.db"
    )
    print(f"Configured MLflow experiment: {args.experiment_name}")
    print(f"SQLite backend: {db_file}")
    print(f"Artifact root: {results_path}")


if __name__ == "__main__":
    main()
