from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _resolve_db_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        return path / "mlflow.db"
    return path


def _load_metric_rows(
    *,
    db_path: Path,
    experiment_name: str,
    metric: str,
    weight_metric: str | None,
) -> dict[str, dict[int, dict[str, float]]]:
    keys = [metric] if weight_metric is None else [metric, weight_metric]
    placeholders = ", ".join("?" for _ in keys)
    query = f"""
        SELECT m.run_uuid, m.key, m.value, m.step, m.timestamp
        FROM metrics AS m
        JOIN runs AS r ON r.run_uuid = m.run_uuid
        JOIN experiments AS e ON e.experiment_id = r.experiment_id
        WHERE e.name = ?
          AND r.lifecycle_stage = 'active'
          AND m.key IN ({placeholders})
    """
    latest = {}
    with sqlite3.connect(db_path) as conn:
        for run_uuid, key, value, step, timestamp in conn.execute(
            query, [experiment_name, *keys]
        ):
            ident = (str(run_uuid), int(step), str(key))
            prev = latest.get(ident)
            if prev is None or int(timestamp) >= prev[0]:
                latest[ident] = (int(timestamp), float(value))

    by_run: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (run_uuid, step, key), (_, value) in latest.items():
        by_run[run_uuid].setdefault(step, {})[key] = value
    return by_run


def _summarize_run(
    steps: dict[int, dict[str, float]],
    *,
    metric: str,
    weight_metric: str | None,
    exclude_initial_cycle: bool,
) -> dict[str, float]:
    selected_steps = sorted(steps)
    if exclude_initial_cycle and selected_steps:
        selected_steps = selected_steps[1:]

    values = []
    weights = []
    for step in selected_steps:
        row = steps[step]
        value = row.get(metric)
        if value is None or not np.isfinite(value):
            continue
        if weight_metric is None:
            weight = 1.0
        else:
            weight = row.get(weight_metric)
            if weight is None or not np.isfinite(weight) or weight <= 0:
                continue
        values.append(float(value))
        weights.append(float(weight))

    if not values:
        return {
            "score": np.nan,
            "n_steps": 0.0,
            "total_weight": 0.0,
        }

    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    return {
        "score": float(np.sum(values_arr * weights_arr) / np.sum(weights_arr)),
        "n_steps": float(len(values)),
        "total_weight": float(np.sum(weights_arr)),
    }


def summarize_experiment(
    *,
    db_path: str | Path,
    experiment_name: str,
    metric: str = "delta_new_pair_acc",
    weight_metric: str | None = "delta_new_pairs",
    exclude_initial_cycle: bool = True,
) -> list[dict[str, str | float]]:
    db_path = _resolve_db_path(db_path)
    by_run = _load_metric_rows(
        db_path=db_path,
        experiment_name=experiment_name,
        metric=metric,
        weight_metric=weight_metric,
    )
    rows = []
    for run_uuid in sorted(by_run):
        summary = _summarize_run(
            by_run[run_uuid],
            metric=metric,
            weight_metric=weight_metric,
            exclude_initial_cycle=exclude_initial_cycle,
        )
        rows.append(
            {
                "run_uuid": run_uuid,
                "metric": metric,
                "weight_metric": "" if weight_metric is None else weight_metric,
                "exclude_initial_cycle": float(bool(exclude_initial_cycle)),
                **summary,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize logged per-cycle MLflow metrics as budget-weighted "
            "run-level scores."
        )
    )
    parser.add_argument("--db", required=True, help="MLflow SQLite DB or results dir.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--metric", default="delta_new_pair_acc")
    parser.add_argument("--weight-metric", default="delta_new_pairs")
    parser.add_argument(
        "--include-initial-cycle",
        action="store_true",
        help="Include cycle 0 instead of dropping the shared random initialization.",
    )
    args = parser.parse_args(argv)

    weight_metric = args.weight_metric if args.weight_metric else None
    rows = summarize_experiment(
        db_path=args.db,
        experiment_name=args.experiment_name,
        metric=args.metric,
        weight_metric=weight_metric,
        exclude_initial_cycle=not args.include_initial_cycle,
    )
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "run_uuid",
            "metric",
            "weight_metric",
            "exclude_initial_cycle",
            "score",
            "n_steps",
            "total_weight",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
