#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src._manifest import (
    DEFAULT_METHOD_DIR,
    DEFAULT_USE_CASE_DIR,
    build_rows,
    load_json,
    resolve_json,
)

LOGGED_OVERRIDE_PREFIXES = (
    "al",
    "assigner",
    "dataset",
    "classifier",
    "embedder",
    "classification_embedder",
    "simulation_embedder",
    "module",
    "pipeline",
    "sample",
    "scorer",
    "simulation",
    "training",
    "seed",
)


@dataclass(frozen=True)
class OverridePredicate:
    raw_override: str
    lhs: str
    candidate_keys: tuple[str, ...]
    expected_value: str | None
    is_delete: bool


@dataclass
class ObservedRun:
    run_id: str
    params: dict[str, str]


def resolve_db_path(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    return path if path.suffix.lower() == ".db" else path / "mlruns.db"


def normalize_value(value: object) -> str:
    text = str(value)
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text

    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        return "0"
    return normalized


def candidate_param_keys(lhs: str) -> tuple[str, ...]:
    candidates = []
    for key in (
        f"choice/{lhs.replace('@', '_')}",
        lhs.replace(".", "/"),
        lhs,
    ):
        if key not in candidates:
            candidates.append(key)
    return tuple(candidates)


def parse_override(override: str) -> OverridePredicate:
    if override.startswith("~"):
        lhs = override[1:].strip()
        if not lhs:
            raise ValueError(f"Invalid delete override: {override!r}")
        return OverridePredicate(
            raw_override=override,
            lhs=lhs,
            candidate_keys=candidate_param_keys(lhs),
            expected_value=None,
            is_delete=True,
        )

    if "=" not in override:
        raise ValueError(f"Unsupported Hydra override format: {override!r}")

    lhs, rhs = override.split("=", 1)
    lhs = lhs.strip()
    if not lhs:
        raise ValueError(f"Invalid override: {override!r}")
    return OverridePredicate(
        raw_override=override,
        lhs=lhs,
        candidate_keys=candidate_param_keys(lhs),
        expected_value=normalize_value(rhs),
        is_delete=False,
    )


def is_logged_override(override: str) -> bool:
    lhs = override[1:] if override.startswith("~") else override.split("=", 1)[0]
    lhs = lhs.strip()
    if not lhs:
        return False

    roots = {lhs.split(".", 1)[0]}
    if "@" in lhs:
        left, right = lhs.split("@", 1)
        roots.add(left.split(".", 1)[0])
        roots.add(right.split(".", 1)[0])

    return any(root in LOGGED_OVERRIDE_PREFIXES for root in roots)


def predicate_matches(
    params: dict[str, str], predicate: OverridePredicate
) -> bool:
    if predicate.is_delete:
        return all(key not in params for key in predicate.candidate_keys)

    for key in predicate.candidate_keys:
        value = params.get(key)
        if value is None:
            continue
        if normalize_value(value) == predicate.expected_value:
            return True
    return False


def row_matches_run_params(
    predicates: Sequence[OverridePredicate], params: dict[str, str]
) -> bool:
    return all(predicate_matches(params, predicate) for predicate in predicates)


def load_experiment_id(conn: sqlite3.Connection, experiment_name: str) -> int | None:
    row = conn.execute(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        (experiment_name,),
    ).fetchone()
    return None if row is None else int(row[0])


def _fetch_finished_run_ids(
    conn: sqlite3.Connection, experiment_id: int
) -> list[str]:
    rows = conn.execute(
        """
        SELECT run_uuid
        FROM runs
        WHERE experiment_id = ?
          AND status = 'FINISHED'
          AND lifecycle_stage = 'active'
        ORDER BY start_time DESC, run_uuid DESC
        """,
        (experiment_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _iter_param_rows(
    conn: sqlite3.Connection, run_ids: Sequence[str], *, chunk_size: int = 500
) -> Iterable[tuple[str, str, str]]:
    if not run_ids:
        return

    for start in range(0, len(run_ids), chunk_size):
        chunk = list(run_ids[start : start + chunk_size])
        placeholders = ",".join("?" for _ in chunk)
        query = f"""
            SELECT run_uuid, key, value
            FROM params
            WHERE run_uuid IN ({placeholders})
            ORDER BY run_uuid, key
        """
        yield from conn.execute(query, chunk)


def load_finished_runs(
    conn: sqlite3.Connection, experiment_id: int
) -> list[ObservedRun]:
    run_ids = _fetch_finished_run_ids(conn, experiment_id)
    params_by_run = {run_id: {} for run_id in run_ids}

    for run_id, key, value in _iter_param_rows(conn, run_ids):
        params_by_run[str(run_id)][str(key)] = str(value)

    return [ObservedRun(run_id=run_id, params=params_by_run[run_id]) for run_id in run_ids]


def build_missing_use_case(
    original_use_case: dict,
    missing_rows: Sequence[dict],
) -> dict:
    original_common = list(original_use_case.get("common_overrides", []))
    keep_common = all(
        row["hydra_overrides"][: len(original_common)] == original_common
        for row in missing_rows
    )
    common_overrides = original_common if keep_common else []

    values = {}
    for row in missing_rows:
        row_overrides = list(row["hydra_overrides"])
        if keep_common:
            row_overrides = row_overrides[len(common_overrides) :]
        values[row["run_id"]] = {
            "overrides": row_overrides,
            "tags": {**row.get("tags", {}), "original_run_id": row["run_id"]},
        }

    return {
        "name": f"{original_use_case['name']}_missing",
        "description": (
            f"Generated missing runs for {original_use_case['name']} from "
            "MLflow coverage checking."
        ),
        "common_overrides": common_overrides,
        "axes": [
            {
                "name": "missing_run",
                "type": "choices",
                "values": values,
            }
        ],
    }


def compile_row_predicates(hydra_overrides: Sequence[str]) -> list[OverridePredicate]:
    return [
        parse_override(item)
        for item in hydra_overrides
        if is_logged_override(item)
    ]


def find_missing_rows(expected_rows: Sequence[dict], observed_runs: Sequence[ObservedRun]) -> list[dict]:
    observed_params = [run.params for run in observed_runs]
    missing_rows = []
    for row in expected_rows:
        predicates = compile_row_predicates(row["hydra_overrides"])
        if not predicates:
            missing_rows.append(row)
            continue
        if not any(row_matches_run_params(predicates, params) for params in observed_params):
            missing_rows.append(row)
    return missing_rows


def write_missing_use_case(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether all runs expected by a use-case JSON are available "
            "as FINISHED runs in an MLflow SQLite database."
        )
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to an MLflow SQLite DB or a directory containing mlruns.db.",
    )
    parser.add_argument(
        "--use-case",
        required=True,
        help="Use-case name from configs/launch/use_cases or a direct JSON path.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output path for the generated missing-runs use-case JSON. "
            "Defaults to <use_case_stem>_missing.json next to the original file."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        db_path = resolve_db_path(args.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"MLflow SQLite DB not found: {db_path}")

        use_case_path = resolve_json(args.use_case, base_dir=DEFAULT_USE_CASE_DIR)
        use_case_cfg = load_json(use_case_path)
        expected_rows = list(build_rows(use_case_cfg, method_dir=DEFAULT_METHOD_DIR))

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else use_case_path.with_name(f"{use_case_path.stem}_missing.json")
        )

        conn = sqlite3.connect(str(db_path))
        try:
            experiment_id = load_experiment_id(conn, use_case_cfg["name"])
            observed_runs = (
                load_finished_runs(conn, experiment_id)
                if experiment_id is not None
                else []
            )
        finally:
            conn.close()

        missing_rows = find_missing_rows(expected_rows, observed_runs)
        matched_count = len(expected_rows) - len(missing_rows)

        print(f"Resolved DB path   : {db_path}")
        print(f"Resolved use case  : {use_case_path}")
        print(f"Experiment name    : {use_case_cfg['name']}")
        print(
            "Experiment ID      : "
            f"{experiment_id if experiment_id is not None else 'not found'}"
        )
        print(f"Expected runs      : {len(expected_rows)}")
        print(f"Matched runs       : {matched_count}")
        print(f"Missing runs       : {len(missing_rows)}")

        if missing_rows:
            payload = build_missing_use_case(use_case_cfg, missing_rows)
            write_missing_use_case(output_path, payload)
            print(f"Missing use case   : {output_path}")
            print("Missing preview    :")
            for row in missing_rows[:5]:
                print(f"  - {row['run_id']}")
            return 1

        print("Missing use case   : none written")
        return 0

    except (FileNotFoundError, KeyError, sqlite3.Error, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
