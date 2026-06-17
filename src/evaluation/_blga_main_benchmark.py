from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src._manifest import DEFAULT_METHOD_DIR, build_rows, load_json


KEY_COLUMNS = ["seed", "dataset", "classifier", "constraint", "method"]
_RUN_AXIS_PARAM_KEYS = {
    "seed": "seed",
    "dataset": "choice/dataset",
    "classifier": "choice/classifier",
    "cap_multiplier": "assigner/actual/max_per_annotator_multiplier",
    "scorer": "choice/scorer_scorer.actual",
}


def load_use_case(path: str | Path) -> dict:
    return load_json(Path(path))


def resolve_mlflow_db_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    return path if path.suffix.lower() == ".db" else path / "mlruns.db"


def _axis_by_name(use_case: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(axis["name"]): axis for axis in use_case.get("axes", [])}


def _first_override_value(overrides: Sequence[str], prefix: str) -> str | None:
    needle = f"{prefix}="
    for item in overrides:
        if item.startswith(needle):
            return item.split("=", 1)[1]
    return None


def _normalize_cap(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return f"{float(value):.1f}"
    except Exception:
        return str(value)


def _build_constraint_maps(use_case: Mapping[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    axes = _axis_by_name(use_case)
    constraint_axis = axes.get("constraint", {})
    values = constraint_axis.get("values", {})
    cap_to_constraint: dict[str, str] = {}
    constraint_to_cap: dict[str, str] = {}
    if not isinstance(values, Mapping):
        return cap_to_constraint, constraint_to_cap

    for label, payload in values.items():
        assert isinstance(payload, Mapping)
        tags = payload.get("tags", {})
        cap = None
        if isinstance(tags, Mapping):
            cap = tags.get("cap_multiplier")
        if cap is None:
            cap = _first_override_value(
                list(payload.get("overrides", [])),
                "assigner.actual.max_per_annotator_multiplier",
            )
        cap_norm = _normalize_cap(cap)
        if cap_norm is None:
            continue
        cap_to_constraint[cap_norm] = str(label)
        constraint_to_cap[str(label)] = cap_norm
    return cap_to_constraint, constraint_to_cap


def _build_method_maps(use_case: Mapping[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    axes = _axis_by_name(use_case)
    method_axis = axes.get("method", {})
    values = method_axis.get("values", {})
    scorer_to_method: dict[str, str] = {}
    method_to_scorer: dict[str, str] = {}
    if not isinstance(values, Mapping):
        return scorer_to_method, method_to_scorer

    for label, payload in values.items():
        assert isinstance(payload, Mapping)
        scorer = _first_override_value(
            list(payload.get("overrides", [])), "scorer@scorer.actual"
        )
        if scorer is None:
            continue
        scorer_to_method[scorer] = str(label)
        method_to_scorer[str(label)] = scorer
    return scorer_to_method, method_to_scorer


def build_expected_grid(use_case: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for row in build_rows(use_case, method_dir=DEFAULT_METHOD_DIR):
        axis_values = row["axis_values"]
        tags = row.get("tags", {})
        overrides = row.get("hydra_overrides", [])
        cap_multiplier = tags.get("cap_multiplier")
        if cap_multiplier is None:
            cap_multiplier = _first_override_value(
                overrides, "assigner.actual.max_per_annotator_multiplier"
            )
        scorer = _first_override_value(overrides, "scorer@scorer.actual")
        rows.append(
            {
                "run_id_expected": row["run_id"],
                "seed": str(axis_values.get("seed")),
                "seed_int": int(axis_values["seed"]),
                "dataset": str(axis_values.get("dataset")),
                "classifier": str(axis_values.get("classifier")),
                "constraint": str(axis_values.get("constraint")),
                "cap_multiplier": _normalize_cap(cap_multiplier),
                "constraint_pressure": tags.get("constraint_pressure"),
                "method": str(axis_values.get("method")),
                "scorer": scorer,
                "method_family": tags.get("method_family"),
                "annotator_source": tags.get("annotator_source"),
                "hydra_overrides": list(overrides),
                "tags": dict(tags),
                "expected": True,
            }
        )
    return pd.DataFrame(rows).sort_values(KEY_COLUMNS).reset_index(drop=True)


def _read_sqlite_table(db_path: Path, query: str, params: Sequence[object] = ()) -> pd.DataFrame:
    db_path = resolve_mlflow_db_path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"MLflow SQLite DB not found: {db_path}")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(query, conn, params=params)


def load_mlflow_runs(
    db_path: str | Path,
    experiment_name: str,
    use_case: Mapping[str, object],
) -> pd.DataFrame:
    scorer_to_method, _ = _build_method_maps(use_case)
    cap_to_constraint, _ = _build_constraint_maps(use_case)
    query = """
        SELECT
            r.run_uuid AS run_id,
            r.status,
            r.lifecycle_stage,
            r.start_time,
            r.end_time,
            p_seed.value AS seed,
            p_dataset.value AS dataset,
            p_classifier.value AS classifier,
            p_cap.value AS cap_multiplier,
            p_scorer.value AS scorer
        FROM runs AS r
        JOIN experiments AS e
          ON e.experiment_id = r.experiment_id
         AND e.name = ?
        LEFT JOIN params AS p_seed
          ON p_seed.run_uuid = r.run_uuid
         AND p_seed.key = 'seed'
        LEFT JOIN params AS p_dataset
          ON p_dataset.run_uuid = r.run_uuid
         AND p_dataset.key = 'choice/dataset'
        LEFT JOIN params AS p_classifier
          ON p_classifier.run_uuid = r.run_uuid
         AND p_classifier.key = 'choice/classifier'
        LEFT JOIN params AS p_cap
          ON p_cap.run_uuid = r.run_uuid
         AND p_cap.key = 'assigner/actual/max_per_annotator_multiplier'
        LEFT JOIN params AS p_scorer
          ON p_scorer.run_uuid = r.run_uuid
         AND p_scorer.key = 'choice/scorer_scorer.actual'
        WHERE r.lifecycle_stage = 'active'
    """
    df = _read_sqlite_table(resolve_mlflow_db_path(db_path), query, [experiment_name])
    if df.empty:
        return df

    df["seed"] = df["seed"].astype("string")
    df["seed_int"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["cap_multiplier"] = df["cap_multiplier"].map(_normalize_cap)
    df["constraint"] = df["cap_multiplier"].map(cap_to_constraint)
    df["method"] = df["scorer"].map(scorer_to_method)
    df["start_time_utc"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
    df["end_time_utc"] = pd.to_datetime(df["end_time"], unit="ms", utc=True)
    return df


def status_overview(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame(columns=["status", "n_runs"])
    return (
        runs_df.groupby("status", dropna=False)["run_id"]
        .nunique()
        .rename("n_runs")
        .reset_index()
        .sort_values("status")
        .reset_index(drop=True)
    )


def _valid_axis_rows(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return runs_df.copy()
    cols = [*KEY_COLUMNS, "run_id"]
    return runs_df.dropna(subset=cols).copy()


def select_latest_finished_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    valid = _valid_axis_rows(runs_df)
    if valid.empty:
        return valid
    finished = valid[valid["status"] == "FINISHED"].copy()
    if finished.empty:
        return finished
    finished = finished.sort_values(
        ["start_time_utc", "end_time_utc", "run_id"],
        ascending=[False, False, False],
        na_position="last",
    )
    return finished.drop_duplicates(subset=KEY_COLUMNS, keep="first").reset_index(drop=True)


def compute_coverage(
    expected_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_finished = select_latest_finished_runs(runs_df)
    observed = latest_finished[[*KEY_COLUMNS, "run_id"]].rename(
        columns={"run_id": "run_id_observed"}
    )
    coverage = expected_df.merge(observed, on=KEY_COLUMNS, how="left")
    coverage["is_finished"] = coverage["run_id_observed"].notna()

    status_counts = (
        runs_df.groupby("status", dropna=False)["run_id"].nunique().to_dict()
        if not runs_df.empty
        else {}
    )
    overview = pd.DataFrame(
        [
            {
                "expected_runs": int(len(expected_df)),
                "finished_matched_runs": int(coverage["is_finished"].sum()),
                "missing_runs": int((~coverage["is_finished"]).sum()),
                "mlflow_runs_total": int(runs_df["run_id"].nunique()) if not runs_df.empty else 0,
                "mlflow_finished_total": int(status_counts.get("FINISHED", 0)),
                "mlflow_failed_total": int(status_counts.get("FAILED", 0)),
                "mlflow_running_total": int(status_counts.get("RUNNING", 0)),
                "coverage_pct": float(100.0 * coverage["is_finished"].mean())
                if len(coverage)
                else 0.0,
            }
        ]
    )
    return coverage, overview


def coverage_by_axis(coverage_df: pd.DataFrame, axis: str | Sequence[str]) -> pd.DataFrame:
    group_cols = [axis] if isinstance(axis, str) else list(axis)
    if coverage_df.empty:
        return pd.DataFrame(columns=[*group_cols, "n_finished", "n_expected", "n_missing", "coverage_pct"])
    out = (
        coverage_df.groupby(group_cols, dropna=False)["is_finished"]
        .agg(n_finished="sum", n_expected="count")
        .reset_index()
    )
    out["n_finished"] = out["n_finished"].astype(int)
    out["n_missing"] = out["n_expected"] - out["n_finished"]
    out["coverage_pct"] = 100.0 * out["n_finished"] / out["n_expected"]
    return out.sort_values([*group_cols]).reset_index(drop=True)


def missing_rows(coverage_df: pd.DataFrame) -> pd.DataFrame:
    return coverage_df[~coverage_df["is_finished"]].copy().reset_index(drop=True)


def build_missing_use_case(
    use_case: Mapping[str, object],
    missing_df: pd.DataFrame,
) -> dict:
    values = {}
    for _, row in missing_df.iterrows():
        run_id = str(row["run_id_expected"])
        overrides = list(row["hydra_overrides"])
        tags = dict(row["tags"])
        tags["original_run_id"] = run_id
        values[run_id] = {
            "overrides": overrides,
            "tags": tags,
        }
    return {
        "name": f"{use_case['name']}_missing",
        "description": f"Generated missing runs for {use_case['name']}.",
        "common_overrides": list(use_case.get("common_overrides", [])),
        "axes": [{"name": "missing_run", "type": "choices", "values": values}],
    }


def _chunks(values: Sequence[str], size: int = 900) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def load_metric_history(
    db_path: str | Path,
    runs_df: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if runs_df.empty or not metrics:
        return pd.DataFrame(
            columns=[
                "run_id",
                "metric",
                "step",
                "value",
                "timestamp",
                *[col for col in runs_df.columns if col not in {"run_id"}],
            ]
        )

    run_ids = runs_df["run_id"].dropna().astype(str).unique().tolist()
    metric_names = [str(metric) for metric in metrics]
    db_path = resolve_mlflow_db_path(db_path)
    frames = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for run_chunk in _chunks(run_ids):
            run_placeholders = ", ".join("?" for _ in run_chunk)
            metric_placeholders = ", ".join("?" for _ in metric_names)
            query = f"""
                SELECT
                    run_uuid AS run_id,
                    key AS metric,
                    step,
                    value,
                    timestamp
                FROM metrics
                WHERE run_uuid IN ({run_placeholders})
                  AND key IN ({metric_placeholders})
            """
            frames.append(
                pd.read_sql_query(query, conn, params=[*run_chunk, *metric_names])
            )

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    if hist.empty:
        return hist

    hist = hist.sort_values(["run_id", "metric", "step", "timestamp"])
    hist = hist.drop_duplicates(
        subset=["run_id", "metric", "step"], keep="last"
    ).reset_index(drop=True)
    axis_cols = [
        "run_id",
        "seed",
        "seed_int",
        "dataset",
        "classifier",
        "constraint",
        "cap_multiplier",
        "method",
        "scorer",
        "status",
    ]
    axis_cols = [col for col in axis_cols if col in runs_df.columns]
    hist = hist.merge(runs_df[axis_cols], on="run_id", how="left")
    return hist.sort_values(["metric", "dataset", "method", "seed_int", "step"]).reset_index(drop=True)


def normalized_aulc(curve_df: pd.DataFrame) -> float:
    curve = curve_df[["step", "value"]].replace([np.inf, -np.inf], np.nan).dropna()
    if curve.empty:
        return float("nan")
    curve = curve.sort_values("step")
    if curve.shape[0] == 1:
        return float(curve["value"].iloc[-1])
    steps = curve["step"].to_numpy(dtype=float)
    values = curve["value"].to_numpy(dtype=float)
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return float(values[-1])
    return float(np.trapezoid(values, steps) / span)


def _standard_error(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def compute_run_metric_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [
        "run_id",
        "metric",
        "seed",
        "seed_int",
        "dataset",
        "classifier",
        "constraint",
        "cap_multiplier",
        "method",
        "scorer",
    ]
    for keys, group in history_df.groupby(group_cols, dropna=False):
        group = group.sort_values("step")
        row = dict(zip(group_cols, keys))
        row["aulc"] = normalized_aulc(group)
        row["final_step"] = int(group["step"].iloc[-1])
        row["final_value"] = float(group["value"].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_by_group(
    run_metric_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    *,
    metric: str,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    group_cols = list(group_cols)
    expected = coverage_by_axis(coverage_df, group_cols)
    if run_metric_df.empty:
        out = expected.copy()
        out["metric"] = metric
        out["n_metric_runs"] = 0
        out["n_seeds"] = 0
        out["aulc_mean"] = np.nan
        out["aulc_se"] = np.nan
        out["final_mean"] = np.nan
        out["final_se"] = np.nan
        return out

    subset = run_metric_df[run_metric_df["metric"] == metric].copy()
    if subset.empty:
        out = expected.copy()
        out["metric"] = metric
        out["n_metric_runs"] = 0
        out["n_seeds"] = 0
        out["aulc_mean"] = np.nan
        out["aulc_se"] = np.nan
        out["final_mean"] = np.nan
        out["final_se"] = np.nan
        return out

    rows = []
    for keys, group in subset.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["metric"] = metric
        row["n_metric_runs"] = int(group["run_id"].nunique())
        row["n_seeds"] = int(group["seed"].nunique())
        row["aulc_mean"] = float(group["aulc"].mean())
        row["aulc_se"] = _standard_error(group["aulc"])
        row["final_mean"] = float(group["final_value"].mean())
        row["final_se"] = _standard_error(group["final_value"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    out = expected.merge(summary, on=group_cols, how="left")
    out["metric"] = out["metric"].fillna(metric)
    out["n_metric_runs"] = out["n_metric_runs"].fillna(0).astype(int)
    out["n_seeds"] = out["n_seeds"].fillna(0).astype(int)
    return out.sort_values(group_cols).reset_index(drop=True)


def paired_delta_vs_baseline(
    run_metric_df: pd.DataFrame,
    *,
    metric: str,
    baseline_method: str = "random",
    group_cols: Sequence[str] = ("dataset", "classifier", "constraint", "method"),
) -> pd.DataFrame:
    if run_metric_df.empty:
        return pd.DataFrame()

    subset = run_metric_df[run_metric_df["metric"] == metric].copy()
    if subset.empty:
        return pd.DataFrame()

    baseline_keys = ["dataset", "classifier", "constraint", "seed"]
    baseline = subset[subset["method"] == baseline_method][
        [*baseline_keys, "aulc", "final_value"]
    ].rename(columns={"aulc": "baseline_aulc", "final_value": "baseline_final"})
    merged = subset.merge(baseline, on=baseline_keys, how="inner", validate="many_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["delta_aulc"] = merged["aulc"] - merged["baseline_aulc"]
    merged["delta_final"] = merged["final_value"] - merged["baseline_final"]

    rows = []
    group_cols = list(group_cols)
    for keys, group in merged.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["metric"] = metric
        row["baseline_method"] = baseline_method
        row["n_pairs"] = int(group.shape[0])
        row["delta_aulc_mean"] = float(group["delta_aulc"].mean())
        row["delta_aulc_se"] = _standard_error(group["delta_aulc"])
        row["delta_final_mean"] = float(group["delta_final"].mean())
        row["delta_final_se"] = _standard_error(group["delta_final"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def filter_frame(
    df: pd.DataFrame,
    *,
    datasets: Sequence[str] | None = None,
    classifiers: Sequence[str] | None = None,
    constraints: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    filters = {
        "dataset": datasets,
        "classifier": classifiers,
        "constraint": constraints,
        "method": methods,
    }
    for column, values in filters.items():
        if values and column in out.columns:
            out = out[out[column].isin(list(values))]
    return out.reset_index(drop=True)


def _common_seeds_for_methods(df: pd.DataFrame, methods: Sequence[str]) -> set[str]:
    seed_sets = []
    for method in methods:
        method_df = df[df["method"] == method]
        seeds = set(method_df["seed"].dropna().astype(str))
        seed_sets.append(seeds)
    if not seed_sets:
        return set()
    return set.intersection(*seed_sets)


def learning_curve_seed_coverage(
    history_df: pd.DataFrame,
    *,
    metric: str,
    datasets: Sequence[str],
    methods: Sequence[str],
    classifier: str | None = None,
    constraint: str | None = None,
    require_common_seeds: bool = False,
) -> pd.DataFrame:
    subset = history_df[history_df["metric"] == metric].copy()
    subset = filter_frame(
        subset,
        datasets=datasets,
        classifiers=[classifier] if classifier else None,
        constraints=[constraint] if constraint else None,
        methods=methods,
    )

    rows = []
    for dataset in datasets:
        dataset_df = subset[subset["dataset"] == dataset]
        common_seeds = _common_seeds_for_methods(dataset_df, methods)
        for method in methods:
            method_df = dataset_df[dataset_df["method"] == method]
            if require_common_seeds:
                method_df = method_df[
                    method_df["seed"].dropna().astype(str).isin(common_seeds)
                ]
            steps = method_df["step"].dropna()
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "constraint": constraint,
                    "method": method,
                    "metric": metric,
                    "n_seeds": int(method_df["seed"].nunique()),
                    "n_common_seeds": int(len(common_seeds)),
                    "n_steps": int(steps.nunique()),
                    "first_step": int(steps.min()) if not steps.empty else np.nan,
                    "last_step": int(steps.max()) if not steps.empty else np.nan,
                    "common_seed_filter": bool(require_common_seeds),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_heatmap(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    value_col: str = "aulc_mean",
    dataset_order: Sequence[str] | None = None,
    method_order: Sequence[str] | None = None,
    classifier: str | None = None,
    constraint: str | None = None,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    subset = summary_df[summary_df["metric"] == metric].copy()
    if classifier is not None and "classifier" in subset.columns:
        subset = subset[subset["classifier"] == classifier]
    if constraint is not None and "constraint" in subset.columns:
        subset = subset[subset["constraint"] == constraint]
    if subset.empty:
        return None

    dataset_order = list(dataset_order or sorted(subset["dataset"].dropna().unique()))
    method_order = list(method_order or sorted(subset["method"].dropna().unique()))
    matrix = np.full((len(dataset_order), len(method_order)), np.nan)
    for i, dataset in enumerate(dataset_order):
        for j, method in enumerate(method_order):
            match = subset[(subset["dataset"] == dataset) & (subset["method"] == method)]
            if not match.empty:
                matrix[i, j] = float(match[value_col].iloc[0])

    fig, ax = plt.subplots(
        figsize=(1.0 * max(len(method_order), 4) + 2.5, 0.45 * len(dataset_order) + 2.2)
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(method_order)), method_order, rotation=35, ha="right")
    ax.set_yticks(range(len(dataset_order)), dataset_order)
    ax.set_xlabel("method")
    ax.set_ylabel("dataset")
    ax.set_title(title or f"{metric} {value_col}")
    for i in range(len(dataset_order)):
        for j in range(len(method_order)):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    return fig


def plot_learning_curves(
    history_df: pd.DataFrame,
    *,
    metric: str,
    datasets: Sequence[str],
    methods: Sequence[str],
    classifier: str | None = None,
    constraint: str | None = None,
    require_common_seeds: bool = False,
):
    import matplotlib.pyplot as plt

    subset = history_df[history_df["metric"] == metric].copy()
    subset = filter_frame(
        subset,
        datasets=datasets,
        classifiers=[classifier] if classifier else None,
        constraints=[constraint] if constraint else None,
        methods=methods,
    )
    if subset.empty:
        return None

    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(7.0, 2.8 * len(datasets)),
        squeeze=False,
        sharex=False,
    )
    for ax, dataset in zip(axes[:, 0], datasets):
        dataset_df = subset[subset["dataset"] == dataset]
        if require_common_seeds:
            common_seeds = _common_seeds_for_methods(dataset_df, methods)
            dataset_df = dataset_df[
                dataset_df["seed"].dropna().astype(str).isin(common_seeds)
            ]
        else:
            common_seeds = None
        for method, group in dataset_df.groupby("method", dropna=False):
            curve = (
                group.groupby("step", dropna=False)["value"]
                .mean()
                .reset_index()
                .sort_values("step")
            )
            ax.plot(curve["step"], curve["value"], marker="o", markersize=2.5, label=str(method))
        title = str(dataset)
        if common_seeds is not None:
            title = f"{title} (common seeds={len(common_seeds)})"
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("budget step")
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    return fig
