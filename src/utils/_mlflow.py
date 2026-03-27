from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlflow
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# MLflow constraints (documented for params / tags / metrics).
# Keys: allowed chars + typical max length 250.
# Values: typical max length 6000 for params (varies by backend store).
# ---------------------------------------------------------------------------

_ALLOWED_KEY_CHARS = re.compile(r"[^0-9A-Za-z_\-\. /]")
_MAX_KEY_LEN = 250
_MAX_PARAM_VAL_LEN = 6000


def _sanitize_key(key: str, *, max_len: int = _MAX_KEY_LEN) -> str:
    """Sanitize a key to satisfy MLflow's allowed character set and max length."""
    key = _ALLOWED_KEY_CHARS.sub("_", str(key))
    return key[:max_len] if len(key) > max_len else key


def _stringify_param_value(
    value: Any, *, max_len: int = _MAX_PARAM_VAL_LEN
) -> str:
    """Convert a value to a stable, reasonably short string for MLflow params."""
    if value is None:
        s = "null"
    elif isinstance(value, (str, int, float, bool)):
        s = str(value)
    else:
        try:
            s = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except Exception:
            s = str(value)
    return s[:max_len] if len(s) > max_len else s


def _is_finite_number(x: Any) -> bool:
    """Return True if x can be converted to a finite float."""
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _sanitize_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Return MLflow-safe metrics: sanitized keys, float values, drop NaN/inf."""
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if _is_finite_number(v):
            out[_sanitize_key(k)] = float(v)
    return out


# ---------------------------------------------------------------------------
# Hydra/OmegaConf -> MLflow params flattening
# ---------------------------------------------------------------------------


def _flatten_for_params(
    obj: Any,
    prefix: str = "",
    *,
    sep: str = "/",
    out: Optional[MutableMapping[str, str]] = None,
    max_depth: int = 12,
    list_mode: str = "json",  # "json" or "indices"
    max_list_elems: int = 50,
) -> Dict[str, str]:
    """Flatten nested dict/list into {key: string_value} for MLflow params.

    Parameters
    ----------
    obj : Any
        Object to flatten (typically dict/list from OmegaConf.to_container()).
    prefix : str, default=""
        Prefix used to build hierarchical keys.
    sep : str, default="/"
        Separator used between key components.
    out : dict, optional
        Output dict to write into. If None, a new dict is created.
    max_depth : int, default=12
        Maximum recursion depth before stopping and stringifying the remaining object.
    list_mode : {"json", "indices"}, default="json"
        If "json", lists are logged as a single JSON string (recommended to avoid
        exploding the number of params). If "indices", list elements become separate
        keys.
    max_list_elems : int, default=50
        Maximum number of list elements to log (either in JSON or indices mode).

    Returns
    -------
    params : dict
        Flat dict of MLflow-safe parameter key/value pairs (values are strings).
    """
    if out is None:
        out = {}

    if max_depth < 0:
        out[_sanitize_key(prefix)] = _stringify_param_value(obj)
        return dict(out)

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            _flatten_for_params(
                v,
                prefix=key,
                sep=sep,
                out=out,
                max_depth=max_depth - 1,
                list_mode=list_mode,
                max_list_elems=max_list_elems,
            )
        return dict(out)

    if isinstance(obj, (list, tuple)):
        if list_mode == "json":
            out[_sanitize_key(prefix)] = _stringify_param_value(
                list(obj)[:max_list_elems]
            )
            return dict(out)

        for i, v in enumerate(list(obj)[:max_list_elems]):
            key = f"{prefix}{sep}{i}" if prefix else str(i)
            _flatten_for_params(
                v,
                prefix=key,
                sep=sep,
                out=out,
                max_depth=max_depth - 1,
                list_mode=list_mode,
                max_list_elems=max_list_elems,
            )
        return dict(out)

    out[_sanitize_key(prefix)] = _stringify_param_value(obj)
    return dict(out)


def log_hydra_config_to_mlflow(
    cfg: DictConfig,
    *,
    log_artifacts: bool = False,
    artifact_dir: str = "config",
    log_params: bool = True,
    include_prefixes: Tuple[str, ...] = (
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
    ),
    exclude_prefixes: Tuple[str, ...] = ("hydra",),
) -> None:
    """Log Hydra/OmegaConf config to MLflow as artifacts and (optionally) params.

    Artifacts
    ---------
    - `<artifact_dir>/hydra_config.yaml` (resolved YAML)
    - `<artifact_dir>/hydra_config.json` (resolved JSON)
    - `<artifact_dir>/.hydra/` (Hydra's config bundle, if available)

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config object.
    artifact_dir : str, default="config"
        MLflow artifact subdirectory for config files.
    log_params : bool, default=True
        Whether to also log a filtered, flattened subset of the resolved config as MLflow params.
    include_roots : tuple of str, default=(...)
        Top-level keys to include when flattening the resolved config for params. Use this to keep
        MLflow params queryable and small.
    exclude_roots : tuple of str, default=("hydra",)
        Top-level keys to exclude from param flattening (Hydra internals are usually noise).

    Notes
    -----
    Logging *everything* as params makes your tracking DB big and your comparisons miserable.
    Prefer: "choices/*" + a small whitelist of true hyperparameters.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)

    if log_artifacts:
        mlflow.log_text(
            OmegaConf.to_yaml(cfg, resolve=True),
            f"{artifact_dir}/hydra_config.yaml",
        )
        mlflow.log_dict(resolved, f"{artifact_dir}/hydra_config.json")

        # Log Hydra's .hydra bundle if present (contains overrides.yaml etc.)
        try:
            out_dir = cfg.hydra.runtime.output_dir
            hydra_dir = Path(str(out_dir)) / ".hydra"
            if hydra_dir.exists():
                mlflow.log_artifacts(
                    str(hydra_dir), artifact_path=f"{artifact_dir}/.hydra"
                )
        except Exception:
            pass

    if not log_params:
        return

    params: Dict[str, str] = {}

    # --- NEW: log chosen config options + "yaml-ish" file names ----------------
    try:
        choices = OmegaConf.to_container(
            cfg.hydra.runtime.choices, resolve=True
        )
    except Exception:
        # Fallback: sometimes cfg doesn't carry hydra.runtime.*
        try:
            from hydra.core.hydra_config import HydraConfig

            choices = OmegaConf.to_container(
                HydraConfig.get().runtime.choices, resolve=True
            )
        except Exception:
            choices = None

    if isinstance(choices, dict):
        for group, choice in choices.items():
            if choice is None:
                continue
            choice_s = str(choice)
            params[_sanitize_key(f"choice/{group}")] = _stringify_param_value(
                choice_s
            )
    # --------------------------------------------------------------------------

    # Flatten a filtered subset of resolved config for queryable params
    for top_k, top_v in resolved.items():
        if top_k in exclude_prefixes:
            continue
        if include_prefixes and top_k not in include_prefixes:
            continue
        _flatten_for_params(
            top_v,
            prefix=top_k,
            sep="/",
            out=params,
            list_mode="json",
            max_depth=12,
        )

    mlflow.log_params(params)


# ---------------------------------------------------------------------------
# MLflow SQLite setup + run logging
# ---------------------------------------------------------------------------

PathLike = Union[str, Path]


def _resolve_under_original_cwd(path: PathLike) -> Path:
    """Resolve a path relative to Hydra's original working directory."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        try:
            base_dir = Path(get_original_cwd())
        except Exception:
            base_dir = Path.cwd()
        p = base_dir / p
    return p.resolve()


def configure_mlflow_sqlite(
    *,
    db_path: PathLike,
    experiment_name: str,
    artifact_root: Optional[PathLike] = None,
) -> None:
    """Configure MLflow to use a SQLite backend store and set the experiment.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Either a directory (in which case `mlruns.db` is used inside it) or a path to a `.db` file.
        Relative paths are resolved against Hydra's original working directory.
    experiment_name : str
        MLflow experiment name. Created if missing.
    artifact_root : str or pathlib.Path, optional
        Artifact root used *only when creating a new experiment*. If the experiment already exists,
        MLflow will keep its existing artifact_location.

    Notes
    -----
    - SQLite URI uses `sqlite:////abs/path/to/file.db` for absolute paths.
    - Experiment artifact location is effectively fixed after creation.
    """
    db_path_resolved = _resolve_under_original_cwd(db_path)
    db_file = db_path_resolved
    if db_file.suffix.lower() != ".db":
        db_file = db_file / "mlruns.db"

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        if artifact_root is not None:
            art_root = _resolve_under_original_cwd(artifact_root)
            art_root.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(
                experiment_name, artifact_location=art_root.as_uri()
            )
        mlflow.set_experiment(experiment_name)
    else:
        if artifact_root is not None:
            warnings.warn(
                f"Experiment '{experiment_name}' already exists; ignoring artifact_root={artifact_root!s}. "
                f"Existing artifact_location is {exp.artifact_location!s}."
            )
        mlflow.set_experiment(experiment_name)


def log_results_to_mlflow(
    cfg: DictConfig,
    cycle_metrics: Sequence[Mapping[str, Any]],
    *,
    experiment_name: str,
    db_path: PathLike,
    artifact_root: Optional[PathLike] = None,
    run_name: Optional[str] = None,
    log_config_params: bool = True,
    config_artifact_dir: str = "config",
    # practical knobs
    log_every: int = 1,
    also_log_cycle_metrics_artifact: bool = False,
    steps: Optional[Sequence[int]] = None,
) -> None:
    """Log an experiment to MLflow (config + per-cycle metrics).

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Hydra config to log.
    cycle_metrics : sequence of mapping
        Per-cycle metric dicts. Each entry is logged with `step=<cycle index>`.
        Metric dicts may include NaN/inf; these are dropped automatically.
    experiment_name : str
        MLflow experiment name to log into.
    db_path : str or pathlib.Path
        Directory or `.db` file path for SQLite backend store. Relative paths resolve against Hydra's
        original working directory.
    artifact_root : str or pathlib.Path, optional
        Artifact root used only when creating a new experiment.
    run_name : str, optional
        MLflow run name.
    log_config_params : bool, default=True
        Whether to log flattened config as MLflow params (in addition to config artifacts).
    config_artifact_dir : str, default="config"
        Artifact folder for config.
    log_every : int, default=1
        Log metrics only every k cycles (1 logs every cycle). Useful when you have thousands of cycles.
        Full metrics can still be stored as an artifact.
    also_log_cycle_metrics_artifact : bool, default=True
        If True, log the full `cycle_metrics` list as a JSON artifact for exact reconstruction.

    Returns
    -------
    None
    """
    # Enable async logging if supported by your MLflow version.
    # API is mlflow.config.enable_async_logging(True). (Your previous call was wrong.)
    try:
        mlflow.config.enable_async_logging(True)
    except Exception:
        pass

    configure_mlflow_sqlite(
        db_path=db_path,
        experiment_name=experiment_name,
        artifact_root=artifact_root,
    )

    with mlflow.start_run(run_name=run_name):
        log_hydra_config_to_mlflow(
            cfg, artifact_dir=config_artifact_dir, log_params=log_config_params
        )

        if also_log_cycle_metrics_artifact:
            # This is the "truth"; MLflow metrics are for plotting/querying.
            mlflow.log_dict(list(cycle_metrics), "cycle_metrics.json")

        if log_every <= 0:
            raise ValueError("log_every must be >= 1.")

        steps = list(range(len(cycle_metrics))) if steps is None else steps
        for step, m in zip(steps, cycle_metrics):
            if step % log_every != 0:
                continue
            safe = _sanitize_metrics(m)
            if safe:
                mlflow.log_metrics(safe, step=step)
