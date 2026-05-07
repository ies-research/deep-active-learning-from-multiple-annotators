from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from skactiveml.utils import compute_vote_vectors, is_labeled


@dataclass(frozen=True)
class TemperatureCalibrationResult:
    temperature: float
    metrics: dict[str, float] = field(default_factory=dict)


def set_classifier_temperature(clf, temperature: float):
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    setattr(clf, "temperature_", temperature)
    return clf


def temperature_scaled_softmax(
    logits: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-15,
) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        temperature = 1.0
    z = logits / max(temperature, eps)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.maximum(exp_z.sum(axis=-1, keepdims=True), eps)


def build_soft_vote_targets(
    y,
    *,
    classes,
    missing_label: Any,
    smoothing_total: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    votes = compute_vote_vectors(
        y,
        classes=np.asarray(classes),
        missing_label=missing_label,
    ).astype(float, copy=False)
    n_votes = votes.sum(axis=1)
    K = votes.shape[1]
    smoothing_total = float(smoothing_total)
    if smoothing_total < 0:
        raise ValueError("smoothing_total must be non-negative.")
    targets = (votes + smoothing_total / K) / np.maximum(
        n_votes[:, None] + smoothing_total,
        1e-15,
    )
    return targets, votes, n_votes


def select_calibration_indices(
    y,
    *,
    missing_label: Any,
    validation_fraction: float,
    min_labeled_samples: int,
    min_validation_samples: int,
    min_votes_per_sample: int,
    random_state=None,
) -> tuple[np.ndarray, dict[str, float]]:
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got shape {y.shape}.")
    validation_fraction = float(validation_fraction)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1).")
    min_labeled_samples = int(min_labeled_samples)
    min_validation_samples = int(min_validation_samples)
    min_votes_per_sample = int(min_votes_per_sample)
    if min_labeled_samples < 0:
        raise ValueError("min_labeled_samples must be non-negative.")
    if min_validation_samples < 1:
        raise ValueError("min_validation_samples must be at least 1.")
    if min_votes_per_sample < 1:
        raise ValueError("min_votes_per_sample must be at least 1.")

    present = is_labeled(y, missing_label=missing_label)
    votes_per_sample = present.sum(axis=1).astype(int, copy=False)
    acquired = np.flatnonzero(votes_per_sample >= 1)
    stats = {
        "calib_candidate_samples": float(acquired.size),
        "calib_selected_samples": 0.0,
        "calib_min_votes_used": float(min_votes_per_sample),
        "calib_enabled": 0.0,
    }
    if acquired.size < min_labeled_samples:
        return np.array([], dtype=int), stats

    eligible = np.flatnonzero(votes_per_sample >= min_votes_per_sample)
    if eligible.size < min_validation_samples and min_votes_per_sample > 1:
        eligible = acquired
        stats["calib_min_votes_used"] = 1.0
    if eligible.size < min_validation_samples:
        return np.array([], dtype=int), stats

    n_validation = int(np.ceil(validation_fraction * acquired.size))
    n_validation = max(n_validation, min_validation_samples)
    n_validation = min(n_validation, eligible.size)

    rng = np.random.default_rng(random_state)
    indices = np.sort(
        rng.choice(eligible, size=n_validation, replace=False).astype(int)
    )
    stats["calib_selected_samples"] = float(indices.size)
    stats["calib_enabled"] = 1.0
    return indices, stats


def _soft_nll(logits, targets, temperature: float) -> float:
    p = temperature_scaled_softmax(logits, temperature=temperature)
    return float(-np.mean(np.sum(targets * np.log(np.clip(p, 1e-15, 1.0)), axis=1)))


def _soft_brier(logits, targets, temperature: float) -> float:
    p = temperature_scaled_softmax(logits, temperature=temperature)
    return float(np.mean(np.sum((p - targets) ** 2, axis=1)))


def _majority_accuracy(logits, vote_counts, temperature: float) -> float:
    y_vote, y_pred = _majority_vote_predictions(
        logits,
        vote_counts,
        temperature,
    )
    if y_vote.size == 0:
        return np.nan
    return float(np.mean(y_pred == y_vote))


def _balanced_majority_accuracy(logits, vote_counts, temperature: float) -> float:
    y_vote, y_pred = _majority_vote_predictions(
        logits,
        vote_counts,
        temperature,
    )
    if y_vote.size == 0:
        return np.nan
    recalls = []
    for cls_idx in np.unique(y_vote):
        in_class = y_vote == cls_idx
        if np.any(in_class):
            recalls.append(float(np.mean(y_pred[in_class] == y_vote[in_class])))
    if not recalls:
        return np.nan
    return float(np.mean(recalls))


def _majority_vote_predictions(logits, vote_counts, temperature: float):
    if vote_counts is None or len(vote_counts) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    vote_counts = np.asarray(vote_counts, dtype=float)
    max_counts = vote_counts.max(axis=1)
    unique_max = (vote_counts == max_counts[:, None]).sum(axis=1) == 1
    has_votes = max_counts > 0
    valid = has_votes & unique_max
    if not np.any(valid):
        return np.array([], dtype=int), np.array([], dtype=int)
    y_vote = np.argmax(vote_counts[valid], axis=1)
    y_pred = np.argmax(
        temperature_scaled_softmax(logits[valid], temperature=temperature),
        axis=1,
    )
    return y_vote.astype(int, copy=False), y_pred.astype(int, copy=False)


def _mean_confidence(logits, temperature: float) -> float:
    p = temperature_scaled_softmax(logits, temperature=temperature)
    return float(np.mean(np.max(p, axis=1)))


def _soft_ece(
    logits,
    targets,
    temperature: float,
    n_bins: int = 15,
) -> float:
    p = temperature_scaled_softmax(logits, temperature=temperature)
    confidence = np.max(p, axis=1)
    prediction = np.argmax(p, axis=1)
    soft_correctness = targets[np.arange(targets.shape[0]), prediction]
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_idx in range(n_bins):
        lo = bin_edges[bin_idx]
        hi = bin_edges[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence > lo) & (confidence <= hi)
        if not np.any(in_bin):
            continue
        weight = float(np.mean(in_bin))
        conf_bin = float(np.mean(confidence[in_bin]))
        acc_bin = float(np.mean(soft_correctness[in_bin]))
        ece += weight * abs(conf_bin - acc_bin)
    return float(ece)


def _objective_value(
    logits,
    targets,
    temperature: float,
    objective: str,
    ece_n_bins: int = 15,
) -> float:
    if objective == "nll":
        return _soft_nll(logits, targets, temperature)
    if objective == "brier":
        return _soft_brier(logits, targets, temperature)
    if objective == "ece":
        return _soft_ece(logits, targets, temperature, n_bins=ece_n_bins)
    raise ValueError("objective must be one of {'nll', 'brier', 'ece'}.")


def _golden_section_search(func, lo: float, hi: float, max_iter: int = 80) -> float:
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    inv_phi_sq = (3.0 - np.sqrt(5.0)) / 2.0
    c = lo + inv_phi_sq * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    fc = func(c)
    fd = func(d)
    for _ in range(max_iter):
        if abs(hi - lo) <= 1e-6:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = lo + inv_phi_sq * (hi - lo)
            fc = func(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + inv_phi * (hi - lo)
            fd = func(d)
    return float((lo + hi) / 2.0)


def tune_temperature_from_logits(
    logits,
    targets,
    *,
    vote_counts=None,
    objective: str = "nll",
    bounds: tuple[float, float] = (0.25, 8.0),
    ece_n_bins: int = 15,
) -> TemperatureCalibrationResult:
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got shape {logits.shape}.")
    if targets.shape != logits.shape:
        raise ValueError(
            f"targets must have shape {logits.shape}, got {targets.shape}."
        )
    if logits.shape[0] == 0:
        return TemperatureCalibrationResult(
            temperature=1.0,
            metrics={
                "calib_temperature": 1.0,
                "calib_nll_before": np.nan,
                "calib_nll_after": np.nan,
                "calib_brier_before": np.nan,
                "calib_brier_after": np.nan,
                "calib_ece_before": np.nan,
                "calib_ece_after": np.nan,
            },
        )
    lo, hi = map(float, bounds)
    if not (0 < lo <= hi):
        raise ValueError("bounds must satisfy 0 < lower <= upper.")
    objective = str(objective).lower()
    ece_n_bins = int(ece_n_bins)

    def func(log_t):
        return _objective_value(
            logits,
            targets,
            np.exp(log_t),
            objective,
            ece_n_bins=ece_n_bins,
        )

    if objective == "ece":
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), 201))
        grid_values = np.array(
            [
                _objective_value(
                    logits,
                    targets,
                    t,
                    objective,
                    ece_n_bins=ece_n_bins,
                )
                for t in grid
            ],
            dtype=float,
        )
        best_t = float(grid[int(np.argmin(grid_values))])
    else:
        best_log_t = _golden_section_search(func, np.log(lo), np.log(hi))
        best_t = float(np.exp(best_log_t))

    candidates = np.array([lo, 1.0, hi, best_t], dtype=float)
    values = np.array(
        [
            _objective_value(
                logits,
                targets,
                t,
                objective,
                ece_n_bins=ece_n_bins,
            )
            for t in candidates
        ],
        dtype=float,
    )
    temperature = float(candidates[int(np.argmin(values))])
    metrics = {
        "calib_temperature": temperature,
        "calib_nll_before": _soft_nll(logits, targets, 1.0),
        "calib_nll_after": _soft_nll(logits, targets, temperature),
        "calib_brier_before": _soft_brier(logits, targets, 1.0),
        "calib_brier_after": _soft_brier(logits, targets, temperature),
        "calib_ece_before": _soft_ece(
            logits, targets, 1.0, n_bins=ece_n_bins
        ),
        "calib_ece_after": _soft_ece(
            logits, targets, temperature, n_bins=ece_n_bins
        ),
        "calib_confidence_before": _mean_confidence(logits, 1.0),
        "calib_confidence_after": _mean_confidence(logits, temperature),
        "calib_majority_acc_before": _majority_accuracy(logits, vote_counts, 1.0),
        "calib_majority_acc_after": _majority_accuracy(
            logits, vote_counts, temperature
        ),
        "calib_majority_balanced_acc_before": _balanced_majority_accuracy(
            logits,
            vote_counts,
            1.0,
        ),
        "calib_majority_balanced_acc_after": _balanced_majority_accuracy(
            logits,
            vote_counts,
            temperature,
        ),
    }
    return TemperatureCalibrationResult(
        temperature=temperature,
        metrics=metrics,
    )
