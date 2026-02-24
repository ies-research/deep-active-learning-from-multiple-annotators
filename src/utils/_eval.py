from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from skactiveml.utils import is_unlabeled, compute_vote_vectors, majority_vote


def _entropy_from_counts(counts: np.ndarray) -> Tuple[float, float]:
    """Return (entropy, normalized_entropy) for nonnegative integer counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts / total
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    Hmax = float(np.log(len(counts))) if len(counts) > 1 else 0.0
    Hn = float(H / Hmax) if Hmax > 0 else 0.0
    return H, Hn


def _gini(x: np.ndarray) -> float:
    """Gini coefficient for nonnegative counts."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini = (n + 1 - 2 * sum_i (cumx_i / cumx_n)) / n
    return float((n + 1 - 2.0 * (cumx / cumx[-1]).sum()) / n)


def _multiclass_brier_ovr(
    y_true: np.ndarray, p: np.ndarray, classes: np.ndarray
) -> float:
    """Multiclass Brier score (OVR style): mean over classes of mean squared error."""
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    classes = np.asarray(classes)

    # Map labels to 0..K-1 indices consistent with p columns (assumed aligned to `classes`)
    # If your p columns are already aligned to sorted unique labels, pass that as `classes`.
    idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([idx[yy] for yy in y_true], dtype=int)

    K = p.shape[1]
    Y = np.zeros((y_true.shape[0], K), dtype=float)
    Y[np.arange(y_true.shape[0]), y_idx] = 1.0
    return float(np.mean((p - Y) ** 2))


def compute_cycle_metrics(
    y_acquired: np.ndarray,
    y_true: np.ndarray,
    *,
    missing_label: Any = np.nan,
    prev_present: Optional[np.ndarray] = None,
    prev_y_acquired: Optional[np.ndarray] = None,  # optional convenience
    random_state: Optional[int] = 0,
    classes: Optional[
        Sequence[int]
    ] = None,  # pass once to avoid np.unique each cycle
    # Optional test-set evaluation (probabilities!)
    p_pred_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute MLflow-friendly per-cycle *scalar* metrics (flat key/value).

    Returns a flat dict[str, float] intended for:
        mlflow.log_metrics(metrics, step=cycle_idx)

    Notes
    -----
    - No nested dicts.
    - No vectors/histograms; those should be logged as artifacts if needed.
    - `cycle_idx` is not included as a metric; use it as MLflow step.
    """
    Y = np.asarray(y_acquired)
    y_true = np.asarray(y_true)

    if Y.ndim != 2:
        raise ValueError(f"y_acquired must be 2D, got shape {Y.shape}.")
    if y_true.ndim != 1 or y_true.shape[0] != Y.shape[0]:
        raise ValueError(
            f"y_true must be shape ({Y.shape[0]},), got {y_true.shape}."
        )

    n_samples, n_annotators = Y.shape

    # Presence mask
    present = ~is_unlabeled(Y, missing_label=missing_label)
    per_sample = present.sum(axis=1).astype(int)
    per_annot = present.sum(axis=0).astype(int)

    total_pairs = int(present.sum())
    n_covered = int((per_sample > 0).sum())
    frac_covered = float(n_covered / n_samples) if n_samples else 0.0

    covered_counts = per_sample[per_sample > 0]
    if covered_counts.size:
        # redundancy across covered samples
        cov_min = float(np.min(covered_counts))
        cov_mean = float(np.mean(covered_counts))
        cov_median = float(np.median(covered_counts))
        cov_p75 = float(np.percentile(covered_counts, 75))
        cov_max = float(np.max(covered_counts))
        frac_cov_ge2 = float(np.mean(covered_counts >= 2))
        frac_cov_ge3 = float(np.mean(covered_counts >= 3))
    else:
        cov_min = cov_mean = cov_median = cov_p75 = cov_max = 0.0
        frac_cov_ge2 = frac_cov_ge3 = 0.0

    # Pair accuracy (micro over all observed labels)
    if total_pairs > 0:
        correct_mask = (Y == y_true[:, None]) & present
        pair_acc = float(correct_mask.sum() / total_pairs)
    else:
        pair_acc = 0.0

    # Majority vote stats
    if classes is None:
        # If y_true is non-integer labels, you can still pass explicit classes.
        classes_arr = np.unique(y_true.astype(int, copy=False))
    else:
        classes_arr = np.asarray(classes)

    V = compute_vote_vectors(
        Y, classes=classes_arr, missing_label=missing_label
    )  # (N, K)
    covered = V.sum(axis=1) > 0

    mv_pred = majority_vote(
        Y,
        classes=classes_arr,
        missing_label=missing_label,
        random_state=random_state,
    )
    mv_present = ~is_unlabeled(mv_pred, missing_label=missing_label)

    if mv_present.any():
        mv_acc = float(
            np.mean(
                mv_pred[mv_present].astype(int)
                == y_true[mv_present].astype(int)
            )
        )
    else:
        mv_acc = 0.0

    # Tie rate among covered samples: multiple maxima in vote counts
    if covered.any():
        vmax = V.max(axis=1)
        n_max = (V == vmax[:, None]).sum(axis=1)
        tie = covered & (vmax > 0) & (n_max > 1)
        mv_tie_rate = float(np.mean(tie[covered]))
    else:
        mv_tie_rate = 0.0

    # Disagreement among samples with >=2 labels: >1 nonzero class in votes
    multi = per_sample >= 2
    if np.any(multi):
        disagree = (V > 0).sum(axis=1) > 1
        disagreement_rate = float(np.mean(disagree[multi]))
    else:
        disagreement_rate = 0.0

    # Allocation concentration across annotators
    H, Hn = _entropy_from_counts(per_annot)
    G = _gini(per_annot)

    # Delta stats (new labels this cycle)
    new_pairs = 0.0
    new_pair_acc = 0.0
    new_unique_samples = 0.0

    if prev_present is None and prev_y_acquired is not None:
        Y_prev = np.asarray(prev_y_acquired)
        if Y_prev.shape != Y.shape:
            raise ValueError(
                f"prev_y_acquired must have shape {Y.shape}, got {Y_prev.shape}."
            )
        prev_present = ~is_unlabeled(Y_prev, missing_label=missing_label)

    if prev_present is not None:
        prev_present = np.asarray(prev_present, dtype=bool)
        if prev_present.shape != Y.shape:
            raise ValueError(
                f"prev_present must have shape {Y.shape}, got {prev_present.shape}."
            )
        new_mask = present & ~prev_present
        npairs = int(new_mask.sum())
        new_pairs = float(npairs)
        if npairs > 0:
            new_correct = ((Y == y_true[:, None]) & new_mask).sum()
            new_pair_acc = float(new_correct / npairs)
            new_unique_samples = float(np.sum(new_mask.any(axis=1)))
        else:
            new_pair_acc = 0.0
            new_unique_samples = 0.0

    # Optional test-set metrics
    test_acc = np.nan
    test_bal_acc = np.nan
    test_log_loss = np.nan
    test_brier = np.nan

    if (p_pred_test is None) ^ (y_test is None):
        raise ValueError("Provide both p_pred_test and y_test, or neither.")
    if p_pred_test is not None:
        p_pred_test = np.asarray(p_pred_test, dtype=float)
        y_test = np.asarray(y_test)

        if p_pred_test.ndim != 2:
            raise ValueError(
                f"p_pred_test must be 2D, got shape {p_pred_test.shape}."
            )
        if y_test.ndim != 1 or y_test.shape[0] != p_pred_test.shape[0]:
            raise ValueError(
                f"y_test must be shape ({p_pred_test.shape[0]},), got {y_test.shape}."
            )

        y_pred_test = np.argmax(p_pred_test, axis=1)
        test_acc = float(accuracy_score(y_test, y_pred_test))
        test_bal_acc = float(balanced_accuracy_score(y_test, y_pred_test))
        test_log_loss = float(
            log_loss(y_test, p_pred_test, labels=classes_arr)
        )
        test_brier = float(
            _multiclass_brier_ovr(y_test, p_pred_test, classes=classes_arr)
        )

    # Flat scalar metrics (MLflow-friendly)
    metrics: Dict[str, float] = {
        # coverage / budget
        "label_total_pairs": float(total_pairs),
        "label_unique_samples": float(n_covered),
        "label_frac_covered": frac_covered,
        "label_labels_per_sample_mean_covered": cov_mean,
        "label_labels_per_sample_median_covered": cov_median,
        "label_labels_per_sample_p75_covered": cov_p75,
        "label_labels_per_sample_min_covered": cov_min,
        "label_labels_per_sample_max_covered": cov_max,
        "label_frac_covered_ge2": frac_cov_ge2,
        "label_frac_covered_ge3": frac_cov_ge3,
        # accuracy / agreement
        "acc_pair_micro": pair_acc,
        "acc_majority_vote": mv_acc,
        "acc_majority_vote_tie_rate": mv_tie_rate,
        "acc_disagreement_rate_multi": disagreement_rate,
        # allocation concentration
        "alloc_entropy": float(H),
        "alloc_entropy_norm": float(Hn),
        "alloc_gini": float(G),
        # delta (new labels)
        "delta_new_pairs": float(new_pairs),
        "delta_new_pair_acc": float(new_pair_acc),
        "delta_new_unique_samples": float(new_unique_samples),
        # test (NaN if not provided)
        "test_acc": float(test_acc) if np.isfinite(test_acc) else np.nan,
        "test_balanced_acc": (
            float(test_bal_acc) if np.isfinite(test_bal_acc) else np.nan
        ),
        "test_log_loss": (
            float(test_log_loss) if np.isfinite(test_log_loss) else np.nan
        ),
        "test_brier_ovr": (
            float(test_brier) if np.isfinite(test_brier) else np.nan
        ),
    }

    # Don’t include cycle_idx as a metric; use it as MLflow step instead.
    # If you *really* want it, log it as a tag, not a metric.

    return metrics
