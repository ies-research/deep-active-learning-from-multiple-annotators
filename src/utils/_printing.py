from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


# ============================================================
# Formatting helpers
# ============================================================
def _format_int(x: float | int) -> str:
    return f"{int(round(float(x))):,}".replace(",", "_")  # 150000 -> 150_000


def _format_float(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _format_pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * float(x):.{digits}f}%"


def _maybe(d: Dict[str, Any], k: str, default: float = float("nan")) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return max(0.0, min(1.0, float(x)))


def _bar01(x: float, width: int = 14) -> str:
    """ASCII bar for values in [0,1]."""
    if not np.isfinite(x):
        return " " * width
    x = _clamp01(x)
    filled = int(round(x * width))
    return "█" * filled + "░" * (width - filled)


def _as_str_list(xs: Sequence[Any], max_items: int = 12) -> str:
    xs = list(xs)
    if len(xs) <= max_items:
        return ", ".join(map(str, xs))
    head = ", ".join(map(str, xs[:max_items]))
    return f"{head}, … (+{len(xs) - max_items} more)"


# ============================================================
# Dataset report (labels + annotator accuracies)
# ============================================================
def pretty_dataset_report(
    *,
    classes: Sequence[Any],
    n_features: int,
    n_samples: int,
    np_arrays: Dict[str, np.ndarray],
    y_key: str = "y_train",
    z_key: str = "z_train",
    sort_annotators_by: str = "acc",  # "acc" or "id"
    digits: int = 2,
    width: int = 84,
) -> None:
    """
    Prints a readable dataset + annotator summary.

    Expects:
      - np_arrays[y_key] shape (n_samples,)
      - np_arrays[z_key] shape (n_samples, n_annotators) OR (n_samples,)
    Handles missing labels in z via -1 (int) or NaN (float).
    """
    y = np.asarray(np_arrays[y_key])
    z = np.asarray(np_arrays[z_key])

    if y.ndim != 1:
        raise ValueError(
            f"{y_key} must be 1D (n_samples,), got shape={y.shape}"
        )

    if z.ndim == 1:
        z = z[:, None]
    elif z.ndim != 2:
        raise ValueError(f"{z_key} must be 1D or 2D, got shape={z.shape}")

    if z.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch: {y_key}.shape[0]={y.shape[0]} != {z_key}.shape[0]={z.shape[0]}"
        )

    n_annotators = z.shape[1]

    # Missing labels: -1 (int) or NaN (float)
    missing = np.zeros_like(z, dtype=bool)
    if np.issubdtype(z.dtype, np.floating):
        missing |= np.isnan(z)
    if np.issubdtype(z.dtype, np.integer):
        missing |= z == -1

    correct = (z == y[:, None]) & (~missing)
    denom_per_ann = np.sum(~missing, axis=0).astype(float)
    denom_overall = float(np.sum(~missing))

    acc_per_ann = np.divide(
        np.sum(correct, axis=0),
        np.maximum(denom_per_ann, 1.0),
        dtype=float,
    )
    acc_overall = float(np.sum(correct)) / max(denom_overall, 1.0)

    if sort_annotators_by == "acc":
        order = np.argsort(-acc_per_ann)
    elif sort_annotators_by == "id":
        order = np.arange(n_annotators)
    else:
        raise ValueError("sort_annotators_by must be 'acc' or 'id'")

    # Header
    title = "Dataset / Annotation Summary"
    print("┏" + "━" * (width - 2) + "┓")
    print("┃" + title.center(width - 2) + "┃")
    print("┗" + "━" * (width - 2) + "┛")

    print(f"Classes       : {_as_str_list(classes)}")
    print(f"# Features    : {_format_int(n_features)}")
    print(f"# Samples     : {_format_int(n_samples)}")
    print(f"# Annotators  : {_format_int(n_annotators)}")
    print()

    # Overall accuracy
    labeled_total = y.shape[0] * n_annotators if n_annotators > 0 else 1
    coverage = denom_overall / float(labeled_total)
    print("Overall label accuracy (ignoring missing):")
    print(
        f"  Accuracy  : {_format_pct(acc_overall, digits)}  {_bar01(acc_overall, 16)}"
    )
    print(
        f"  Coverage  : {_format_pct(coverage, digits)} of sample×annotator labels present"
    )
    print()

    # Per-annotator table
    idx_w = max(3, len(str(max(n_annotators - 1, 0))))
    lab_w = 12
    acc_w = 10
    header = (
        f"{'id':>{idx_w}}  {'labeled':>{lab_w}}  {'acc':>{acc_w}}  {'bar':>16}"
    )
    print(
        "Per-annotator accuracy (sorted):"
        if sort_annotators_by == "acc"
        else "Per-annotator accuracy:"
    )
    print(header)
    print("-" * len(header))

    per_annotator_accuracies = []
    for j in order:
        labeled = int(denom_per_ann[j])
        acc = float(acc_per_ann[j])
        per_annotator_accuracies.append(acc)
        print(
            f"{j:>{idx_w}d}  "
            f"{_format_int(labeled):>{lab_w}}  "
            f"{_format_pct(acc, digits):>{acc_w}}  "
            f"{_bar01(acc, 16)}"
        )

    print()
    print(
        f"Average Annotator Accuracy:  "
        f"{_format_pct(float(np.mean(per_annotator_accuracies)), digits)}"
        f"+-{_format_pct(float(np.std(per_annotator_accuracies)), digits)}"
    )
    print()


# ============================================================
# Active learning cycle metrics (dashboard + grouped block)
# ============================================================
IMPORTANT_KEYS = [
    "label_total_pairs",
    "label_unique_samples",
    "label_frac_covered",
    "label_labels_per_sample_mean_covered",
    "label_frac_covered_ge2",
    "acc_pair_micro",
    "acc_majority_vote",
    "acc_majority_vote_tie_rate",
    "acc_disagreement_rate_multi",
    "alloc_entropy_norm",
    "alloc_gini",
    "delta_new_pairs",
    "delta_new_pair_acc",
    "delta_new_unique_samples",
    "test_acc",
    "test_balanced_acc",
    "test_log_loss",
    "test_brier_ovr",
]


def pretty_cycle_metrics(
    m: Dict[str, Any],
    *,
    cycle: Optional[int] = None,
    pairs_per_cycle: Optional[int] = None,
    width: int = 84,
    digits_pct: int = 1,
) -> None:
    """
    Prints an easy-to-scan summary of key AL metrics for one cycle.
    """

    # Coverage / budget
    total_pairs = _maybe(m, "label_total_pairs")
    uniq = _maybe(m, "label_unique_samples")
    covered = _maybe(m, "label_frac_covered")
    lps_mean = _maybe(m, "label_labels_per_sample_mean_covered")
    frac_ge2 = _maybe(m, "label_frac_covered_ge2")

    # Label quality / noise
    acc_pair = _maybe(m, "acc_pair_micro")
    acc_mv = _maybe(m, "acc_majority_vote")
    tie = _maybe(m, "acc_majority_vote_tie_rate")
    disagree = _maybe(m, "acc_disagreement_rate_multi")

    # Fairness / allocation
    ent_norm = _maybe(m, "alloc_entropy_norm")
    gini = _maybe(m, "alloc_gini")

    # Deltas
    d_pairs = _maybe(m, "delta_new_pairs")
    d_pairs_acc = _maybe(m, "delta_new_pair_acc")

    # Model
    test_acc = _maybe(m, "test_acc")
    test_bal = _maybe(m, "test_balanced_acc")
    test_ll = _maybe(m, "test_log_loss")
    test_brier = _maybe(m, "test_brier_ovr")

    tag = (
        f"Active Learning Metrics | Cycle {cycle}"
        if cycle is not None
        else "Active Learning Metrics"
    )
    print("┏" + "━" * (width - 2) + "┓")
    print("┃" + tag.center(width - 2) + "┃")
    print("┗" + "━" * (width - 2) + "┛")

    # Dashboard line
    cov_txt = _format_pct(covered, 2) if np.isfinite(covered) else "n/a"
    dash = (
        f"Pairs {_format_int(total_pairs)}"
        + (f" (+{_format_int(d_pairs)})" if np.isfinite(d_pairs) else "")
        + f" | Covered {cov_txt}"
        + f" | PairAcc {_format_pct(acc_pair, digits_pct)}"
        + f" | TestBalAcc {_format_pct(test_bal, digits_pct)}"
    )
    print(dash[:width])
    print()

    # Grouped blocks
    print("Coverage")
    print(f"  total_pairs                 : {_format_int(total_pairs)}")
    if pairs_per_cycle is not None and np.isfinite(d_pairs):
        print(
            f"  new_pairs_this_cycle        : {_format_int(d_pairs)} / {_format_int(pairs_per_cycle)}"
        )
    else:
        print(f"  new_pairs_this_cycle        : {_format_int(d_pairs)}")
    print(f"  unique_samples_covered      : {_format_int(uniq)}")
    print(f"  frac_covered                : {cov_txt}")
    print(f"  labels_per_sample_mean      : {_format_float(lps_mean, 2)}")
    print(f"  frac_samples_with_≥2_labels : {_format_pct(frac_ge2, 1)}")

    print("\nLabel quality")
    print(f"  acc_pair_micro              : {_format_pct(acc_pair, 1)}")
    print(f"  acc_majority_vote           : {_format_pct(acc_mv, 1)}")
    print(f"  majority_vote_tie_rate      : {_format_pct(tie, 1)}")
    print(f"  disagreement_rate_multi     : {_format_pct(disagree, 1)}")
    if np.isfinite(d_pairs_acc):
        print(f"  acc_of_new_pairs            : {_format_pct(d_pairs_acc, 1)}")

    print("\nAllocation fairness")
    print(f"  entropy_norm                : {_format_float(ent_norm, 3)}")
    print(f"  gini                        : {_format_float(gini, 3)}")

    print("\nModel performance")
    print(f"  test_acc                    : {_format_pct(test_acc, 2)}")
    print(f"  test_balanced_acc           : {_format_pct(test_bal, 2)}")
    print(f"  test_log_loss               : {_format_float(test_ll, 4)}")
    print(f"  test_brier_ovr              : {_format_float(test_brier, 6)}")
    print()


@dataclass
class MetricHistory:
    """Stores metrics across cycles and can print a compact trend table."""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, cycle: int, m: Dict[str, Any]) -> None:
        r = {"cycle": cycle}
        for k in IMPORTANT_KEYS:
            r[k] = m.get(k, float("nan"))
        self.rows.append(r)

    def print_table(self, last: int = 10) -> None:
        if not self.rows:
            print("(no history)")
            return

        rows = self.rows[-last:]
        cols = [
            "cycle",
            "label_total_pairs",
            "label_frac_covered",
            "acc_pair_micro",
            "acc_majority_vote",
            "alloc_entropy_norm",
            "alloc_gini",
            "test_acc",
            "test_balanced_acc",
            "test_log_loss",
        ]

        header = (
            f"{'cy':>3}  {'pairs':>9}  {'cov':>7}  {'pair':>7}  {'mv':>7}  "
            f"{'ent':>6}  {'gini':>6}  {'tacc':>7}  {'tbacc':>7}  {'ll':>8}"
        )
        print(header)
        print("-" * len(header))

        for r in rows:
            cy = int(r["cycle"])
            pairs = _format_int(_maybe(r, "label_total_pairs"))
            cov = _format_pct(_maybe(r, "label_frac_covered"), 2)
            pair = _format_pct(_maybe(r, "acc_pair_micro"), 1)
            mv = _format_pct(_maybe(r, "acc_majority_vote"), 1)
            ent = _format_float(_maybe(r, "alloc_entropy_norm"), 3)
            gini = _format_float(_maybe(r, "alloc_gini"), 3)
            tacc = _format_pct(_maybe(r, "test_acc"), 1)
            tbacc = _format_pct(_maybe(r, "test_balanced_acc"), 1)
            ll = _format_float(_maybe(r, "test_log_loss"), 3)

            print(
                f"{cy:>3}  {pairs:>9}  {cov:>7}  {pair:>7}  {mv:>7}  "
                f"{ent:>6}  {gini:>6}  {tacc:>7}  {tbacc:>7}  {ll:>8}"
            )


# ============================================================
# Convenience wrapper: call once at start + each cycle
# ============================================================
def print_al_reports(
    *,
    classes: Sequence[Any],
    n_features: int,
    n_samples: int,
    np_arrays: Dict[str, np.ndarray],
    cycle_metrics: Optional[Dict[str, Any]] = None,
    cycle: Optional[int] = None,
    pairs_per_cycle: Optional[int] = None,
    history: Optional[MetricHistory] = None,
    print_history_every: Optional[int] = None,
) -> None:
    """
    One function to print:
      - dataset report (always)
      - cycle metrics (if provided)
      - history table (if history provided)
    """

    pretty_dataset_report(
        classes=classes,
        n_features=n_features,
        n_samples=n_samples,
        np_arrays=np_arrays,
        y_key="y_train",
        z_key="z_train",
        sort_annotators_by="acc",
        digits=2,
    )

    if cycle_metrics is not None:
        pretty_cycle_metrics(
            cycle_metrics,
            cycle=cycle,
            pairs_per_cycle=pairs_per_cycle,
        )

        if history is not None and cycle is not None:
            history.add(cycle, cycle_metrics)
            if print_history_every is not None and print_history_every > 0:
                if (cycle + 1) % print_history_every == 0:
                    history.print_table(last=10)
