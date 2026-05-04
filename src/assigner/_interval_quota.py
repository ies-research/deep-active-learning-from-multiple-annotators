import numpy as np

from ._base import PairAssigner
from ._constraints import coerce_annotator_vector


class IntervalQuotaPairAssigner(PairAssigner):
    """
    Quota-style assigner with interval-based annotator non-dominance.

    If pair-level utility lower/upper bounds are provided, the assigner first
    computes a batch-global feasible-pair summary per annotator. Annotators are
    preferred when their optimistic summary is not below the best pessimistic
    summary. Assignment is quota-balanced among those non-dominated annotators,
    with dominated annotators used as backfill when the preferred set cannot
    satisfy the remaining budget and constraints.

    If interval metadata is unavailable, the assigner falls back to ordinary
    quota balancing unless ``require_intervals=True``.
    """

    def __init__(
        self,
        coverage="none",
        *,
        max_per_sample=None,
        max_per_annotator=None,
        require_intervals=False,
        fallback="quota",
    ):
        coverage = str(coverage)
        fallback = str(fallback)
        if coverage not in {"none", "hard"}:
            raise ValueError("coverage must be one of {'none', 'hard'}")
        if fallback not in {"quota"}:
            raise ValueError("fallback must be 'quota'")
        if max_per_sample is not None and int(max_per_sample) <= 0:
            raise ValueError("max_per_sample must be positive or None.")
        if max_per_annotator is not None and int(max_per_annotator) <= 0:
            raise ValueError("max_per_annotator must be positive or None.")

        self.coverage = coverage
        self.max_per_sample = (
            None if max_per_sample is None else int(max_per_sample)
        )
        self.max_per_annotator = (
            None if max_per_annotator is None else int(max_per_annotator)
        )
        self.require_intervals = bool(require_intervals)
        self.fallback = fallback

        self.last_used_intervals_ = False
        self.last_eligible_annotators_ = None
        self.last_backfill_count_ = 0
        self.last_interval_lcb_ = None
        self.last_interval_ucb_ = None

    def _assign(
        self,
        sample_indices,
        annotator_indices,
        utilities,
        budget,
        annotator_label_counts=None,
        annotator_remaining_counts=None,
        utility_lcb=None,
        utility_ucb=None,
    ):
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        U = np.asarray(utilities, dtype=float).copy()

        S, A = U.shape
        budget = int(budget)
        self.last_used_intervals_ = False
        self.last_eligible_annotators_ = np.asarray([], dtype=int)
        self.last_backfill_count_ = 0
        self.last_interval_lcb_ = None
        self.last_interval_ucb_ = None

        if budget <= 0 or S == 0 or A == 0:
            return np.empty((0, 2), dtype=int)

        hist = self._coerce_annotator_counts(
            annotator_indices, annotator_label_counts
        )
        remaining = self._coerce_annotator_remaining(
            annotator_indices, annotator_remaining_counts
        )

        eligible = self._resolve_interval_eligible_mask(
            U=U,
            annotator_indices=annotator_indices,
            utility_lcb=utility_lcb,
            utility_ucb=utility_ucb,
        )

        batch_a = np.zeros(A, dtype=int)
        batch_s = np.zeros(S, dtype=int)
        selected = []

        for _ in range(budget):
            feasible = self._feasible_mask(U, batch_s, batch_a, remaining)
            if not feasible.any():
                break

            primary = feasible & eligible[None, :]
            backfill = False
            if primary.any():
                active = primary
            else:
                active = feasible
                backfill = True

            s_loc, a_loc = self._pick_quota_pair(
                U=U,
                active=active,
                hist=hist,
                batch_a=batch_a,
                batch_s=batch_s,
            )
            if s_loc is None:
                break

            selected.append(
                (int(sample_indices[s_loc]), int(annotator_indices[a_loc]))
            )
            if backfill:
                self.last_backfill_count_ += 1

            U[s_loc, a_loc] = np.nan
            batch_a[a_loc] += 1
            batch_s[s_loc] += 1

        return np.asarray(selected, dtype=int).reshape(-1, 2)

    def _resolve_interval_eligible_mask(
        self,
        *,
        U,
        annotator_indices,
        utility_lcb,
        utility_ucb,
    ):
        A = U.shape[1]
        has_intervals = utility_lcb is not None and utility_ucb is not None
        if not has_intervals:
            if self.require_intervals:
                raise ValueError(
                    "IntervalQuotaPairAssigner requires utility_lcb and "
                    "utility_ucb when require_intervals=True."
                )
            eligible = np.ones(A, dtype=bool)
            self.last_eligible_annotators_ = annotator_indices.copy()
            return eligible

        lcb = np.asarray(utility_lcb, dtype=float)
        ucb = np.asarray(utility_ucb, dtype=float)
        if lcb.shape != U.shape or ucb.shape != U.shape:
            raise ValueError(
                "utility_lcb and utility_ucb must match utilities shape "
                f"{U.shape}."
            )

        feasible = np.isfinite(U)
        valid_lcb = feasible & np.isfinite(lcb)
        valid_ucb = feasible & np.isfinite(ucb)
        valid = valid_lcb & valid_ucb

        lcb_a = np.full(A, -np.inf, dtype=float)
        ucb_a = np.full(A, -np.inf, dtype=float)
        has_valid = valid.any(axis=0)
        for a_loc in np.flatnonzero(has_valid):
            lcb_a[a_loc] = float(np.min(lcb[valid[:, a_loc], a_loc]))
            ucb_a[a_loc] = float(np.max(ucb[valid[:, a_loc], a_loc]))

        if not np.any(has_valid):
            if self.require_intervals:
                raise ValueError(
                    "No finite interval metadata is available for feasible pairs."
                )
            eligible = np.ones(A, dtype=bool)
        else:
            threshold = float(np.max(lcb_a[has_valid]))
            eligible = has_valid & (ucb_a >= threshold)
            if not np.any(eligible):
                eligible = has_valid

        self.last_used_intervals_ = bool(np.any(has_valid))
        self.last_interval_lcb_ = lcb_a
        self.last_interval_ucb_ = ucb_a
        self.last_eligible_annotators_ = annotator_indices[eligible].copy()
        return eligible

    def _feasible_mask(self, U, batch_s, batch_a, remaining):
        feasible = np.isfinite(U)
        if self.max_per_sample is not None:
            feasible &= batch_s[:, None] < self.max_per_sample
        if self.max_per_annotator is not None:
            feasible &= batch_a[None, :] < self.max_per_annotator
        if remaining is not None:
            feasible &= batch_a[None, :] < remaining[None, :]
        return feasible

    def _pick_quota_pair(self, *, U, active, hist, batch_a, batch_s):
        feasible_cols = np.flatnonzero(active.any(axis=0))
        if feasible_cols.size == 0:
            return None, None

        eff_counts = hist + batch_a
        min_eff = eff_counts[feasible_cols].min()
        cand_cols = feasible_cols[eff_counts[feasible_cols] == min_eff]

        if self.coverage == "none":
            return self._pick_max_utility_pair(U, active, cand_cols)
        return self._pick_coverage_then_utility_pair(
            U,
            active,
            cand_cols,
            batch_s,
        )

    @staticmethod
    def _pick_max_utility_pair(U, active, cand_cols):
        mask = active[:, cand_cols]
        if not mask.any():
            return None, None
        sub = np.where(mask, U[:, cand_cols], -np.inf)
        flat = int(np.argmax(sub))
        s_loc, a_loc_sub = np.unravel_index(flat, sub.shape)
        if not np.isfinite(sub[s_loc, a_loc_sub]):
            return None, None
        return int(s_loc), int(cand_cols[a_loc_sub])

    @staticmethod
    def _pick_coverage_then_utility_pair(U, active, cand_cols, batch_s):
        row_ok = active[:, cand_cols].any(axis=1)
        rows = np.flatnonzero(row_ok)
        if rows.size == 0:
            return None, None

        min_count = batch_s[rows].min()
        rows = rows[batch_s[rows] == min_count]
        mask = active[np.ix_(rows, cand_cols)]
        sub = np.where(mask, U[np.ix_(rows, cand_cols)], -np.inf)
        flat = int(np.argmax(sub))
        r_loc, a_loc_sub = np.unravel_index(flat, sub.shape)
        if not np.isfinite(sub[r_loc, a_loc_sub]):
            return None, None
        return int(rows[r_loc]), int(cand_cols[a_loc_sub])

    @staticmethod
    def _coerce_annotator_counts(annotator_indices, annotator_label_counts):
        out = coerce_annotator_vector(
            annotator_indices,
            annotator_label_counts,
            name="annotator_label_counts",
        )
        if out is None:
            return np.zeros(len(annotator_indices), dtype=int)
        return out

    @staticmethod
    def _coerce_annotator_remaining(
        annotator_indices,
        annotator_remaining_counts,
    ):
        return coerce_annotator_vector(
            annotator_indices,
            annotator_remaining_counts,
            name="annotator_remaining_counts",
        )
