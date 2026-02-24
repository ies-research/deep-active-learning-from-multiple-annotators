from __future__ import annotations

import heapq
from typing import Any, List, Optional, Tuple

import numpy as np

from ._base import PairAssigner


class GainPairAssigner(PairAssigner):
    """
    Assign sample-annotator pairs under a pair-budget to maximize sum of per-sample scores.

    Utilities are interpreted as estimated per-sample annotator accuracies p in [0, 1].
    Unavailable pairs are indicated by np.nan.

    The score for a sample is computed from the probability that a majority vote is correct
    after k queried annotators (Poisson-binomial, updated incrementally).

    Parameters
    ----------
    budget_policy : {"stop_if_nonpositive", "force_budget"}, default="stop_if_nonpositive"
        - "stop_if_nonpositive": stop early if the best available marginal gain is <= 0.
        - "force_budget": keep assigning until budget is exhausted or no feasible pairs remain.
    score_mode : {"raw_correct_prob", "expected_correct_minus_incorrect", "normalized_advantage"}, default="raw_correct_prob"
        - "raw_correct_prob": score = q where q = P(majority correct)
        - "expected_correct_minus_incorrect": score = 2*q - 1  (binary-style ±1 utility)
        - "normalized_advantage": score = (q - 1/C) / (1 - 1/C)  requires n_classes
    n_classes : int or None, default=None
        Number of classes C for "normalized_advantage".
    max_labels_per_sample : int or None, default=None
        Optional cap on how many annotators can be assigned per sample.
    clip_p : float, default=1e-6
        Clip accuracies to [clip_p, 1-clip_p] for numerical stability.
    """

    def __init__(
        self,
        budget_policy: str = "force_budget",
        score_mode: str = "expected_correct_minus_incorrect",
        n_classes: Optional[int] = None,
        max_labels_per_sample: Optional[int] = None,
        clip_p: float = 1e-6,
    ):
        budget_policy = str(budget_policy)
        if budget_policy not in {"stop_if_nonpositive", "force_budget"}:
            raise ValueError(f"Invalid budget_policy={budget_policy!r}.")
        score_mode = str(score_mode)
        if score_mode not in {
            "raw_correct_prob",
            "expected_correct_minus_incorrect",
            "normalized_advantage",
        }:
            raise ValueError(f"Invalid score_mode={score_mode!r}.")
        if score_mode == "normalized_advantage":
            if n_classes is None or int(n_classes) < 2:
                raise ValueError(
                    "n_classes must be an int >= 2 when score_mode='normalized_advantage'."
                )
        if (
            max_labels_per_sample is not None
            and int(max_labels_per_sample) < 1
        ):
            raise ValueError(
                "max_labels_per_sample must be None or an int >= 1."
            )

        self.budget_policy = budget_policy
        self.score_mode = score_mode
        self.n_classes = None if n_classes is None else int(n_classes)
        self.max_labels_per_sample = (
            None
            if max_labels_per_sample is None
            else int(max_labels_per_sample)
        )
        self.clip_p = float(clip_p)

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _update_dp(dp: np.ndarray, p: float) -> np.ndarray:
        """Given dp[m]=P(M=m), update with one Bernoulli(p) correctness event."""
        k = dp.size - 1
        new = np.empty(k + 2, dtype=float)
        new[0] = dp[0] * (1.0 - p)
        new[1 : k + 1] = dp[1:] * (1.0 - p) + dp[:-1] * p
        new[k + 1] = dp[k] * p
        return new

    @staticmethod
    def _prob_majority_correct(dp: np.ndarray) -> float:
        """dp[m]=P(M=m) where M is number correct among k=dp.size-1; return P(M > k/2)."""
        k = dp.size - 1
        thr = (k // 2) + 1
        return float(dp[thr:].sum())

    def _score_from_q(self, q: float, k: int) -> float:
        # No labels => no inference => score 0 (as you wanted).
        if k <= 0:
            return 0.0
        if self.score_mode == "raw_correct_prob":
            return float(q)
        if self.score_mode == "expected_correct_minus_incorrect":
            return float(q - 5 * (1 - q))
        # normalized_advantage
        C = self.n_classes
        baseline = 1.0 / C
        return float((q - baseline) / (1.0 - baseline))

    def _gain_if_add(
        self, dp: np.ndarray, cur_score: float, p_next: float
    ) -> Tuple[float, float, np.ndarray]:
        """Compute gain and return (gain, new_score, new_dp) if adding annotator with p_next."""
        new_dp = self._update_dp(dp, p_next)
        q_new = self._prob_majority_correct(new_dp)
        k_new = new_dp.size - 1
        new_score = self._score_from_q(q_new, k_new)
        return new_score - cur_score, new_score, new_dp

    # ---- main API ---------------------------------------------------------

    def _assign(
        self,
        sample_indices,
        annotator_indices,
        utilities,
        budget,
        **kwargs,
    ):
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        U = np.asarray(utilities, dtype=float)

        S, A = U.shape
        budget = int(budget)
        if budget <= 0 or S == 0 or A == 0:
            return np.empty((0, 2), dtype=int)

        # Build per-sample sorted candidate annotator lists (local indices) by decreasing p
        cand_a: List[np.ndarray] = []
        cand_p: List[np.ndarray] = []
        for s in range(S):
            row = U[s]
            mask = ~np.isnan(row)
            if not np.any(mask):
                cand_a.append(np.empty((0,), dtype=int))
                cand_p.append(np.empty((0,), dtype=float))
                continue
            a_loc = np.flatnonzero(mask)
            p = row[a_loc].astype(float, copy=False)
            p = np.clip(p, self.clip_p, 1.0 - self.clip_p)
            order = np.argsort(p)[::-1]  # descending
            cand_a.append(a_loc[order])
            cand_p.append(p[order])

        # State per sample
        ptr = np.zeros(S, dtype=int)  # next candidate pointer
        k_used = np.zeros(S, dtype=int)  # labels assigned to sample so far
        dp_list: List[np.ndarray] = [
            np.array([1.0], dtype=float) for _ in range(S)
        ]  # dp for M correct
        score = np.zeros(S, dtype=float)
        version = np.zeros(S, dtype=int)

        # Heap entries: (-gain, tie_breaker, s, version_at_push)
        # tie_breaker: prefer samples with fewer labels if gains tie (mild coverage bias)
        heap: List[Tuple[float, int, int, int]] = []

        def can_add(s: int) -> bool:
            if ptr[s] >= cand_a[s].size:
                return False
            if (
                self.max_labels_per_sample is not None
                and k_used[s] >= self.max_labels_per_sample
            ):
                return False
            return True

        def push_gain(s: int) -> None:
            if not can_add(s):
                return
            p_next = float(cand_p[s][ptr[s]])
            g, _, _ = self._gain_if_add(dp_list[s], float(score[s]), p_next)
            # We push even negative gains; policy decides whether to stop.
            heapq.heappush(heap, (-g, int(k_used[s]), int(s), int(version[s])))

        for s in range(S):
            push_gain(s)

        selected_pairs: List[Tuple[int, int]] = []
        feasible_total = int(
            np.isfinite(np.where(np.isnan(U), -np.inf, U)).ravel().size
        )  # not used
        # Actually feasible pair count:
        feasible_pairs = int((~np.isnan(U)).sum())
        B = min(budget, feasible_pairs)

        for _ in range(B):
            while heap:
                neg_g, _, s, v = heapq.heappop(heap)
                if v == version[s]:
                    break
            else:
                break  # heap empty

            g_est = -neg_g
            if self.budget_policy == "stop_if_nonpositive" and g_est <= 0.0:
                break

            if not can_add(s):
                continue

            # Assign the next best annotator for this sample
            a_loc = int(cand_a[s][ptr[s]])
            p_next = float(cand_p[s][ptr[s]])

            # Update exact state (recompute gain to be consistent)
            g, new_score, new_dp = self._gain_if_add(
                dp_list[s], float(score[s]), p_next
            )

            # If policy is stop_if_nonpositive, also stop on actual gain <= 0
            if self.budget_policy == "stop_if_nonpositive" and g <= 0.0:
                break

            selected_pairs.append(
                (int(sample_indices[s]), int(annotator_indices[a_loc]))
            )

            dp_list[s] = new_dp
            score[s] = float(new_score)
            ptr[s] += 1
            k_used[s] += 1
            version[s] += 1

            push_gain(s)

        return np.asarray(selected_pairs, dtype=int)
