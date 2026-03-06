from __future__ import annotations

import numpy as np
from ._base import PairAssigner


class BudgetEntropyTopKAssigner(PairAssigner):
    r"""
    Batch assigner using budget-aware exploration from per-sample annotator entropy.

    This assigner implements the unconstrained strategy:
    for each sample i with annotator utilities u_i in [0, 1],

    1. compute the normalized entropy h_i of the row utilities,
    2. compute an exploration level
           e_i = (sqrt(b_eff) / (1 + sqrt(b_eff))) * h_i,
       where b_eff is an effective budget-per-annotator derived from both
       the local batch budget and the remaining global budget,
    3. set
           k_i = 1 + ceil((A_i - 1) * e_i),
       where A_i is the number of feasible annotators for sample i,
    4. build a top-k mixture policy
           q_i = (1 - e_i) * delta_best + e_i * Exploration(Top-k_i),
    5. sample one annotator from q_i for each selected sample.

    If `budget < n_samples`, the assigner first ranks samples by their expected
    utility under q_i and only keeps the top `budget` samples.

    Notes
    -----
    - This class returns at most one pair per sample, hence at most
      `min(budget, n_samples)` pairs.
    - Non-finite utilities are treated as unavailable entries.
    - Utilities are expected to lie in [0, 1].

    Parameters
    ----------
    eps : float, default=1e-12
        Numerical stability constant used in entropy and normalization steps.
    sample_score : {"expected", "max"}, default="expected"
        Criterion used to rank samples when `budget < n_samples`.
        - "expected": rank by expected utility under q_i.
        - "max": rank by the best row utility.
    exploration_distribution : {"uniform", "utility"}, default="uniform"
        Distribution used for the exploratory mass over the top-k annotators.
        - "uniform": spread exploration mass uniformly over top-k.
        - "utility": distribute exploration mass proportionally to utilities
          within top-k.
    exploration_budget_mode : {"remaining", "local", "min", "geometric_mean"}, default="remaining"
        How local batch budget and remaining global budget are combined to form
        the effective budget-per-annotator controlling exploration.
        Let
            b_local = budget / n_annotators
            b_remaining = remaining_budget / n_annotators
        Then:
        - "remaining":      b_eff = b_remaining
        - "local":          b_eff = b_local
        - "min":            b_eff = min(b_local, b_remaining)
        - "geometric_mean": b_eff = sqrt(b_local * b_remaining)
    random_state : None, int, or numpy.random.Generator, default=None
        Random state used when no `rng` keyword argument is passed to `__call__`.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-12,
        sample_score: str = "expected",
        exploration_distribution: str = "uniform",
        exploration_budget_mode: str = "remaining",
        random_state=None,
    ):
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError(f"`eps` must be > 0, got {self.eps}.")

        if sample_score not in {"expected", "max"}:
            raise ValueError(
                "`sample_score` must be one of {'expected', 'max'}, "
                f"got {sample_score!r}."
            )
        self.sample_score = sample_score

        if exploration_distribution not in {"uniform", "utility"}:
            raise ValueError(
                "`exploration_distribution` must be one of "
                "{'uniform', 'utility'}, "
                f"got {exploration_distribution!r}."
            )
        self.exploration_distribution = exploration_distribution

        if exploration_budget_mode not in {
            "remaining",
            "local",
            "min",
            "geometric_mean",
        }:
            raise ValueError(
                "`exploration_budget_mode` must be one of "
                "{'remaining', 'local', 'min', 'geometric_mean'}, "
                f"got {exploration_budget_mode!r}."
            )
        self.exploration_budget_mode = exploration_budget_mode

        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    def _assign(
        self,
        utilities,
        sample_indices,
        annotator_indices,
        budget,
        remaining_budget=None,
    ):
        S, A = utilities.shape
        if budget == 0 or S == 0 or A == 0:
            return np.empty((0, 2), dtype=int)

        rng = self._rng

        if remaining_budget is None:
            remaining_budget = budget
        remaining_budget = self._validate_budget_scalar(
            remaining_budget, name="remaining_budget"
        )

        b_local = budget / A
        b_remaining = remaining_budget / A
        b_effective = self._combine_budget_per_annotator(
            b_local=b_local,
            b_remaining=b_remaining,
        )

        row_infos = []
        row_scores = np.full(S, -np.inf, dtype=float)

        for i in range(S):
            info = self._build_row_policy(
                utilities[i],
                budget_per_annotator=b_effective,
                rng=rng,
            )
            row_infos.append(info)

            if info is None:
                continue

            if self.sample_score == "expected":
                row_scores[i] = info["expected_utility"]
            else:  # "max"
                row_scores[i] = info["best_utility"]

        valid_rows = np.flatnonzero(np.isfinite(row_scores))
        if valid_rows.size == 0:
            return np.empty((0, 2), dtype=int)

        n_select = min(budget, valid_rows.size)
        selected_rows = self._argsort_desc_with_random_ties(
            row_scores, rng=rng, valid_only=True
        )[:n_select]

        query_pairs = np.empty((n_select, 2), dtype=int)
        for out_idx, i in enumerate(selected_rows):
            info = row_infos[i]
            chosen_local_annotator = info["feasible_cols"][
                rng.choice(len(info["feasible_cols"]), p=info["policy"])
            ]
            query_pairs[out_idx, 0] = sample_indices[i]
            query_pairs[out_idx, 1] = annotator_indices[chosen_local_annotator]

        return query_pairs

    def _build_row_policy(
        self,
        row_utilities: np.ndarray,
        *,
        budget_per_annotator: float,
        rng: np.random.Generator,
    ):
        feasible_cols = np.flatnonzero(np.isfinite(row_utilities))
        if feasible_cols.size == 0:
            return None

        u = np.asarray(row_utilities[feasible_cols], dtype=float)

        if np.any((u < -self.eps) | (u > 1.0 + self.eps)):
            raise ValueError(
                "This assigner expects utilities in [0, 1]. Found row values "
                f"outside that range: {u}."
            )

        u = np.clip(u, 0.0, 1.0)
        n_feasible = len(feasible_cols)

        if n_feasible == 1:
            policy = np.array([1.0], dtype=float)
            return {
                "feasible_cols": feasible_cols,
                "policy": policy,
                "entropy": 0.0,
                "exploration": 0.0,
                "k_top": 1,
                "best_utility": float(u[0]),
                "expected_utility": float(u[0]),
            }

        # Normalized row entropy h_i in [0, 1].
        mass = u + self.eps
        weights = mass / mass.sum()
        entropy = -np.sum(weights * np.log(weights)) / np.log(n_feasible)
        entropy = float(np.clip(entropy, 0.0, 1.0))

        # Exploration level e_i = sqrt(b_eff)/(1+sqrt(b_eff)) * h_i.
        root_b = np.sqrt(max(budget_per_annotator, 0.0))
        exploration = ((root_b / (1.0 + root_b)) * entropy)**3 * 0.1
        exploration = float(np.clip(exploration, 0.0, 1.0))

        # Top-k size.
        k_top = 1 + int(np.ceil((n_feasible - 1) * exploration))
        k_top = max(1, min(k_top, n_feasible))

        order = self._argsort_desc_with_random_ties(u, rng=rng, valid_only=False)
        best_pos = int(order[0])
        topk_pos = np.asarray(order[:k_top], dtype=int)

        policy = np.zeros(n_feasible, dtype=float)
        policy[best_pos] += 1.0 - exploration

        if self.exploration_distribution == "uniform":
            policy[topk_pos] += exploration / k_top
        else:  # "utility"
            topk_mass = u[topk_pos] + self.eps
            topk_mass = topk_mass / topk_mass.sum()
            policy[topk_pos] += exploration * topk_mass

        policy = np.maximum(policy, 0.0)
        policy = policy / policy.sum()

        expected_utility = float(np.dot(policy, u))
        best_utility = float(u[best_pos])

        return {
            "feasible_cols": feasible_cols,
            "policy": policy,
            "entropy": entropy,
            "exploration": exploration,
            "k_top": k_top,
            "best_utility": best_utility,
            "expected_utility": expected_utility,
        }

    def _combine_budget_per_annotator(
        self,
        *,
        b_local: float,
        b_remaining: float,
    ) -> float:
        if self.exploration_budget_mode == "remaining":
            return b_remaining
        if self.exploration_budget_mode == "local":
            return b_local
        if self.exploration_budget_mode == "min":
            return min(b_local, b_remaining)
        if self.exploration_budget_mode == "geometric_mean":
            return float(np.sqrt(max(b_local, 0.0) * max(b_remaining, 0.0)))

        raise RuntimeError(
            f"Unknown exploration_budget_mode: {self.exploration_budget_mode!r}"
        )

    @staticmethod
    def _argsort_desc_with_random_ties(
        values: np.ndarray,
        *,
        rng: np.random.Generator,
        valid_only: bool,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float)

        if valid_only:
            idx = np.flatnonzero(np.isfinite(values))
        else:
            idx = np.arange(len(values))

        if idx.size == 0:
            return idx

        perm = rng.permutation(idx)
        return perm[np.argsort(-values[perm], kind="mergesort")]

    @staticmethod
    def _validate_budget_scalar(value, *, name: str) -> float:
        value = float(value)
        if value < 0:
            raise ValueError(f"`{name}` must be >= 0, got {value}.")
        return value

    def _resolve_rng(self, rng):
        if rng is None:
            return self._rng
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)