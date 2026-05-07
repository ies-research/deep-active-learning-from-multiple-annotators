import numpy as np
from sklearn.utils import check_random_state

from ._base import PairAssigner
from ._constraints import coerce_annotator_vector


class _CosineCallSchedule:
    """
    Cosine schedule evaluated once per call.

    value(t) = end + (start-end) * 0.5 * (1 + cos(pi * min(t, T) / T))

    After T calls, it stays at `end`.
    If T <= 0, it stays at `start`.
    """

    def __init__(self, start: float, end: float, T: int, kind: str = "cosine"):
        self.start = float(start)
        self.end = float(end)
        self.T = int(T)
        self.kind = str(kind)
        self.t = 0  # number of calls already *used*

        if self.kind not in {"constant", "cosine"}:
            raise ValueError(f"Unknown schedule kind: {self.kind!r}.")

    def value(self) -> float:
        if self.kind == "constant" or self.T <= 0:
            return float(self.start)

        # clamp to [0, T] so it saturates at end
        x = int(np.clip(self.t, 0, self.T))
        return float(
            self.end
            + (self.start - self.end)
            * 0.5
            * (1.0 + np.cos(np.pi * (x / self.T)))
        )

    def step(self) -> None:
        self.t += 1

    def reset(self, t0: int = 0) -> None:
        self.t = int(t0)


class GreedyPairAssigner(PairAssigner):
    """
    Greedy assigner for selecting (sample, annotator) pairs from a utility matrix.

    Key point:
    - epsilon/temperature cosine annealing happens *across calls* to `_assign`.

    Parameters (annealed across calls)
    ---------------------------------
    selection : {"greedy","epsilon_greedy","softmax", /
            "posterior_best_annotator"}, default="greedy"

    epsilon_max, epsilon_min : float
        Epsilon endpoints for epsilon-greedy.
    epsilon_T : int, default=100
        Number of `_assign` calls over which epsilon goes from max -> min.
        After that, it stays at epsilon_min.
    epsilon_schedule : {"constant","cosine"}, default="cosine"

    temperature_max, temperature_min : float
        Temperature endpoints for softmax.
    temperature_T : int, default=100
        Number of `_assign` calls over which temperature goes from max -> min.
    temperature_schedule : {"constant","cosine"}, default="cosine"
    explore_top_m : int or None, default=None
        Restrict epsilon exploration to the best M candidates. The scope is
        controlled by `explore_top_m_scope`.
    explore_top_m_scope : {"pair","annotator"}, default="pair"
        If "pair", keep the best M feasible sample-annotator pairs globally.
        If "annotator", keep the best M feasible annotators within each
        feasible sample row, then sample uniformly over the retained pairs.
    """

    def __init__(
        self,
        selection="greedy",
        # epsilon annealing across calls
        epsilon_max=0.2,
        epsilon_min=None,
        epsilon_T=100,
        epsilon_schedule="cosine",
        # temperature annealing across calls
        temperature_max=1.0,
        temperature_min=None,  # if None -> same as max
        temperature_T=100,
        temperature_schedule="cosine",
        coverage="none",
        soft_coverage_lambda=0.0,
        max_per_sample=None,
        max_per_annotator=None,
        explore_top_m=None,
        explore_top_m_scope="pair",
        posterior_best_temperature=1.0,
        posterior_best_floor=0.0,
        random_state=None,
    ):
        selection = str(selection)
        coverage = str(coverage)
        explore_top_m_scope = str(explore_top_m_scope)

        if selection not in {
            "greedy",
            "epsilon_greedy",
            "softmax",
            "posterior_best_annotator",
        }:
            raise ValueError(f"Invalid selection={selection!r}.")
        if coverage not in {"none", "hard", "soft"}:
            raise ValueError(f"Invalid coverage={coverage!r}.")
        if explore_top_m_scope not in {"pair", "annotator"}:
            raise ValueError(
                "explore_top_m_scope must be one of {'pair', 'annotator'}."
            )

        eps_max = float(epsilon_max)
        eps_min = eps_max if epsilon_min is None else float(epsilon_min)
        if not (0.0 <= eps_min <= eps_max <= 1.0):
            raise ValueError("Require 0 <= epsilon_min <= epsilon_max <= 1.")

        tmax = float(temperature_max)
        tmin = tmax if temperature_min is None else float(temperature_min)
        if tmax <= 0.0 or tmin <= 0.0:
            raise ValueError("temperature_max/min must be > 0.")

        if float(soft_coverage_lambda) < 0.0:
            raise ValueError("soft_coverage_lambda must be >= 0.")

        if max_per_sample is not None and int(max_per_sample) <= 0:
            raise ValueError("max_per_sample must be positive or None.")
        if max_per_annotator is not None and int(max_per_annotator) <= 0:
            raise ValueError("max_per_annotator must be positive or None.")
        if explore_top_m is not None and int(explore_top_m) <= 0:
            raise ValueError("explore_top_m must be positive or None.")
        posterior_best_temperature = float(posterior_best_temperature)
        posterior_best_floor = float(posterior_best_floor)
        if posterior_best_temperature <= 0.0:
            raise ValueError("posterior_best_temperature must be > 0.")
        if posterior_best_floor < 0.0:
            raise ValueError("posterior_best_floor must be >= 0.")

        self.selection = selection
        self.coverage = coverage
        self.soft_coverage_lambda = float(soft_coverage_lambda)
        self.max_per_sample = (
            None if max_per_sample is None else int(max_per_sample)
        )
        self.max_per_annotator = (
            None if max_per_annotator is None else int(max_per_annotator)
        )
        self.explore_top_m = (
            None if explore_top_m is None else int(explore_top_m)
        )
        self.explore_top_m_scope = explore_top_m_scope
        self.posterior_best_temperature = posterior_best_temperature
        self.posterior_best_floor = posterior_best_floor
        self.random_state = check_random_state(random_state)

        # Call-based schedules (stateful)
        self._eps_sched = _CosineCallSchedule(
            start=eps_max,
            end=eps_min,
            T=int(epsilon_T),
            kind=str(epsilon_schedule),
        )
        self._temp_sched = _CosineCallSchedule(
            start=tmax,
            end=tmin,
            T=int(temperature_T),
            kind=str(temperature_schedule),
        )

    @property
    def n_assign_calls_(self) -> int:
        """How many times `_assign` has been called (counting completed calls)."""
        return int(self._eps_sched.t)

    def reset_annealing(self, call_index: int = 0) -> None:
        """Reset call-based annealing counters."""
        self._eps_sched.reset(call_index)
        self._temp_sched.reset(call_index)

    def state_dict(self) -> dict:
        """Minimal state for reproducibility/checkpointing."""
        return {
            "eps_t": int(self._eps_sched.t),
            "temp_t": int(self._temp_sched.t),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore annealing counters."""
        self._eps_sched.reset(int(state.get("eps_t", 0)))
        self._temp_sched.reset(int(state.get("temp_t", 0)))

    def _assign(
        self, sample_indices, annotator_indices, utilities, budget, **kwargs
    ):
        # Evaluate schedules ONCE per call
        eps = self._eps_sched.value()
        temp = max(self._temp_sched.value(), 1e-12)

        # Always advance schedules once per call, even if we early-return.
        # If you hate this behavior, you can move step() to just before return.
        try:
            sample_indices = np.asarray(sample_indices, dtype=int)
            annotator_indices = np.asarray(annotator_indices, dtype=int)
            U = np.asarray(utilities, dtype=float).copy()

            S, A = U.shape
            budget = int(budget)
            if budget <= 0 or S == 0 or A == 0:
                return np.empty((0, 2), dtype=int)
            utility_draws = self._validate_utility_draws(
                kwargs.get("utility_draws"),
                U.shape,
            )

            rng = self.random_state
            sel = []
            remaining = self._coerce_annotator_remaining(
                annotator_indices, kwargs.get("annotator_remaining_counts")
            )

            # Per-batch counts (coverage/caps)
            c_s = np.zeros(S, dtype=int)
            c_a = np.zeros(A, dtype=int)

            for _ in range(budget):
                feasible = ~np.isnan(U)

                # Apply caps by masking feasibility
                if self.max_per_sample is not None:
                    feasible &= c_s[:, None] < self.max_per_sample
                if self.max_per_annotator is not None:
                    feasible &= c_a[None, :] < self.max_per_annotator
                if remaining is not None:
                    feasible &= c_a[None, :] < remaining[None, :]

                if not feasible.any():
                    break

                score = U.copy()

                if self.coverage == "soft" and self.soft_coverage_lambda > 0.0:
                    score = score - self.soft_coverage_lambda * c_s[:, None]

                score[~feasible] = -np.inf

                if self.coverage == "hard":
                    feas_rows = np.flatnonzero(feasible.any(axis=1))
                    min_count = c_s[feas_rows].min()
                    rows = feas_rows[c_s[feas_rows] == min_count]

                    row_ok = np.zeros(S, dtype=bool)
                    row_ok[rows] = True
                    feasible &= row_ok[:, None]
                    score[~feasible] = -np.inf

                if self.selection == "greedy":
                    flat_idx = int(np.argmax(score))
                    if not np.isfinite(score.ravel()[flat_idx]):
                        break

                elif self.selection == "epsilon_greedy":
                    if rng.rand() < eps:
                        flat_idx = self._sample_uniform_or_topm(
                            score, feasible, rng
                        )
                    else:
                        flat_idx = int(np.argmax(score))
                        if not np.isfinite(score.ravel()[flat_idx]):
                            break

                elif self.selection == "softmax":
                    flat_idx = self._sample_softmax(
                        score, feasible, rng, temperature=temp
                    )
                else:  # posterior_best_annotator
                    flat_idx = self._sample_posterior_best_annotator(
                        score=score,
                        feasible=feasible,
                        utility_draws=utility_draws,
                        rng=rng,
                    )

                s_loc, a_loc = np.unravel_index(flat_idx, (S, A))
                sel.append(
                    (int(sample_indices[s_loc]), int(annotator_indices[a_loc]))
                )

                U[s_loc, a_loc] = np.nan
                c_s[s_loc] += 1
                c_a[a_loc] += 1

            return np.asarray(sel, dtype=int).reshape(-1, 2)

        finally:
            self._eps_sched.step()
            self._temp_sched.step()

    @staticmethod
    def _coerce_annotator_remaining(
        annotator_indices, annotator_remaining_counts
    ):
        return coerce_annotator_vector(
            annotator_indices,
            annotator_remaining_counts,
            name="annotator_remaining_counts",
        )

    def _sample_uniform_or_topm(self, score, feasible, rng):
        feas_idx = np.flatnonzero(feasible.ravel())
        if feas_idx.size == 0:
            return 0

        if (
            self.explore_top_m is not None
            and self.explore_top_m_scope == "annotator"
        ):
            return self._sample_uniform_over_row_topm_annotators(
                score,
                feasible,
                rng,
            )

        if (
            self.explore_top_m is not None
            and feas_idx.size > self.explore_top_m
        ):
            feas_scores = score.ravel()[feas_idx]
            top = np.argpartition(feas_scores, -self.explore_top_m)[
                -self.explore_top_m :
            ]
            feas_idx = feas_idx[top]

        return int(feas_idx[rng.randint(feas_idx.size)])

    def _sample_uniform_over_row_topm_annotators(self, score, feasible, rng):
        score = np.asarray(score, dtype=float)
        feasible = np.asarray(feasible, dtype=bool)
        S, A = score.shape
        keep = np.zeros_like(feasible, dtype=bool)
        m = int(self.explore_top_m)

        for s in np.flatnonzero(feasible.any(axis=1)):
            a_idx = np.flatnonzero(feasible[s])
            if a_idx.size <= m:
                keep[s, a_idx] = True
                continue

            row_scores = score[s, a_idx]
            top = np.argpartition(row_scores, -m)[-m:]
            keep[s, a_idx[top]] = True

        top_idx = np.flatnonzero(keep.ravel())
        if top_idx.size == 0:
            return 0
        return int(top_idx[rng.randint(top_idx.size)])

    def _sample_softmax(self, score, feasible, rng, temperature: float):
        feas_idx = np.flatnonzero(feasible.ravel())
        if feas_idx.size == 0:
            return 0

        feas_scores = score.ravel()[feas_idx]
        finite = np.isfinite(feas_scores)
        if not finite.any():
            return int(feas_idx[rng.randint(feas_idx.size)])

        feas_idx = feas_idx[finite]
        feas_scores = feas_scores[finite]

        logits = feas_scores / float(max(temperature, 1e-12))
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        probs = exps / np.sum(exps)
        j = int(rng.choice(feas_idx.size, p=probs))
        return int(feas_idx[j])

    def _validate_utility_draws(self, utility_draws, shape):
        if self.selection != "posterior_best_annotator":
            return None
        if utility_draws is None:
            raise ValueError(
                "selection='posterior_best_annotator' requires utility_draws."
            )
        utility_draws = np.asarray(utility_draws, dtype=float)
        if utility_draws.ndim != 3:
            raise ValueError(
                "utility_draws must have shape "
                "(n_samples, n_annotators, n_draws)."
            )
        if utility_draws.shape[:2] != tuple(shape):
            raise ValueError(
                "utility_draws must agree with utilities on samples and "
                f"annotators, got {utility_draws.shape[:2]} and {shape}."
            )
        if utility_draws.shape[2] < 2:
            raise ValueError(
                "posterior_best_annotator requires at least two utility draws."
            )
        return utility_draws

    def _sample_posterior_best_annotator(
        self,
        *,
        score,
        feasible,
        utility_draws,
        rng,
    ):
        row_score = np.full(score.shape[0], -np.inf, dtype=float)
        for s in np.flatnonzero(feasible.any(axis=1)):
            row_values = score[s, feasible[s]]
            finite = np.isfinite(row_values)
            if finite.any():
                row_score[s] = float(np.max(row_values[finite]))
        feasible_rows = np.flatnonzero(np.isfinite(row_score))
        if feasible_rows.size == 0:
            feas_idx = np.flatnonzero(feasible.ravel())
            return int(feas_idx[rng.randint(feas_idx.size)])

        max_row_score = np.max(row_score[feasible_rows])
        best_rows = feasible_rows[np.isclose(row_score[feasible_rows], max_row_score)]
        s_loc = int(best_rows[rng.randint(best_rows.size)])
        a_candidates = np.flatnonzero(feasible[s_loc])
        p_best = self._posterior_best_probabilities_for_row(
            utility_draws[s_loc],
            a_candidates,
        )
        p = p_best + self.posterior_best_floor
        if not np.isfinite(p).all() or p.sum() <= 0.0:
            p = np.ones_like(p, dtype=float)
        p = p / p.sum()
        logits = np.log(np.clip(p, 1e-300, None)) / self.posterior_best_temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        a_loc = int(a_candidates[int(rng.choice(a_candidates.size, p=probs))])
        return int(np.ravel_multi_index((s_loc, a_loc), score.shape))

    @staticmethod
    def _posterior_best_probabilities_for_row(row_draws, a_candidates):
        draws = np.asarray(row_draws, dtype=float)[a_candidates]
        finite = np.isfinite(draws)
        valid_draw = finite.any(axis=0)
        if not np.any(valid_draw):
            return np.full(a_candidates.size, 1.0 / a_candidates.size)

        draws = draws[:, valid_draw]
        finite = finite[:, valid_draw]
        draws = np.where(finite, draws, -np.inf)
        max_draw = np.max(draws, axis=0, keepdims=True)
        is_best = np.isclose(draws, max_draw) & finite
        n_best = np.maximum(is_best.sum(axis=0, keepdims=True), 1)
        credit = is_best / n_best
        p_best = credit.mean(axis=1)
        if p_best.sum() <= 0.0:
            return np.full(a_candidates.size, 1.0 / a_candidates.size)
        return p_best / p_best.sum()
