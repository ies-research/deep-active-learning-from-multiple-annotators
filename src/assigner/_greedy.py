import numpy as np
from sklearn.utils import check_random_state

from ._base import PairAssigner


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
    selection : {"greedy","epsilon_greedy","softmax"}, default="greedy"

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
        random_state=None,
    ):
        selection = str(selection)
        coverage = str(coverage)

        if selection not in {"greedy", "epsilon_greedy", "softmax"}:
            raise ValueError(f"Invalid selection={selection!r}.")
        if coverage not in {"none", "hard", "soft"}:
            raise ValueError(f"Invalid coverage={coverage!r}.")

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

            rng = self.random_state
            sel = []

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

                else:  # softmax
                    flat_idx = self._sample_softmax(
                        score, feasible, rng, temperature=temp
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

    def _sample_uniform_or_topm(self, score, feasible, rng):
        feas_idx = np.flatnonzero(feasible.ravel())
        if feas_idx.size == 0:
            return 0

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
