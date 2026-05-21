import numpy as np

from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from ._base import PairAssigner
from ._constraints import coerce_annotator_vector


class OptimalPairAssigner(PairAssigner):
    """
    Exact utility-maximizing pair assigner under linear batch constraints.

    The assigner solves a binary linear program over the candidate
    sample-annotator grid. Higher utility values are preferred; infeasible pairs
    are represented by NaN utilities.
    """

    def __init__(
        self,
        *,
        min_per_sample=1,
        max_per_sample=None,
        min_per_annotator=None,
        max_per_annotator=None,
        max_per_annotator_mode="fixed",
        max_per_annotator_multiplier=1.0,
        budget_constraint="exact",
        on_infeasible="raise",
        time_limit=None,
        mip_rel_gap=None,
    ):
        max_per_annotator_mode = str(max_per_annotator_mode)
        budget_constraint = str(budget_constraint)
        on_infeasible = str(on_infeasible)

        if min_per_sample is not None and int(min_per_sample) < 0:
            raise ValueError("min_per_sample must be non-negative or None.")
        if max_per_sample is not None and int(max_per_sample) <= 0:
            raise ValueError("max_per_sample must be positive or None.")
        if min_per_annotator is not None and int(min_per_annotator) < 0:
            raise ValueError("min_per_annotator must be non-negative or None.")
        if max_per_annotator is not None and int(max_per_annotator) <= 0:
            raise ValueError("max_per_annotator must be positive or None.")
        if max_per_annotator_mode not in {"fixed", "relative_batch_share"}:
            raise ValueError(
                "max_per_annotator_mode must be one of "
                "{'fixed', 'relative_batch_share'}."
            )
        if float(max_per_annotator_multiplier) <= 0.0:
            raise ValueError("max_per_annotator_multiplier must be > 0.")
        if budget_constraint not in {"exact", "at_most"}:
            raise ValueError("budget_constraint must be one of {'exact', 'at_most'}.")
        if on_infeasible not in {"raise", "relax"}:
            raise ValueError("on_infeasible must be one of {'raise', 'relax'}.")
        if time_limit is not None and float(time_limit) <= 0.0:
            raise ValueError("time_limit must be positive or None.")
        if mip_rel_gap is not None and float(mip_rel_gap) < 0.0:
            raise ValueError("mip_rel_gap must be non-negative or None.")

        self.min_per_sample = (
            None if min_per_sample is None else int(min_per_sample)
        )
        self.max_per_sample = (
            None if max_per_sample is None else int(max_per_sample)
        )
        self.min_per_annotator = (
            None if min_per_annotator is None else int(min_per_annotator)
        )
        self.max_per_annotator = (
            None if max_per_annotator is None else int(max_per_annotator)
        )
        self.max_per_annotator_mode = max_per_annotator_mode
        self.max_per_annotator_multiplier = float(max_per_annotator_multiplier)
        self.budget_constraint = budget_constraint
        self.on_infeasible = on_infeasible
        self.time_limit = None if time_limit is None else float(time_limit)
        self.mip_rel_gap = None if mip_rel_gap is None else float(mip_rel_gap)
        self.last_result_ = None
        self.last_relaxed_ = False
        self.last_constraint_pressure_ = None

    def _assign(
        self,
        sample_indices,
        annotator_indices,
        utilities,
        budget,
        annotator_remaining_counts=None,
        **kwargs,
    ):
        del kwargs
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        U = np.asarray(utilities, dtype=float)

        S, A = U.shape
        budget = int(budget)
        if budget <= 0 or S == 0 or A == 0:
            self.last_result_ = None
            self.last_relaxed_ = False
            return np.empty((0, 2), dtype=int)

        feasible = np.isfinite(U)
        if not feasible.any():
            if self.budget_constraint == "exact" and self.on_infeasible == "raise":
                raise RuntimeError("OptimalPairAssigner has no feasible pairs.")
            self.last_result_ = None
            self.last_relaxed_ = self.on_infeasible == "relax"
            return np.empty((0, 2), dtype=int)

        remaining = self._coerce_annotator_remaining(
            annotator_indices,
            annotator_remaining_counts,
        )
        row_lower, row_upper = self._sample_bounds(feasible)
        col_lower, col_upper = self._annotator_bounds(
            feasible=feasible,
            budget=budget,
            remaining=remaining,
        )

        target_budget = budget
        if self.on_infeasible == "relax" and self.budget_constraint == "exact":
            target_budget = min(
                target_budget,
                int(feasible.sum()),
                int(row_upper.sum()),
                int(col_upper.sum()),
            )

        attempts = [(row_lower, col_lower, target_budget, False)]
        if self.on_infeasible == "relax":
            relaxed_row_lower = np.minimum(row_lower, row_upper)
            relaxed_col_lower = np.minimum(col_lower, col_upper)
            attempts.append((relaxed_row_lower, relaxed_col_lower, target_budget, True))
            attempts.append(
                (
                    np.zeros_like(row_lower),
                    np.zeros_like(col_lower),
                    target_budget,
                    True,
                )
            )

        last_result = None
        for row_lb, col_lb, target, relaxed in attempts:
            if self.budget_constraint == "exact" and target < 0:
                continue
            result = self._solve_milp(
                U=U,
                feasible=feasible,
                row_lower=row_lb,
                row_upper=row_upper,
                col_lower=col_lb,
                col_upper=col_upper,
                budget=target,
            )
            last_result = result
            if result.success:
                self.last_result_ = result
                self.last_relaxed_ = bool(relaxed)
                X = result.x.reshape(S, A)
                selected = np.argwhere(X > 0.5)
                return np.asarray(
                    [
                        (int(sample_indices[s]), int(annotator_indices[a]))
                        for s, a in selected
                    ],
                    dtype=int,
                ).reshape(-1, 2)

        self.last_result_ = last_result
        self.last_relaxed_ = self.on_infeasible == "relax"
        message = "OptimalPairAssigner MILP was infeasible."
        if last_result is not None:
            message += f" status={last_result.status}, message={last_result.message}"
        if self.on_infeasible == "raise":
            raise RuntimeError(message)
        return np.empty((0, 2), dtype=int)

    def _solve_milp(
        self,
        *,
        U,
        feasible,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
        budget,
    ):
        S, A = U.shape
        n_vars = S * A
        utilities = np.where(feasible, U, 0.0)
        c = -utilities.reshape(-1)
        bounds = Bounds(
            lb=np.zeros(n_vars, dtype=float),
            ub=feasible.reshape(-1).astype(float),
        )
        integrality = np.ones(n_vars, dtype=int)

        n_constraints = 1 + S + A
        matrix = lil_matrix((n_constraints, n_vars), dtype=float)
        lb = np.zeros(n_constraints, dtype=float)
        ub = np.zeros(n_constraints, dtype=float)

        matrix[0, :] = 1.0
        if self.budget_constraint == "exact":
            lb[0] = float(budget)
            ub[0] = float(budget)
        else:
            lb[0] = 0.0
            ub[0] = float(budget)

        for s in range(S):
            row = 1 + s
            matrix[row, s * A : (s + 1) * A] = 1.0
            lb[row] = float(row_lower[s])
            ub[row] = float(row_upper[s])

        for a in range(A):
            row = 1 + S + a
            matrix[row, a::A] = 1.0
            lb[row] = float(col_lower[a])
            ub[row] = float(col_upper[a])

        options = {"disp": False}
        if self.time_limit is not None:
            options["time_limit"] = self.time_limit
        if self.mip_rel_gap is not None:
            options["mip_rel_gap"] = self.mip_rel_gap

        return milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=LinearConstraint(matrix.tocsr(), lb=lb, ub=ub),
            options=options,
        )

    def _sample_bounds(self, feasible):
        row_feasible = feasible.sum(axis=1).astype(int)
        lower = np.zeros(feasible.shape[0], dtype=int)
        if self.min_per_sample is not None:
            lower[:] = self.min_per_sample
        upper = row_feasible.copy()
        if self.max_per_sample is not None:
            upper = np.minimum(upper, self.max_per_sample)
        return lower, upper

    def _annotator_bounds(self, *, feasible, budget, remaining):
        col_feasible = feasible.sum(axis=0).astype(int)
        lower = np.zeros(feasible.shape[1], dtype=int)
        if self.min_per_annotator is not None:
            lower[:] = self.min_per_annotator
        upper = col_feasible.copy()
        max_per_annotator = self._resolve_max_per_annotator(
            budget=budget,
            n_annotators=feasible.shape[1],
        )
        if max_per_annotator is not None:
            upper = np.minimum(upper, max_per_annotator)
        if remaining is not None:
            upper = np.minimum(upper, remaining)
        return lower, upper

    def _resolve_max_per_annotator(self, *, budget: int, n_annotators: int):
        if self.max_per_annotator_mode == "fixed":
            return self.max_per_annotator
        if n_annotators <= 0:
            return None
        cap = int(
            np.ceil(
                self.max_per_annotator_multiplier
                * float(max(int(budget), 0))
                / float(n_annotators)
            )
        )
        return max(cap, 1)

    @staticmethod
    def _coerce_annotator_remaining(annotator_indices, annotator_remaining_counts):
        return coerce_annotator_vector(
            annotator_indices,
            annotator_remaining_counts,
            name="annotator_remaining_counts",
        )

    def constraint_pressure(
        self,
        *,
        budget,
        annotator_indices=None,
        annotator_remaining_counts=None,
        **kwargs,
    ) -> float:
        del kwargs
        budget = int(budget)
        if budget <= 0:
            return 0.0
        if annotator_indices is None:
            if annotator_remaining_counts is None:
                return 0.0
            annotator_indices = np.arange(len(annotator_remaining_counts))
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        if annotator_indices.size <= 1:
            return 0.0

        remaining = self._coerce_annotator_remaining(
            annotator_indices,
            annotator_remaining_counts,
        )
        upper = np.full(annotator_indices.size, budget, dtype=float)
        max_per_annotator = self._resolve_max_per_annotator(
            budget=budget,
            n_annotators=annotator_indices.size,
        )
        if max_per_annotator is not None:
            upper = np.minimum(upper, float(max_per_annotator))
        if remaining is not None:
            upper = np.minimum(upper, remaining.astype(float, copy=False))

        upper = upper[upper > 0.0]
        if upper.size <= 1:
            pressure = 0.0
        else:
            effective_budget = min(float(budget), float(upper.sum()))
            if effective_budget <= 0.0:
                pressure = 0.0
            else:
                min_needed = self._min_annotators_needed(
                    upper=upper,
                    effective_budget=effective_budget,
                )
                pressure = (min_needed - 1.0) / max(float(upper.size - 1), 1.0)
        self.last_constraint_pressure_ = float(np.clip(pressure, 0.0, 1.0))
        return self.last_constraint_pressure_

    @staticmethod
    def _min_annotators_needed(*, upper: np.ndarray, effective_budget: float) -> float:
        remaining = float(effective_budget)
        used = 0
        for capacity in np.sort(upper)[::-1]:
            if remaining <= 1e-12:
                break
            remaining -= float(capacity)
            used += 1
        return float(used)
