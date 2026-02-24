import numpy as np

from ._base import PairAssigner


class QuotaPairAssigner(PairAssigner):
    """
    Greedy assigner that balances labels per annotator using historical counts.

    At each selection step, the assigner prioritizes annotators with the lowest
    (historical_count + batch_count) among annotators that still have feasible
    pairs. Within that restricted set, it selects the next pair either by:
      - preferring sample coverage (`prefer="coverage"`), or
      - maximizing overall utility (`prefer="utility"`).

    Parameters
    ----------
    coverage : {"none", "hard"}, default="utility"
        Tie-breaking preference *after* balancing annotators:
        - "none": pick the highest-utility feasible pair among the
          most-underused annotators.
        - "hard": among the most-underused annotators, prefer samples with
          the fewest labels assigned within this batch (0 first, then 1, ...);
          then pick the best utility.
    """

    def __init__(self, coverage="utility"):
        prefer = str(coverage)
        if prefer not in {"none", "hard"}:
            raise ValueError(
                f"`prefer` must be 'none' or 'hard', got {prefer!r}."
            )
        self.coverage = coverage

    def _assign(
        self,
        sample_indices,
        annotator_indices,
        utilities,
        budget,
        annotator_label_counts=None,
    ):
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        U = np.asarray(utilities, dtype=float).copy()

        S, A = U.shape
        budget = int(budget)
        if budget <= 0 or S == 0 or A == 0:
            return np.empty((0, 2), dtype=int)

        # Map historical counts to LOCAL annotator order
        hist = self._coerce_annotator_counts(
            annotator_indices, annotator_label_counts
        )

        batch_a = np.zeros(A, dtype=int)  # labels per annotator in this batch
        batch_s = np.zeros(S, dtype=int)  # labels per sample in this batch

        selected = []

        for _ in range(budget):
            feasible = ~np.isnan(U)
            if not feasible.any():
                break

            feasible_cols = np.flatnonzero(feasible.any(axis=0))
            if feasible_cols.size == 0:
                break

            eff_counts = hist + batch_a
            min_eff = eff_counts[feasible_cols].min()
            cand_cols = feasible_cols[eff_counts[feasible_cols] == min_eff]

            # Choose (s, a) among candidate annotators
            if self.coverage == "none":
                s_loc, a_loc = self._pick_max_utility_pair(U, cand_cols)
            else:  # prefer == "hard"
                s_loc, a_loc = self._pick_coverage_then_utility_pair(
                    U, cand_cols, batch_s
                )

            if s_loc is None:
                break  # no feasible pair among candidates (should be rare)

            selected.append(
                (int(sample_indices[s_loc]), int(annotator_indices[a_loc]))
            )

            # Remove selected pair (no replacement)
            U[s_loc, a_loc] = np.nan
            batch_a[a_loc] += 1
            batch_s[s_loc] += 1

        return np.asarray(selected, dtype=int).reshape(-1, 2)

    @staticmethod
    def _coerce_annotator_counts(annotator_indices, annotator_label_counts):
        A = len(annotator_indices)
        if annotator_label_counts is None:
            return np.zeros(A, dtype=int)

        # Case 1: dict mapping global annotator id -> count
        if isinstance(annotator_label_counts, dict):
            out = np.zeros(A, dtype=int)
            for j, a_glob in enumerate(annotator_indices):
                out[j] = int(annotator_label_counts.get(int(a_glob), 0))
            if (out < 0).any():
                raise ValueError(
                    "Annotator label counts must be non-negative."
                )
            return out

        # Case 2: array-like aligned with annotator_indices
        arr = np.asarray(annotator_label_counts)
        if arr.shape != (A,):
            raise ValueError(
                f"`annotator_label_counts` must be dict or array of shape "
                f"({A},), got {arr.shape}."
            )
        arr = arr.astype(int, copy=False)
        if (arr < 0).any():
            raise ValueError("Annotator label counts must be non-negative.")
        return arr

    @staticmethod
    def _pick_max_utility_pair(U, cand_cols):
        sub = U[:, cand_cols]
        if np.all(np.isnan(sub)):
            return None, None
        flat = int(np.nanargmax(sub))
        s_loc, a_loc_sub = np.unravel_index(flat, sub.shape)
        a_loc = int(cand_cols[a_loc_sub])
        return int(s_loc), a_loc

    @staticmethod
    def _pick_coverage_then_utility_pair(U, cand_cols, batch_s):
        # Feasible samples (within candidate annotators)
        feas_row = ~np.isnan(U[:, cand_cols]).all(axis=1)
        rows = np.flatnonzero(feas_row)
        if rows.size == 0:
            return None, None

        # Prefer samples with minimal batch count
        min_c = batch_s[rows].min()
        rows = rows[batch_s[rows] == min_c]

        sub = U[rows][:, cand_cols]
        if np.all(np.isnan(sub)):
            # Defensive: if chosen rows have no feasible entries, fall back to
            # utility-only within cand_cols
            return PairAssigner._pick_max_utility_pair(U, cand_cols)

        flat = int(np.nanargmax(sub))
        r_local, a_local = np.unravel_index(flat, sub.shape)
        s_loc = int(rows[r_local])
        a_loc = int(cand_cols[a_local])
        return s_loc, a_loc
