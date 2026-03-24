import numpy as np

from ._base import PairScorer


class PerformancePairScorer(PairScorer):
    """
    Pair scorer that uses estimated annotator correctness directly.

    For each candidate sample, this scorer calls ``clf.predict_proba`` with
    ``extra_outputs="annotator_perf"`` and returns the estimated probability
    that each candidate annotator labels the sample correctly.
    """

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf=None,
        **kwargs,
    ):
        del kwargs
        if clf is None:
            raise ValueError("`clf` must be provided.")

        X_cand = X[sample_indices]
        out = clf.predict_proba(X_cand, extra_outputs="annotator_perf")
        if not isinstance(out, tuple) or len(out) < 2:
            raise ValueError(
                "clf.predict_proba must return a tuple when "
                "`extra_outputs='annotator_perf'` is requested."
            )

        annotator_perf = np.asarray(out[1], dtype=float)
        annotator_perf = self._take_selected_annotators(
            annotator_perf,
            axis=1,
            annotator_indices=annotator_indices,
            n_annotators_total=y.shape[1],
            name="annotator_perf",
        )

        if available_mask is not None:
            annotator_perf = np.where(
                available_mask, annotator_perf, np.nan
            )
        return annotator_perf

    @staticmethod
    def _take_selected_annotators(
        arr,
        *,
        axis,
        annotator_indices,
        n_annotators_total,
        name,
    ):
        if arr.shape[axis] == len(annotator_indices):
            return arr
        if arr.shape[axis] == n_annotators_total:
            return np.take(arr, indices=annotator_indices, axis=axis)
        raise ValueError(
            f"{name} has incompatible annotator axis size {arr.shape[axis]}; "
            f"expected {len(annotator_indices)} or {n_annotators_total}."
        )
