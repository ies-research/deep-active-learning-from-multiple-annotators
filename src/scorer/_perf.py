import numpy as np

from sklearn.utils import check_random_state
from ._base import PairScorer


class PerformancePairScorer(PairScorer):
    """
    Random utility for (sample, annotator) pairs.

    Utilities are sampled i.i.d. from a uniform distribution on ``[low, high)``.
    Pairs masked as unavailable receive ``fill_value`` (default: ``np.nan``).

    Notes
    -----
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    """

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf,
    ):
        """
        Parameters
        ----------
        X : array-like
            Unused (kept for API compatibility).
        y : array-like
            Unused (kept for API compatibility).
        sample_indices : array-like of shape (n_sel_samples,), dtype=int
            Selected sample indices (only length matters here).
        annotator_indices : array-like of shape (n_sel_annotators,), dtype=int
            Selected annotator indices (only length matters here).
        available_mask : None or array-like of shape (n_sel_samples, n_sel_annotators), dtype=bool
            Availability mask for candidate pairs.

        Returns
        -------
        utilities : numpy.ndarray of shape (n_sel_samples, n_sel_annotators)
            Random utilities.
        """
        X_cand = X[sample_indices]
        P, P_perf, P_annot = clf.predict_proba(
            X_cand, extra_outputs=["annotator_perf", "annotator_class"]
        )
        U = P_perf  # information_gain(P, P_perf, P_annot)
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)

        return U


