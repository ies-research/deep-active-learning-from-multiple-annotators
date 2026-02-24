import numpy as np

from sklearn.utils import check_random_state
from ._base import PairScorer


class RandomPairScorer(PairScorer):
    """
    Random utility for (sample, annotator) pairs.

    Utilities are sampled i.i.d. from a uniform distribution on ``[low, high)``.
    Pairs masked as unavailable receive ``fill_value`` (default: ``np.nan``).

    Parameters
    ----------
    low : float, default=0.0
        Lower bound of the uniform distribution (inclusive).
    high : float, default=1.0
        Upper bound of the uniform distribution (exclusive).
    fill_value : float, default=-np.inf
        Utility value assigned to unavailable pairs when ``available_mask`` is
        given. Using ``-np.inf`` makes downstream argmax/greedy selection safe.
    random_state : None or int, default=None
        Seed for reproducible randomness.

    Notes
    -----
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    """

    def __init__(self, low=0.0, high=1.0, random_state=None):
        self.low = float(low)
        self.high = float(high)
        self.random_state = check_random_state(random_state)

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        **kwargs,
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
        **kwargs : dict
            May contain ``rng : numpy.random.Generator`` to override the internal RNG.

        Returns
        -------
        utilities : numpy.ndarray of shape (n_sel_samples, n_sel_annotators)
            Random utilities.
        """
        S = len(sample_indices)
        A = len(annotator_indices)
        U = self.random_state.uniform(self.low, self.high, size=(S, A)).astype(
            float, copy=False
        )
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)

        return U
