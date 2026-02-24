import numpy as np

from abc import ABC, abstractmethod


class PairAssigner(ABC):
    """
    Base class for selecting (sample, annotator) query pairs given utilities.

    The assigner consumes a utility matrix over candidate samples and candidate
    annotators and returns a set of selected query pairs as global indices.

    Notes
    -----
    - Utilities are assumed to be in *local* matrix order induced by
      `sample_indices` and `annotator_indices`, i.e., row i corresponds to
      `sample_indices[i]` and column j corresponds to `annotator_indices[j]`.
    - `available_mask` is a feasibility mask in the same local coordinates.
    - The returned pairs must always be global indices referencing the original
      sample/annotator ids.

    """

    def __call__(
        self,
        utilities,
        *,
        sample_indices=None,
        annotator_indices=None,
        budget=1,
        **kwargs,
    ):
        """
        Select query pairs.

        Parameters
        ----------
        sample_indices : array-like of shape (n_sel_samples,), dtype=int
            Global indices of candidate samples.
        annotator_indices : array-like of shape (n_sel_annotators,), dtype=int
            Global indices of candidate annotators.
        utilities : array-like of shape (n_sel_samples, n_sel_annotators)
            Utility matrix in local order. Higher values indicate more desirable
            pairs.
        budget : int, default=1
            Maximum number of (sample, annotator) pairs to return.
        **kwargs : dict
            Additional keyword arguments for concrete assigners (e.g., rng).

        Returns
        -------
        query_indices : numpy.ndarray of shape (k, 2), dtype=int
            Selected query pairs as global indices, where `k <= budget`.
            Column 0 contains sample indices, column 1 contains annotator indices.

        """
        utilities = np.asarray(utilities, dtype=float)
        sample_indices = (
            np.arange(utilities.shape[0])
            if sample_indices is None
            else np.asarray(sample_indices)
        )
        annotator_indices = (
            np.arange(utilities.shape[1])
            if annotator_indices is None
            else np.asarray(annotator_indices)
        )

        S = len(sample_indices)
        A = len(annotator_indices)

        if utilities.shape != (S, A):
            raise ValueError(
                f"`utilities` must have shape {(S, A)}, got {utilities.shape}."
            )

        budget = int(budget)
        if budget < 0:
            raise ValueError(f"`budget` must be >= 0, got {budget}.")

        return self._assign(
            utilities=utilities,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            budget=budget,
            **kwargs,
        )

    @abstractmethod
    def _assign(
        self,
        utilities,
        sample_indices,
        annotator_indices,
        budget,
        **kwargs,
    ):
        """
        Implementation of pair assignment.

        Parameters
        ----------
        sample_indices : numpy.ndarray of shape (n_sel_samples,), dtype=int
            See ``__call__``.
        annotator_indices : numpy.ndarray of shape (n_sel_annotators,), dtype=int
            See ``__call__``.
        utilities : numpy.ndarray of shape (n_sel_samples, n_sel_annotators)
            See ``__call__``.
        available_mask : numpy.ndarray of shape (n_sel_samples, n_sel_annotators), dtype=bool
            See ``__call__``.
        budget : int
            See ``__call__``.
        **kwargs : dict
            See ``__call__``.

        Returns
        -------
        query_indices : numpy.ndarray of shape (k, 2), dtype=int
            See ``__call__``.

        """
        raise NotImplementedError
