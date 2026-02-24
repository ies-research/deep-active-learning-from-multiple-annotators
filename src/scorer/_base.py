import numpy as np

from abc import ABC, abstractmethod


class PairScorer(ABC):
    """
    Base class for computing utilities of candidate (sample, annotator) pairs.

    Given a subset of candidate samples and candidate annotators, implementations
    return a utility matrix ``U`` where ``U[i, j]`` is the utility of querying
    annotator ``annotator_indices[j]`` for sample ``sample_indices[i]``.

    Notes
    -----
    - The returned utilities are in the *local* matrix order induced by the
      passed ``sample_indices`` and ``annotator_indices``. That is, row ``i``
      corresponds to ``sample_indices[i]`` and column ``j`` to
      ``annotator_indices[j]``.
    - ``available_mask`` is a feasibility mask in local coordinates. It may be
      used as an optimization hint (skip expensive computations) and/or to mark
      infeasible pairs (e.g., set them to ``-np.inf``).
    - Implementations must return a full matrix of shape
      ``(len(sample_indices), len(annotator_indices))``.

    """

    def __call__(
        self,
        X,
        y,
        *,
        sample_indices=None,
        annotator_indices=None,
        available_mask=None,
        **kwargs,
    ):
        """
        Compute utilities for candidate (sample, annotator) pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, ...)
            Training data set. The representation may be feature vectors,
            embeddings, or raw inputs depending on the downstream model.
        y : array-like of shape (n_samples, n_annotators)
            Current multi-annotator labels with missing labels encoded via a
            common convention (e.g., ``missing_label`` in the calling strategy).
        sample_indices : array-like of shape (n_sel_samples,), dtype=int
            Global indices of candidate samples for which utilities are computed.
            These refer to rows in ``X`` and ``y``.
        annotator_indices : array-like of shape (n_sel_annotators,), dtype=int
            Global indices of candidate annotators for which utilities are
            computed. These refer to columns in ``y``.
        available_mask : None or array-like of shape (n_sel_samples, n_sel_annotators), dtype=bool
            Boolean mask indicating which (sample, annotator) pairs are
            admissible candidates in the *local* block. Implementations may use
            this to skip computation for inadmissible pairs, but must still
            return a full utility matrix.
        **kwargs : dict
            Additional keyword arguments passed through by the calling query
            strategy (e.g., a fitted model, predicted probabilities, costs).

        Returns
        -------
        utilities : numpy.ndarray of shape (n_sel_samples, n_sel_annotators)
            Utility matrix in local order. Higher values indicate more desirable
            (sample, annotator) pairs.

        """
        sample_indices = (
            np.arange(y.shape[0])
            if sample_indices is None
            else np.asarray(sample_indices)
        )
        annotator_indices = (
            np.arange(y.shape[1])
            if annotator_indices is None
            else np.asarray(annotator_indices)
        )
        if available_mask is not None:
            available_mask = np.asarray(available_mask, dtype=bool)
            expected_shape = (len(sample_indices), len(annotator_indices))
            if available_mask.shape != expected_shape:
                raise ValueError(
                    f"`available_mask` must have shape "
                    f"{(len(sample_indices), len(annotator_indices))}, got "
                    f"{available_mask.shape}."
                )
        utilities = self._compute(
            X=X,
            y=y,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
            **kwargs,
        )

        utilities = np.asarray(utilities, dtype=float)
        if utilities.shape != (len(sample_indices), len(annotator_indices)):
            raise ValueError(
                "Expected utilities of shape "
                f"({len(sample_indices)}, {len(annotator_indices)}), "
                f"got {utilities.shape}."
            )
        return utilities

    @abstractmethod
    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        *args,
        **kwargs,
    ):
        """
        Implementation of utility computation.

        Parameters
        ----------
        X : array-like
            See ``__call__``.
        y : array-like
            See ``__call__``.
        sample_indices : array-like of shape (n_sel_samples,), dtype=int
            See ``__call__``.
        annotator_indices : array-like of shape (n_sel_annotators,), dtype=int
            See ``__call__``.
        available_mask : None or array-like of shape (n_sel_samples, n_sel_annotators), dtype=bool
            See ``__call__``.
        **kwargs : dict
            See ``__call__``.

        Returns
        -------
        utilities : numpy.ndarray of shape (n_sel_samples, n_sel_annotators)
            Utility matrix.

        """
        raise NotImplementedError
