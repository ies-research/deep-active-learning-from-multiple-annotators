from __future__ import annotations

import numpy as np

from sklearn.utils import check_random_state
from skactiveml.utils import is_labeled

from ._base import PairScorer


def _get_missing_label(clf):
    return getattr(clf, "missing_label", np.nan)


class LabelMinorityPairScorer(PairScorer):
    """
    Pair scorer based on minority-label affinity in annotator histories.

    This scorer implements the Label Minority baseline from
    "Annotator-Centric Active Learning for Subjective NLP Tasks". It uses only
    the labels assigned by annotators so far: the globally minority class is
    determined from the observed annotations in ``y``, and each candidate
    annotator receives a utility equal to the relative frequency with which
    they have assigned that label in their history.

    The utility is sample-independent for a fixed annotation matrix ``y``. For
    a set of candidate samples, the same annotator score is broadcast across
    rows and only the optional availability mask can create row-wise
    differences. To mirror the released reference code more closely, if an
    available annotator has no history yet, one such annotator is chosen
    uniformly at random for the candidate sample and receives a dominant
    cold-start utility.

    Parameters
    ----------
    classes : array-like of shape (n_classes,), default=None
        Accepted for API compatibility. The current implementation follows the
        released reference code and determines the minority label from the
        observed annotations only, so unseen classes are ignored until they
        appear in ``y``.
    random_state : None or int, default=None
        Seed for reproducible random tie-breaking among available annotators
        without history.

    Notes
    -----
    - The scorer does not call ``clf.predict_proba``. A classifier is only
      consulted for ``missing_label`` metadata.
    - When multiple observed labels share the minimum count, the first one in
      row-major observation order is selected, matching the released reference
      code's use of ``min(all_annotations, key=all_annotations.count)``.
    - If any available annotator has an empty history for a sample, the scorer
      follows the released reference code and randomly selects among those
      annotators instead of comparing against seen annotators.
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    """

    def __init__(self, *, classes=None, random_state=None):
        self.classes = (
            None if classes is None else np.asarray(classes).copy()
        )
        self.random_state = check_random_state(random_state)

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
        del X

        rng = kwargs.get("rng", None)
        if rng is None:
            rng = self.random_state

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        y = np.asarray(y)
        observed_mask = is_labeled(y=y, missing_label=_get_missing_label(clf))
        available = (
            np.ones((n_sel_s, n_sel_a), dtype=bool)
            if available_mask is None
            else np.asarray(available_mask, dtype=bool)
        )
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        has_history = np.any(observed_mask[:, annotator_indices], axis=0)

        selected_scores = np.zeros(n_sel_a, dtype=float)
        if np.any(observed_mask):
            observed_labels = y[observed_mask]
            unique_labels, class_counts = np.unique(
                observed_labels, return_counts=True
            )
            min_count = int(class_counts.min())
            minority_candidates = set(unique_labels[class_counts == min_count].tolist())
            minority_label = next(
                label for label in observed_labels if label in minority_candidates
            )

            history_counts = np.sum(observed_mask, axis=0, dtype=float)
            minority_counts = np.sum(
                observed_mask & (y == minority_label),
                axis=0,
                dtype=float,
            )
            minority_bias = np.divide(
                minority_counts,
                history_counts,
                out=np.zeros_like(minority_counts, dtype=float),
                where=history_counts > 0,
            )
            selected_scores = minority_bias[annotator_indices]

        U = np.broadcast_to(selected_scores[None, :], (n_sel_s, n_sel_a)).astype(
            float, copy=True
        )
        for s_pos in range(n_sel_s):
            unseen_mask = available[s_pos] & ~has_history
            if np.any(unseen_mask):
                U[s_pos] = 0.0
                chosen_pos = int(rng.choice(np.flatnonzero(unseen_mask)))
                U[s_pos, chosen_pos] = 2.0
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U
