from __future__ import annotations

import numpy as np

from sklearn.utils import check_random_state
from skactiveml.utils import is_labeled

from ._base import PairScorer


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _predict_embeddings(
    clf, X: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Return L2-normalized sample embeddings from ``clf.predict_proba``."""
    out = clf.predict_proba(X, extra_outputs=["embeddings"])
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError(
            "clf.predict_proba must return a tuple when "
            "`extra_outputs=['embeddings']` is requested."
        )
    return _l2_normalize(np.asarray(out[1], dtype=float), eps=eps)


def _get_missing_label(clf):
    return getattr(clf, "missing_label", np.nan)


def _resolve_classes(
    clf,
    y: np.ndarray,
    observed_mask: np.ndarray,
    classes=None,
) -> np.ndarray:
    if classes is not None:
        return np.asarray(classes)
    classes = getattr(clf, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    return np.unique(np.asarray(y)[observed_mask])


def _resolve_rng(random_state, kwargs):
    rng = kwargs.get("rng", None)
    return random_state if rng is None else rng


def _apply_random_cold_start_override(
    U: np.ndarray,
    *,
    available_mask,
    has_history: np.ndarray,
    rng,
    cold_start_score: float,
) -> np.ndarray:
    """Mirror the reference code's random choice among available unseen annotators."""
    available = (
        np.ones(U.shape, dtype=bool)
        if available_mask is None
        else np.asarray(available_mask, dtype=bool)
    )
    for s_pos in range(U.shape[0]):
        unseen_mask = available[s_pos] & ~has_history
        if np.any(unseen_mask):
            U[s_pos] = 0.0
            chosen_pos = int(rng.choice(np.flatnonzero(unseen_mask)))
            U[s_pos, chosen_pos] = cold_start_score
    return U


class SemanticDiversityPairScorer(PairScorer):
    """
    Semantic-diversity scorer over annotator labeling histories.

    This scorer implements the semantic diversity heuristic from
    "Annotator-Centric Active Learning for Subjective NLP Tasks". For a
    candidate pair ``(x_i, a_j)``, it compares the embedding of ``x_i`` with
    the embeddings of all samples previously labeled by annotator ``a_j`` and
    returns the average cosine distance to that history.

    In the paper, the annotator with the lowest average cosine similarity is
    selected. This implementation returns the corresponding average cosine
    distance instead so that larger values remain better utilities for the
    generic :class:`~src.scorer._base.PairScorer` interface. To mirror the
    released reference code more closely, if an available annotator has no
    history yet, one such annotator is chosen uniformly at random for the
    candidate sample and receives the cold-start utility.

    Parameters
    ----------
    cold_start_score : float, default=2.0
        Utility assigned to the randomly selected available annotator without
        any observed labels. The default is larger than the maximum cosine
        distance between normalized vectors, which prioritizes unexplored
        annotators.
    eps : float, default=1e-12
        Lower bound used during L2 normalization of sample embeddings.
    random_state : None or int, default=None
        Seed for reproducible random tie-breaking among available annotators
        without history.

    Notes
    -----
    - Only the samples already annotated by the candidate annotator are used;
      observed labels themselves do not affect this score.
    - Sample embeddings are obtained from
      ``clf.predict_proba(X, extra_outputs=["embeddings"])`` and normalized
      before cosine distances are computed.
    - If any available annotator has an empty history for a sample, the scorer
      follows the released reference code and randomly selects among those
      annotators instead of comparing against seen annotators.
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    """

    def __init__(
        self,
        *,
        cold_start_score: float = 2.0,
        eps: float = 1e-12,
        random_state=None,
    ):
        self.cold_start_score = float(cold_start_score)
        self.eps = float(eps)
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
        if clf is None:
            raise ValueError("`clf` must be provided.")
        rng = _resolve_rng(self.random_state, kwargs)

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        observed_mask = is_labeled(y=y, missing_label=_get_missing_label(clf))
        history_samples = np.flatnonzero(np.any(observed_mask, axis=1))
        required_samples = np.unique(
            np.concatenate([np.asarray(sample_indices, dtype=int), history_samples])
        )
        required_embeddings = _predict_embeddings(
            clf, X[required_samples], eps=self.eps
        )

        cand_pos = np.searchsorted(required_samples, sample_indices)
        X_cand_emb = required_embeddings[cand_pos]

        U = np.zeros((n_sel_s, n_sel_a), dtype=float)
        has_history = np.zeros(n_sel_a, dtype=bool)
        for a_pos, a_id in enumerate(annotator_indices):
            history_a = np.flatnonzero(observed_mask[:, a_id])
            if history_a.size == 0:
                continue

            has_history[a_pos] = True
            history_pos = np.searchsorted(required_samples, history_a)
            X_hist_a = required_embeddings[history_pos]
            distances = 1.0 - np.clip(X_cand_emb @ X_hist_a.T, -1.0, 1.0)
            U[:, a_pos] = distances.mean(axis=1)

        U = _apply_random_cold_start_override(
            U,
            available_mask=available_mask,
            has_history=has_history,
            rng=rng,
            cold_start_score=self.cold_start_score,
        )
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U


class RepresentationDiversityPairScorer(PairScorer):
    """
    Representation-diversity scorer over annotator profiles.

    This scorer implements the representation diversity heuristic from
    "Annotator-Centric Active Learning for Subjective NLP Tasks". It first
    builds one representation per annotator by averaging vectors that combine
    the embedding of each labeled sample with the corresponding observed label.
    For a candidate pair ``(x_i, a_j)``, the returned utility is the average
    cosine distance between annotator ``a_j`` and the other annotators that are
    available for sample ``x_i``.

    In the paper, the annotator with the lowest average cosine similarity to
    the other available annotators is selected. This implementation returns the
    equivalent average cosine distance so that larger utilities are preferred
    by downstream maximization-based query strategies. As in the released
    reference code, if an available annotator has no history yet, one such
    annotator is chosen uniformly at random for the candidate sample and
    receives the cold-start utility.

    Parameters
    ----------
    label_weight : float, default=1.0
        Multiplicative weight applied to the label-feature component before the
        per-observation annotator vectors are averaged. Larger values increase
        the contribution of label identity relative to sample semantics.
    classes : None or array-like of shape (n_classes,), default=None
        Optional explicit class order. This is mainly useful when
        ``label_embeddings`` is provided and should be aligned to a class order
        independent of ``clf.classes_`` or the labels currently observed in
        ``y``.
    label_embeddings : None or array-like of shape (n_classes, n_label_features), default=None
        Optional precomputed label-feature matrix. Row ``k`` is the feature
        vector for ``classes[k]`` after class resolution. If ``None``, labels
        are represented via one-hot vectors.
    cold_start_score : float, default=2.0
        Utility assigned to the randomly selected available annotator without
        any observed labels. The default prioritizes annotators whose
        representation cannot yet be estimated.
    eps : float, default=1e-12
        Lower bound used during L2 normalization of sample and annotator
        representations.
    random_state : None or int, default=None
        Seed for reproducible random tie-breaking among available annotators
        without history.

    Notes
    -----
    - Annotator representations are formed from concatenated
      ``[embedding, label_features(label)]`` observation vectors, averaged over
      the samples labeled by that annotator.
    - The score is sample-independent except for the per-sample availability
      mask, which determines the peer annotators included in the average.
    - The released reference code uses external encoded label vectors together
      with an SVD reduction step. This scorer now supports precomputed label
      embeddings directly, but still does not perform the additional SVD
      reduction.
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    """

    def __init__(
        self,
        *,
        label_weight: float = 1.0,
        classes=None,
        label_embeddings=None,
        cold_start_score: float = 2.0,
        eps: float = 1e-12,
        random_state=None,
    ):
        self.label_weight = float(label_weight)
        self.classes = None if classes is None else np.asarray(classes).copy()
        self.label_embeddings = (
            None
            if label_embeddings is None
            else np.asarray(label_embeddings, dtype=float).copy()
        )
        self.cold_start_score = float(cold_start_score)
        self.eps = float(eps)
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
        if clf is None:
            raise ValueError("`clf` must be provided.")
        rng = _resolve_rng(self.random_state, kwargs)

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        observed_mask = is_labeled(y=y, missing_label=_get_missing_label(clf))
        history_samples = np.flatnonzero(np.any(observed_mask, axis=1))
        if history_samples.size > 0:
            history_embeddings = _predict_embeddings(
                clf, X[history_samples], eps=self.eps
            )
        else:
            history_embeddings = np.empty((0, 0), dtype=float)

        classes = _resolve_classes(
            clf, y, observed_mask, classes=self.classes
        )
        class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}
        label_features = self._resolve_label_features(classes)

        reps = None
        rep_valid = np.zeros(n_sel_a, dtype=bool)
        if history_samples.size > 0:
            rep_dim = history_embeddings.shape[1] + label_features.shape[1]
            reps = np.zeros((n_sel_a, rep_dim), dtype=float)
            history_row_by_sample = np.full(y.shape[0], -1, dtype=int)
            history_row_by_sample[history_samples] = np.arange(history_samples.size)

            for a_pos, a_id in enumerate(annotator_indices):
                sample_ids = np.flatnonzero(observed_mask[:, a_id])
                if sample_ids.size == 0:
                    continue

                rows = history_row_by_sample[sample_ids]
                X_hist_a = history_embeddings[rows]
                label_idx = np.fromiter(
                    (class_to_idx[label] for label in y[sample_ids, a_id]),
                    dtype=int,
                    count=sample_ids.size,
                )
                label_repr = label_features[label_idx]
                obs_repr = _l2_normalize(
                    np.concatenate([X_hist_a, label_repr], axis=1),
                    eps=self.eps,
                )
                reps[a_pos] = _l2_normalize(
                    obs_repr.mean(axis=0, keepdims=True),
                    eps=self.eps,
                )[0]
                rep_valid[a_pos] = True

        available = (
            np.ones((n_sel_s, n_sel_a), dtype=bool)
            if available_mask is None
            else np.asarray(available_mask, dtype=bool)
        )
        U = np.zeros((n_sel_s, n_sel_a), dtype=float)
        for s_pos in range(n_sel_s):
            available_with_repr = available[s_pos] & rep_valid
            for a_pos in range(n_sel_a):
                if not rep_valid[a_pos]:
                    continue

                peer_mask = available_with_repr.copy()
                peer_mask[a_pos] = False
                if not np.any(peer_mask):
                    U[s_pos, a_pos] = 0.0
                    continue

                similarities = np.clip(reps[peer_mask] @ reps[a_pos], -1.0, 1.0)
                U[s_pos, a_pos] = (1.0 - similarities).mean()

        U = _apply_random_cold_start_override(
            U,
            available_mask=available_mask,
            has_history=rep_valid,
            rng=rng,
            cold_start_score=self.cold_start_score,
        )
        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U

    def _resolve_label_features(self, classes: np.ndarray) -> np.ndarray:
        if self.label_embeddings is None:
            return self.label_weight * np.eye(len(classes), dtype=float)

        label_embeddings = np.asarray(self.label_embeddings, dtype=float)
        if label_embeddings.ndim != 2:
            raise ValueError(
                "`label_embeddings` must be a 2D array of shape "
                "(n_classes, n_label_features)."
            )
        if label_embeddings.shape[0] != len(classes):
            raise ValueError(
                "`label_embeddings` must provide one row per resolved class; "
                f"expected {len(classes)}, got {label_embeddings.shape[0]}."
            )
        return self.label_weight * label_embeddings
