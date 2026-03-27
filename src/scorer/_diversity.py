from __future__ import annotations

import numpy as np

from skactiveml.utils import is_labeled

from ._base import PairScorer


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _predict_embeddings(clf, X: np.ndarray) -> np.ndarray:
    out = clf.predict_proba(X, extra_outputs=["embeddings"])
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise ValueError(
            "clf.predict_proba must return a tuple when "
            "`extra_outputs=['embeddings']` is requested."
        )
    return _l2_normalize(np.asarray(out[1], dtype=float))


def _get_missing_label(clf):
    return getattr(clf, "missing_label", np.nan)


def _resolve_classes(clf, y: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    classes = getattr(clf, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    return np.unique(np.asarray(y)[observed_mask])


class SemanticDiversityPairScorer(PairScorer):
    """
    Pair scorer based on semantic diversity of annotator history.

    For a candidate pair ``(x, a)``, the utility is the average cosine
    distance between the candidate sample embedding and the embeddings of
    samples previously labeled by annotator ``a``. Annotators without any
    labeling history receive the configurable `cold_start_score`.

    This mirrors the semantic-diversity annotator selection heuristic from
    annotator-centric active learning, adapted to return pairwise utilities.
    """

    def __init__(self, *, cold_start_score: float = 2.0, eps: float = 1e-12):
        self.cold_start_score = float(cold_start_score)
        self.eps = float(eps)

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

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        observed_mask = is_labeled(y=y, missing_label=_get_missing_label(clf))
        history_samples = np.flatnonzero(np.any(observed_mask, axis=1))
        required_samples = np.unique(
            np.concatenate([np.asarray(sample_indices, dtype=int), history_samples])
        )
        required_embeddings = _predict_embeddings(clf, X[required_samples])

        cand_pos = np.searchsorted(required_samples, sample_indices)
        X_cand_emb = required_embeddings[cand_pos]

        U = np.empty((n_sel_s, n_sel_a), dtype=float)
        for a_pos, a_id in enumerate(annotator_indices):
            history_a = np.flatnonzero(observed_mask[:, a_id])
            if history_a.size == 0:
                U[:, a_pos] = self.cold_start_score
                continue

            history_pos = np.searchsorted(required_samples, history_a)
            X_hist_a = required_embeddings[history_pos]
            distances = 1.0 - np.clip(X_cand_emb @ X_hist_a.T, -1.0, 1.0)
            U[:, a_pos] = distances.mean(axis=1)

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U


class RepresentationDiversityPairScorer(PairScorer):
    """
    Pair scorer based on diversity of annotator representations.

    Each annotator is represented by the average of normalized observation
    vectors built from the embeddings of samples they labeled together with a
    one-hot encoding of the observed class label. The utility of annotator
    ``a`` is the average cosine distance between this representation and the
    representations of the other available annotators.

    Annotators without any labeling history receive the configurable
    `cold_start_score`, which prioritizes exploring previously unused
    annotators.
    """

    def __init__(
        self,
        *,
        label_weight: float = 1.0,
        cold_start_score: float = 2.0,
        eps: float = 1e-12,
    ):
        self.label_weight = float(label_weight)
        self.cold_start_score = float(cold_start_score)
        self.eps = float(eps)

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

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        observed_mask = is_labeled(y=y, missing_label=_get_missing_label(clf))
        history_samples = np.flatnonzero(np.any(observed_mask, axis=1))
        if history_samples.size > 0:
            history_embeddings = _predict_embeddings(clf, X[history_samples])
        else:
            history_embeddings = np.empty((0, 0), dtype=float)

        classes = _resolve_classes(clf, y, observed_mask)
        class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}

        reps = None
        rep_valid = np.zeros(n_sel_a, dtype=bool)
        if history_samples.size > 0:
            rep_dim = history_embeddings.shape[1] + len(classes)
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
                label_one_hot = np.zeros(
                    (sample_ids.size, len(classes)),
                    dtype=float,
                )
                label_one_hot[np.arange(sample_ids.size), label_idx] = (
                    self.label_weight
                )
                obs_repr = _l2_normalize(
                    np.concatenate([X_hist_a, label_one_hot], axis=1),
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
        U = np.empty((n_sel_s, n_sel_a), dtype=float)
        for s_pos in range(n_sel_s):
            available_with_repr = available[s_pos] & rep_valid
            for a_pos in range(n_sel_a):
                if not rep_valid[a_pos]:
                    U[s_pos, a_pos] = self.cold_start_score
                    continue

                peer_mask = available_with_repr.copy()
                peer_mask[a_pos] = False
                if not np.any(peer_mask):
                    U[s_pos, a_pos] = 0.0
                    continue

                similarities = np.clip(reps[peer_mask] @ reps[a_pos], -1.0, 1.0)
                U[s_pos, a_pos] = (1.0 - similarities).mean()

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U
