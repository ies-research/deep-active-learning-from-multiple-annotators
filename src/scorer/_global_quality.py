from __future__ import annotations

import numpy as np

from skactiveml.utils import is_labeled

from ._base import PairScorer


def _get_missing_label(clf, configured_missing_label):
    if configured_missing_label is not None:
        return configured_missing_label
    return getattr(clf, "missing_label", np.nan)


def _resolve_classes(clf, y: np.ndarray, observed_mask: np.ndarray, classes):
    if classes is not None:
        return np.asarray(classes)
    clf_classes = getattr(clf, "classes_", None)
    if clf_classes is not None:
        return np.asarray(clf_classes)
    if not np.any(observed_mask):
        raise ValueError("classes must be provided when no labels are observed.")
    return np.unique(y[observed_mask])


def _encode_observed_labels(
    *,
    y: np.ndarray,
    observed_mask: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    class_to_idx = {label: i for i, label in enumerate(classes)}
    y_idx = np.full(y.shape, -1, dtype=int)
    for idx in zip(*np.where(observed_mask)):
        label = y[idx]
        try:
            y_idx[idx] = class_to_idx[label]
        except KeyError as exc:
            raise ValueError(
                f"Observed label {label!r} is not present in classes."
            ) from exc
    return y_idx


def _predict_proba(clf, X: np.ndarray, n_classes: int) -> np.ndarray:
    if clf is None:
        raise ValueError("clf must be provided.")
    P = np.asarray(clf.predict_proba(X), dtype=float)
    if P.ndim != 2 or P.shape[1] != n_classes:
        raise ValueError(
            "clf.predict_proba(X) must return probabilities with shape "
            f"(n_samples, {n_classes}), got {P.shape}."
        )
    P = np.clip(P, 0.0, None)
    row_sum = P.sum(axis=1, keepdims=True)
    if np.any(row_sum[:, 0] <= 0):
        raise ValueError("Classifier probabilities must have positive row sums.")
    return P / row_sum


def _label_probability_scores(
    *,
    P: np.ndarray,
    y_idx: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(y_idx.shape, dtype=float)
    obs_i, obs_a = np.where(observed_mask)
    if obs_i.size:
        scores[obs_i, obs_a] = P[obs_i, y_idx[obs_i, obs_a]]
    return scores


def _leave_one_out_agreement_scores(
    *,
    y_idx: np.ndarray,
    observed_mask: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    counts = np.zeros((y_idx.shape[0], n_classes), dtype=float)
    obs_i, _ = np.where(observed_mask)
    if obs_i.size:
        np.add.at(counts, (obs_i, y_idx[observed_mask]), 1.0)

    scores = np.full(y_idx.shape, np.nan, dtype=float)
    obs_i, obs_a = np.where(observed_mask)
    for i, a in zip(obs_i, obs_a):
        label = int(y_idx[i, a])
        other_counts = counts[i].copy()
        other_counts[label] -= 1.0
        total = float(other_counts.sum())
        if total > 0:
            scores[i, a] = other_counts[label] / total
    return scores


def _shrunken_annotator_mean(
    *,
    values: np.ndarray,
    observed_mask: np.ndarray,
    prior_mean: float,
    prior_strength: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    observed_values = observed_mask & np.isfinite(values)
    sums = np.where(observed_values, values, 0.0).sum(axis=0)
    counts = observed_values.sum(axis=0).astype(float)
    strength = max(float(prior_strength), 0.0)
    denom = strength + counts
    return (strength * float(prior_mean) + sums) / np.maximum(denom, 1e-12)


def _broadcast_global_scores(
    *,
    scores: np.ndarray,
    sample_indices: np.ndarray,
    annotator_indices: np.ndarray,
    available_mask,
) -> np.ndarray:
    U = np.broadcast_to(
        scores[np.asarray(annotator_indices, dtype=int)][None, :],
        (len(sample_indices), len(annotator_indices)),
    ).astype(float, copy=True)
    if available_mask is not None:
        U = np.where(available_mask, U, np.nan)
    return U


class LabelQualityGlobalPairScorer(PairScorer):
    """
    Global annotator scorer based on classifier label quality.

    Each annotator receives one scalar score: the shrunken average classifier
    probability assigned to the labels previously provided by that annotator.
    The score is then broadcast to all candidate samples.
    """

    def __init__(
        self,
        *,
        classes=None,
        prior_quality: float = 0.5,
        prior_strength: float = 1.0,
        missing_label=None,
    ):
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")
        self.classes = None if classes is None else np.asarray(classes).copy()
        self.prior_quality = float(prior_quality)
        self.prior_strength = float(prior_strength)
        self.missing_label = missing_label

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
        y = np.asarray(y)
        observed_mask = is_labeled(
            y=y, missing_label=_get_missing_label(clf, self.missing_label)
        )
        classes = _resolve_classes(clf, y, observed_mask, self.classes)
        y_idx = _encode_observed_labels(
            y=y, observed_mask=observed_mask, classes=classes
        )
        P = _predict_proba(clf, np.asarray(X), n_classes=len(classes))
        label_quality = _label_probability_scores(
            P=P, y_idx=y_idx, observed_mask=observed_mask
        )
        scores = _shrunken_annotator_mean(
            values=label_quality,
            observed_mask=observed_mask,
            prior_mean=self.prior_quality,
            prior_strength=self.prior_strength,
        )
        return _broadcast_global_scores(
            scores=scores,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
        )


class AgreementGlobalPairScorer(PairScorer):
    """
    Global annotator scorer based on leave-one-out annotator agreement.

    For every previously observed label, the scorer computes the fraction of
    other annotators on the same sample that provided the same label. Labels
    without another annotator on that sample contribute no evidence and are
    handled by the prior.
    """

    def __init__(
        self,
        *,
        classes=None,
        prior_quality: float = 0.5,
        prior_strength: float = 1.0,
        missing_label=None,
    ):
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")
        self.classes = None if classes is None else np.asarray(classes).copy()
        self.prior_quality = float(prior_quality)
        self.prior_strength = float(prior_strength)
        self.missing_label = missing_label

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
        del X, kwargs
        y = np.asarray(y)
        observed_mask = is_labeled(
            y=y, missing_label=_get_missing_label(clf, self.missing_label)
        )
        classes = _resolve_classes(clf, y, observed_mask, self.classes)
        y_idx = _encode_observed_labels(
            y=y, observed_mask=observed_mask, classes=classes
        )
        agreement = _leave_one_out_agreement_scores(
            y_idx=y_idx,
            observed_mask=observed_mask,
            n_classes=len(classes),
        )
        scores = _shrunken_annotator_mean(
            values=agreement,
            observed_mask=observed_mask,
            prior_mean=self.prior_quality,
            prior_strength=self.prior_strength,
        )
        return _broadcast_global_scores(
            scores=scores,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
        )


class CrowdLabGlobalPairScorer(PairScorer):
    """
    CROWDLAB-style global annotator scorer.

    The scorer forms a classifier-and-annotator consensus distribution for
    observed samples and estimates each annotator's global quality by the
    average leave-one-out consensus probability of that annotator's labels.
    With single-labeled samples, the score falls back to classifier label
    quality; with overlapping labels, annotator agreement contributes as well.
    """

    def __init__(
        self,
        *,
        classes=None,
        model_weight: float = 1.0,
        prior_quality: float = 0.5,
        prior_strength: float = 1.0,
        n_iter: int = 5,
        missing_label=None,
        eps: float = 1e-12,
    ):
        if model_weight < 0:
            raise ValueError("model_weight must be non-negative.")
        if prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")
        if int(n_iter) <= 0:
            raise ValueError("n_iter must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.classes = None if classes is None else np.asarray(classes).copy()
        self.model_weight = float(model_weight)
        self.prior_quality = float(prior_quality)
        self.prior_strength = float(prior_strength)
        self.n_iter = int(n_iter)
        self.missing_label = missing_label
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
        y = np.asarray(y)
        observed_mask = is_labeled(
            y=y, missing_label=_get_missing_label(clf, self.missing_label)
        )
        classes = _resolve_classes(clf, y, observed_mask, self.classes)
        y_idx = _encode_observed_labels(
            y=y, observed_mask=observed_mask, classes=classes
        )
        P = _predict_proba(clf, np.asarray(X), n_classes=len(classes))
        scores = self._crowdlab_quality(
            P=P,
            y_idx=y_idx,
            observed_mask=observed_mask,
            n_classes=len(classes),
        )
        return _broadcast_global_scores(
            scores=scores,
            sample_indices=sample_indices,
            annotator_indices=annotator_indices,
            available_mask=available_mask,
        )

    def _crowdlab_quality(
        self,
        *,
        P: np.ndarray,
        y_idx: np.ndarray,
        observed_mask: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        label_quality = _label_probability_scores(
            P=P, y_idx=y_idx, observed_mask=observed_mask
        )
        quality = _shrunken_annotator_mean(
            values=label_quality,
            observed_mask=observed_mask,
            prior_mean=self.prior_quality,
            prior_strength=self.prior_strength,
        )

        obs_i, obs_a = np.where(observed_mask)
        if obs_i.size == 0:
            return quality

        for _ in range(self.n_iter):
            label_weighted_votes = np.zeros(
                (observed_mask.shape[0], n_classes), dtype=float
            )
            for i, a in zip(obs_i, obs_a):
                label_weighted_votes[i, y_idx[i, a]] += quality[a]

            values = np.full(y_idx.shape, np.nan, dtype=float)
            for i, a in zip(obs_i, obs_a):
                label = int(y_idx[i, a])
                loo_votes = label_weighted_votes[i].copy()
                loo_votes[label] -= quality[a]
                numer = self.model_weight * P[i] + loo_votes
                denom = float(numer.sum())
                if denom <= self.eps:
                    values[i, a] = self.prior_quality
                else:
                    values[i, a] = numer[label] / denom

            quality = _shrunken_annotator_mean(
                values=values,
                observed_mask=observed_mask,
                prior_mean=self.prior_quality,
                prior_strength=self.prior_strength,
            )
        return quality
