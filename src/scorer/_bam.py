from __future__ import annotations

import warnings

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state
from skactiveml.utils import is_labeled

from ._base import PairScorer


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class BetaModelPairScorer(PairScorer):
    """
    Beta Annotators Model scorer for sample-annotator pairs.

    The scorer estimates a local correctness probability for each annotator.
    For annotator ``a``, a multiclass Parzen estimator is built from the labels
    of all other annotators. On samples already labeled by ``a``, this
    leave-one-annotator-out estimator provides a pseudo label. Agreement with
    that pseudo label is converted into weighted binary correctness evidence,
    which is then kernel-smoothed to the candidate samples and combined with a
    Beta prior.

    If ``representation="classifier_embeddings"``, the classifier is used only
    to provide embeddings through ``extra_outputs=["embeddings"]``; class
    probabilities are ignored.
    """

    def __init__(
        self,
        *,
        prior=(1.0, 0.1),
        gamma="mean_gamma",
        gamma_scope="global",
        weights_type="entropy",
        representation="input",
        normalize_embeddings=False,
        use_ess=False,
        tau=1.0,
        random_state=None,
        eps=1e-12,
    ):
        self.prior = prior
        self.gamma = gamma
        self.gamma_scope = str(gamma_scope)
        self.weights_type = str(weights_type)
        self.representation = str(representation)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.use_ess = bool(use_ess)
        self.tau = float(tau)
        self.random_state = check_random_state(random_state)
        self.eps = float(eps)

        prior_arr = np.asarray(prior, dtype=float)
        if prior_arr.shape != (2,):
            raise ValueError("prior must have shape (2,).")
        if np.any(prior_arr < 0) or prior_arr.sum() <= 0:
            raise ValueError("prior entries must be non-negative with positive sum.")
        if self.weights_type not in {"entropy", "margin"}:
            raise ValueError("weights_type must be one of {'entropy', 'margin'}.")
        if self.gamma_scope not in {"global", "per_annotator"}:
            raise ValueError("gamma_scope must be one of {'global', 'per_annotator'}.")
        if self.representation not in {"input", "classifier_embeddings"}:
            raise ValueError(
                "representation must be one of "
                "{'input', 'classifier_embeddings'}."
            )
        if self.tau <= 0:
            raise ValueError("tau must be > 0.")
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

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
            raise ValueError("clf must be provided.")

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        classes = np.asarray(getattr(clf, "classes_", None))
        if classes.ndim != 1 or classes.size < 2:
            raise ValueError("clf.classes_ must contain at least two classes.")
        n_classes = int(classes.size)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        y = np.asarray(y)
        missing_label = getattr(clf, "missing_label", np.nan)
        observed_mask = is_labeled(y=y, missing_label=missing_label)
        y_idx = self._encode_observed_labels(
            y=y,
            observed_mask=observed_mask,
            class_to_idx=class_to_idx,
        )
        prior = np.asarray(self.prior, dtype=float)
        prior_mean = prior[0] / prior.sum()
        if not np.any(observed_mask):
            U = np.full((n_sel_s, n_sel_a), prior_mean, dtype=float)
            if available_mask is not None:
                U = np.where(available_mask, U, np.nan)
            return U

        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        obs_sample_indices = np.flatnonzero(observed_mask.any(axis=1))
        needed_indices = np.unique(
            np.concatenate([sample_indices, obs_sample_indices])
        )
        embeddings = self._sample_representations(
            clf=clf,
            X=np.asarray(X),
            indices=needed_indices,
        )
        row_by_index = {idx: pos for pos, idx in enumerate(needed_indices)}
        X_repr_cand = embeddings[[row_by_index[idx] for idx in sample_indices]]
        X_obs_all_emb = embeddings[
            [row_by_index[idx] for idx in obs_sample_indices]
        ]

        gamma = self._resolve_gamma_from_embeddings(X_obs_all_emb, self.gamma)

        U = np.full((n_sel_s, n_sel_a), prior_mean, dtype=float)
        for j, annotator_idx in enumerate(annotator_indices):
            U[:, j] = self._annotator_posterior_mean(
                annotator_idx=int(annotator_idx),
                X_cand_emb=X_repr_cand,
                embeddings=embeddings,
                row_by_index=row_by_index,
                observed_mask=observed_mask,
                y_idx=y_idx,
                n_classes=n_classes,
                global_gamma=gamma,
                prior=prior,
            )

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U

    def _annotator_posterior_mean(
        self,
        *,
        annotator_idx: int,
        X_cand_emb: np.ndarray,
        embeddings: np.ndarray,
        row_by_index: dict[int, int],
        observed_mask: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
        global_gamma: float,
        prior: np.ndarray,
    ) -> np.ndarray:
        obs_a = np.flatnonzero(observed_mask[:, annotator_idx])
        if obs_a.size == 0:
            return np.full(X_cand_emb.shape[0], prior[0] / prior.sum(), dtype=float)

        other_observed = observed_mask.copy()
        other_observed[:, annotator_idx] = False
        train_samples = np.flatnonzero(other_observed.any(axis=1))
        if train_samples.size == 0:
            return np.full(X_cand_emb.shape[0], prior[0] / prior.sum(), dtype=float)

        train_counts = self._vote_counts(
            samples=train_samples,
            observed_mask=other_observed,
            y_idx=y_idx,
            n_classes=n_classes,
        )
        X_train_emb = embeddings[[row_by_index[idx] for idx in train_samples]]
        X_obs_a_emb = embeddings[[row_by_index[idx] for idx in obs_a]]
        gamma = self._resolve_annotator_gamma(
            X_obs_a_emb=X_obs_a_emb,
            global_gamma=global_gamma,
        )

        obs_scores = self._parzen_class_scores(
            X_query=X_obs_a_emb,
            X_train=X_train_emb,
            train_counts=train_counts,
            gamma=gamma,
        )
        obs_proba, obs_mass = self._normalize_scores(obs_scores)
        pseudo_labels = np.argmax(obs_proba, axis=1)
        confidence = self._confidence(obs_proba)
        confidence = np.where(obs_mass > self.eps, confidence, 0.0)

        correct = y_idx[obs_a, annotator_idx] == pseudo_labels
        correctness = correct.astype(float)

        local_kernel = rbf_kernel(X_cand_emb, X_obs_a_emb, gamma=gamma)
        alpha, beta = self._beta_posterior_from_local_evidence(
            local_kernel=local_kernel,
            correctness=correctness,
            confidence=confidence,
            prior=prior,
        )
        return alpha / np.maximum(alpha + beta, self.eps)

    def _resolve_annotator_gamma(
        self,
        *,
        X_obs_a_emb: np.ndarray,
        global_gamma: float,
    ) -> float:
        if self.gamma_scope == "global":
            return float(global_gamma)
        if X_obs_a_emb.shape[0] < 2:
            return float(global_gamma)
        return self._resolve_gamma_from_embeddings(X_obs_a_emb, self.gamma)

    def _beta_posterior_from_local_evidence(
        self,
        *,
        local_kernel: np.ndarray,
        correctness: np.ndarray,
        confidence: np.ndarray,
        prior: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights = np.asarray(local_kernel, dtype=float) * confidence[None, :]
        success = weights @ correctness
        failure = weights @ (1.0 - correctness)
        mass = success + failure

        if not self.use_ess:
            return prior[0] + success, prior[1] + failure

        mean = np.divide(
            success,
            mass,
            out=np.zeros_like(success, dtype=float),
            where=mass > self.eps,
        )
        weight_sq = np.sum(weights * weights, axis=1)
        n_eff = np.divide(
            mass * mass,
            weight_sq,
            out=np.zeros_like(mass, dtype=float),
            where=weight_sq > self.eps,
        )
        concentration = self.tau * n_eff
        alpha = prior[0] + concentration * mean
        beta = prior[1] + concentration * (1.0 - mean)
        return alpha, beta

    @staticmethod
    def _vote_counts(
        *,
        samples: np.ndarray,
        observed_mask: np.ndarray,
        y_idx: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        sample_observed = observed_mask[samples]
        row_pos, annotator_pos = np.where(sample_observed)
        counts = np.zeros((len(samples), n_classes), dtype=float)
        if row_pos.size == 0:
            return counts
        labels = y_idx[samples[row_pos], annotator_pos]
        np.add.at(counts, (row_pos, labels), 1.0)
        return counts

    @staticmethod
    def _parzen_class_scores(
        *,
        X_query: np.ndarray,
        X_train: np.ndarray,
        train_counts: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        if X_train.shape[0] == 0:
            return np.zeros((X_query.shape[0], train_counts.shape[1]), dtype=float)
        return rbf_kernel(X_query, X_train, gamma=gamma) @ train_counts

    def _normalize_scores(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mass = np.asarray(scores, dtype=float).sum(axis=1, keepdims=True)
        proba = np.full_like(scores, 1.0 / scores.shape[1], dtype=float)
        valid = mass[:, 0] > self.eps
        if np.any(valid):
            proba[valid] = scores[valid] / mass[valid]
        return proba, mass[:, 0]

    def _confidence(self, proba: np.ndarray) -> np.ndarray:
        proba = np.clip(np.asarray(proba, dtype=float), self.eps, 1.0)
        proba = proba / np.maximum(proba.sum(axis=1, keepdims=True), self.eps)
        if self.weights_type == "entropy":
            entropy = -np.sum(proba * np.log(proba), axis=1) / np.log(
                proba.shape[1]
            )
            return np.clip(1.0 - entropy, 0.0, 1.0)
        proba_sort = np.sort(proba, axis=1)
        return np.clip(proba_sort[:, -1] - proba_sort[:, -2], 0.0, 1.0)

    def _sample_representations(
        self, *, clf, X: np.ndarray, indices: np.ndarray
    ) -> np.ndarray:
        if self.representation == "input":
            features = np.asarray(X[indices], dtype=float)
            features = features.reshape(features.shape[0], -1)
            return (
                _l2_normalize(features, eps=self.eps)
                if self.normalize_embeddings
                else features
            )

        out = clf.predict_proba(X[indices], extra_outputs=["embeddings"])
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                "clf.predict_proba must return embeddings when "
                "extra_outputs=['embeddings'] is requested."
            )
        embeddings = np.asarray(out[1], dtype=float)
        if embeddings.shape[0] != len(indices):
            raise ValueError(
                "embeddings must have the same number of rows as requested samples."
            )
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        return (
            _l2_normalize(embeddings, eps=self.eps)
            if self.normalize_embeddings
            else embeddings
        )

    @staticmethod
    def _encode_observed_labels(*, y, observed_mask, class_to_idx):
        y_idx = np.full(y.shape, -1, dtype=int)
        for idx in zip(*np.where(observed_mask)):
            label = y[idx]
            try:
                y_idx[idx] = class_to_idx[label]
            except KeyError as exc:
                raise ValueError(
                    f"Observed label {label!r} is not present in clf.classes_."
                ) from exc
        return y_idx

    @staticmethod
    def _resolve_gamma_from_embeddings(E: np.ndarray, mode):
        E = np.asarray(E, dtype=float)
        if mode in {"mean_gamma", "original_mean_gamma", "bam_mean_gamma"}:
            return BetaModelPairScorer._calculate_mean_gamma(
                N=E.shape[0],
                variance=np.var(E, axis=0) if E.shape[0] > 0 else np.asarray([0.0]),
                n_features=E.shape[1] if E.ndim == 2 and E.shape[1] > 0 else 1,
            )
        if E.shape[0] < 2:
            return 1.0

        norms = (E * E).sum(axis=1, keepdims=True)
        d2 = norms + norms.T - 2.0 * (E @ E.T)
        np.fill_diagonal(d2, np.nan)
        d = np.sqrt(np.maximum(d2, 0.0))

        if mode == "median":
            s = np.nanmedian(d)
            s = max(float(s), 1e-3)
            return s ** (-2)
        if mode == "mean":
            s = np.nanmean(d)
            s = max(float(s), 1e-3)
            return s ** (-2)
        if mode == "minimum":
            s = np.nanmin(d)
            s = max(float(s), 1e-3)
            return s ** (-2)

        gamma = float(mode)
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        return gamma

    @staticmethod
    def _calculate_mean_gamma(
        N,
        variance,
        n_features,
        delta=(np.sqrt(2) * 1e-6),
    ):
        denominator = 2 * N * np.sum(variance)
        numerator = (N - 1) * np.log((N - 1) / delta**2) if N > 1 else 0.0
        if denominator <= 0:
            gamma = 1 / n_features
            warnings.warn(
                "The variance of the provided data is 0. Bandwidth of "
                + f"1/n_features={gamma} is used instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            gamma = 0.5 * numerator / denominator
        return float(gamma)
