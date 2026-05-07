from __future__ import annotations

import numpy as np

from skactiveml.utils import is_labeled

from ._base import PairScorer


class PostHocConfusionInformationGain(PairScorer):
    """
    Post-hoc full-confusion information-gain scorer.

    The scorer estimates one instance-independent confusion matrix per
    annotator from the current label matrix and classifier posteriors, then
    computes the mutual information between each candidate's latent class and
    each annotator's possible response.
    """

    def __init__(
        self,
        *,
        classes=None,
        missing_label=None,
        alpha: float | np.ndarray = 1.0,
        class_prior: str = "classifier",
        top_m: int | None = None,
        eps: float = 1e-12,
        log_base: float = 2.0,
    ):
        self.classes = classes
        self.missing_label = missing_label
        self.alpha = alpha
        self.class_prior = str(class_prior)
        self.top_m = None if top_m is None else int(top_m)
        self.eps = float(eps)
        self.log_base = float(log_base)

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

        X = np.asarray(X)
        y = np.asarray(y)
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        if y.ndim != 2:
            raise ValueError("`y` must have shape (n_samples, n_annotators).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` must agree on the sample axis.")

        classes = self._resolve_classes(clf)
        K = len(classes)
        if K < 2:
            raise ValueError("Information gain requires at least two classes.")
        missing_label = self._resolve_missing_label(clf)

        P_cand_raw = np.asarray(
            clf.predict_proba(X[sample_indices]), dtype=float
        )
        P_cand = self._resolve_class_prior(P_cand_raw)

        observed = is_labeled(y=y, missing_label=missing_label)
        obs_s = np.flatnonzero(observed.any(axis=1))
        if obs_s.size == 0:
            soft_counts = np.zeros((y.shape[1], K, K), dtype=float)
        else:
            P_obs = np.asarray(clf.predict_proba(X[obs_s]), dtype=float)
            P_obs = self._normalize_probabilities(
                P_obs, name="observed probabilities"
            )
            soft_counts = self._compute_soft_counts(
                y=y,
                observed=observed,
                obs_sample_indices=obs_s,
                P_obs=P_obs,
                classes=classes,
            )

        C = self._posterior_mean_confusion(
            soft_counts=soft_counts,
            alpha=self.alpha,
            n_classes=K,
        )
        U_all = self._compute_information_gain(P=P_cand, C=C)
        U = U_all[:, annotator_indices]

        if available_mask is not None:
            U = np.where(np.asarray(available_mask, dtype=bool), U, np.nan)
        return U

    def _resolve_classes(self, clf) -> np.ndarray:
        classes = self.classes
        if classes is None:
            classes = getattr(clf, "classes_", None)
        if classes is None:
            raise ValueError(
                "`classes` must be provided or available as `clf.classes_`."
            )
        classes = np.asarray(classes)
        if classes.ndim != 1:
            raise ValueError("`classes` must be one-dimensional.")
        return classes

    def _resolve_missing_label(self, clf):
        if self.missing_label is not None:
            return self.missing_label
        return getattr(clf, "missing_label", np.nan)

    def _resolve_class_prior(self, P: np.ndarray) -> np.ndarray:
        P = self._normalize_probabilities(P, name="candidate probabilities")
        prior = self.class_prior.lower()
        if prior == "classifier":
            return P
        if prior == "uniform":
            return np.full_like(P, 1.0 / P.shape[1], dtype=float)
        if prior == "top_m":
            if self.top_m is None:
                raise ValueError("class_prior='top_m' requires `top_m`.")
            m = int(self.top_m)
            K = P.shape[1]
            if not (1 <= m <= K):
                raise ValueError(f"`top_m` must be in [1, {K}], got {m}.")
            out = np.zeros_like(P, dtype=float)
            top = np.argpartition(P, -m, axis=1)[:, -m:]
            rows = np.arange(P.shape[0])[:, None]
            out[rows, top] = 1.0 / m
            return out
        raise ValueError(
            "class_prior must be one of {'classifier', 'uniform', 'top_m'}."
        )

    def _normalize_probabilities(self, P: np.ndarray, *, name: str) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2:
            raise ValueError(f"Expected {name} with shape (n_samples, n_classes).")
        if not np.all(np.isfinite(P)):
            raise ValueError(f"{name} must be finite.")
        P = np.clip(P, 0.0, None)
        row_sum = P.sum(axis=1, keepdims=True)
        if np.any(row_sum <= self.eps):
            raise ValueError(f"{name} rows must have positive mass.")
        return P / row_sum

    def _compute_soft_counts(
        self,
        *,
        y: np.ndarray,
        observed: np.ndarray,
        obs_sample_indices: np.ndarray,
        P_obs: np.ndarray,
        classes: np.ndarray,
    ) -> np.ndarray:
        K = len(classes)
        class_to_idx = {label: i for i, label in enumerate(classes.tolist())}
        row_by_sample = {int(s): i for i, s in enumerate(obs_sample_indices)}
        N = np.zeros((y.shape[1], K, K), dtype=float)

        obs_s, obs_m = np.where(observed)
        for s, m in zip(obs_s, obs_m):
            raw_label = y[s, m]
            try:
                label = raw_label.item() if hasattr(raw_label, "item") else raw_label
                c = class_to_idx[label]
            except KeyError as exc:
                raise ValueError(
                    f"Observed label {raw_label!r} is not present in classes."
                ) from exc
            N[int(m), :, int(c)] += P_obs[row_by_sample[int(s)]]
        return N

    def _posterior_mean_confusion(
        self,
        *,
        soft_counts: np.ndarray,
        alpha,
        n_classes: int,
    ) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=float)
        if alpha_arr.ndim == 0:
            alpha_arr = np.full((1, n_classes, n_classes), float(alpha_arr))
        elif alpha_arr.shape == (n_classes, n_classes):
            alpha_arr = np.broadcast_to(alpha_arr[None, :, :], soft_counts.shape)
        elif alpha_arr.shape != soft_counts.shape:
            raise ValueError(
                "`alpha` must be scalar, shape (n_classes, n_classes), or "
                "shape (n_annotators, n_classes, n_classes)."
            )
        if np.any(alpha_arr < 0):
            raise ValueError("`alpha` entries must be non-negative.")

        posterior = soft_counts + alpha_arr
        row_sum = posterior.sum(axis=2, keepdims=True)
        if np.any(row_sum <= self.eps):
            raise ValueError("Each confusion row must have positive posterior mass.")
        return posterior / row_sum

    def _compute_information_gain(
        self, *, P: np.ndarray, C: np.ndarray
    ) -> np.ndarray:
        P = self._normalize_probabilities(P, name="candidate probabilities")
        C = np.asarray(C, dtype=float)
        C = C / np.maximum(C.sum(axis=2, keepdims=True), self.eps)

        p_y = np.einsum("nz,mzc->nmc", P, C)
        denom = np.clip(p_y[:, :, None, :], self.eps, None)
        C_safe = np.clip(C[None, :, :, :], self.eps, None)
        log_ratio = np.log(C_safe / denom)
        if self.log_base != np.e:
            log_ratio = log_ratio / np.log(self.log_base)
        terms = P[:, None, :, None] * C[None, :, :, :] * log_ratio
        gain = terms.sum(axis=(2, 3))
        return np.maximum(gain, 0.0)
