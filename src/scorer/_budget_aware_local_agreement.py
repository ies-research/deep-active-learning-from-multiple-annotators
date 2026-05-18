from __future__ import annotations

import numpy as np

from scipy.special import betaln, gammaln, logsumexp
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state

from ._base import PairScorer


def _l2_normalize(X: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, eps)


class BudgetAwareLocalAgreementScorer(PairScorer):
    """Budget-aware local Beta scorer for classifier-conditioned agreement.

    The scorer estimates the probability that an annotator agrees with confident
    classifier pseudo-labels near each candidate sample. The core estimate is a
    scalar local Beta posterior with empirical-Bayes pool/global shrinkage and a
    reference-radius correction for broad neighborhoods.

    Optionally, utilities can be model-averaged with a global response-bias
    model. This is intended for constant-label spammers and random guessers, and
    is disabled by default for clean ablations.

    Parameters
    ----------
    score_mode : {"mean", "thompson"}, default="thompson"
        How the local Beta posterior is converted to a utility. ``"mean"`` uses
        the posterior mean and is deterministic. ``"thompson"`` draws from the
        posterior and averages ``thompson_samples`` draws.
    thompson_samples : int, default=1
        Number of posterior samples used when ``score_mode="thompson"``. A value
        of 1 is standard Thompson sampling. Larger values smooth the random
        utility and approach the posterior mean.
    bias_model_correction : {"none", "model_average"}, default="none"
        Optional global response-bias branch. ``"none"`` uses only the responsive
        agreement posterior. ``"model_average"`` mixes the responsive posterior
        with a sample-independent response distribution posterior for each
        annotator.
    locality_mode : {"local", "global"}, default="local"
        Controls whether responsive agreement is candidate-local. ``"local"``
        uses budget-aware kNN evidence around each candidate and shrinks it
        toward the annotator-global prior. ``"global"`` disables kNN evidence and
        broadcasts each annotator's global responsive posterior to all candidate
        samples.
    base_prior_strength : float, default=1.0
        Total pseudo-count strength of the chance-level base prior. The prior
        mean is ``1 / n_classes``.
    pool_prior_scale : float, default=1.0
        Multiplier for the population-to-annotator shrinkage strength. The
        effective population prior strength is
        ``pool_prior_scale * k_star[m]``.
    local_prior_scale : float, default=1.0
        Multiplier for the annotator-global-to-local prior strength in local
        mode. The base local strength is proportional to
        ``local_prior_scale * min(k_star[m], G_m)`` before radius correction.
    local_prior_min : float, default=1.0
        Minimum annotator-global-to-local prior strength in local mode.
    normalize_embeddings : bool, default=False
        Whether to L2-normalize classifier embeddings before distance
        computations.
    metric : str, default="euclidean"
        Distance metric passed to :func:`sklearn.metrics.pairwise_distances` for
        kNN and reference-radius computations.
    missing_label : object, default=None
        Missing-label marker. If ``None``, the scorer uses ``clf.missing_label``
        when available and falls back to ``np.nan``.
    exclude_self : bool, default=True
        Whether to exclude the candidate sample itself from local evidence and
        full-dataset reference-neighbor computations when the candidate already
        appears in the training pool.
    store_neighbor_diagnostics : bool, default=False
        Whether to store neighbor-level diagnostic arrays. This is useful for
        toy visualizations but can be memory-heavy in benchmark runs.
    random_state : int, RandomState instance, or None, default=None
        Random state used for Thompson sampling and response-bias posterior
        sampling.
    eps : float, default=1e-12
        Numerical stability constant used in divisions and logarithms.
    """

    def __init__(
        self,
        *,
        score_mode: str = "thompson",
        thompson_samples: int = 1,
        bias_model_correction: str = "none",
        locality_mode: str = "local",
        base_prior_strength: float = 1.0,
        pool_prior_scale: float = 1.0,
        local_prior_scale: float = 1.0,
        local_prior_min: float = 1.0,
        normalize_embeddings: bool = False,
        metric: str = "euclidean",
        missing_label=None,
        exclude_self: bool = True,
        store_neighbor_diagnostics: bool = False,
        random_state=None,
        eps: float = 1e-12,
    ):
        if score_mode not in {"mean", "thompson"}:
            raise ValueError("score_mode must be one of {'mean', 'thompson'}.")
        if int(thompson_samples) <= 0:
            raise ValueError("thompson_samples must be positive.")
        if bias_model_correction not in {"none", "model_average"}:
            raise ValueError(
                "bias_model_correction must be one of {'none', 'model_average'}."
            )
        if locality_mode not in {"local", "global"}:
            raise ValueError("locality_mode must be one of {'local', 'global'}.")
        if base_prior_strength <= 0:
            raise ValueError("base_prior_strength must be positive.")
        if pool_prior_scale < 0:
            raise ValueError("pool_prior_scale must be non-negative.")
        if local_prior_scale < 0:
            raise ValueError("local_prior_scale must be non-negative.")
        if local_prior_min <= 0:
            raise ValueError("local_prior_min must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.score_mode = str(score_mode)
        self.thompson_samples = int(thompson_samples)
        self.bias_model_correction = str(bias_model_correction)
        self.locality_mode = str(locality_mode)
        self.base_prior_strength = float(base_prior_strength)
        self.pool_prior_scale = float(pool_prior_scale)
        self.local_prior_scale = float(local_prior_scale)
        self.local_prior_min = float(local_prior_min)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.metric = str(metric)
        self.missing_label = missing_label
        self.exclude_self = bool(exclude_self)
        self.store_neighbor_diagnostics = bool(store_neighbor_diagnostics)
        self.random_state = check_random_state(random_state)
        self.eps = float(eps)
        self._reset_diagnostics()

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf=None,
        remaining_budget=None,
        **kwargs,
    ):
        del kwargs
        if clf is None:
            raise ValueError("clf must be provided.")

        X = np.asarray(X)
        y = np.asarray(y)
        sample_indices = np.asarray(sample_indices, dtype=int)
        annotator_indices = np.asarray(annotator_indices, dtype=int)
        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        if n_sel_s == 0 or n_sel_a == 0:
            return np.empty((n_sel_s, n_sel_a), dtype=float)

        P, E = self._predict_probabilities_and_embeddings(clf, X)
        n_samples, n_classes = P.shape
        if n_classes < 2:
            raise ValueError(
                "BudgetAwareLocalAgreementScorer requires at least two classes."
            )
        if y.shape[0] != n_samples:
            raise ValueError(
                "y and classifier probabilities must have the same n_samples."
            )
        if y.ndim != 2:
            raise ValueError("y must have shape (n_samples, n_annotators).")

        classes = np.asarray(getattr(clf, "classes_", None))
        if classes.ndim != 1 or classes.size != n_classes:
            raise ValueError(
                "clf.classes_ must be a 1D array matching predict_proba columns."
            )

        observed_mask = self._observed_mask(y, clf)
        y_idx = self._encode_observed_labels(
            y=y,
            observed_mask=observed_mask,
            classes=classes,
        )
        pseudo_labels = np.argmax(P, axis=1)
        confidence = self._entropy_confidence(P)

        success, failure = self._agreement_evidence(
            y_idx=y_idx,
            observed_mask=observed_mask,
            pseudo_labels=pseudo_labels,
            confidence=confidence,
        )
        k_star = self._budget_k_star(
            observed_counts=observed_mask.sum(axis=0),
            total_observed=int(np.count_nonzero(observed_mask)),
            n_annotators=y.shape[1],
            remaining_budget=remaining_budget,
        )
        alpha0, beta0 = self._base_beta_prior(n_classes)
        global_prior = self._global_priors(
            success=success,
            failure=failure,
            k_star=k_star,
            alpha0=alpha0,
            beta0=beta0,
        )
        if self.locality_mode == "local":
            local = self._local_evidence(
                E=E,
                sample_indices=sample_indices,
                annotator_indices=annotator_indices,
                observed_mask=observed_mask,
                success=success,
                failure=failure,
                confidence=confidence,
                k_star=k_star,
            )
            nu = global_prior["nu_base"][annotator_indices][None, :] * np.maximum(
                1.0, local["rho"]
            )
            alpha = (
                nu * global_prior["mu_global"][annotator_indices][None, :]
                + local["success"]
            )
            beta = (
                nu * (1.0 - global_prior["mu_global"][annotator_indices][None, :])
                + local["failure"]
            )
        else:
            local = self._empty_local_diagnostics(n_sel_s, n_sel_a)
            nu = np.broadcast_to(
                global_prior["tau_pool"][annotator_indices][None, :],
                (n_sel_s, n_sel_a),
            ).copy()
            alpha = np.broadcast_to(
                global_prior["alpha_global"][annotator_indices][None, :],
                (n_sel_s, n_sel_a),
            ).copy()
            beta = np.broadcast_to(
                global_prior["beta_global"][annotator_indices][None, :],
                (n_sel_s, n_sel_a),
            ).copy()
        raw_score = alpha / np.maximum(alpha + beta, self.eps)

        bias = None
        if self.bias_model_correction == "model_average":
            bias = self._bias_model(
                P=P,
                y_idx=y_idx,
                observed_mask=observed_mask,
                confidence=confidence,
                pseudo_labels=pseudo_labels,
                n_classes=n_classes,
                alpha0=alpha0,
                beta0=beta0,
                sample_indices=sample_indices,
                annotator_indices=annotator_indices,
            )

        utilities = self._score_utilities(
            alpha=alpha,
            beta=beta,
            P_candidates=P[sample_indices],
            annotator_indices=annotator_indices,
            bias=bias,
        )

        feasible = ~observed_mask[np.ix_(sample_indices, annotator_indices)]
        if available_mask is not None:
            feasible &= np.asarray(available_mask, dtype=bool)
        utilities = np.where(feasible, utilities, np.nan)

        self._store_diagnostics(
            alpha=alpha,
            beta=beta,
            raw_score=raw_score,
            final_score=utilities,
            local=local,
            nu=nu,
            global_prior=global_prior,
            bias=bias,
        )
        return utilities

    def _predict_probabilities_and_embeddings(
        self, clf, X
    ) -> tuple[np.ndarray, np.ndarray]:
        out = clf.predict_proba(X, extra_outputs=["embeddings"])
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise ValueError(
                "clf.predict_proba(X, extra_outputs=['embeddings']) must return "
                "(probabilities, embeddings)."
            )
        P = self._normalize_probabilities(np.asarray(out[0], dtype=float))
        E = np.asarray(out[1], dtype=float)
        if E.shape[0] != P.shape[0]:
            raise ValueError("embeddings must have one row per probability row.")
        E = E.reshape(E.shape[0], -1)
        if self.normalize_embeddings:
            E = _l2_normalize(E, eps=self.eps)
        return P, E

    def _observed_mask(self, y: np.ndarray, clf) -> np.ndarray:
        missing_label = (
            getattr(clf, "missing_label", np.nan)
            if self.missing_label is None
            else self.missing_label
        )
        try:
            if bool(np.isnan(missing_label)):
                return ~np.isnan(y)
        except TypeError:
            pass
        return y != missing_label

    @staticmethod
    def _encode_observed_labels(*, y, observed_mask, classes):
        class_to_idx = {label: i for i, label in enumerate(classes)}
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

    def _agreement_evidence(
        self,
        *,
        y_idx: np.ndarray,
        observed_mask: np.ndarray,
        pseudo_labels: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        agreement = observed_mask & (y_idx == pseudo_labels[:, None])
        mass = confidence[:, None] * observed_mask
        success = mass * agreement
        failure = mass * (observed_mask & ~agreement)
        return success.astype(float), failure.astype(float)

    def _budget_k_star(
        self,
        *,
        observed_counts: np.ndarray,
        total_observed: int,
        n_annotators: int,
        remaining_budget,
    ) -> np.ndarray:
        if remaining_budget is None:
            T = float(total_observed) / max(n_annotators, 1)
            return np.full(n_annotators, int(np.ceil(np.sqrt(T))), dtype=int)

        budget = np.asarray(remaining_budget, dtype=float)
        if budget.ndim == 0:
            T = (float(total_observed) + max(float(budget), 0.0)) / max(n_annotators, 1)
            return np.full(n_annotators, int(np.ceil(np.sqrt(T))), dtype=int)
        if budget.shape != observed_counts.shape:
            raise ValueError(
                "remaining_budget must be scalar or have one value per annotator."
            )
        T_m = observed_counts.astype(float) + np.maximum(budget, 0.0)
        return np.ceil(np.sqrt(T_m)).astype(int)

    def _base_beta_prior(self, n_classes: int) -> tuple[float, float]:
        p0 = 1.0 / float(n_classes)
        return self.base_prior_strength * p0, self.base_prior_strength * (1.0 - p0)

    def _global_priors(
        self,
        *,
        success: np.ndarray,
        failure: np.ndarray,
        k_star: np.ndarray,
        alpha0: float,
        beta0: float,
    ) -> dict[str, np.ndarray | float]:
        S_pool = float(success.sum())
        F_pool = float(failure.sum())
        mu_pool = (alpha0 + S_pool) / max(alpha0 + beta0 + S_pool + F_pool, self.eps)

        S_m = success.sum(axis=0)
        F_m = failure.sum(axis=0)
        G_m = S_m + F_m
        tau = self.pool_prior_scale * k_star.astype(float)
        denom = tau + G_m
        mu_global = np.full(success.shape[1], mu_pool, dtype=float)
        valid = denom > self.eps
        mu_global[valid] = (tau[valid] * mu_pool + S_m[valid]) / denom[valid]
        nu_base = np.maximum(
            self.local_prior_min,
            self.local_prior_scale
            * np.minimum(np.maximum(k_star.astype(float), 0.0), G_m),
        )
        alpha_global = tau * mu_pool + S_m
        beta_global = tau * (1.0 - mu_pool) + F_m
        return {
            "mu_pool": float(mu_pool),
            "mu_global": mu_global,
            "nu_base": nu_base,
            "tau_pool": tau,
            "alpha_global": alpha_global,
            "beta_global": beta_global,
        }

    @staticmethod
    def _empty_local_diagnostics(n_sel_s: int, n_sel_a: int) -> dict[str, np.ndarray]:
        return {
            "success": np.zeros((n_sel_s, n_sel_a), dtype=float),
            "failure": np.zeros((n_sel_s, n_sel_a), dtype=float),
            "h_actual": np.full((n_sel_s, n_sel_a), np.nan, dtype=float),
            "h_ref": np.full((n_sel_s, n_sel_a), np.nan, dtype=float),
            "rho": np.ones((n_sel_s, n_sel_a), dtype=float),
            "k_local": np.zeros((n_sel_s, n_sel_a), dtype=int),
        }

    def _local_evidence(
        self,
        *,
        E: np.ndarray,
        sample_indices: np.ndarray,
        annotator_indices: np.ndarray,
        observed_mask: np.ndarray,
        success: np.ndarray,
        failure: np.ndarray,
        confidence: np.ndarray,
        k_star: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        out = {
            "success": np.zeros((n_sel_s, n_sel_a), dtype=float),
            "failure": np.zeros((n_sel_s, n_sel_a), dtype=float),
            "h_actual": np.full((n_sel_s, n_sel_a), np.nan, dtype=float),
            "h_ref": np.full((n_sel_s, n_sel_a), np.nan, dtype=float),
            "rho": np.ones((n_sel_s, n_sel_a), dtype=float),
            "k_local": np.zeros((n_sel_s, n_sel_a), dtype=int),
        }

        max_k = int(np.max(k_star[annotator_indices])) if n_sel_a else 0
        if self.store_neighbor_diagnostics and max_k > 0:
            out["neighbor_indices"] = np.full((n_sel_s, n_sel_a, max_k), -1, dtype=int)
            out["neighbor_distances"] = np.full((n_sel_s, n_sel_a, max_k), np.nan)
            out["neighbor_success"] = np.full((n_sel_s, n_sel_a, max_k), np.nan)
            out["neighbor_failure"] = np.full((n_sel_s, n_sel_a, max_k), np.nan)
            out["neighbor_confidence"] = np.full((n_sel_s, n_sel_a, max_k), np.nan)

        full_dist = pairwise_distances(E[sample_indices], E, metric=self.metric)
        if self.exclude_self:
            for row, sample_index in enumerate(sample_indices):
                if 0 <= sample_index < E.shape[0]:
                    full_dist[row, sample_index] = np.inf

        for local_j, annotator_index in enumerate(annotator_indices):
            obs_idx = np.flatnonzero(observed_mask[:, annotator_index])
            L_m = obs_idx.size
            k_m = int(min(k_star[annotator_index], L_m))
            if k_m <= 0:
                continue

            R_ref = int(
                np.clip(np.ceil(k_m * E.shape[0] / L_m), 1, max(E.shape[0] - 1, 1))
            )
            ref_sorted = np.sort(full_dist, axis=1)
            if E.shape[0] > 1:
                out["h_ref"][:, local_j] = ref_sorted[:, R_ref - 1]

            d_obs = pairwise_distances(
                E[sample_indices], E[obs_idx], metric=self.metric
            )
            if self.exclude_self:
                for row, sample_index in enumerate(sample_indices):
                    pos = np.flatnonzero(obs_idx == sample_index)
                    if pos.size:
                        d_obs[row, pos[0]] = np.inf

            for row in range(n_sel_s):
                finite = np.flatnonzero(np.isfinite(d_obs[row]))
                row_k = min(k_m, finite.size)
                if row_k <= 0:
                    continue
                order = finite[np.argsort(d_obs[row, finite])[:row_k]]
                neighbors = obs_idx[order]
                out["k_local"][row, local_j] = row_k
                out["h_actual"][row, local_j] = float(d_obs[row, order[-1]])
                out["success"][row, local_j] = float(
                    success[neighbors, annotator_index].sum()
                )
                out["failure"][row, local_j] = float(
                    failure[neighbors, annotator_index].sum()
                )
                h_ref = out["h_ref"][row, local_j]
                if np.isfinite(h_ref):
                    out["rho"][row, local_j] = out["h_actual"][row, local_j] / (
                        h_ref + self.eps
                    )

                if self.store_neighbor_diagnostics and max_k > 0:
                    sl = slice(0, row_k)
                    out["neighbor_indices"][row, local_j, sl] = neighbors
                    out["neighbor_distances"][row, local_j, sl] = d_obs[row, order]
                    out["neighbor_success"][row, local_j, sl] = success[
                        neighbors, annotator_index
                    ]
                    out["neighbor_failure"][row, local_j, sl] = failure[
                        neighbors, annotator_index
                    ]
                    out["neighbor_confidence"][row, local_j, sl] = confidence[neighbors]
        return out

    def _bias_model(
        self,
        *,
        P: np.ndarray,
        y_idx: np.ndarray,
        observed_mask: np.ndarray,
        confidence: np.ndarray,
        pseudo_labels: np.ndarray,
        n_classes: int,
        alpha0: float,
        beta0: float,
        sample_indices: np.ndarray,
        annotator_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        success, failure = self._agreement_evidence(
            y_idx=y_idx,
            observed_mask=observed_mask,
            pseudo_labels=pseudo_labels,
            confidence=confidence,
        )
        S_m = success.sum(axis=0)
        F_m = failure.sum(axis=0)

        response_counts = np.zeros((observed_mask.shape[1], n_classes), dtype=float)
        obs_s, obs_m = np.where(observed_mask)
        if obs_s.size:
            np.add.at(response_counts, (obs_m, y_idx[obs_s, obs_m]), confidence[obs_s])
        eta = np.full(n_classes, 1.0 / n_classes, dtype=float)
        eta_sum = float(eta.sum())
        Q_m = response_counts.sum(axis=1)

        log_resp = (
            betaln(alpha0 + S_m, beta0 + F_m)
            - betaln(alpha0, beta0)
            - F_m * np.log(float(n_classes - 1))
        )
        log_bias = (
            gammaln(eta_sum)
            - gammaln(eta_sum + Q_m)
            + np.sum(
                gammaln(eta[None, :] + response_counts) - gammaln(eta[None, :]), axis=1
            )
        )
        stacked = np.vstack([log_resp, log_bias])
        log_norm = logsumexp(stacked, axis=0)
        p_resp = np.exp(log_resp - log_norm)
        p_bias = 1.0 - p_resp
        response_mean = (eta[None, :] + response_counts) / np.maximum(
            eta_sum + Q_m[:, None],
            self.eps,
        )
        bias_score = P[sample_indices] @ response_mean[annotator_indices].T
        return {
            "p_resp": p_resp,
            "p_bias": p_bias,
            "log_resp": log_resp,
            "log_bias": log_bias,
            "response_counts": response_counts,
            "response_mean": response_mean,
            "bias_score": bias_score,
            "eta": eta,
        }

    def _score_utilities(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        P_candidates: np.ndarray,
        annotator_indices: np.ndarray,
        bias: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        mean = alpha / np.maximum(alpha + beta, self.eps)
        if self.score_mode == "mean":
            if bias is None:
                return mean
            p_resp = bias["p_resp"][annotator_indices][None, :]
            return p_resp * mean + (1.0 - p_resp) * bias["bias_score"]

        shape = (self.thompson_samples,) + alpha.shape
        alpha_draw = np.broadcast_to(alpha, shape)
        beta_draw = np.broadcast_to(beta, shape)
        theta = self.random_state.beta(alpha_draw, beta_draw)
        if bias is None:
            return theta.mean(axis=0)

        utility_draws = theta.copy()
        p_resp = bias["p_resp"][annotator_indices]
        z_resp = self.random_state.binomial(
            1, p_resp[None, :], size=(self.thompson_samples, len(annotator_indices))
        ).astype(bool)
        for local_j, annotator_index in enumerate(annotator_indices):
            params = bias["eta"] + bias["response_counts"][annotator_index]
            r_draws = self.random_state.dirichlet(params, size=self.thompson_samples)
            bias_draws = (P_candidates @ r_draws.T).T
            utility_draws[:, :, local_j] = np.where(
                z_resp[:, None, local_j],
                theta[:, :, local_j],
                bias_draws,
            )
        return utility_draws.mean(axis=0)

    @staticmethod
    def _normalize_probabilities(P: np.ndarray) -> np.ndarray:
        if P.ndim != 2:
            raise ValueError("classifier probabilities must be a 2D array.")
        P = np.clip(P, 0.0, None)
        row_sum = P.sum(axis=1, keepdims=True)
        if np.any(row_sum[:, 0] <= 0):
            raise ValueError("classifier probabilities must have positive row sums.")
        return P / row_sum

    def _entropy_confidence(self, P: np.ndarray) -> np.ndarray:
        safe = np.clip(P, self.eps, 1.0)
        entropy = -np.sum(P * np.log(safe), axis=1)
        return np.clip(1.0 - entropy / np.log(P.shape[1]), 0.0, 1.0)

    def _store_diagnostics(
        self,
        *,
        alpha,
        beta,
        raw_score,
        final_score,
        local,
        nu,
        global_prior,
        bias,
    ) -> None:
        self.last_alpha_ = alpha
        self.last_beta_ = beta
        self.last_raw_score_ = raw_score
        self.last_final_score_ = final_score
        self.last_h_actual_ = local["h_actual"]
        self.last_h_ref_ = local["h_ref"]
        self.last_rho_ = local["rho"]
        self.last_nu_ = nu
        self.last_k_local_ = local["k_local"]
        self.last_local_success_ = local["success"]
        self.last_local_failure_ = local["failure"]
        self.last_mu_pool_ = global_prior["mu_pool"]
        self.last_mu_global_ = global_prior["mu_global"]
        self.last_alpha_global_ = global_prior["alpha_global"]
        self.last_beta_global_ = global_prior["beta_global"]
        self.last_tau_pool_ = global_prior["tau_pool"]
        self.last_p_responsive_ = None if bias is None else bias["p_resp"]
        self.last_p_bias_ = None if bias is None else bias["p_bias"]
        self.last_log_likelihood_responsive_ = (
            None if bias is None else bias["log_resp"]
        )
        self.last_log_likelihood_bias_ = None if bias is None else bias["log_bias"]
        self.last_bias_score_ = None if bias is None else bias["bias_score"]
        self.last_neighbor_indices_ = local.get("neighbor_indices")
        self.last_neighbor_distances_ = local.get("neighbor_distances")
        self.last_neighbor_success_ = local.get("neighbor_success")
        self.last_neighbor_failure_ = local.get("neighbor_failure")
        self.last_neighbor_confidence_ = local.get("neighbor_confidence")

    def _reset_diagnostics(self) -> None:
        self.last_alpha_ = None
        self.last_beta_ = None
        self.last_raw_score_ = None
        self.last_final_score_ = None
        self.last_h_actual_ = None
        self.last_h_ref_ = None
        self.last_rho_ = None
        self.last_nu_ = None
        self.last_k_local_ = None
        self.last_local_success_ = None
        self.last_local_failure_ = None
        self.last_mu_pool_ = None
        self.last_mu_global_ = None
        self.last_alpha_global_ = None
        self.last_beta_global_ = None
        self.last_tau_pool_ = None
        self.last_p_responsive_ = None
        self.last_p_bias_ = None
        self.last_log_likelihood_responsive_ = None
        self.last_log_likelihood_bias_ = None
        self.last_bias_score_ = None
        self.last_neighbor_indices_ = None
        self.last_neighbor_distances_ = None
        self.last_neighbor_success_ = None
        self.last_neighbor_failure_ = None
        self.last_neighbor_confidence_ = None
