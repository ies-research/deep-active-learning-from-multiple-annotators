from __future__ import annotations

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel

from skactiveml.utils import is_labeled

from ._ks_bag import KernelSmoothedBayesianAnnotatorGain, _l2_normalize
from ._local_response_bias_mixture import LocalResponseBiasMixtureGain
from ._utils import expected_score_gain


class LikelihoodLocalResponseBiasMixtureGain(LocalResponseBiasMixtureGain):
    """
    Local response-bias mixture scorer with likelihood-based theta inference.

    For a candidate pair (x, a), this scorer estimates a local response-bias
    distribution g and a local task-dependence strength theta under

        p(L = l | Z = c, x, a) = theta * 1[l = c] + (1 - theta) * g_l.

    Unlike :class:`LocalResponseBiasMixtureGain`, theta is inferred directly
    from the observed-label likelihood on a one-dimensional grid.
    """

    def __init__(
        self,
        *,
        gamma_x="median",
        gamma_x_scope: str = "global",
        gamma_a="median",
        use_annotator_embeddings: bool = True,
        annotator_lambda: float = 0.0,
        class_prior: str = "classifier",
        class_prior_strength: float = 1.0,
        class_prior_lambda: float = 0.0,
        use_ess_class_prior: bool = False,
        tau_class_prior: float = 1.0,
        sample_class_prior: bool = False,
        response_dirichlet_strength: float = 1.0,
        use_ess_response_dirichlet: bool = False,
        tau_response_dirichlet: float = 1.0,
        sample_response_dirichlet: bool = False,
        sample_response_dirichlet_for_theta: bool | None = None,
        theta_alpha0: float = 1.0,
        theta_beta0: float = 1.0,
        theta_grid_size: int = 64,
        theta_grid_eps: float = 1e-4,
        sample_observed_class_probabilities_for_theta: bool = False,
        observed_class_dirichlet_mode: str = "fixed",
        observed_class_dirichlet_strength: float = 10.0,
        n_mc_samples: int = 1,
        gain_ucb_quantile: float | None = None,
        gain_batch_size: int | None = None,
        use_mi_cap: bool = True,
        use_response_entropy_cap: bool = False,
        response_entropy_cap_lambda: float = 1.0,
        random_state=None,
    ):
        super().__init__(
            gamma_x=gamma_x,
            gamma_x_scope=gamma_x_scope,
            gamma_a=gamma_a,
            use_annotator_embeddings=use_annotator_embeddings,
            annotator_lambda=annotator_lambda,
            class_prior=class_prior,
            class_prior_strength=class_prior_strength,
            class_prior_lambda=class_prior_lambda,
            use_ess_class_prior=use_ess_class_prior,
            tau_class_prior=tau_class_prior,
            sample_class_prior=sample_class_prior,
            response_dirichlet_strength=response_dirichlet_strength,
            use_ess_response_dirichlet=use_ess_response_dirichlet,
            tau_response_dirichlet=tau_response_dirichlet,
            sample_response_dirichlet=sample_response_dirichlet,
            theta_alpha0=theta_alpha0,
            theta_beta0=theta_beta0,
            theta_min_denom=0.0,
            theta_ess_max=None,
            tau_theta=1.0,
            n_mc_samples=n_mc_samples,
            gain_ucb_quantile=gain_ucb_quantile,
            gain_batch_size=gain_batch_size,
            random_state=random_state,
        )
        self.theta_grid_size = int(theta_grid_size)
        self.theta_grid_eps = float(theta_grid_eps)
        self.sample_response_dirichlet_for_theta = (
            self.sample_response_dirichlet
            if sample_response_dirichlet_for_theta is None
            else bool(sample_response_dirichlet_for_theta)
        )
        self.sample_observed_class_probabilities_for_theta = bool(
            sample_observed_class_probabilities_for_theta
        )
        self.observed_class_dirichlet_mode = str(observed_class_dirichlet_mode)
        self.observed_class_dirichlet_strength = float(
            observed_class_dirichlet_strength
        )
        self.use_mi_cap = bool(use_mi_cap)
        self.use_response_entropy_cap = bool(use_response_entropy_cap)
        self.response_entropy_cap_lambda = float(response_entropy_cap_lambda)

        if self.theta_grid_size < 2:
            raise ValueError("theta_grid_size must be >= 2")
        if not (0.0 < self.theta_grid_eps < 0.5):
            raise ValueError("theta_grid_eps must be in (0, 0.5)")
        if self.observed_class_dirichlet_mode not in {"fixed", "kernel"}:
            raise ValueError(
                "observed_class_dirichlet_mode must be one of "
                "{'fixed', 'kernel'}"
            )
        if self.observed_class_dirichlet_strength <= 0:
            raise ValueError("observed_class_dirichlet_strength must be > 0")
        if self.response_entropy_cap_lambda < 0.0:
            raise ValueError("response_entropy_cap_lambda must be >= 0")

    def _compute(
        self,
        X,
        y,
        sample_indices,
        annotator_indices,
        available_mask,
        clf,
        **kwargs,
    ):
        rng = kwargs.get("rng", None)
        if rng is None:
            rng = np.random.default_rng(
                self.random_state.randint(0, 2**32 - 1)
            )

        classes = np.asarray(clf.classes_)
        K = len(classes)
        if K < 2:
            raise ValueError(
                "Likelihood local response-bias mixture requires >=2 classes."
            )

        cand_extra_outputs = ["embeddings"]
        if self.use_annotator_embeddings:
            cand_extra_outputs.append("annotator_embeddings")
        cand_out = clf.predict_proba(
            X[sample_indices],
            extra_outputs=cand_extra_outputs,
        )
        if not isinstance(cand_out, (tuple, list)) or len(cand_out) < 2:
            raise ValueError(
                "clf.predict_proba must return class probabilities and "
                "embeddings when extra_outputs are requested."
            )

        r_cand = self._normalize_probabilities(cand_out[0])
        X_cand_emb = _l2_normalize(np.asarray(cand_out[1], dtype=float))
        A_pred = (
            cand_out[2]
            if (self.use_annotator_embeddings and len(cand_out) > 2)
            else None
        )

        is_lbld = is_labeled(y=y, missing_label=clf.missing_label)
        obs_s, obs_a = np.where(is_lbld)
        if obs_s.size == 0:
            U = np.full(
                (len(sample_indices), len(annotator_indices)),
                np.nan,
                dtype=float,
            )
            if available_mask is not None:
                U = np.where(available_mask, U, np.nan)
            return U

        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_obs_raw = np.asarray(y[obs_s, obs_a])
        try:
            y_obs_idx = np.array([class_to_idx[v] for v in y_obs_raw], dtype=int)
        except KeyError as e:
            raise ValueError(f"Observed label {e.args[0]!r} not found in clf.classes_")
        y_obs_oh = np.eye(K, dtype=float)[y_obs_idx]

        obs_out = clf.predict_proba(X[obs_s], extra_outputs=["embeddings"])
        if not isinstance(obs_out, (tuple, list)) or len(obs_out) < 2:
            raise ValueError(
                "clf.predict_proba must return class probabilities and "
                "embeddings for observed annotation evidence."
            )
        r_obs = self._normalize_probabilities(obs_out[0])
        X_obs_emb = _l2_normalize(np.asarray(obs_out[1], dtype=float))

        _, obs_first_idx = np.unique(obs_s, return_index=True)
        X_obs_cls_emb = X_obs_emb[obs_first_idx]
        r_obs_cls = r_obs[obs_first_idx]

        n_annotators_total = y.shape[1]
        A_all = None
        A_obs_emb = None
        use_annotator_kernel = False
        gamma_a_val = None
        if A_pred is not None:
            A_pred = np.asarray(A_pred, dtype=float)
            if A_pred.ndim == 2 and A_pred.shape[0] == n_annotators_total:
                A_all = _l2_normalize(A_pred)
                A_obs_emb = A_all[obs_a]
                use_annotator_kernel = True

        gamma_x_global = KernelSmoothedBayesianAnnotatorGain._resolve_gamma_from_embeddings(
            X_obs_emb,
            self.gamma_x,
        )
        if use_annotator_kernel:
            gamma_a_val = KernelSmoothedBayesianAnnotatorGain._resolve_gamma_from_embeddings(
                A_all,
                self.gamma_a,
            )

        r_cand_prior = self._resolve_class_prior(
            r=r_cand,
            X_cand_emb=X_cand_emb,
            X_obs_cls_emb=X_obs_cls_emb,
            r_obs_cls=r_obs_cls,
            gamma_x=gamma_x_global,
            rng=rng,
        )
        observed_class_alpha_all = None
        if (
            self.sample_observed_class_probabilities_for_theta
            and self.observed_class_dirichlet_mode == "kernel"
        ):
            observed_class_alpha_all = self._observed_class_dirichlet_posterior(
                X_query_emb=X_obs_emb,
                X_obs_cls_emb=X_obs_cls_emb,
                r_obs_cls=r_obs_cls,
                gamma_x=gamma_x_global,
            )

        Kx_obs_cand_local_global = rbf_kernel(
            X_obs_emb, X_cand_emb, gamma=gamma_x_global
        )
        gamma0 = np.full(K, self.response_dirichlet_strength / K, dtype=float)

        U = np.empty((len(sample_indices), len(annotator_indices)), dtype=float)
        obs_indices_by_annotator = None
        if not use_annotator_kernel:
            obs_indices_by_annotator = [
                np.flatnonzero(obs_a == idx)
                for idx in range(n_annotators_total)
            ]

        for j_a, a in enumerate(annotator_indices):
            a = int(a)
            if a < 0 or a >= n_annotators_total:
                raise ValueError(
                    f"Annotator index {a} out of bounds for y with "
                    f"{n_annotators_total} annotators."
                )

            if self.gamma_x_scope == "per_annotator":
                obs_mask_a = obs_a == a
                if np.count_nonzero(obs_mask_a) >= 1:
                    gamma_x_a = KernelSmoothedBayesianAnnotatorGain._resolve_gamma_from_embeddings(
                        X_obs_emb[obs_mask_a],
                        self.gamma_x,
                    )
                else:
                    gamma_x_a = gamma_x_global
                Kx_obs_cand_local = rbf_kernel(
                    X_obs_emb, X_cand_emb, gamma=gamma_x_a
                )
            else:
                Kx_obs_cand_local = Kx_obs_cand_local_global

            Kx_obs_cand = KernelSmoothedBayesianAnnotatorGain._mix_with_global_sample_kernel(
                Kx_obs_cand_local,
                lam=self.annotator_lambda,
            )
            if use_annotator_kernel:
                Ka_obs = rbf_kernel(
                    A_obs_emb, A_all[[a]], gamma=gamma_a_val
                ).reshape(-1)
                K_pair = Kx_obs_cand * Ka_obs[:, None]
                y_pair_oh = y_obs_oh
                y_pair_idx = y_obs_idx
                r_pair = r_obs
                observed_class_alpha_pair = observed_class_alpha_all
            else:
                obs_idx_a = obs_indices_by_annotator[a]
                K_pair = Kx_obs_cand[obs_idx_a]
                y_pair_oh = y_obs_oh[obs_idx_a]
                y_pair_idx = y_obs_idx[obs_idx_a]
                r_pair = r_obs[obs_idx_a]
                observed_class_alpha_pair = (
                    None
                    if observed_class_alpha_all is None
                    else observed_class_alpha_all[obs_idx_a]
                )

            gamma_response, _ = (
                KernelSmoothedBayesianAnnotatorGain.parzen_dirichlet_posterior(
                    K=K_pair,
                    Y=y_pair_oh,
                    gamma0=gamma0,
                    use_ess=self.use_ess_response_dirichlet,
                    tau=self.tau_response_dirichlet,
                )
            )

            U_col = self._likelihood_gain_batch(
                r=r_cand_prior,
                K=K_pair,
                r_obs=r_pair,
                y_obs_idx=y_pair_idx,
                gamma=gamma_response,
                observed_class_alpha=observed_class_alpha_pair,
                rng=rng,
            )
            if available_mask is not None:
                U_col = np.where(available_mask[:, j_a], U_col, np.nan)
            U[:, j_a] = U_col

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U

    def _likelihood_gain_batch(
        self,
        *,
        r: np.ndarray,
        K: np.ndarray,
        r_obs: np.ndarray,
        y_obs_idx: np.ndarray,
        gamma: np.ndarray,
        rng: np.random.Generator,
        observed_class_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        K = np.asarray(K, dtype=float)
        r_obs = np.asarray(r_obs, dtype=float)
        y_obs_idx = np.asarray(y_obs_idx, dtype=int)
        gamma = np.asarray(gamma, dtype=float)
        observed_class_alpha = (
            None
            if observed_class_alpha is None
            else np.asarray(observed_class_alpha, dtype=float)
        )

        if r.ndim != 3:
            raise ValueError("r must have shape (n_samples, n_draws, n_classes).")
        if K.ndim != 2 or K.shape[1] != r.shape[0]:
            raise ValueError("K must have shape (n_obs, n_samples).")
        if r_obs.ndim != 2 or r_obs.shape[0] != K.shape[0]:
            raise ValueError("r_obs must have shape (n_obs, n_classes).")
        if y_obs_idx.ndim != 1 or y_obs_idx.shape[0] != K.shape[0]:
            raise ValueError("y_obs_idx must have shape (n_obs,).")
        if gamma.ndim != 2 or gamma.shape[0] != r.shape[0]:
            raise ValueError("gamma must have shape (n_samples, n_classes).")
        if observed_class_alpha is not None:
            if observed_class_alpha.shape != r_obs.shape:
                raise ValueError(
                    "observed_class_alpha must have shape (n_obs, n_classes)."
                )

        batch_size = self.gain_batch_size
        if batch_size is not None and r.shape[0] > batch_size:
            gains = np.empty(r.shape[0], dtype=float)
            for start in range(0, r.shape[0], batch_size):
                stop = min(start + batch_size, r.shape[0])
                gains[start:stop] = self._likelihood_gain_batch_inner(
                    r=r[start:stop],
                    K=K[:, start:stop],
                    r_obs=r_obs,
                    y_obs_idx=y_obs_idx,
                    gamma=gamma[start:stop],
                    observed_class_alpha=observed_class_alpha,
                    rng=rng,
                )
            return gains

        return self._likelihood_gain_batch_inner(
            r=r,
            K=K,
            r_obs=r_obs,
            y_obs_idx=y_obs_idx,
            gamma=gamma,
            observed_class_alpha=observed_class_alpha,
            rng=rng,
        )

    def _likelihood_gain_batch_inner(
        self,
        *,
        r: np.ndarray,
        K: np.ndarray,
        r_obs: np.ndarray,
        y_obs_idx: np.ndarray,
        gamma: np.ndarray,
        observed_class_alpha: np.ndarray | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T = r.shape[1]
        g_gain = self._sample_response_distribution_batch(
            gamma=gamma,
            rng=rng,
            n_draws=T,
        )
        g_theta = self._response_distribution_for_theta(
            gamma=gamma,
            g_gain=g_gain,
            n_draws=T,
        )
        r_obs_theta = self._sample_observed_class_probabilities_for_theta(
            r_obs=r_obs,
            observed_class_alpha=observed_class_alpha,
            rng=rng,
            n_draws=T,
        )
        theta = self._sample_theta_from_likelihood_grid(
            K=K,
            r_obs=r_obs_theta,
            y_obs_idx=y_obs_idx,
            g=g_theta,
            rng=rng,
        )
        C = self.response_bias_mixture_confusion(theta=theta, g=g_gain)
        gain_draws = expected_score_gain(
            r,
            C=C,
            score="entropy",
            normalize=True,
            check_input=False,
            batch_size=self.gain_batch_size,
        )
        gain_draws = self._apply_gain_caps(gain_draws, r=r, C=C, g=g_gain)
        return self._aggregate_gain_draws(gain_draws)

    def _response_distribution_for_theta(
        self,
        *,
        gamma: np.ndarray,
        g_gain: np.ndarray,
        n_draws: int,
    ) -> np.ndarray:
        if self.sample_response_dirichlet_for_theta:
            return g_gain
        mean = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
        return np.repeat(mean[:, None, :], n_draws, axis=1)

    def _sample_theta_from_likelihood_grid(
        self,
        *,
        K: np.ndarray,
        r_obs: np.ndarray,
        y_obs_idx: np.ndarray,
        g: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        theta_grid = self._theta_grid()
        log_post = self.theta_log_posterior_grid(
            theta_grid=theta_grid,
            K=K,
            r_obs=r_obs,
            y_obs_idx=y_obs_idx,
            g=g,
            theta_alpha0=self.theta_alpha0,
            theta_beta0=self.theta_beta0,
        )
        probs = self.normalize_log_weights(log_post)
        if self.n_mc_samples <= 0:
            return np.sum(probs * theta_grid[None, None, :], axis=-1)

        cdf = np.cumsum(probs, axis=-1)
        draws = rng.random(cdf.shape[:-1] + (1,))
        idx = np.sum(cdf < draws, axis=-1)
        idx = np.minimum(idx, theta_grid.size - 1)
        return theta_grid[idx]

    def _sample_observed_class_probabilities_for_theta(
        self,
        *,
        r_obs: np.ndarray,
        rng: np.random.Generator,
        n_draws: int,
        observed_class_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        r_obs = np.asarray(r_obs, dtype=float)
        if not self.sample_observed_class_probabilities_for_theta:
            return r_obs

        if self.observed_class_dirichlet_mode == "fixed":
            alpha = self.observed_class_dirichlet_strength * r_obs
        elif self.observed_class_dirichlet_mode == "kernel":
            if observed_class_alpha is None:
                raise ValueError(
                    "observed_class_alpha is required when "
                    "observed_class_dirichlet_mode='kernel'."
                )
            alpha = np.asarray(observed_class_alpha, dtype=float)
        else:
            raise ValueError(
                "observed_class_dirichlet_mode must be one of "
                "{'fixed', 'kernel'}"
            )

        alpha = np.clip(alpha, 1e-12, None)
        if self.n_mc_samples <= 0:
            return alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)

        alpha = alpha[:, None, :]
        if n_draws != 1:
            alpha = np.repeat(alpha, n_draws, axis=1)
        x = rng.gamma(shape=alpha, scale=1.0)
        return x / np.maximum(x.sum(axis=-1, keepdims=True), 1e-12)

    def _observed_class_dirichlet_posterior(
        self,
        *,
        X_query_emb: np.ndarray,
        X_obs_cls_emb: np.ndarray,
        r_obs_cls: np.ndarray,
        gamma_x: float,
    ) -> np.ndarray:
        K_cls_local = rbf_kernel(
            X_obs_cls_emb,
            X_query_emb,
            gamma=float(gamma_x),
        )
        K_cls = KernelSmoothedBayesianAnnotatorGain._mix_with_global_sample_kernel(
            K_cls_local,
            lam=self.class_prior_lambda,
        )
        alpha0 = np.full(
            r_obs_cls.shape[1],
            self.class_prior_strength / r_obs_cls.shape[1],
            dtype=float,
        )
        alpha, _ = KernelSmoothedBayesianAnnotatorGain.parzen_dirichlet_posterior(
            K=K_cls,
            Y=r_obs_cls,
            gamma0=alpha0,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )
        return alpha

    @staticmethod
    def theta_log_posterior_grid(
        *,
        theta_grid: np.ndarray,
        K: np.ndarray,
        r_obs: np.ndarray,
        y_obs_idx: np.ndarray,
        g: np.ndarray,
        theta_alpha0: float,
        theta_beta0: float,
        eps: float = 1e-12,
        max_elements: int = 5_000_000,
    ) -> np.ndarray:
        theta_grid = np.asarray(theta_grid, dtype=float)
        K = np.asarray(K, dtype=float)
        r_obs = np.asarray(r_obs, dtype=float)
        y_obs_idx = np.asarray(y_obs_idx, dtype=int)
        g = np.asarray(g, dtype=float)

        if K.ndim == 1:
            K = K[:, None]
        if g.ndim == 2:
            g = g[:, None, :]
        if theta_grid.ndim != 1:
            raise ValueError("theta_grid must be 1D.")
        if K.ndim != 2:
            raise ValueError("K must have shape (n_obs, n_samples).")
        if r_obs.ndim not in {2, 3}:
            raise ValueError(
                "r_obs must have shape (n_obs, n_classes) or "
                "(n_obs, n_draws, n_classes)."
            )
        if y_obs_idx.ndim != 1:
            raise ValueError("y_obs_idx must have shape (n_obs,).")
        if K.shape[0] != r_obs.shape[0] or K.shape[0] != y_obs_idx.shape[0]:
            raise ValueError("K, r_obs, and y_obs_idx must agree on observations.")
        if g.ndim != 3:
            raise ValueError("g must have shape (n_samples, n_draws, n_classes).")
        if g.shape[0] != K.shape[1]:
            raise ValueError("g and K must agree on n_samples.")
        r_obs_classes = r_obs.shape[-1]
        if g.shape[2] != r_obs_classes:
            raise ValueError("g and r_obs must agree on n_classes.")
        if r_obs.ndim == 3 and r_obs.shape[1] != g.shape[1]:
            raise ValueError("3D r_obs must agree with g on n_draws.")
        if theta_alpha0 <= 0 or theta_beta0 <= 0:
            raise ValueError("theta_alpha0 and theta_beta0 must be > 0")
        if np.any((theta_grid <= 0.0) | (theta_grid >= 1.0)):
            raise ValueError("theta_grid entries must be in (0, 1).")

        B, T, _ = g.shape
        R = theta_grid.size
        log_prior = (
            (theta_alpha0 - 1.0) * np.log(theta_grid)
            + (theta_beta0 - 1.0) * np.log1p(-theta_grid)
        )
        log_post = np.broadcast_to(log_prior, (B, T, R)).copy()
        if K.shape[0] == 0:
            return log_post

        if r_obs.ndim == 2:
            h = r_obs[np.arange(y_obs_idx.size), y_obs_idx]
        else:
            h = r_obs[np.arange(y_obs_idx.size), :, y_obs_idx]
        obs_chunk = max(1, int(max_elements // max(B * T * R, 1)))
        for start in range(0, K.shape[0], obs_chunk):
            stop = min(start + obs_chunk, K.shape[0])
            labels = y_obs_idx[start:stop]
            h_chunk = h[start:stop]
            g_label = g[:, :, labels]
            if h.ndim == 1:
                h_term = h_chunk[None, None, :, None]
            else:
                h_term = np.swapaxes(h_chunk, 0, 1)[None, :, :, None]
            likelihood = (
                theta_grid[None, None, None, :] * h_term
                + (1.0 - theta_grid[None, None, None, :])
                * g_label[:, :, :, None]
            )
            log_like = np.log(np.clip(likelihood, eps, 1.0))
            log_post += np.einsum(
                "ob,btor->btr",
                K[start:stop],
                log_like,
                optimize=True,
            )

        return log_post

    @staticmethod
    def normalize_log_weights(
        log_w: np.ndarray,
        *,
        eps: float = 1e-12,
    ) -> np.ndarray:
        log_w = np.asarray(log_w, dtype=float)
        max_log_w = np.max(log_w, axis=-1, keepdims=True)
        weights = np.exp(log_w - max_log_w)
        total = weights.sum(axis=-1, keepdims=True)
        invalid = (~np.isfinite(total)) | (total <= eps)
        probs = weights / np.maximum(total, eps)
        if np.any(invalid):
            probs = np.where(invalid, 1.0 / log_w.shape[-1], probs)
        return probs

    @staticmethod
    def theta_grid_summary(
        theta_grid: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_grid = np.asarray(theta_grid, dtype=float)
        probs = np.asarray(probs, dtype=float)
        mean = np.sum(probs * theta_grid, axis=-1)
        var = np.sum(probs * (theta_grid - mean[..., None]) ** 2, axis=-1)
        mode = theta_grid[np.argmax(probs, axis=-1)]
        return mean, var, mode

    def _apply_gain_caps(
        self,
        gain: np.ndarray,
        *,
        r: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        gain = np.maximum(np.asarray(gain, dtype=float), 0.0)
        if self.use_mi_cap:
            q = np.einsum("...k,...ky->...y", r, C)
            cap = np.minimum(
                self._entropy_bits(r),
                self._entropy_bits(q),
            )
            gain = np.minimum(gain, cap)
        if self.use_response_entropy_cap:
            g_cap = self.response_entropy_cap_lambda * self._entropy_bits(g)
            gain = np.minimum(gain, g_cap)
        return gain

    def _theta_grid(self) -> np.ndarray:
        return np.linspace(
            self.theta_grid_eps,
            1.0 - self.theta_grid_eps,
            self.theta_grid_size,
            dtype=float,
        )
