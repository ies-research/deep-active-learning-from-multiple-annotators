from __future__ import annotations

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled

from ._base import PairScorer
from ._ks_bag import KernelSmoothedBayesianAnnotatorGain, _l2_normalize
from ._utils import expected_score_gain


class LocalResponseBiasMixtureGain(PairScorer):
    """
    Local response-bias mixture scorer.

    For a candidate pair (x, a), this scorer estimates a local annotator
    response-bias distribution g and a local task-dependence strength theta,
    then computes entropy mutual information under

        p(L = l | Z = c, x, a) = theta * 1[l = c] + (1 - theta) * g_l.

    The theta posterior is a Beta approximation built from classifier
    probabilities assigned to previously observed annotator labels, corrected
    for the response-bias baseline.
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
        theta_alpha0: float = 1.0,
        theta_beta0: float = 1.0,
        theta_min_denom: float = 1e-6,
        theta_ess_max: float | None = None,
        tau_theta: float = 1.0,
        n_mc_samples: int = 1,
        gain_ucb_quantile: float | None = None,
        gain_batch_size: int | None = None,
        random_state=None,
    ):
        self.gamma_x = gamma_x
        self.gamma_x_scope = str(gamma_x_scope)
        self.gamma_a = gamma_a
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.annotator_lambda = float(annotator_lambda)
        self.class_prior = str(class_prior)
        self.class_prior_strength = float(class_prior_strength)
        self.class_prior_lambda = float(class_prior_lambda)
        self.use_ess_class_prior = bool(use_ess_class_prior)
        self.tau_class_prior = float(tau_class_prior)
        self.sample_class_prior = bool(sample_class_prior)
        self.response_dirichlet_strength = float(response_dirichlet_strength)
        self.use_ess_response_dirichlet = bool(use_ess_response_dirichlet)
        self.tau_response_dirichlet = float(tau_response_dirichlet)
        self.sample_response_dirichlet = bool(sample_response_dirichlet)
        self.theta_alpha0 = float(theta_alpha0)
        self.theta_beta0 = float(theta_beta0)
        self.theta_min_denom = float(theta_min_denom)
        self.theta_ess_max = (
            None if theta_ess_max is None else float(theta_ess_max)
        )
        self.tau_theta = float(tau_theta)
        self.n_mc_samples = int(n_mc_samples)
        self.gain_ucb_quantile = (
            None if gain_ucb_quantile is None else float(gain_ucb_quantile)
        )
        self.gain_batch_size = (
            None if gain_batch_size is None else int(gain_batch_size)
        )
        self.random_state = check_random_state(random_state)

        if self.gamma_x_scope not in {"global", "per_annotator"}:
            raise ValueError(
                "gamma_x_scope must be one of {'global', 'per_annotator'}"
            )
        if not (0.0 <= self.annotator_lambda <= 1.0):
            raise ValueError("annotator_lambda must be in [0, 1]")
        if self.class_prior not in {"classifier", "uniform", "kernel"}:
            raise ValueError(
                "class_prior must be one of {'classifier', 'uniform', 'kernel'}"
            )
        if not (0.0 <= self.class_prior_lambda <= 1.0):
            raise ValueError("class_prior_lambda must be in [0, 1]")
        if self.class_prior_strength <= 0:
            raise ValueError("class_prior_strength must be > 0")
        if self.tau_class_prior <= 0:
            raise ValueError("tau_class_prior must be > 0")
        if self.sample_class_prior and self.class_prior != "kernel":
            raise ValueError(
                "sample_class_prior=True requires class_prior='kernel'"
            )
        if self.class_prior != "kernel":
            if self.class_prior_strength != 1.0:
                raise ValueError(
                    "class_prior_strength is only used when class_prior='kernel'"
                )
            if self.use_ess_class_prior:
                raise ValueError(
                    "use_ess_class_prior is only supported when class_prior='kernel'"
                )
            if self.tau_class_prior != 1.0:
                raise ValueError(
                    "tau_class_prior is only used when class_prior='kernel'"
                )
            if self.class_prior_lambda != 0.0:
                raise ValueError(
                    "class_prior_lambda is only used when class_prior='kernel'"
                )
        if self.response_dirichlet_strength <= 0:
            raise ValueError("response_dirichlet_strength must be > 0")
        if self.tau_response_dirichlet <= 0:
            raise ValueError("tau_response_dirichlet must be > 0")
        if self.theta_alpha0 <= 0 or self.theta_beta0 <= 0:
            raise ValueError("theta_alpha0 and theta_beta0 must be > 0")
        if self.theta_min_denom < 0:
            raise ValueError("theta_min_denom must be >= 0")
        if self.theta_ess_max is not None and self.theta_ess_max <= 0:
            raise ValueError("theta_ess_max must be positive or None")
        if self.tau_theta <= 0:
            raise ValueError("tau_theta must be > 0")
        if self.gain_ucb_quantile is not None:
            if not (0.0 < self.gain_ucb_quantile < 1.0):
                raise ValueError("gain_ucb_quantile must be in (0, 1)")
            if self.n_mc_samples <= 0:
                raise ValueError("gain_ucb_quantile requires n_mc_samples > 0")
        if self.gain_batch_size is not None and self.gain_batch_size <= 0:
            raise ValueError("gain_batch_size must be positive or None")

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
            raise ValueError("Local response-bias mixture requires >=2 classes.")

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
            else:
                obs_idx_a = obs_indices_by_annotator[a]
                K_pair = Kx_obs_cand[obs_idx_a]
                y_pair_oh = y_obs_oh[obs_idx_a]
                y_pair_idx = y_obs_idx[obs_idx_a]
                r_pair = r_obs[obs_idx_a]

            gamma_response, _ = (
                KernelSmoothedBayesianAnnotatorGain.parzen_dirichlet_posterior(
                    K=K_pair,
                    Y=y_pair_oh,
                    gamma0=gamma0,
                    use_ess=self.use_ess_response_dirichlet,
                    tau=self.tau_response_dirichlet,
                )
            )
            g_mean = gamma_response / np.maximum(
                gamma_response.sum(axis=1, keepdims=True),
                1e-12,
            )

            alpha_theta, beta_theta = self.theta_posterior_from_bias_moments(
                K=K_pair,
                r_obs=r_pair,
                y_obs_idx=y_pair_idx,
                g_mean=g_mean,
                theta_alpha0=self.theta_alpha0,
                theta_beta0=self.theta_beta0,
                theta_min_denom=self.theta_min_denom,
                theta_ess_max=self.theta_ess_max,
                tau_theta=self.tau_theta,
            )

            U_col = self._gain_batch(
                r=r_cand_prior,
                alpha=alpha_theta,
                beta=beta_theta,
                gamma=gamma_response,
                rng=rng,
            )
            if available_mask is not None:
                U_col = np.where(available_mask[:, j_a], U_col, np.nan)
            U[:, j_a] = U_col

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)
        return U

    def _resolve_class_prior(
        self,
        r: np.ndarray,
        *,
        X_cand_emb: np.ndarray | None = None,
        X_obs_cls_emb: np.ndarray | None = None,
        r_obs_cls: np.ndarray | None = None,
        gamma_x: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        T = self._mc_draw_count()
        if self.class_prior == "classifier":
            return np.repeat(r[:, None, :], T, axis=1)
        K = r.shape[1]
        if self.class_prior == "uniform":
            return np.full((r.shape[0], T, K), 1.0 / K, dtype=float)
        if (
            X_cand_emb is None
            or X_obs_cls_emb is None
            or r_obs_cls is None
            or gamma_x is None
        ):
            raise ValueError(
                "class_prior='kernel' requires candidate embeddings, "
                "deduplicated labeled-sample embeddings, labeled-sample "
                "posteriors, and gamma_x."
            )

        K_cls_local = rbf_kernel(
            X_obs_cls_emb, X_cand_emb, gamma=float(gamma_x)
        )
        K_cls = KernelSmoothedBayesianAnnotatorGain._mix_with_global_sample_kernel(
            K_cls_local,
            lam=self.class_prior_lambda,
        )
        alpha0 = np.full(K, self.class_prior_strength / K, dtype=float)
        alpha, _ = KernelSmoothedBayesianAnnotatorGain.parzen_dirichlet_posterior(
            K=K_cls,
            Y=r_obs_cls,
            gamma0=alpha0,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )
        if self.sample_class_prior and self.n_mc_samples > 0:
            if rng is None:
                raise ValueError("sample_class_prior=True requires an RNG.")
            alpha_bt = np.clip(alpha[:, None, :], 1e-12, None)
            if T != 1:
                alpha_bt = np.repeat(alpha_bt, T, axis=1)
            X = rng.gamma(shape=alpha_bt, scale=1.0)
            return X / np.maximum(X.sum(axis=2, keepdims=True), 1e-12)

        mean = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        return np.repeat(mean[:, None, :], T, axis=1)

    @staticmethod
    def theta_posterior_from_bias_moments(
        *,
        K: np.ndarray,
        r_obs: np.ndarray,
        y_obs_idx: np.ndarray,
        g_mean: np.ndarray,
        theta_alpha0: float,
        theta_beta0: float,
        theta_min_denom: float,
        theta_ess_max: float | None,
        tau_theta: float,
        eps: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray]:
        K = np.asarray(K, dtype=float)
        r_obs = np.asarray(r_obs, dtype=float)
        y_obs_idx = np.asarray(y_obs_idx, dtype=int)
        g_mean = np.asarray(g_mean, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}.")
        if r_obs.ndim != 2:
            raise ValueError(f"r_obs must be 2D, got shape {r_obs.shape}.")
        if y_obs_idx.ndim != 1:
            raise ValueError("y_obs_idx must be 1D.")
        if K.shape[0] != r_obs.shape[0] or K.shape[0] != y_obs_idx.shape[0]:
            raise ValueError("K, r_obs, and y_obs_idx must agree on observations.")
        if g_mean.ndim != 2 or g_mean.shape[1] != r_obs.shape[1]:
            raise ValueError("g_mean must have shape (n_candidates, n_classes).")
        if g_mean.shape[0] != K.shape[1]:
            raise ValueError("g_mean must agree with K on n_candidates.")

        n_candidates = K.shape[1]
        if K.shape[0] == 0:
            return (
                np.full(n_candidates, theta_alpha0, dtype=float),
                np.full(n_candidates, theta_beta0, dtype=float),
            )

        h = r_obs[np.arange(y_obs_idx.size), y_obs_idx]
        r2 = np.sum(r_obs * r_obs, axis=1)
        baseline = r_obs @ g_mean.T
        denom = r2[:, None] - baseline
        abs_denom = np.abs(denom)
        valid = abs_denom > theta_min_denom

        theta_hat = np.zeros_like(denom, dtype=float)
        np.divide(
            h[:, None] - baseline,
            denom,
            out=theta_hat,
            where=valid,
        )
        theta_hat = np.clip(theta_hat, 0.0, 1.0)
        raw_weight = np.where(valid, K * abs_denom, 0.0)

        mass = raw_weight.sum(axis=0)
        weighted_theta = (raw_weight * theta_hat).sum(axis=0)
        theta_bar = np.divide(
            weighted_theta,
            np.maximum(mass, eps),
            out=np.zeros_like(weighted_theta),
            where=mass > eps,
        )

        weight_sq = (raw_weight * raw_weight).sum(axis=0)
        n_eff = np.divide(
            mass * mass,
            np.maximum(weight_sq, eps),
            out=np.zeros_like(mass),
            where=mass > eps,
        )
        if theta_ess_max is not None:
            n_eff = np.minimum(n_eff, theta_ess_max)
        conc = tau_theta * n_eff

        alpha = theta_alpha0 + conc * theta_bar
        beta = theta_beta0 + conc * (1.0 - theta_bar)
        no_evidence = mass <= eps
        if np.any(no_evidence):
            alpha[no_evidence] = theta_alpha0
            beta[no_evidence] = theta_beta0
        return alpha, beta

    def _gain_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        if r.ndim != 3:
            raise ValueError("r must have shape (n_samples, n_draws, n_classes).")
        if gamma.ndim != 2 or gamma.shape[0] != r.shape[0]:
            raise ValueError("gamma must have shape (n_samples, n_classes).")

        batch_size = self.gain_batch_size
        if batch_size is not None and r.shape[0] > batch_size:
            gains = np.empty(r.shape[0], dtype=float)
            for start in range(0, r.shape[0], batch_size):
                stop = min(start + batch_size, r.shape[0])
                gains[start:stop] = self._gain_batch_inner(
                    r=r[start:stop],
                    alpha=alpha[start:stop],
                    beta=beta[start:stop],
                    gamma=gamma[start:stop],
                    rng=rng,
                )
            return gains

        return self._gain_batch_inner(
            r=r,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rng=rng,
        )

    def _gain_batch_inner(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        T = r.shape[1]
        theta = self._sample_theta_batch(
            alpha=alpha,
            beta=beta,
            rng=rng,
            n_draws=T,
        )
        g = self._sample_response_distribution_batch(
            gamma=gamma,
            rng=rng,
            n_draws=T,
        )
        C = self.response_bias_mixture_confusion(theta=theta, g=g)
        gain_draws = expected_score_gain(
            r,
            C=C,
            score="entropy",
            normalize=True,
            check_input=False,
            batch_size=self.gain_batch_size,
        )
        gain_draws = self._clip_entropy_gain(gain_draws, r=r, C=C)
        return self._aggregate_gain_draws(gain_draws)

    @staticmethod
    def response_bias_mixture_confusion(
        *,
        theta: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        g = np.asarray(g, dtype=float)
        if theta.ndim != 2:
            raise ValueError("theta must have shape (n_samples, n_draws).")
        if g.ndim != 3:
            raise ValueError("g must have shape (n_samples, n_draws, n_classes).")
        if theta.shape != g.shape[:2]:
            raise ValueError("theta and g must agree on samples and draws.")

        theta = np.clip(theta, 0.0, 1.0)
        g = np.clip(g, 1e-12, 1.0)
        g = g / np.maximum(g.sum(axis=-1, keepdims=True), 1e-12)
        K = g.shape[-1]
        C = np.broadcast_to(
            (1.0 - theta)[..., None, None] * g[..., None, :],
            theta.shape + (K, K),
        ).copy()
        idx = np.arange(K)
        C[..., idx, idx] += theta[..., None]
        return C

    def _sample_theta_batch(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
        n_draws: int,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if self.n_mc_samples <= 0:
            return self._theta_point_estimate(alpha=alpha, beta=beta)[:, None]
        return rng.beta(
            alpha[:, None],
            beta[:, None],
            size=(alpha.shape[0], n_draws),
        ).astype(float)

    def _sample_response_distribution_batch(
        self,
        *,
        gamma: np.ndarray,
        rng: np.random.Generator,
        n_draws: int,
    ) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=float)
        if self._use_mc_response_dirichlet():
            alpha = np.clip(gamma[:, None, :], 1e-12, None)
            if n_draws != 1:
                alpha = np.repeat(alpha, n_draws, axis=1)
            x = rng.gamma(shape=alpha, scale=1.0)
            return x / np.maximum(x.sum(axis=-1, keepdims=True), 1e-12)

        mean = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
        return np.repeat(mean[:, None, :], n_draws, axis=1)

    @staticmethod
    def _theta_point_estimate(
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        return np.clip(alpha / np.maximum(alpha + beta, 1e-12), 0.0, 1.0)

    def _aggregate_gain_draws(self, gain_draws: np.ndarray) -> np.ndarray:
        gain_draws = np.asarray(gain_draws, dtype=float)
        if gain_draws.ndim != 2:
            raise ValueError("gain_draws must have shape (n_samples, n_draws).")
        if self.gain_ucb_quantile is None:
            return gain_draws.mean(axis=1)
        return np.quantile(gain_draws, self.gain_ucb_quantile, axis=1)

    @staticmethod
    def _clip_entropy_gain(
        gain: np.ndarray,
        *,
        r: np.ndarray,
        C: np.ndarray,
        eps: float = 1e-12,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        q = np.einsum("...k,...ky->...y", r, C)
        cap = np.minimum(
            LocalResponseBiasMixtureGain._entropy_bits(r, eps=eps),
            LocalResponseBiasMixtureGain._entropy_bits(q, eps=eps),
        )
        return np.minimum(np.maximum(gain, 0.0), cap)

    @staticmethod
    def _entropy_bits(P: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        P = np.clip(P, eps, 1.0)
        P = P / np.maximum(P.sum(axis=-1, keepdims=True), eps)
        return -(P * (np.log(P) / np.log(2.0))).sum(axis=-1)

    @staticmethod
    def _normalize_probabilities(P: np.ndarray, eps: float = 1e-15) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        P = np.clip(P, eps, 1.0)
        return P / np.maximum(P.sum(axis=1, keepdims=True), eps)

    def _mc_draw_count(self) -> int:
        return 1 if self.n_mc_samples <= 0 else self.n_mc_samples

    def _use_mc_response_dirichlet(self) -> bool:
        return self.sample_response_dirichlet and self.n_mc_samples > 0
