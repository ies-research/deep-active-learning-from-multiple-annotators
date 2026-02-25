from __future__ import annotations

import numpy as np

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled
from ._base import PairScorer
from ._perf import information_gain


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class IGKernelChannelPairScorer(PairScorer):
    """
    Pair scorer using information gain under a kernel-smoothed annotator model.

    For a candidate pair (x, a), this scorer supports multiple channel variants:

    - "channel" (default, historical):
        Z ~ r(·) = p(Z|x)                 from clf
        theta ~ Beta(alpha(x,a), beta(x,a))
        g ~ Dirichlet(gamma(x,a))
        Y = Z              with prob theta
        Y ~ Categorical(g) with prob (1-theta)

    - "scalar_uniform_confusion":
        estimate a single accuracy scalar theta and define a proper confusion matrix
        with uniform off-diagonal mass:
            C[z,z] = theta
            C[z,y!=z] = (1-theta)/(K-1)

    - "diag_uniform_confusion":
        estimate per-class diagonal accuracies theta_z and define rows
            C[z,z] = theta_z
            C[z,y!=z] = (1-theta_z)/(K-1)

    - "full_confusion":
        estimate the full row-stochastic confusion matrix C[z,y].

    Kernel-smoothed posterior parameters are built from observed annotations
    (x_i, a_i, y_i):

    - Beta correctness model (soft counts using classifier probabilities)
        m_i = p_clf(y_i | x_i)
        s(x,a) = sum_i w_i(x,a) * m_i
        f(x,a) = sum_i w_i(x,a) * (1 - m_i)
        alpha = alpha0 + s,  beta = beta0 + f
      (optionally ESS-scaled via `use_ess_beta=True`)

    - Dirichlet label model (kernel-weighted label counts)
        gamma_k(x,a) = gamma0_k + sum_i w_i(x,a) * 1[y_i = k]
      (optionally ESS-scaled via `use_ess_label_dirichlet=True`)

    where the pair weight factorizes as:
        w_i(x,a) = k_x(x_i, x) * k_a(a_i, a)

    `k_a` uses annotator embeddings when available. If annotator embeddings are not
    available (or are not global), the scorer falls back to exact annotator identity
    weighting: k_a(a_i, a) = 1[a_i = a].

    Utility is the mutual information:
        IG(x,a) = I(Z; Y | x, a)
    evaluated via Monte Carlo samples from Beta/Dirichlet (or using posterior means).

    Parameters
    ----------
    accuracy_mean : float, default=0.95
        Prior mean accuracy. Used as the Beta prior mean for variants with scalar/diagonal
        accuracies and as the diagonal prior mean of each Dirichlet confusion row for
        `channel_variant="full_confusion"`.
    accuracy_strength : float, default=10.0
        Prior strength for accuracy parameters. Used for Beta priors in variants with
        scalar/diagonal accuracies and as the total concentration per confusion row in
        `channel_variant="full_confusion"`.
    gamma_x : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the sample-embedding RBF kernel.
    gamma_a : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the annotator-embedding RBF kernel (if used).
    use_annotator_embeddings : bool, default=True
        If True, request and use global annotator embeddings (when provided by `clf`)
        to smooth across annotators. If False, always use exact annotator identity
        weighting.
    channel_label_dirichlet_strength : float, default=1.0
        Symmetric Dirichlet prior concentration for the fallback label distribution `g`
        in `channel_variant="channel"` only.
    channel_variant : {"channel","scalar_uniform_confusion","diag_uniform_confusion","full_confusion"}, default="channel"
        Annotator noise parameterization used for IG computation.
    use_ess_beta : bool, default=False
        If True, map the kernel-weighted correctness evidence to a Beta posterior using
        ESS-based concentration instead of raw weighted counts.
    tau_beta : float, default=1.0
        Discount factor for ESS-based Beta concentration (only used if `use_ess_beta=True`).
    use_ess_label_dirichlet : bool, default=False
        If True, map kernel-weighted label evidence to a Dirichlet posterior using
        ESS-based concentration instead of raw weighted counts.
    tau_label_dirichlet : float, default=1.0
        Discount factor for ESS-based Dirichlet concentration (only used if
        `use_ess_label_dirichlet=True`).
    top_m : int or None, default=2
        If not None, approximate IG in top-M + "other" reduced label space.
        Currently used only for `channel_variant="channel"`.
    n_theta_samples : int, default=1
        Number of Monte Carlo draws for latent channel parameters. For variants with
        Beta accuracies, this controls Beta draws. If <=0, posterior means are used
        (when applicable).
    sample_label_dirichlet : bool, default=False
        If True, sample Dirichlet-distributed label parameters (`g` in `channel`,
        confusion rows in `full_confusion`); otherwise use posterior means.
    random_state : None or int, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        accuracy_mean: float = 0.95,
        accuracy_strength: float = 10.0,
        gamma_x="median",
        gamma_a="median",
        use_annotator_embeddings: bool = True,
        channel_label_dirichlet_strength: float = 1.0,
        channel_variant: str = "channel",
        use_ess_beta: bool = False,
        tau_beta: float = 1.0,
        use_ess_label_dirichlet: bool = False,
        tau_label_dirichlet: float = 1.0,
        top_m: int | None = 2,
        n_theta_samples: int = 1,
        sample_label_dirichlet: bool = False,
        random_state=None,
    ):
        self.accuracy_mean = float(accuracy_mean)
        self.accuracy_strength = float(accuracy_strength)
        self.gamma_x = gamma_x
        self.gamma_a = gamma_a
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.channel_label_dirichlet_strength = float(
            channel_label_dirichlet_strength
        )
        self.channel_variant = str(channel_variant)
        self.use_ess_beta = bool(use_ess_beta)
        self.tau_beta = float(tau_beta)
        self.use_ess_label_dirichlet = bool(use_ess_label_dirichlet)
        self.tau_label_dirichlet = float(tau_label_dirichlet)
        self.top_m = top_m
        self.n_theta_samples = int(n_theta_samples)
        self.sample_label_dirichlet = bool(sample_label_dirichlet)
        self.random_state = check_random_state(random_state)

        if not (0.0 < self.accuracy_mean < 1.0):
            raise ValueError("accuracy_mean must be in (0, 1)")
        if self.accuracy_strength <= 0:
            raise ValueError("accuracy_strength must be > 0")
        if self.channel_label_dirichlet_strength <= 0:
            raise ValueError("channel_label_dirichlet_strength must be > 0")

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

        classes = clf.classes_
        K = len(classes)
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        valid_variants = {
            "channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
            "full_confusion",
        }
        if self.channel_variant not in valid_variants:
            raise ValueError(
                f"Unknown channel_variant={self.channel_variant!r}. "
                f"Expected one of {sorted(valid_variants)}."
            )

        # Candidate sample posteriors/embeddings and (optionally) annotator embeddings.
        cand_extra_outputs = ["embeddings"]
        if self.use_annotator_embeddings:
            cand_extra_outputs.append("annotator_embeddings")
        cand_out = clf.predict_proba(
            X[sample_indices],
            extra_outputs=cand_extra_outputs,
        )
        if not isinstance(cand_out, (tuple, list)):
            raise ValueError(
                "clf.predict_proba must return a tuple when extra_outputs are requested."
            )
        if len(cand_out) < 2:
            raise ValueError(
                "clf.predict_proba returned too few outputs for requested embeddings."
            )
        r_cand = cand_out[0]
        X_cand_emb = cand_out[1]
        A_pred = cand_out[2] if (self.use_annotator_embeddings and len(cand_out) > 2) else None
        r_cand = np.asarray(r_cand, dtype=float)
        r_cand = np.clip(r_cand, 1e-15, 1.0)
        r_cand = r_cand / np.maximum(r_cand.sum(axis=1, keepdims=True), 1e-15)
        X_cand_emb = _l2_normalize(np.asarray(X_cand_emb, dtype=float))

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

        y_obs = y[obs_s, obs_a].astype(int)
        y_obs_oh = np.eye(K, dtype=float)[y_obs]

        # Observed sample embeddings + classifier probabilities for soft correctness counts.
        r_obs, X_obs_emb = clf.predict_proba(
            X[obs_s], extra_outputs=["embeddings"]
        )
        r_obs = np.asarray(r_obs, dtype=float)
        r_obs = np.clip(r_obs, 1e-15, 1.0)
        r_obs = r_obs / np.maximum(r_obs.sum(axis=1, keepdims=True), 1e-15)
        X_obs_emb = _l2_normalize(np.asarray(X_obs_emb, dtype=float))
        m_obs = r_obs[np.arange(obs_s.size), y_obs]
        m_obs = np.clip(m_obs, 0.0, 1.0)

        # Global annotator embeddings are optional. If unavailable, use exact annotator identity weights.
        A_all = None
        A_obs_emb = None
        use_annotator_kernel = False
        gamma_a_val = None
        if A_pred is not None:
            A_pred = np.asarray(A_pred, dtype=float)
            if A_pred.ndim == 2 and A_pred.shape[0] == y.shape[1]:
                A_all = _l2_normalize(A_pred)
                A_obs_emb = A_all[obs_a]
                use_annotator_kernel = True

        gamma_x_val = self._resolve_gamma_from_embeddings(
            X_obs_emb, self.gamma_x
        )
        if use_annotator_kernel:
            gamma_a_val = self._resolve_gamma_from_embeddings(
                A_all, self.gamma_a
            )

        # Sample-kernel weights from observed pairs to candidate samples.
        Kx_obs_cand = rbf_kernel(X_obs_emb, X_cand_emb, gamma=gamma_x_val)

        alpha0 = self.accuracy_mean * self.accuracy_strength
        beta0 = (1.0 - self.accuracy_mean) * self.accuracy_strength
        gamma0 = np.full(
            K, self.channel_label_dirichlet_strength / K, dtype=float
        )
        delta0_full = self._full_confusion_dirichlet_prior(
            K=K,
            accuracy_mean=self.accuracy_mean,
            row_strength=self.accuracy_strength,
        )

        U = np.empty(
            (len(sample_indices), len(annotator_indices)), dtype=float
        )

        for j_a, a in enumerate(annotator_indices):
            if use_annotator_kernel:
                Ka_obs = rbf_kernel(
                    A_obs_emb, A_all[[a]], gamma=gamma_a_val
                ).reshape(-1)
            else:
                Ka_obs = (obs_a == a).astype(float)

            K_obs_cand = Kx_obs_cand * Ka_obs[:, None]

            if self.channel_variant in {
                "channel",
                "scalar_uniform_confusion",
                "diag_uniform_confusion",
            }:
                alpha, beta, _ = self.parzen_beta_posterior(
                    K=K_obs_cand,
                    p=m_obs,
                    alpha0=alpha0,
                    beta0=beta0,
                    use_ess=self.use_ess_beta,
                    tau=self.tau_beta,
                )
            else:
                alpha = beta = None

            gamma_cand = None
            if self.channel_variant == "channel":
                gamma_cand, _ = self.parzen_dirichlet_posterior(
                    K=K_obs_cand,
                    Y=y_obs_oh,
                    gamma0=gamma0,
                    use_ess=self.use_ess_label_dirichlet,
                    tau=self.tau_label_dirichlet,
                )

                # Deterministic full-K path: reuse the shared closed-form IG implementation.
                if (
                    self.n_theta_samples <= 0
                    and not self.sample_label_dirichlet
                    and (self.top_m is None or self.top_m >= K)
                ):
                    theta_mean = (alpha / np.maximum(alpha + beta, 1e-12))[:, None]
                    U[:, j_a] = information_gain(
                        r_cand,
                        theta_mean,
                        gamma_cand[:, None, :],
                        normalize=True,
                        check_input=False,
                    )[:, 0]
                    continue

            if self.channel_variant == "diag_uniform_confusion":
                alpha_diag = np.empty((len(sample_indices), K), dtype=float)
                beta_diag = np.empty((len(sample_indices), K), dtype=float)
                y_eq = y_obs_oh
                for z in range(K):
                    K_row = K_obs_cand * r_obs[:, [z]]
                    a_z, b_z, _ = self.parzen_beta_posterior(
                        K=K_row,
                        p=y_eq[:, z],
                        alpha0=alpha0,
                        beta0=beta0,
                        use_ess=self.use_ess_beta,
                        tau=self.tau_beta,
                    )
                    alpha_diag[:, z] = a_z
                    beta_diag[:, z] = b_z
            else:
                alpha_diag = beta_diag = None

            if self.channel_variant == "full_confusion":
                confusion_rows = np.empty(
                    (len(sample_indices), K, K), dtype=float
                )
                for z in range(K):
                    K_row = K_obs_cand * r_obs[:, [z]]
                    row_post, _ = self.parzen_dirichlet_posterior(
                        K=K_row,
                        Y=y_obs_oh,
                        gamma0=delta0_full[z],
                        use_ess=self.use_ess_label_dirichlet,
                        tau=self.tau_label_dirichlet,
                    )
                    confusion_rows[:, z, :] = row_post
            else:
                confusion_rows = None

            for j_x in range(len(sample_indices)):
                if available_mask is not None and not available_mask[j_x, j_a]:
                    U[j_x, j_a] = np.nan
                    continue

                r = r_cand[j_x]

                if self.channel_variant == "channel":
                    if self.top_m is None or self.top_m >= K:
                        U[j_x, j_a] = self._ig_channel_full(
                            r=r,
                            alpha=float(alpha[j_x]),
                            beta=float(beta[j_x]),
                            gamma=gamma_cand[j_x],
                            rng=rng,
                        )
                    else:
                        U[j_x, j_a] = self._ig_channel_topm(
                            r=r,
                            alpha=float(alpha[j_x]),
                            beta=float(beta[j_x]),
                            gamma=gamma_cand[j_x],
                            top_m=int(self.top_m),
                            rng=rng,
                        )
                elif self.channel_variant == "scalar_uniform_confusion":
                    U[j_x, j_a] = self._ig_scalar_uniform_confusion(
                        r=r,
                        alpha=float(alpha[j_x]),
                        beta=float(beta[j_x]),
                        rng=rng,
                    )
                elif self.channel_variant == "diag_uniform_confusion":
                    U[j_x, j_a] = self._ig_diag_uniform_confusion(
                        r=r,
                        alpha=alpha_diag[j_x],
                        beta=beta_diag[j_x],
                        rng=rng,
                    )
                else:  # full_confusion
                    U[j_x, j_a] = self._ig_full_confusion(
                        r=r,
                        delta=confusion_rows[j_x],
                        rng=rng,
                    )

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)

        return U

    # -------------------------
    # IG computation
    # -------------------------
    def _ig_channel_full(
        self,
        *,
        r: np.ndarray,
        alpha: float,
        beta: float,
        gamma: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        """
        Full K-class IG under channel:
            p(y|z,theta,g) = theta*1[y=z] + (1-theta)*g_y
        with theta ~ Beta(alpha,beta), g ~ Dirichlet(gamma).
        """
        if self.n_theta_samples <= 0:
            thetas = np.array([alpha / (alpha + beta)], dtype=float)
        else:
            thetas = rng.beta(alpha, beta, size=self.n_theta_samples).astype(
                float
            )

        if self.sample_label_dirichlet:
            gs = rng.dirichlet(gamma, size=thetas.size)
        else:
            g_mean = gamma / np.maximum(gamma.sum(), 1e-12)
            gs = np.tile(g_mean[None, :], (thetas.size, 1))

        H_prior = self._entropy(r)

        igs = []
        for theta, g in zip(thetas, gs):
            py = theta * r + (1.0 - theta) * g
            py = np.clip(py, 1e-15, 1.0)
            py = py / py.sum()

            H_post_exp = 0.0
            base = (1.0 - theta) * g
            for ell in range(r.size):
                L = np.full_like(r, base[ell])
                L[ell] = theta + base[ell]
                post = r * L
                post = post / np.maximum(post.sum(), 1e-15)
                H_post_exp += py[ell] * self._entropy(post)

            igs.append(H_prior - H_post_exp)

        return float(np.mean(igs))

    def _ig_channel_topm(
        self,
        *,
        r: np.ndarray,
        alpha: float,
        beta: float,
        gamma: np.ndarray,
        top_m: int,
        rng: np.random.Generator,
    ) -> float:
        """
        Top-M + 'other' IG approximation.
        """
        if top_m <= 0:
            raise ValueError("top_m must be >= 1 when used.")

        idx = np.array(np.argsort(-r)[:top_m], dtype=int)

        r_top = r[idx]
        r_other = 1.0 - r_top.sum()
        r_red = np.concatenate([r_top, [max(r_other, 0.0)]], axis=0)
        r_red = np.clip(r_red, 1e-15, 1.0)
        r_red = r_red / r_red.sum()
        Kr = r_red.size

        if self.n_theta_samples <= 0:
            thetas = np.array([alpha / (alpha + beta)], dtype=float)
        else:
            thetas = rng.beta(alpha, beta, size=self.n_theta_samples).astype(
                float
            )

        if self.sample_label_dirichlet:
            g_fulls = rng.dirichlet(gamma, size=thetas.size)
        else:
            g_mean_full = gamma / np.maximum(gamma.sum(), 1e-12)
            g_fulls = np.tile(g_mean_full[None, :], (thetas.size, 1))

        H_prior = self._entropy(r_red)

        igs = []
        for theta, g_full in zip(thetas, g_fulls):
            g_top = g_full[idx]
            g_other = 1.0 - g_top.sum()
            g_red = np.concatenate([g_top, [max(g_other, 0.0)]], axis=0)
            g_red = np.clip(g_red, 1e-15, 1.0)
            g_red = g_red / g_red.sum()

            py = theta * r_red + (1.0 - theta) * g_red
            py = np.clip(py, 1e-15, 1.0)
            py = py / py.sum()

            base = (1.0 - theta) * g_red
            H_post_exp = 0.0
            for ell in range(Kr):
                L = np.full_like(r_red, base[ell])
                L[ell] = theta + base[ell]
                post = r_red * L
                post = post / np.maximum(post.sum(), 1e-15)
                H_post_exp += py[ell] * self._entropy(post)

            igs.append(H_prior - H_post_exp)

        return float(np.mean(igs))

    def _ig_scalar_uniform_confusion(
        self,
        *,
        r: np.ndarray,
        alpha: float,
        beta: float,
        rng: np.random.Generator,
    ) -> float:
        if self.n_theta_samples <= 0:
            thetas = np.array([alpha / (alpha + beta)], dtype=float)
        else:
            thetas = rng.beta(alpha, beta, size=self.n_theta_samples).astype(
                float
            )

        K = r.size
        igs = []
        for theta in thetas:
            C = self._confusion_from_scalar_theta(K=K, theta=float(theta))
            igs.append(self._ig_from_confusion(r=r, C=C))
        return float(np.mean(igs))

    def _ig_diag_uniform_confusion(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if self.n_theta_samples <= 0:
            thetas = (alpha / np.maximum(alpha + beta, 1e-12))[None, :]
        else:
            thetas = rng.beta(alpha, beta, size=(self.n_theta_samples, r.size))
            thetas = np.asarray(thetas, dtype=float)

        igs = []
        for theta_vec in thetas:
            C = self._confusion_from_diag_thetas(theta=np.asarray(theta_vec))
            igs.append(self._ig_from_confusion(r=r, C=C))
        return float(np.mean(igs))

    def _ig_full_confusion(
        self,
        *,
        r: np.ndarray,
        delta: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        delta = np.asarray(delta, dtype=float)
        if delta.ndim != 2 or delta.shape[0] != delta.shape[1]:
            raise ValueError(
                f"delta must be square (K,K), got shape {delta.shape}"
            )

        if self.sample_label_dirichlet:
            n_draws = max(1, self.n_theta_samples if self.n_theta_samples > 0 else 1)
            Cs = np.empty((n_draws, delta.shape[0], delta.shape[1]), dtype=float)
            for t in range(n_draws):
                for z in range(delta.shape[0]):
                    Cs[t, z, :] = rng.dirichlet(delta[z])
        else:
            C_mean = delta / np.maximum(delta.sum(axis=1, keepdims=True), 1e-12)
            Cs = C_mean[None, :, :]

        igs = [self._ig_from_confusion(r=r, C=C) for C in Cs]
        return float(np.mean(igs))

    @classmethod
    def _confusion_from_scalar_theta(cls, *, K: int, theta: float) -> np.ndarray:
        if K < 2:
            raise ValueError("K must be >= 2")
        theta = float(np.clip(theta, 0.0, 1.0))
        off = (1.0 - theta) / (K - 1)
        C = np.full((K, K), off, dtype=float)
        np.fill_diagonal(C, theta)
        return C

    @classmethod
    def _confusion_from_diag_thetas(cls, *, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        K = theta.size
        if K < 2:
            raise ValueError("theta must have length >= 2")
        theta = np.clip(theta, 0.0, 1.0)
        off = (1.0 - theta) / (K - 1)
        C = np.repeat(off[:, None], K, axis=1)
        C[np.arange(K), np.arange(K)] = theta
        return C

    @classmethod
    def _full_confusion_dirichlet_prior(
        cls, *, K: int, accuracy_mean: float, row_strength: float
    ) -> np.ndarray:
        if K < 2:
            raise ValueError("K must be >= 2")
        if not (0.0 < accuracy_mean < 1.0):
            raise ValueError("accuracy_mean must be in (0, 1)")
        if row_strength <= 0:
            raise ValueError("row_strength must be > 0")
        off = (1.0 - accuracy_mean) / (K - 1)
        prior_mean = np.full((K, K), off, dtype=float)
        np.fill_diagonal(prior_mean, accuracy_mean)
        return row_strength * prior_mean

    @classmethod
    def _ig_from_confusion(cls, *, r: np.ndarray, C: np.ndarray) -> float:
        r = np.asarray(r, dtype=float)
        C = np.asarray(C, dtype=float)
        r = np.clip(r, 1e-15, 1.0)
        r = r / np.maximum(r.sum(), 1e-15)
        C = np.clip(C, 1e-15, 1.0)
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-15)

        py = r @ C
        py = np.clip(py, 1e-15, 1.0)
        py = py / py.sum()

        H_prior = cls._entropy(r)
        H_cond = 0.0
        for y_idx in range(C.shape[1]):
            post = r * C[:, y_idx]
            post = post / np.maximum(post.sum(), 1e-15)
            H_cond += py[y_idx] * cls._entropy(post)
        return float(max(H_prior - H_cond, 0.0))

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-15, 1.0)
        p = p / p.sum()
        return float(-(p * np.log2(p)).sum())

    # -------------------------
    # Gamma resolution
    # -------------------------
    @staticmethod
    def _resolve_gamma_from_embeddings(E: np.ndarray, mode):
        E = np.asarray(E, dtype=float)
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
        return float(mode)

    @staticmethod
    def parzen_beta_posterior(
        K: np.ndarray,
        p: np.ndarray,
        *,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        use_ess: bool = False,
        tau: float = 1.0,
        eps: float = 1e-12,
    ):
        K = np.asarray(K, dtype=float)
        p = np.asarray(p, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if p.ndim != 1:
            raise ValueError(f"p must be 1D, got shape {p.shape}")
        if K.shape[0] != p.shape[0]:
            raise ValueError(
                f"Shape mismatch: K rows {K.shape[0]} vs p {p.shape[0]}"
            )
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("alpha0 and beta0 must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        p = np.clip(p, 0.0, 1.0)
        mass = K.sum(axis=0)
        s = p @ K
        f = (1.0 - p) @ K
        mu = np.where(mass > eps, s / np.maximum(mass, eps), 0.5)

        if not use_ess:
            alpha = alpha0 + s
            beta = beta0 + f
            info = {"mu": mu, "mass": mass, "s": s, "f": f}
            return alpha, beta, info

        m2 = (K**2).sum(axis=0)
        n_eff = (mass**2) / np.maximum(m2, eps)
        conc = tau * n_eff

        alpha = alpha0 + conc * mu
        beta = beta0 + conc * (1.0 - mu)

        info = {"mu": mu, "mass": mass, "n_eff": n_eff}
        return alpha, beta, info

    @staticmethod
    def parzen_dirichlet_posterior(
        K: np.ndarray,
        Y: np.ndarray,
        *,
        gamma0: np.ndarray,
        use_ess: bool = False,
        tau: float = 1.0,
        eps: float = 1e-12,
    ):
        K = np.asarray(K, dtype=float)
        Y = np.asarray(Y, dtype=float)
        gamma0 = np.asarray(gamma0, dtype=float)

        if K.ndim != 2:
            raise ValueError(f"K must be 2D, got shape {K.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D, got shape {Y.shape}")
        if K.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Shape mismatch: K rows {K.shape[0]} vs Y rows {Y.shape[0]}"
            )
        if Y.shape[1] != gamma0.shape[0]:
            raise ValueError(
                f"Y classes {Y.shape[1]} must equal gamma0 length {gamma0.shape[0]}"
            )
        if np.any(gamma0 <= 0):
            raise ValueError("gamma0 entries must be > 0")
        if tau <= 0:
            raise ValueError("tau must be > 0")

        counts = K.T @ Y
        mass = counts.sum(axis=1)
        mu = counts / np.maximum(mass[:, None], eps)

        if not use_ess:
            gamma = gamma0[None, :] + counts
            info = {"counts": counts, "mass": mass, "mu": mu}
            return gamma, info

        k_mass = K.sum(axis=0)
        k_m2 = (K**2).sum(axis=0)
        n_eff = (k_mass**2) / np.maximum(k_m2, eps)
        conc = tau * n_eff
        gamma = gamma0[None, :] + conc[:, None] * mu
        info = {"mass": mass, "mu": mu, "n_eff": n_eff}
        return gamma, info
