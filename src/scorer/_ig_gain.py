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
    Pair scorer using information gain under a kernel-smoothed channel model.

    For a candidate pair (x, a):
        Z ~ r(·) = p(Z|x)                 from clf
        theta ~ Beta(alpha(x,a), beta(x,a))
        g ~ Dirichlet(gamma(x,a))
        Y = Z              with prob theta
        Y ~ Categorical(g) with prob (1-theta)

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
    mean : float, default=0.95
        Prior mean for theta.
    strength : float, default=10.0
        Prior strength for theta: alpha0 = mean*strength, beta0=(1-mean)*strength.
    gamma_x : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the sample-embedding RBF kernel.
    gamma_a : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the annotator-embedding RBF kernel (if used).
    dirichlet_strength : float, default=1.0
        Symmetric Dirichlet prior concentration. Each class gets gamma0 = dirichlet_strength / K.
    use_ess_beta : bool, default=False
        If True, map the kernel-weighted correctness evidence to a Beta posterior using
        ESS-based concentration instead of raw weighted counts.
    tau_beta : float, default=1.0
        Discount factor for ESS-based Beta concentration (only used if `use_ess_beta=True`).
    top_m : int or None, default=2
        If not None, approximate IG in top-M + "other" reduced label space.
    n_theta_samples : int, default=1
        Number of Beta(theta) draws. If <=0, use the posterior mean of theta.
    sample_g : bool, default=False
        If True, sample g from Dirichlet; otherwise use its posterior mean.
    random_state : None or int, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        mean: float = 0.95,
        strength: float = 10.0,
        gamma_x="median",
        gamma_a="median",
        dirichlet_strength: float = 1.0,
        use_ess_beta: bool = False,
        tau_beta: float = 1.0,
        top_m: int | None = 2,
        n_theta_samples: int = 1,
        sample_g: bool = False,
        random_state=None,
    ):
        self.mean = float(mean)
        self.strength = float(strength)
        self.gamma_x = gamma_x
        self.gamma_a = gamma_a
        self.dirichlet_strength = float(dirichlet_strength)
        self.use_ess_beta = bool(use_ess_beta)
        self.tau_beta = float(tau_beta)
        self.top_m = top_m
        self.n_theta_samples = int(n_theta_samples)
        self.sample_g = bool(sample_g)
        self.random_state = check_random_state(random_state)

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

        # Candidate sample posteriors/embeddings and annotator embeddings.
        r_cand, X_cand_emb, A_pred = clf.predict_proba(
            X[sample_indices],
            extra_outputs=["embeddings", "annotator_embeddings"],
        )
        A_pred = None
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

        alpha0 = self.mean * self.strength
        beta0 = (1.0 - self.mean) * self.strength
        gamma0 = np.full(K, self.dirichlet_strength / K, dtype=float)

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

            alpha, beta, _ = self.parzen_beta_posterior(
                K=K_obs_cand,
                p=m_obs,
                alpha0=alpha0,
                beta0=beta0,
                use_ess=self.use_ess_beta,
                tau=self.tau_beta,
            )

            gamma_cand = gamma0[None, :] + (K_obs_cand.T @ y_obs_oh)

            # Deterministic full-K path: reuse the shared closed-form IG implementation.
            if (
                self.n_theta_samples <= 0
                and not self.sample_g
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

            for j_x in range(len(sample_indices)):
                if available_mask is not None and not available_mask[j_x, j_a]:
                    U[j_x, j_a] = np.nan
                    continue

                r = r_cand[j_x]

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

        if self.sample_g:
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

        if self.sample_g:
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
