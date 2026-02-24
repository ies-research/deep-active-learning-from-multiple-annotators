from __future__ import annotations

import numpy as np

from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel

from skactiveml.utils import is_labeled
from ._base import PairScorer


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class IGKernelChannelPairScorer(PairScorer):
    """
    Pair scorer using an information-gain (IG) utility under a low-parameter
    generative channel model with Beta (correctness) + Dirichlet (fallback label).

    Channel model for a candidate pair (x, a):
        Z ~ r(·) = p(Z|x)  from clf
        theta ~ Beta(alpha(x,a), beta(x,a))  (kernel-smoothed, TS-ready)
        g ~ Dirichlet(gamma(x,a))            (kernel-smoothed label counts)
        Y = Z          with prob theta
        Y ~ Categorical(g) with prob (1-theta)

    The utility is IG:
        IG(x,a) = I(Z;Y | x,a)

    Correctness evidence for building Beta posteriors is a mixture of:
      - model evidence m_i = p_clf(y_i | x_i) for observed pair (x_i, a_i)
      - LOWO peer credence c_i = q_{-a_i}(y_i | x_i, a_i) from nearby other-annotator labels

    Mixed evidence:
        e_i = lambda_i * c_i + (1-lambda_i) * m_i
    where lambda_i depends on ESS of the LOWO neighborhood.

    Parameters
    ----------
    mean : float, default=0.95
        Prior mean for theta.
    strength : float, default=10
        Prior strength for theta: alpha0 = mean*strength, beta0=(1-mean)*strength.
    gamma_x : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the *sample* embedding RBF kernel.
    gamma_a : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the *annotator* embedding RBF kernel.
    dirichlet_strength : float, default=100.0
        Prior concentration for g. Prior is symmetric: gamma0 = dirichlet_strength / K.
    lambda_c0 : float, default=10.0
        ESS -> lambda mapping: lambda = ESS/(ESS + lambda_c0).
    tau_beta : float, default=1.0
        Discount factor for ESS-based Beta concentration (see parzen_beta_posterior).
    top_m : int, default=2
        Compute IG in a reduced label space containing top_m classes from r plus one "other" bucket.
        Use top_m=None to compute full K-class IG (can be expensive for K=100).
    n_theta_samples : int, default=1
        Number of theta draws for TS-style IG. If 1, a single draw. If 0, use mean theta.
    sample_g : bool, default=False
        If True, sample g from Dirichlet for each IG evaluation. Otherwise use E[g] (more stable).
    random_state : None or int, default=None
        Seed for reproducibility.

    Notes
    -----
    - Callers may override RNG by passing `rng: numpy.random.Generator` via **kwargs.
    - Requires clf.predict_proba(..., extra_outputs=["embeddings","annotator_embeddings"]).
      If annotator_embeddings are not available, falls back to using only sample kernel.
    """

    def __init__(
            self,
            *,
            mean: float = 0.95,
            strength: float = 10.0,
            gamma_x="median",
            gamma_a="median",
            dirichlet_strength: float = 10000.0,
            lambda_c0: float = 10.0,
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
        self.lambda_c0 = float(lambda_c0)
        self.tau_beta = float(tau_beta)
        self.top_m = top_m
        self.n_theta_samples = int(n_theta_samples)
        self.sample_g = bool(sample_g)
        self.random_state = check_random_state(random_state)

    # -------------------------
    # Main API
    # -------------------------
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
            # use RandomState-backed Generator-like behavior
            rng = np.random.default_rng(self.random_state.randint(0, 2**32 - 1))

        classes = clf.classes_
        K = len(classes)
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")

        # Candidate sample embeddings + their posteriors r + annotator embeddings.
        r_cand, X_cand_emb, A_all = clf.predict_proba(X[sample_indices], extra_outputs=["embeddings", "annotator_embeddings"])
        X_cand_emb = _l2_normalize(np.asarray(X_cand_emb, dtype=float))

        use_annotator_kernel = A_all is not None
        if use_annotator_kernel:
            A_all = np.asarray(A_all, dtype=float)
            A_all = _l2_normalize(A_all)
            A_cand_emb = A_all[annotator_indices]
        else:
            # dummy: annotator kernel becomes 1
            A_cand_emb = None

        # Build global observed pair list (training pairs) across all annotators
        is_lbld = is_labeled(y=y, missing_label=clf.missing_label)  # (n_samples, n_annotators)
        obs_s, obs_a = np.where(is_lbld)
        if obs_s.size == 0:
            # no labels anywhere -> all priors
            U = np.full((len(sample_indices), len(annotator_indices)), np.nan, dtype=float)
            if available_mask is not None:
                U = np.where(available_mask, U, np.nan)
            return U

        y_obs = y[obs_s, obs_a].astype(int)

        # Get embeddings for observed samples
        r_obs, X_obs_emb = clf.predict_proba(X[obs_s], extra_outputs=["embeddings"])
        X_obs_emb = _l2_normalize(np.asarray(X_obs_emb, dtype=float))

        # Model evidence per observed pair: m_i = p_clf(y_i | x_i)
        one_hot = np.eye(K, dtype=float)
        y_obs_oh = one_hot[y_obs]  # (n_obs, K)
        m_obs = (r_obs * y_obs_oh).sum(axis=1)  # (n_obs,)

        # Annotator embeddings for observed annotators (if available)
        if use_annotator_kernel:
            A_obs_emb = A_all[obs_a]  # (n_obs, d_a)
        else:
            A_obs_emb = None

        # Precompute gamma(s) for sample kernel from observed sample embeddings
        gamma_x_val = self._resolve_gamma_from_embeddings(X_obs_emb, self.gamma_x)

        # Precompute gamma for annotator kernel from observed annotator embeddings (if available)
        gamma_a_val = None
        if use_annotator_kernel:
            gamma_a_val = self._resolve_gamma_from_embeddings(A_obs_emb, self.gamma_a)

        # Precompute sample-kernel similarities:
        # Kx_obs_cand: (n_obs, n_cand_samples)
        Kx_obs_cand = rbf_kernel(X_obs_emb, X_cand_emb, gamma=gamma_x_val)

        # If we have annotator embeddings, we’ll build Ka_obs_cand per candidate annotator on the fly.
        # If not, Ka = 1.

        # Compute LOWO peer credence for each observed pair i:
        # c_i = q_{-a_i}(y_i | x_i, a_i) using neighborhood over other-annotator observed pairs.
        c_obs, ess_obs = self._compute_lowo_credence_for_observed(
            y_obs=y_obs,
            obs_a=obs_a,
            Kx_self=None,  # compute from X_obs_emb internally
            X_obs_emb=X_obs_emb,
            A_obs_emb=A_obs_emb,
            gamma_x=gamma_x_val,
            gamma_a=gamma_a_val,
        )
        # lambda per observed pair from ESS
        lam_obs = ess_obs / (ess_obs + self.lambda_c0)

        # Mixed evidence targets for Beta updates
        e_obs = lam_obs * c_obs + (1.0 - lam_obs) * m_obs
        e_obs = np.clip(e_obs, 0.0, 1.0)

        # Beta prior
        alpha0 = self.mean * self.strength
        beta0 = (1.0 - self.mean) * self.strength

        # Dirichlet prior (symmetric)
        gamma0 = np.full(K, self.dirichlet_strength / K, dtype=float)

        # Output utilities
        U = np.empty((len(sample_indices), len(annotator_indices)), dtype=float)

        # Precompute entropy H(r) for candidates (top-m reduction later)
        for j_a, a in enumerate(annotator_indices):
            # Ka for obs vs this candidate annotator
            if use_annotator_kernel:
                Ka_obs = rbf_kernel(A_obs_emb, A_all[[a]], gamma=gamma_a_val).reshape(-1)  # (n_obs,)
            else:
                Ka_obs = np.ones(obs_s.shape[0], dtype=float)

            # LOWO relative to candidate annotator: exclude obs from same annotator a
            mask_lowo = (obs_a != a)
            Ka_obs_lowo = Ka_obs * mask_lowo.astype(float)

            # Joint kernel weights between observed pairs and candidate samples for this annotator:
            # K_obs_cand = (n_obs, n_cand_samples)
            K_obs_cand = Kx_obs_cand * Ka_obs_lowo[:, None]

            # Beta posterior per candidate sample
            alpha, beta, beta_info = self.parzen_beta_posterior(
                K=K_obs_cand,
                p=e_obs,
                alpha0=alpha0,
                beta0=beta0,
                use_ess=True,
                tau=self.tau_beta,
            )

            # Dirichlet posterior for g per candidate sample:
            # gamma_j,ℓ = gamma0_ℓ + sum_i K[i,j] * 1[y_i = ℓ]
            # Efficient: accumulate via one-hot
            # (K_obs_cand.T @ y_obs_oh) gives (n_cand_samples, K)
            gamma_cand = gamma0[None, :] + (K_obs_cand.T @ y_obs_oh)

            # Compute IG utilities for each candidate sample with this annotator
            for j_x in range(len(sample_indices)):
                if available_mask is not None and not available_mask[j_x, j_a]:
                    U[j_x, j_a] = np.nan
                    continue

                r = np.asarray(r_cand[j_x], dtype=float)
                r = np.clip(r, 1e-15, 1.0)
                r = r / r.sum()

                # Decide evaluation label space (top-m + other) for speed
                if self.top_m is None or self.top_m >= K:
                    # Full K
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
    # LOWO credence helper
    # -------------------------
    def _compute_lowo_credence_for_observed(
            self,
            *,
            y_obs: np.ndarray,
            obs_a: np.ndarray,
            Kx_self,
            X_obs_emb: np.ndarray,
            A_obs_emb: np.ndarray | None,
            gamma_x: float,
            gamma_a: float | None,
            eps: float = 1e-12,
    ):
        """
        Compute for each observed pair i:
            c_i = q_{-a_i}(y_i | x_i, a_i)
        using kernel-weighted label distribution over observed pairs with annotator != a_i.

        Returns
        -------
        c_obs : ndarray (n_obs,)
            LOWO credence per observed pair.
        ess_obs : ndarray (n_obs,)
            ESS of the LOWO neighborhood per observed pair.
        """
        n_obs = y_obs.shape[0]
        # sample similarity among observed samples
        Kx = rbf_kernel(X_obs_emb, X_obs_emb, gamma=gamma_x)  # (n_obs, n_obs)

        if A_obs_emb is not None and gamma_a is not None:
            Ka = rbf_kernel(A_obs_emb, A_obs_emb, gamma=gamma_a)  # (n_obs, n_obs)
            W = Kx * Ka
        else:
            W = Kx

        # Exclude self
        np.fill_diagonal(W, 0.0)

        # Exclude same annotator (LOWO)
        same_a = (obs_a[:, None] == obs_a[None, :])
        W = np.where(same_a, 0.0, W)

        # credence for the *observed* label y_i:
        # q_{-a_i}(y_i) = sum_j W[i,j] 1[y_j=y_i] / sum_j W[i,j]
        # Compute numerator by grouping labels
        K = int(np.max(y_obs)) + 1
        # But safer: use unique labels in y_obs and map -> 0..K-1 if needed.
        # Here assume y_obs already 0..K-1 class indices.
        y_obs = y_obs.astype(int)

        mass = W.sum(axis=1)  # (n_obs,)
        # numerator: sum over neighbors with same label
        # Build one-hot for observed labels; n_obs x K
        one_hot = np.eye(K, dtype=float)[y_obs]
        num = (W @ one_hot)  # (n_obs, K)
        c = num[np.arange(n_obs), y_obs] / np.maximum(mass, eps)
        c = np.where(mass > eps, c, 0.5)

        # ESS per i
        m2 = (W**2).sum(axis=1)
        ess = (mass**2) / np.maximum(m2, eps)
        ess = np.where(mass > eps, ess, 0.0)

        c = np.clip(c, 0.0, 1.0)
        return c, ess

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
        # Sample or mean theta
        if self.n_theta_samples <= 0:
            thetas = np.array([alpha / (alpha + beta)], dtype=float)
        else:
            thetas = rng.beta(alpha, beta, size=self.n_theta_samples).astype(float)

        # Sample or mean g
        if self.sample_g:
            # one g per theta sample (could also share; either is fine)
            gs = rng.dirichlet(gamma, size=thetas.size)
        else:
            g_mean = gamma / np.maximum(gamma.sum(), 1e-12)
            gs = np.tile(g_mean[None, :], (thetas.size, 1))

        H_prior = self._entropy(r)

        igs = []
        for theta, g in zip(thetas, gs):
            # p(y) = theta*r + (1-theta)*g
            py = theta * r + (1.0 - theta) * g
            py = np.clip(py, 1e-15, 1.0)
            py = py / py.sum()

            # posterior for each observed y=ℓ:
            # p(z=k | y=ℓ) ∝ r_k * [theta*1[ℓ=k] + (1-theta)*g_ℓ]
            # For fixed ℓ, the likelihood term is:
            #   L_k = (1-theta)*g_ℓ for k!=ℓ, and theta + (1-theta)*g_ℓ for k=ℓ
            # So we can compute each posterior in O(K) per ℓ (=> O(K^2)).
            # For K=100, prefer top_m mode.
            H_post_exp = 0.0
            base = (1.0 - theta) * g  # vector over ℓ
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

        Build reduced label space S = {top_m classes} ∪ {other}.
        Channel is applied in this reduced space by aggregating r and g accordingly.
        """
        K = r.size
        if top_m <= 0:
            raise ValueError("top_m must be >= 1 when used.")

        idx = np.argsort(-r)[:top_m]
        idx = np.array(idx, dtype=int)

        r_top = r[idx]
        r_other = 1.0 - r_top.sum()
        r_red = np.concatenate([r_top, [max(r_other, 0.0)]], axis=0)
        r_red = np.clip(r_red, 1e-15, 1.0)
        r_red = r_red / r_red.sum()
        Kr = r_red.size  # top_m + 1

        # Reduce g similarly from Dirichlet mean/sample
        if self.n_theta_samples <= 0:
            thetas = np.array([alpha / (alpha + beta)], dtype=float)
        else:
            thetas = rng.beta(alpha, beta, size=self.n_theta_samples).astype(float)

        # For g: either sample full and reduce, or reduce mean
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

            # p(y) in reduced space
            py = theta * r_red + (1.0 - theta) * g_red
            py = np.clip(py, 1e-15, 1.0)
            py = py / py.sum()

            # posterior per reduced label ℓ
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

        # Use pairwise squared distances cheaply: for median/mean we need distances.
        # For simplicity and safety, use euclidean distances via dot trick:
        # d^2 = ||u||^2 + ||v||^2 - 2 u·v
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
        # numeric gamma
        return float(mode)

    # -------------------------
    # Beta posterior (same as your old class)
    # -------------------------
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
            raise ValueError(f"Shape mismatch: K rows {K.shape[0]} vs p {p.shape[0]}")
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