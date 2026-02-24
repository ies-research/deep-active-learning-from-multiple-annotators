import numpy as np

from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

from skactiveml.utils import is_labeled
from ._base import PairScorer


class BetaModelPairScorer(PairScorer):
    """
    Random utility for (sample, annotator) pairs.

    Utilities are generated via Thompson sampling from a Beta distribution
    inferred via a Parzen estimator for each sample-annotator pair.

    Parameters
    ----------
    mean : float, default=0.95
        Prior mean of annotator correctness.
    strength : float, default=10
        Prior strength (pseudo-count concentration).
    gamma : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection rule (or fixed value) for the RBF kernel over *sample*
        embeddings in the Parzen estimator.
    annotator_similarity : {"none","cosine","rbf"}, default="none"
        If not "none", pool labeled evidence from other annotators weighted by
        similarity in annotator embedding space `A_cand` of shape
        (n_annotators, annotator_embed_dim).
    annotator_gamma : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection rule (or fixed value) for the RBF kernel over *annotator*
        embeddings if `annotator_similarity="rbf"`.
    annotator_sim_power : float, default=1.0
        Exponent applied to similarity weights: w <- w ** annotator_sim_power.
        Values >1 sharpen, values <1 flatten.
    random_state : None or int, default=None
        Seed for reproducible randomness.

    Notes
    -----
    - Callers may override the internal RNG by passing
      ``rng : numpy.random.Generator`` via ``**kwargs``.
    - If `annotator_similarity` is enabled but `A_cand` does not include embeddings
      for all annotators (i.e., its first dimension differs from y.shape[1]),
      pooling is restricted to the provided `annotator_indices`.
    """

    def __init__(
            self,
            mean=0.95,
            strength=10,
            gamma="median",
            annotator_similarity="none",
            annotator_gamma="median",
            annotator_sim_power=1.0,
            random_state=None,
    ):
        self.mean = mean
        self.strength = strength
        self.gamma = gamma

        self.annotator_similarity = annotator_similarity
        self.annotator_gamma = annotator_gamma
        self.annotator_sim_power = annotator_sim_power

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
        _, X_cand, A_cand = clf.predict_proba(
            X[sample_indices], extra_outputs=["embeddings", "annotator_embeddings"]
        )

        # Candidate sample embeddings: normalize for distance/kernel stability
        X_cand = np.asarray(X_cand, dtype=float)
        X_cand /= np.maximum(np.linalg.norm(X_cand, axis=1, keepdims=True), 1e-12)

        is_lbld = is_labeled(y=y, missing_label=clf.missing_label)  # (n_samples, n_annotators_total)

        # RNG: prefer numpy Generator if passed.
        rng = kwargs.get("rng", None)
        if rng is None:
            rng = self.random_state

        n_sel_s = len(sample_indices)
        n_sel_a = len(annotator_indices)
        U = np.empty((n_sel_s, n_sel_a), dtype=float)

        one_hot = np.eye(len(clf.classes_))

        # ---- annotator similarity matrix (optional) ----
        A_cand = None if A_cand is None else np.asarray(A_cand, dtype=float)

        use_ann_sim = self.annotator_similarity is not None and self.annotator_similarity != "none"
        ann_sim_is_global = False
        S_ann = None
        ann_id_to_row = None

        if use_ann_sim:
            if A_cand is None or A_cand.ndim != 2:
                raise ValueError(
                    "annotator_similarity enabled, but A_cand is missing or not 2D."
                )

            # If A_cand covers all annotators, we can pool from all. Otherwise restrict.
            ann_sim_is_global = (A_cand.shape[0] == y.shape[1])

            if ann_sim_is_global:
                S_ann = self._annotator_similarity_matrix(
                    A_cand,
                    method=self.annotator_similarity,
                    gamma=self.annotator_gamma,
                )
            else:
                # Pool only among the selected annotator_indices.
                # Build map annotator_id -> row in A_cand. We assume A_cand rows align with annotator_indices order.
                # If your clf returns A_cand in some other order, fix it there, not here.
                if A_cand.shape[0] != n_sel_a:
                    raise ValueError(
                        f"A_cand has shape {A_cand.shape}, but expected "
                        f"(n_sel_annotators={n_sel_a}, d) when not global."
                    )
                S_ann = self._annotator_similarity_matrix(
                    A_cand,
                    method=self.annotator_similarity,
                    gamma=self.annotator_gamma,
                )
                ann_id_to_row = {ann_id: r for r, ann_id in enumerate(annotator_indices)}

        # Prior
        alpha0 = self.mean * self.strength
        beta0 = (1.0 - self.mean) * self.strength

        # Optional overrides for posterior behavior
        use_ess = kwargs.get("use_ess", True)
        tau = kwargs.get("tau", 1.0)
        eps = kwargs.get("eps", 1e-12)

        for a_pos, a_id in enumerate(annotator_indices):
            if not use_ann_sim:
                # Original behavior: only use labels from annotator a_id.
                mask_a = is_lbld[:, a_id]
                alpha, beta = self._posterior_from_single_annotator(
                    clf=clf,
                    X=X,
                    y=y,
                    mask_a=mask_a,
                    a_id=a_id,
                    one_hot=one_hot,
                    X_cand=X_cand,
                    alpha0=alpha0,
                    beta0=beta0,
                    use_ess=use_ess,
                    tau=tau,
                    eps=eps,
                )
            else:
                # Pooled behavior: sum evidence from annotators b weighted by sim(b, a_id).
                if ann_sim_is_global:
                    # Consider any annotator with at least one label.
                    src_ids = np.flatnonzero(is_lbld.any(axis=0))
                    w = S_ann[src_ids, a_id]
                else:
                    # Pool only within selected annotators.
                    src_ids = np.array([b_id for b_id in annotator_indices if is_lbld[:, b_id].any()], dtype=int)
                    w = np.array([S_ann[ann_id_to_row[b_id], ann_id_to_row[a_id]] for b_id in src_ids], dtype=float)

                # Enforce nonnegative weights. Cosine can go negative.
                w = np.clip(w, 0.0, None)
                if self.annotator_sim_power != 1.0:
                    w = w ** float(self.annotator_sim_power)

                # Always keep self weight at least 1 (otherwise you can get "I learned nothing from myself").
                # If self isn't in src_ids (no labels), this does nothing.
                self_mask = (src_ids == a_id)
                if np.any(self_mask):
                    w[self_mask] = np.maximum(w[self_mask], 1.0)

                alpha, beta = self._posterior_from_pooled_annotators(
                    clf=clf,
                    X=X,
                    y=y,
                    is_lbld=is_lbld,
                    src_ids=src_ids,
                    src_w=w,
                    one_hot=one_hot,
                    X_cand=X_cand,
                    alpha0=alpha0,
                    beta0=beta0,
                    use_ess=use_ess,
                    tau=tau,
                    eps=eps,
                )

            # Thompson sample utilities
            U[:, a_pos] = rng.beta(alpha, beta)

        if available_mask is not None:
            U = np.where(available_mask, U, np.nan)

        return U

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _posterior_from_single_annotator(
            self,
            *,
            clf,
            X,
            y,
            mask_a,
            a_id,
            one_hot,
            X_cand,
            alpha0,
            beta0,
            use_ess,
            tau,
            eps,
    ):
        # If no labels, fall back to prior.
        if not np.any(mask_a):
            alpha = np.full(X_cand.shape[0], alpha0, dtype=float)
            beta = np.full(X_cand.shape[0], beta0, dtype=float)
            return alpha, beta

        p_class, X_lbld_a = clf.predict_proba(X[mask_a], extra_outputs=["embeddings"])
        y_a = y[mask_a][:, a_id]
        y_a_one_hot = one_hot[y_a]
        p_perf_lbld_a = (p_class * y_a_one_hot).sum(axis=1)

        X_lbld_a = np.asarray(X_lbld_a, dtype=float)
        X_lbld_a /= np.maximum(np.linalg.norm(X_lbld_a, axis=1, keepdims=True), 1e-12)

        gamma_a = self._resolve_gamma(self.gamma, X_lbld_a)
        K = rbf_kernel(X_lbld_a, X_cand, gamma=gamma_a)

        alpha, beta, _ = self.parzen_beta_posterior(
            K=K,
            p=p_perf_lbld_a,
            alpha0=alpha0,
            beta0=beta0,
            use_ess=use_ess,
            tau=tau,
            eps=eps,
        )
        return alpha, beta

    def _posterior_from_pooled_annotators(
            self,
            *,
            clf,
            X,
            y,
            is_lbld,
            src_ids,
            src_w,
            one_hot,
            X_cand,
            alpha0,
            beta0,
            use_ess,
            tau,
            eps,
    ):
        n_pred = X_cand.shape[0]
        mass = np.zeros(n_pred, dtype=float)
        s = np.zeros(n_pred, dtype=float)
        f = np.zeros(n_pred, dtype=float)
        m2 = np.zeros(n_pred, dtype=float)  # for ESS

        for b_id, w_b in zip(src_ids, src_w):
            if w_b <= 0.0:
                continue
            mask_b = is_lbld[:, b_id]
            if not np.any(mask_b):
                continue

            p_class, X_lbld_b = clf.predict_proba(X[mask_b], extra_outputs=["embeddings"])
            y_b = y[mask_b][:, b_id]
            y_b_one_hot = one_hot[y_b]
            p_perf_lbld_b = (p_class * y_b_one_hot).sum(axis=1)

            X_lbld_b = np.asarray(X_lbld_b, dtype=float)
            X_lbld_b /= np.maximum(np.linalg.norm(X_lbld_b, axis=1, keepdims=True), 1e-12)

            gamma_b = self._resolve_gamma(self.gamma, X_lbld_b)
            K_b = rbf_kernel(X_lbld_b, X_cand, gamma=gamma_b)

            # Aggregate Parzen stats as if K_total were concatenation of (w_b * K_b)
            Ksum = K_b.sum(axis=0)
            mass += w_b * Ksum

            s += w_b * (p_perf_lbld_b @ K_b)
            f += w_b * ((1.0 - p_perf_lbld_b) @ K_b)

            if use_ess:
                m2 += (w_b ** 2) * (K_b ** 2).sum(axis=0)

        # Convert aggregated stats into Beta posterior (same math as parzen_beta_posterior)
        mu = np.where(mass > eps, s / np.maximum(mass, eps), 0.5)

        if not use_ess:
            alpha = alpha0 + s
            beta = beta0 + f
            return alpha, beta

        n_eff = (mass ** 2) / np.maximum(m2, eps)
        conc = tau * n_eff
        alpha = alpha0 + conc * mu
        beta = beta0 + conc * (1.0 - mu)
        return alpha, beta

    def _resolve_gamma(self, gamma_rule, X_emb):
        # Same idea as your code, but isolated.
        D = euclidean_distances(X_emb)
        np.fill_diagonal(D, np.nan)

        if gamma_rule == "median":
            g = np.nanmax([np.nanmedian(D), 1e-3]) ** (-2)
        elif gamma_rule == "mean":
            g = np.nanmax([np.nanmean(D), 1e-3]) ** (-2)
        elif gamma_rule == "minimum":
            g = np.nanmax([np.nanmin(D), 1e-3]) ** (-2)
        else:
            g = float(gamma_rule)
        return g

    def _annotator_similarity_matrix(self, A, *, method, gamma):
        A = np.asarray(A, dtype=float)
        A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

        if method == "cosine":
            # Cosine similarity in [-1, 1]
            S = A @ A.T
            return S

        if method == "rbf":
            g = self._resolve_gamma(gamma, A)
            S = rbf_kernel(A, A, gamma=g)
            return S

        raise ValueError(f"Unknown annotator_similarity={method!r}")

    # -------------------------------------------------------------------------
    # Your original posterior function stays unchanged
    # -------------------------------------------------------------------------

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
            raise ValueError(
                f"K must be 2D (n_train, n_pred), got shape {K.shape}"
            )
        if p.ndim != 1:
            raise ValueError(f"p must be 1D (n_train,), got shape {p.shape}")
        if K.shape[0] != p.shape[0]:
            raise ValueError(
                f"Shape mismatch: K has {K.shape[0]} rows but p has length {p.shape[0]}"
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