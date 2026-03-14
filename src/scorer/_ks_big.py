from __future__ import annotations

import numpy as np

from scipy.stats import beta as beta_distribution
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_random_state

from skactiveml.utils import is_labeled
from ._base import PairScorer
from ._utils import expected_score_gain


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


class KernelSmoothedBayesianGain(PairScorer):
    """
    Pair scorer using expected uncertainty reduction under a
    kernel-smoothed annotator model.

    For a candidate pair (x, a), this scorer supports multiple channel
    variants:

    - "channel":
        Z ~ r(·) = p(Z|x)                 from clf
        theta ~ Beta(alpha(x,a), beta(x,a))
        g ~ Dirichlet(gamma(x,a))
        p(Y=Z | Z) = theta
        p(Y=y!=Z | Z) ∝ g_y
      i.e., `g` is conditioned on being incorrect.

    - "scalar_uniform_confusion":
        estimate a single accuracy scalar theta and define a proper confusion
        matrix with uniform off-diagonal mass:
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
      (ESS-scaled when `use_ess_beta=True`)

    - Dirichlet label model (kernel-weighted label counts)
        gamma_k(x,a) = gamma0_k + sum_i w_i(x,a) * 1[y_i = k]
      (ESS-scaled when `use_ess_label_dirichlet=True`)

    where the pair weight factorizes as:
        w_i(x,a) = k_x(x_i, x) * k_a(a_i, a)

    `k_a` uses annotator embeddings when available. If annotator embeddings are
    not available (or are not global), the scorer falls back to exact annotator
    identity weighting: k_a(a_i, a) = 1[a_i = a].

    Utility is an expected reduction in predictive uncertainty, evaluated via
    Monte Carlo samples from Beta/Dirichlet (or using posterior means). The
    uncertainty functional is controlled by ``gain_type``.

    Parameters
    ----------
    accuracy_mean : float or {"global_observed", "per_annotator_observed"}, \
            default="global_observed"
        Accuracy-prior specification. A float uses a fixed prior mean. The
        observed modes infer the prior mean from the average soft correctness:
        - "global_observed": use the average soft correctness across all
          observed annotations.
        - "per_annotator_observed": use the average soft correctness over all
          instances labeled by the respective annotator; if an annotator has no
          observations, fallback to the global observed average.
    accuracy_strength : float, default=10.0
        Prior strength for accuracy parameters. Used for Beta priors in
        variants with scalar/diagonal accuracies and as the total concentration
        per confusion row in `channel_variant="full_confusion"`.
    gamma_x : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the sample-embedding RBF kernel.
    gamma_x_scope : {"global","per_annotator"}, default="global"
        Scope used to resolve `gamma_x`:
        - "global": estimate a single bandwidth from all observed sample
          embeddings.
        - "per_annotator": estimate bandwidth separately per target annotator
          from that annotator's observed sample embeddings (fallback to global
          if <2 points).
    gamma_a : float or {"median","mean","minimum"}, default="median"
        Bandwidth selection for the annotator-embedding RBF kernel (if used).
    use_annotator_embeddings : bool, default=True
        If True, request and use global annotator embeddings (when provided by
        `clf`) to smooth across annotators. If False, always use exact
        annotator identity weighting.
    annotator_lambda : float, default=0.0
        Convex weight on an instance-independent sample kernel in the
        annotator-side posterior updates. `0.0` keeps the local kernel only,
        while `1.0` yields instance-independent worker estimates.
    channel_label_dirichlet_strength : float, default=1.0
        Symmetric Dirichlet prior concentration for the fallback label
        distribution `g` in `channel_variant="channel"` only.
    gain_type : {"entropy", "margin", "brier"}, default="entropy"
        Uncertainty functional reduced in expectation after observing an
        annotator label. ``"entropy"`` recovers standard information gain.
    channel_variant : {"channel", "scalar_uniform_confusion", /
            "diag_uniform_confusion","full_confusion"}, default="channel"
        Annotator noise parameterization used for gain computation.
    class_prior : {"classifier", "uniform", "kernel"}, default="classifier"
        Prior used for the latent class in the IG computation:
        - "classifier": use the classifier posterior ``p(Y|x)``.
        - "uniform": use a uniform prior over classes.
        - "kernel": use a kernel-smoothed Dirichlet prior built from the
          classifier posteriors on labeled samples.
    class_prior_strength : float, default=1.0
        Symmetric Dirichlet prior concentration for the kernelized class prior.
        Used only when `class_prior="kernel"`.
    class_prior_lambda : float, default=0.0
        Convex weight on an instance-independent sample kernel in the
        kernelized class prior. `0.0` keeps the local kernel only, while
        `1.0` collapses the class prior to a global sample smoother.
    use_ess_class_prior : bool, default=False
        If True, map kernel-weighted class evidence to the class-prior
        Dirichlet posterior using ESS-based concentration instead of raw
        weighted counts.
    tau_class_prior : float, default=1.0
        Discount factor for ESS-based class-prior Dirichlet concentration
        (only used if `use_ess_class_prior=True`).
    sample_class_prior : bool, default=False
        If True and `class_prior="kernel"`, sample the class prior from the
        kernelized Dirichlet instead of using its posterior mean.
    use_ess_beta : bool, default=False
        If True, map the kernel-weighted correctness evidence to a Beta 
        posterior using ESS-based concentration instead of raw weighted counts.
    tau_beta : float, default=1.0
        Discount factor for ESS-based Beta concentration
        (only used if `use_ess_beta=True`).
    use_ess_label_dirichlet : bool, default=False
        If True, map kernel-weighted label evidence to a Dirichlet posterior
        using ESS-based concentration instead of raw weighted counts.
    tau_label_dirichlet : float, default=1.0
        Discount factor for ESS-based Dirichlet concentration
        (only used if `use_ess_label_dirichlet=True`).
    top_m : int or None, default=None
        If not None, approximate entropy gain in top-M + "other" reduced label
        space. Currently supported only for `gain_type="entropy"` with
        `channel_variant="channel"` together with
        `class_prior` in `{"classifier", "kernel"}`.
    n_theta_samples : int, default=1
        Number of Monte Carlo draws for latent channel parameters. For variants
        with Beta accuracies, this controls Beta draws. If <=0, posterior means
        are used (when applicable).
    theta_ucb_quantile : float or None, default=None
        Deterministic optimistic quantile for Beta-based accuracy parameters
        when `n_theta_samples <= 0`. If provided, the point estimate becomes
        the Beta posterior quantile `q` instead of the posterior mean. Used
        only for `channel`, `scalar_uniform_confusion`, and
        `diag_uniform_confusion`.
    sample_label_dirichlet : bool, default=False
        If True, sample Dirichlet-distributed label parameters
        (`g` in `channel`, confusion rows in `full_confusion`); otherwise use
        posterior means.
    channel_wrong_label_mode : {"normalize", "sample_dirichlet_wrong"}, /
            default="normalize"
        Wrong-label construction for `channel_variant="channel"`:
        - "normalize": use shared `g` and condition on being wrong by removing
          the assumed true class and renormalizing.
        - "sample_dirichlet_wrong": for each assumed true class z, draw
          (or use mean of) a Dirichlet over wrong labels only,
          `Dir(gamma_{-z})`.
    random_state : None or int, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        *,
        accuracy_mean: float | str = "global_observed",
        accuracy_strength: float = 10.0,
        gamma_x="median",
        gamma_x_scope: str = "global",
        gamma_a="median",
        use_annotator_embeddings: bool = True,
        annotator_lambda: float = 0.0,
        channel_label_dirichlet_strength: float = 1.0,
        gain_type: str = "entropy",
        channel_variant: str = "channel",
        class_prior: str = "classifier",
        class_prior_strength: float = 1.0,
        class_prior_lambda: float = 0.0,
        use_ess_class_prior: bool = False,
        tau_class_prior: float = 1.0,
        sample_class_prior: bool = False,
        use_ess_beta: bool = False,
        tau_beta: float = 1.0,
        use_ess_label_dirichlet: bool = False,
        tau_label_dirichlet: float = 1.0,
        top_m: int | None = None,
        n_theta_samples: int = 1,
        theta_ucb_quantile: float | None = None,
        sample_label_dirichlet: bool = False,
        channel_wrong_label_mode: str = "normalize",
        random_state=None,
    ):
        if isinstance(accuracy_mean, str):
            self.accuracy_mean = str(accuracy_mean)
            self._accuracy_mean_mode = self.accuracy_mean
        else:
            self.accuracy_mean = float(accuracy_mean)
            self._accuracy_mean_mode = "fixed"
        self.accuracy_strength = float(accuracy_strength)
        self.gamma_x = gamma_x
        self.gamma_x_scope = str(gamma_x_scope)
        self.gamma_a = gamma_a
        self.use_annotator_embeddings = bool(use_annotator_embeddings)
        self.annotator_lambda = float(annotator_lambda)
        self.channel_label_dirichlet_strength = float(
            channel_label_dirichlet_strength
        )
        self.gain_type = str(gain_type)
        self.channel_variant = str(channel_variant)
        self.class_prior = str(class_prior)
        self.class_prior_strength = float(class_prior_strength)
        self.class_prior_lambda = float(class_prior_lambda)
        self.use_ess_class_prior = bool(use_ess_class_prior)
        self.tau_class_prior = float(tau_class_prior)
        self.sample_class_prior = bool(sample_class_prior)
        self.use_ess_beta = bool(use_ess_beta)
        self.tau_beta = float(tau_beta)
        self.use_ess_label_dirichlet = bool(use_ess_label_dirichlet)
        self.tau_label_dirichlet = float(tau_label_dirichlet)
        self.top_m = None if top_m is None else int(top_m)
        self.n_theta_samples = int(n_theta_samples)
        self.theta_ucb_quantile = (
            None
            if theta_ucb_quantile is None
            else float(theta_ucb_quantile)
        )
        self.sample_label_dirichlet = bool(sample_label_dirichlet)
        self.channel_wrong_label_mode = str(channel_wrong_label_mode)
        self.random_state = check_random_state(random_state)

        if self._accuracy_mean_mode == "fixed":
            if not (0.0 < self.accuracy_mean < 1.0):
                raise ValueError("accuracy_mean must be in (0, 1)")
        elif self._accuracy_mean_mode not in {
            "global_observed",
            "per_annotator_observed",
        }:
            raise ValueError(
                "accuracy_mean must be a float in (0, 1) or one of "
                "{'global_observed', 'per_annotator_observed'}"
            )
        if self.accuracy_strength <= 0:
            raise ValueError("accuracy_strength must be > 0")
        if self.theta_ucb_quantile is not None and not (
            0.0 < self.theta_ucb_quantile < 1.0
        ):
            raise ValueError("theta_ucb_quantile must be in (0, 1)")
        if self.channel_label_dirichlet_strength <= 0:
            raise ValueError("channel_label_dirichlet_strength must be > 0")
        if self.gamma_x_scope not in {"global", "per_annotator"}:
            raise ValueError(
                "gamma_x_scope must be one of {'global', 'per_annotator'}"
            )
        if not (0.0 <= self.annotator_lambda <= 1.0):
            raise ValueError("annotator_lambda must be in [0, 1]")
        if not (0.0 <= self.class_prior_lambda <= 1.0):
            raise ValueError("class_prior_lambda must be in [0, 1]")
        if self.channel_wrong_label_mode not in {
            "normalize",
            "sample_dirichlet_wrong",
        }:
            raise ValueError(
                "channel_wrong_label_mode must be one of "
                "{'normalize', 'sample_dirichlet_wrong'}"
            )
        if self.channel_variant not in {
            "channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
            "full_confusion",
        }:
            raise ValueError(
                "channel_variant must be one of "
                "{'channel', 'scalar_uniform_confusion', "
                "'diag_uniform_confusion', 'full_confusion'}"
            )
        if self.class_prior not in {"classifier", "uniform", "kernel"}:
            raise ValueError(
                "class_prior must be one of {'classifier', 'uniform', 'kernel'}"
            )
        if self.gain_type not in {"entropy", "margin", "brier"}:
            raise ValueError(
                "gain_type must be one of {'entropy', 'margin', 'brier'}"
            )
        if self.class_prior_strength <= 0:
            raise ValueError("class_prior_strength must be > 0")
        if self.tau_class_prior <= 0:
            raise ValueError("tau_class_prior must be > 0")
        if self.tau_beta <= 0:
            raise ValueError("tau_beta must be > 0")
        if self.tau_label_dirichlet <= 0:
            raise ValueError("tau_label_dirichlet must be > 0")
        if self.sample_class_prior and self.class_prior != "kernel":
            raise ValueError(
                "sample_class_prior=True requires class_prior='kernel'"
            )
        if self.top_m is not None:
            if self.top_m <= 0:
                raise ValueError("top_m must be positive or None")
            if self.channel_variant != "channel":
                raise ValueError(
                    "top_m is only supported with channel_variant='channel'"
                )
            if self.gain_type != "entropy":
                raise ValueError(
                    "top_m is only supported with gain_type='entropy'"
                )
            if self.class_prior not in {"classifier", "kernel"}:
                raise ValueError(
                    "top_m is only supported with class_prior in "
                    "{'classifier', 'kernel'}"
                )

        uses_beta = self.channel_variant in {
            "channel",
            "scalar_uniform_confusion",
            "diag_uniform_confusion",
        }
        if not uses_beta:
            if self.use_ess_beta:
                raise ValueError(
                    "use_ess_beta is only supported for channel variants "
                    "with Beta accuracy posteriors"
                )
            if self.tau_beta != 1.0:
                raise ValueError(
                    "tau_beta is only used for channel variants with Beta "
                    "accuracy posteriors"
                )
            if self.theta_ucb_quantile is not None:
                raise ValueError(
                    "theta_ucb_quantile is only supported for channel "
                    "variants with Beta accuracy posteriors"
                )

        if (
            self.theta_ucb_quantile is not None
            and self.n_theta_samples > 0
        ):
            raise ValueError(
                "theta_ucb_quantile requires n_theta_samples <= 0"
            )

        uses_label_dirichlet = self.channel_variant in {
            "channel",
            "full_confusion",
        }
        if not uses_label_dirichlet:
            if self.use_ess_label_dirichlet:
                raise ValueError(
                    "use_ess_label_dirichlet is only supported for "
                    "channel and full_confusion variants"
                )
            if self.tau_label_dirichlet != 1.0:
                raise ValueError(
                    "tau_label_dirichlet is only used for channel and "
                    "full_confusion variants"
                )
            if self.sample_label_dirichlet:
                raise ValueError(
                    "sample_label_dirichlet is only supported for "
                    "channel and full_confusion variants"
                )

        if self.channel_variant != "channel":
            if self.channel_label_dirichlet_strength != 1.0:
                raise ValueError(
                    "channel_label_dirichlet_strength is only used for "
                    "channel_variant='channel'"
                )
            if self.channel_wrong_label_mode != "normalize":
                raise ValueError(
                    "channel_wrong_label_mode is only used for "
                    "channel_variant='channel'"
                )

        if self.class_prior != "kernel":
            if self.class_prior_strength != 1.0:
                raise ValueError(
                    "class_prior_strength is only used when "
                    "class_prior='kernel'"
                )
            if self.use_ess_class_prior:
                raise ValueError(
                    "use_ess_class_prior is only supported when "
                    "class_prior='kernel'"
                )
            if self.tau_class_prior != 1.0:
                raise ValueError(
                    "tau_class_prior is only used when class_prior='kernel'"
                )
            if self.class_prior_lambda != 0.0:
                raise ValueError(
                    "class_prior_lambda is only used when "
                    "class_prior='kernel'"
                )

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
        if self.top_m is not None and self.channel_variant != "channel":
            raise ValueError(
                "top_m is only supported with channel_variant='channel'."
            )
        if self.top_m is not None and self.class_prior == "uniform":
            raise ValueError(
                "top_m is only supported with class_prior in "
                "{'classifier', 'kernel'} for channel_variant='channel'."
            )
        if self.top_m is not None and self.gain_type != "entropy":
            raise ValueError(
                "top_m is only supported with gain_type='entropy' "
                "for channel_variant='channel'."
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

        classes = np.asarray(clf.classes_)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        y_obs_raw = np.asarray(y[obs_s, obs_a])
        try:
            y_obs_idx = np.array([class_to_idx[v] for v in y_obs_raw], dtype=int)
        except KeyError as e:
            raise ValueError(f"Observed label {e.args[0]!r} not found in clf.classes_")

        y_obs_oh = np.eye(K, dtype=float)[y_obs_idx]

        # Observed sample embeddings + classifier probabilities for soft correctness counts.
        r_obs, X_obs_emb = clf.predict_proba(
            X[obs_s], extra_outputs=["embeddings"]
        )
        r_obs = np.asarray(r_obs, dtype=float)
        r_obs = np.clip(r_obs, 1e-15, 1.0)
        r_obs = r_obs / np.maximum(r_obs.sum(axis=1, keepdims=True), 1e-15)
        X_obs_emb = _l2_normalize(np.asarray(X_obs_emb, dtype=float))
        m_obs = r_obs[np.arange(obs_s.size), y_obs_idx]
        m_obs = np.clip(m_obs, 0.0, 1.0)
        _, obs_first_idx = np.unique(obs_s, return_index=True)
        X_obs_cls_emb = X_obs_emb[obs_first_idx]
        r_obs_cls = r_obs[obs_first_idx]

        eps_prior = 1e-6
        global_obs_acc_mean = float(np.mean(m_obs))
        global_obs_acc_mean = float(
            np.clip(global_obs_acc_mean, eps_prior, 1.0 - eps_prior)
        )

        n_annotators_total = y.shape[1]
        obs_count_by_annotator = np.bincount(
            obs_a, minlength=n_annotators_total
        ).astype(float)
        obs_sum_by_annotator = np.bincount(
            obs_a, weights=m_obs, minlength=n_annotators_total
        ).astype(float)
        obs_mean_by_annotator = np.divide(
            obs_sum_by_annotator,
            np.maximum(obs_count_by_annotator, 1.0),
        )
        obs_mean_by_annotator = np.clip(
            obs_mean_by_annotator, eps_prior, 1.0 - eps_prior
        )

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

        gamma_x_global = self._resolve_gamma_from_embeddings(
            X_obs_emb, self.gamma_x
        )
        if use_annotator_kernel:
            gamma_a_val = self._resolve_gamma_from_embeddings(
                A_all, self.gamma_a
            )
        r_cand_prior = self._resolve_class_prior(
            r=r_cand,
            X_cand_emb=X_cand_emb,
            X_obs_cls_emb=X_obs_cls_emb,
            r_obs_cls=r_obs_cls,
            gamma_x=gamma_x_global,
            rng=rng,
        )

        # Local sample-kernel weights from observed pairs to candidate samples.
        Kx_obs_cand_local_global = rbf_kernel(
            X_obs_emb, X_cand_emb, gamma=gamma_x_global
        )

        if self._accuracy_mean_mode == "fixed":
            prior_acc_global = float(self.accuracy_mean)
        else:
            prior_acc_global = global_obs_acc_mean
        prior_acc_global = float(
            np.clip(prior_acc_global, eps_prior, 1.0 - eps_prior)
        )
        alpha0_global = prior_acc_global * self.accuracy_strength
        beta0_global = (1.0 - prior_acc_global) * self.accuracy_strength
        gamma0 = np.full(
            K, self.channel_label_dirichlet_strength / K, dtype=float
        )
        delta0_full_global = self._full_confusion_dirichlet_prior(
            K=K,
            accuracy_mean=prior_acc_global,
            row_strength=self.accuracy_strength,
        )

        U = np.empty(
            (len(sample_indices), len(annotator_indices)), dtype=float
        )

        for j_a, a in enumerate(annotator_indices):
            if self._accuracy_mean_mode == "per_annotator_observed":
                if a < 0 or a >= n_annotators_total:
                    raise ValueError(
                        f"Annotator index {a} out of bounds for y with "
                        f"{n_annotators_total} annotators."
                    )
                if obs_count_by_annotator[a] > 0:
                    prior_acc = float(obs_mean_by_annotator[a])
                else:
                    prior_acc = global_obs_acc_mean
                prior_acc = float(np.clip(prior_acc, eps_prior, 1.0 - eps_prior))
                alpha0 = prior_acc * self.accuracy_strength
                beta0 = (1.0 - prior_acc) * self.accuracy_strength
                delta0_full = self._full_confusion_dirichlet_prior(
                    K=K,
                    accuracy_mean=prior_acc,
                    row_strength=self.accuracy_strength,
                )
            else:
                alpha0 = alpha0_global
                beta0 = beta0_global
                delta0_full = delta0_full_global

            if self.gamma_x_scope == "per_annotator":
                obs_mask_a = obs_a == a
                if np.count_nonzero(obs_mask_a) >= 1:
                    gamma_x_a = self._resolve_gamma_from_embeddings(
                        X_obs_emb[obs_mask_a], self.gamma_x
                    )
                else:
                    gamma_x_a = gamma_x_global
                Kx_obs_cand_local = rbf_kernel(
                    X_obs_emb, X_cand_emb, gamma=gamma_x_a
                )
            else:
                Kx_obs_cand_local = Kx_obs_cand_local_global

            Kx_obs_cand = self._mix_with_global_sample_kernel(
                Kx_obs_cand_local,
                lam=self.annotator_lambda,
            )

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

                # Deterministic full-K path: reuse the shared closed-form gain helper.
                if (
                    self.n_theta_samples <= 0
                    and not self.sample_label_dirichlet
                    and (self.top_m is None or self.top_m >= K)
                ):
                    U_col = self._ig_channel_full_batch(
                        r=r_cand_prior,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma_cand,
                        rng=rng,
                    )
                    if available_mask is not None:
                        U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                    U[:, j_a] = U_col
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

            # Vectorized fast paths for full-K variants.
            if (
                self.channel_variant == "channel"
                and (self.top_m is None or self.top_m >= K)
            ):
                U_col = self._ig_channel_full_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma_cand,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if (
                self.channel_variant == "channel"
                and self.top_m is not None
                and self.top_m < K
            ):
                U_col = self._ig_channel_topm_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma_cand,
                    top_m=int(self.top_m),
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "scalar_uniform_confusion":
                U_col = self._ig_scalar_uniform_confusion_batch(
                    r=r_cand_prior,
                    alpha=alpha,
                    beta=beta,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "diag_uniform_confusion":
                U_col = self._ig_diag_uniform_confusion_batch(
                    r=r_cand_prior,
                    alpha=alpha_diag,
                    beta=beta_diag,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            if self.channel_variant == "full_confusion":
                U_col = self._ig_full_confusion_batch(
                    r=r_cand_prior,
                    delta=confusion_rows,
                    rng=rng,
                )
                if available_mask is not None:
                    U_col = np.where(available_mask[:, j_a], U_col, np.nan)
                U[:, j_a] = U_col
                continue

            raise RuntimeError(
                "Unhandled channel variant branch in fast-path computation."
            )

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
        if r.ndim != 2:
            raise ValueError(
                f"r must have shape (n_samples, n_classes), got {r.shape}."
            )
        if self.class_prior == "classifier":
            return r
        K = r.shape[1]
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        if self.class_prior == "uniform":
            return np.full_like(r, 1.0 / K, dtype=float)

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

        X_cand_emb = np.asarray(X_cand_emb, dtype=float)
        X_obs_cls_emb = np.asarray(X_obs_cls_emb, dtype=float)
        r_obs_cls = np.asarray(r_obs_cls, dtype=float)
        if X_cand_emb.ndim != 2 or X_obs_cls_emb.ndim != 2:
            raise ValueError("X_cand_emb and X_obs_cls_emb must be 2D.")
        if r_obs_cls.ndim != 2 or r_obs_cls.shape[1] != K:
            raise ValueError(
                f"r_obs_cls must have shape (n_obs_samples, {K}), got {r_obs_cls.shape}."
            )
        if X_obs_cls_emb.shape[0] != r_obs_cls.shape[0]:
            raise ValueError(
                "X_obs_cls_emb and r_obs_cls must have the same number of rows."
            )
        if X_cand_emb.shape[0] != r.shape[0]:
            raise ValueError(
                "X_cand_emb must have the same number of rows as r."
            )

        K_cls_local = rbf_kernel(
            X_obs_cls_emb, X_cand_emb, gamma=float(gamma_x)
        )
        K_cls = self._mix_with_global_sample_kernel(
            K_cls_local,
            lam=self.class_prior_lambda,
        )
        alpha0 = np.full(K, self.class_prior_strength / K, dtype=float)
        alpha, _ = self.parzen_dirichlet_posterior(
            K=K_cls,
            Y=r_obs_cls,
            gamma0=alpha0,
            use_ess=self.use_ess_class_prior,
            tau=self.tau_class_prior,
        )

        if self.sample_class_prior:
            if rng is None:
                raise ValueError(
                    "sample_class_prior=True requires an RNG."
                )
            alpha_bt = np.clip(alpha, 1e-12, None)
            X = rng.gamma(shape=alpha_bt, scale=1.0)
            return X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)

        return alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)

    # -------------------------
    # Gain computation
    # -------------------------
    def _ig_channel_full_batch(
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

        S, K = r.shape
        thetas = self._sample_theta_batch(alpha=alpha, beta=beta, rng=rng)
        T = thetas.shape[1]

        if self.channel_wrong_label_mode == "sample_dirichlet_wrong":
            Cs = self._channel_confusion_from_wrong_dirichlet_batch(
                gamma=gamma,
                theta=thetas,
                rng=rng,
                sample=self.sample_label_dirichlet,
            )
            r_rep = np.repeat(r[:, None, :], T, axis=1)
            ig_draws = self._pair_gain(
                r_rep,
                C=Cs,
            )
            return ig_draws.mean(axis=1)

        if self.sample_label_dirichlet:
            g_alpha = np.clip(gamma[:, None, :], 1e-12, None)
            if T != 1:
                g_alpha = np.repeat(g_alpha, T, axis=1)
            g = rng.gamma(shape=g_alpha, scale=1.0)
            g = g / np.maximum(g.sum(axis=-1, keepdims=True), 1e-12)
        else:
            g_mean = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-12)
            g = np.repeat(g_mean[:, None, :], T, axis=1)

        r_rep = np.repeat(r[:, None, :], T, axis=1)
        ig_draws = self._pair_gain(
            r_rep.reshape(-1, K),
            P_perf=thetas.reshape(-1, 1),
            P_annot=g.reshape(-1, 1, K),
        ).reshape(S, T)
        return ig_draws.mean(axis=1)

    def _ig_channel_topm_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        top_m: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        gamma = np.asarray(gamma, dtype=float)

        r_red, gamma_red = self._reduce_topm_vectors_batch(
            r=r, gamma=gamma, top_m=top_m
        )

        thetas = self._sample_theta_batch(alpha=alpha, beta=beta, rng=rng)
        T = thetas.shape[1]

        if self.channel_wrong_label_mode == "sample_dirichlet_wrong":
            Cs = self._channel_confusion_from_wrong_dirichlet_batch(
                gamma=gamma_red,
                theta=thetas,
                rng=rng,
                sample=self.sample_label_dirichlet,
            )
            r_rep = np.repeat(r_red[:, None, :], T, axis=1)
            ig_draws = self._pair_gain(
                r_rep,
                C=Cs,
            )
            return ig_draws.mean(axis=1)

        if self.sample_label_dirichlet:
            g_alpha = np.clip(gamma_red[:, None, :], 1e-12, None)
            if T != 1:
                g_alpha = np.repeat(g_alpha, T, axis=1)
            g_red = rng.gamma(shape=g_alpha, scale=1.0)
            g_red = g_red / np.maximum(g_red.sum(axis=-1, keepdims=True), 1e-12)
        else:
            g_mean_red = gamma_red / np.maximum(
                gamma_red.sum(axis=1, keepdims=True), 1e-12
            )
            g_red = np.repeat(g_mean_red[:, None, :], T, axis=1)

        S, K_red = r_red.shape
        r_rep = np.repeat(r_red[:, None, :], T, axis=1)
        ig_draws = self._pair_gain(
            r_rep.reshape(-1, K_red),
            P_perf=thetas.reshape(-1, 1),
            P_annot=g_red.reshape(-1, 1, K_red),
        ).reshape(S, T)
        return ig_draws.mean(axis=1)

    def _ig_scalar_uniform_confusion_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        S, K = r.shape
        thetas = self._sample_theta_batch(alpha=alpha, beta=beta, rng=rng)
        T = thetas.shape[1]

        eye = np.eye(K, dtype=float)[None, None, :, :]
        off_base = (
            (np.ones((K, K), dtype=float) - np.eye(K, dtype=float)) / (K - 1)
        )[None, None, :, :]
        Cs = (1.0 - thetas)[..., None, None] * off_base + thetas[
            ..., None, None
        ] * eye

        r_rep = np.repeat(r[:, None, :], T, axis=1)
        ig_draws = self._pair_gain(
            r_rep,
            C=Cs,
        )
        return ig_draws.mean(axis=1)

    def _ig_diag_uniform_confusion_batch(
        self,
        *,
        r: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)

        S, K = r.shape
        if self.n_theta_samples <= 0:
            thetas = self._theta_ucb_point_estimate(
                alpha=alpha,
                beta=beta,
            )[:, None, :]
        else:
            thetas = rng.beta(
                alpha[:, None, :],
                beta[:, None, :],
                size=(S, self.n_theta_samples, K),
            ).astype(float)

        T = thetas.shape[1]
        off = (1.0 - thetas) / (K - 1)
        Cs = np.repeat(off[..., None], K, axis=-1)
        idx = np.arange(K)
        Cs[..., idx, idx] = thetas

        r_rep = np.repeat(r[:, None, :], T, axis=1)
        ig_draws = self._pair_gain(
            r_rep,
            C=Cs,
        )
        return ig_draws.mean(axis=1)

    def _ig_full_confusion_batch(
        self,
        *,
        r: np.ndarray,
        delta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        delta = np.asarray(delta, dtype=float)

        if delta.ndim != 3 or delta.shape[1] != delta.shape[2]:
            raise ValueError(
                "delta must have shape (n_samples, K, K) in batch full_confusion."
            )

        if not self.sample_label_dirichlet:
            C_mean = delta / np.maximum(delta.sum(axis=2, keepdims=True), 1e-12)
            Cs = C_mean[:, None, :, :]
        else:
            T = max(self.n_theta_samples, 1)
            alpha = np.clip(delta[:, None, :, :], 1e-12, None)
            if T != 1:
                alpha = np.repeat(alpha, T, axis=1)
            X = rng.gamma(shape=alpha, scale=1.0)
            Cs = X / np.maximum(X.sum(axis=3, keepdims=True), 1e-12)

        T = Cs.shape[1]
        r_rep = np.repeat(r[:, None, :], T, axis=1)
        ig_draws = self._pair_gain(
            r_rep,
            C=Cs,
        )
        return ig_draws.mean(axis=1)

    def _pair_gain(
        self,
        P: np.ndarray,
        *,
        P_perf: np.ndarray | None = None,
        P_annot: np.ndarray | None = None,
        C: np.ndarray | None = None,
    ) -> np.ndarray:
        return expected_score_gain(
            P,
            P_perf=P_perf,
            P_annot=P_annot,
            C=C,
            score=self.gain_type,
            normalize=True,
            check_input=False,
        )

    def _sample_theta_batch(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        if self.n_theta_samples <= 0:
            return self._theta_ucb_point_estimate(
                alpha=alpha,
                beta=beta,
            )[:, None]
        return rng.beta(
            alpha[:, None],
            beta[:, None],
            size=(alpha.shape[0], self.n_theta_samples),
        ).astype(float)

    def _theta_ucb_point_estimate(
        self,
        *,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        alpha = np.asarray(alpha, dtype=float)
        beta = np.asarray(beta, dtype=float)
        denom = np.maximum(alpha + beta, 1e-12)
        mean = alpha / denom
        if self.theta_ucb_quantile is None:
            return mean

        theta_q = beta_distribution.ppf(
            self.theta_ucb_quantile,
            np.clip(alpha, 1e-12, None),
            np.clip(beta, 1e-12, None),
        )
        return np.clip(theta_q, 0.0, 1.0)

    @staticmethod
    def _reduce_topm_vectors_batch(
        *, r: np.ndarray, gamma: np.ndarray, top_m: int, eps: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(r, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        if r.ndim != 2 or gamma.ndim != 2 or r.shape != gamma.shape:
            raise ValueError(
                "r and gamma must be 2D with identical shape (n_samples, n_classes)."
            )
        S, K = r.shape
        if K < 2:
            raise ValueError("IG requires at least 2 classes.")
        if not (1 <= top_m < K):
            raise ValueError("top_m must satisfy 1 <= top_m < n_classes.")

        idx_part = np.argpartition(-r, kth=top_m - 1, axis=1)[:, :top_m]
        r_top_part = np.take_along_axis(r, idx_part, axis=1)
        order = np.argsort(-r_top_part, axis=1)
        idx = np.take_along_axis(idx_part, order, axis=1)

        r_top = np.take_along_axis(r, idx, axis=1)
        r_other = np.maximum(1.0 - r_top.sum(axis=1), 0.0)
        r_red = np.concatenate([r_top, r_other[:, None]], axis=1)
        r_red = np.clip(r_red, eps, 1.0)
        r_red = r_red / np.maximum(r_red.sum(axis=1, keepdims=True), eps)

        gamma_top = np.take_along_axis(gamma, idx, axis=1)
        gamma_other = np.maximum(
            gamma.sum(axis=1) - gamma_top.sum(axis=1), eps
        )
        gamma_red = np.concatenate([gamma_top, gamma_other[:, None]], axis=1)
        gamma_red = np.clip(gamma_red, eps, None)

        return r_red, gamma_red

    @staticmethod
    def _channel_confusion_from_wrong_dirichlet_batch(
        *,
        gamma: np.ndarray,
        theta: np.ndarray,
        rng: np.random.Generator,
        sample: bool,
        eps: float = 1e-12,
    ) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if gamma.ndim != 2:
            raise ValueError(
                f"gamma must have shape (n_samples, K), got {gamma.shape}."
            )
        if theta.ndim != 2 or theta.shape[0] != gamma.shape[0]:
            raise ValueError(
                f"theta must have shape (n_samples, n_draws), got {theta.shape}."
            )

        S, K = gamma.shape
        T = theta.shape[1]
        gamma = np.clip(gamma, eps, None)
        theta = np.clip(theta, 0.0, 1.0)

        C = np.zeros((S, T, K, K), dtype=float)
        idx = np.arange(K)
        C[..., idx, idx] = theta[:, :, None]
        off_scale = (1.0 - theta)[:, :, None]

        for z in range(K):
            off_idx = idx != z
            alpha = gamma[:, off_idx]
            if sample:
                alpha_bt = alpha[:, None, :]
                if T != 1:
                    alpha_bt = np.repeat(alpha_bt, T, axis=1)
                x = rng.gamma(shape=alpha_bt, scale=1.0)
                off = x / np.maximum(x.sum(axis=-1, keepdims=True), eps)
            else:
                off = alpha / np.maximum(alpha.sum(axis=-1, keepdims=True), eps)
                off = off[:, None, :]
                if T != 1:
                    off = np.repeat(off, T, axis=1)
            C[:, :, z, off_idx] = off_scale * off

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

    # -------------------------
    # Gamma resolution
    # -------------------------
    @staticmethod
    def _mix_with_global_sample_kernel(
        K_local: np.ndarray,
        *,
        lam: float,
    ) -> np.ndarray:
        K_local = np.asarray(K_local, dtype=float)
        if K_local.ndim != 2:
            raise ValueError(
                f"K_local must be 2D, got shape {K_local.shape}."
            )
        if not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0, 1].")
        if lam == 0.0:
            return K_local
        if lam == 1.0:
            return np.ones_like(K_local, dtype=float)
        return lam + (1.0 - lam) * K_local

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
